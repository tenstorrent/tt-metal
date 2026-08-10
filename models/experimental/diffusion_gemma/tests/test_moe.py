# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoE unit tests: expert activations, prefill routing/dispatch, the ragged packer and group
ladder, and the concat-experts denoise MoE against a torch oracle on device."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.tt import concat_moe
from models.experimental.diffusion_gemma.tt import expert_operations
from models.experimental.diffusion_gemma.tt import prefill_moe as PM
from models.experimental.diffusion_gemma.tt import sparse_moe as SM


# --- expert activations ------------------------------------------------------------------------


def test_diffusion_gemma_gelu_uses_the_checkpoint_tanh_variant(monkeypatch):
    calls = []
    monkeypatch.setattr(
        expert_operations.ttnn,
        "gelu",
        lambda value, **kwargs: calls.append((value, kwargs)) or "activated",
    )

    assert expert_operations.apply_gelu("gate") == "activated"
    assert calls == [("gate", {"variant": expert_operations.ttnn.GeluVariant.Tanh})]


def test_legacy_geglu_releases_the_activation_without_editing_shared_gemma4(monkeypatch):
    """The no-context fallback must free its own temporary.

    DiffusionGemma used to get this free by adding ``activated.deallocate(True)`` to
    ``models/demos/gemma4/tt/experts/operations.py``. That shared edit was reverted on
    2026-07-30; this test is what keeps the release from going with it.
    """
    released = []

    class _Activated:
        def deallocate(self, force=False):
            released.append(force)

    activated = _Activated()
    monkeypatch.setattr(expert_operations.ttnn, "gelu", lambda value, **kwargs: activated)
    monkeypatch.setattr(expert_operations.ttnn, "mul", lambda a, b: "down_input")

    assert expert_operations._legacy_geglu_with_release("gate", "up") == "down_input"
    assert released == [True], "the fallback leaked the GELU activation"


def test_dense_expert_dispatch_is_context_local_and_resets(monkeypatch):
    monkeypatch.setattr(expert_operations, "_legacy_geglu_with_release", lambda gate, up: ("legacy", gate, up))
    monkeypatch.setattr(expert_operations, "apply_geglu", lambda gate, up: ("tanh", gate, up))

    assert expert_operations._contextual_geglu("g", "u") == ("legacy", "g", "u")
    with expert_operations.use_tanh_expert_activations(True):
        assert expert_operations._contextual_geglu("g", "u") == ("tanh", "g", "u")
    assert expert_operations._contextual_geglu("g", "u") == ("legacy", "g", "u")


# --- tuned prefill MoE: flags, geometry, dispatch ----------------------------------------------


def _model(
    *,
    hidden_size=2816,
    intermediate_size=192,
    moe_intermediate_size=704,
    num_experts=128,
    top_k=8,
    dtype="bf16",
    arch="blackhole",
    mesh_shape=(1, 4),
    num_devices=4,
    grid=(11, 10),
    tp=4,
    ep=1,
    sp=1,
    tp_axis=1,
    mismatched_second_layer=False,
):
    def make_experts(layer_hidden_size):
        weight = SimpleNamespace(get_dtype=lambda: dtype)
        return SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=layer_hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
            ),
            weights=SimpleNamespace(
                intermediate_size_per_device=intermediate_size,
                gate_proj=weight,
                up_proj=weight,
                down_proj=weight,
            ),
        )

    mesh = SimpleNamespace(
        arch=lambda: arch,
        shape=mesh_shape,
        get_num_devices=lambda: num_devices,
        compute_with_storage_grid_size=lambda: SimpleNamespace(x=grid[0], y=grid[1]),
    )
    mesh_config = SimpleNamespace(
        mesh_shape=mesh_shape,
        tp_axis=tp_axis,
        prefill=SimpleNamespace(tp=tp, ep=ep, sp=sp),
    )
    layer_hidden_sizes = [hidden_size, 2048 if mismatched_second_layer else hidden_size]
    return SimpleNamespace(
        mesh_device=mesh,
        mesh_config=mesh_config,
        layers=[
            SimpleNamespace(moe=SimpleNamespace(experts=make_experts(layer_hidden_size)))
            for layer_hidden_size in layer_hidden_sizes
        ],
    )


@pytest.fixture
def fake_ttnn(monkeypatch):
    fake = SimpleNamespace(
        TILE_SIZE=32,
        bfloat16="bf16",
        bfloat8_b="bfp8",
        device=SimpleNamespace(Arch=SimpleNamespace(BLACKHOLE="blackhole")),
        CoreCoord=lambda x, y: (x, y),
        MatmulMultiCoreReuseMultiCast1DProgramConfig=lambda **kwargs: kwargs,
    )
    monkeypatch.setattr(PM, "ttnn", fake)
    return fake


@pytest.fixture
def contextual_builder(monkeypatch, fake_ttnn):
    def original(m, n, in0_block_w=1):
        return ("original", m, n, in0_block_w)

    monkeypatch.setattr(PM, "_original_builder", original)
    monkeypatch.setattr(PM.gemma4_prefill, "_build_sparse_matmul_config", PM._contextual_config_builder)
    return PM._contextual_config_builder


@pytest.mark.parametrize(
    "flag,enabled",
    [
        pytest.param(PM.FLAG, PM.tuned_prefill_moe_enabled, id="tuned"),
        pytest.param(PM.RAGGED_FLAG, PM.ragged_prefill_moe_enabled, id="ragged"),
        pytest.param(PM.RAGGED_LONG_FLAG, PM.ragged_long_prefill_enabled, id="ragged_long"),
    ],
)
def test_prefill_moe_flags_default_on_and_can_be_disabled(monkeypatch, flag, enabled):
    monkeypatch.delenv(flag, raising=False)
    assert enabled()
    monkeypatch.setenv(flag, "0")
    assert not enabled()
    monkeypatch.setenv(flag, "1")
    assert enabled()


def test_tuned_prefill_moe_uses_measured_qb2_geometry(monkeypatch, contextual_builder):
    monkeypatch.setenv(PM.FLAG, "1")

    with PM.use_tuned_prefill_moe(_model()):
        builder = PM.gemma4_prefill._build_sparse_matmul_config
        gate = builder(32, 192)
        down = builder(32, 2816)
        fallback = builder(64, 192, 7)

    assert gate["compute_with_storage_grid_size"] == (6, 1)
    assert gate["in0_block_w"] == 44
    assert gate["per_core_N"] == 1
    assert down["compute_with_storage_grid_size"] == (11, 4)
    assert down["in0_block_w"] == 3
    assert down["per_core_N"] == 2
    assert fallback == ("original", 64, 192, 7)
    assert PM.gemma4_prefill._build_sparse_matmul_config is contextual_builder
    assert contextual_builder(32, 192) == ("original", 32, 192, 1)


@pytest.mark.parametrize(
    "model",
    [
        _model(hidden_size=2048),
        _model(intermediate_size=256),
        _model(moe_intermediate_size=768),
        _model(num_experts=64),
        _model(top_k=4),
        _model(dtype="bfp8"),
        _model(arch="wormhole"),
        _model(mesh_shape=(2, 2)),
        _model(num_devices=8),
        _model(tp=2),
        _model(ep=2),
        _model(sp=2),
        _model(tp_axis=0),
        _model(grid=(8, 8)),
        _model(mismatched_second_layer=True),
        SimpleNamespace(layers=[]),
    ],
)
def test_tuned_prefill_moe_leaves_unsupported_models_unchanged(monkeypatch, contextual_builder, model):
    monkeypatch.setenv(PM.FLAG, "1")

    with PM.use_tuned_prefill_moe(model):
        assert contextual_builder(32, 192) == ("original", 32, 192, 1)


def test_tuned_prefill_moe_requires_measured_chunk_size(monkeypatch, contextual_builder):
    monkeypatch.setenv(PM.FLAG, "1")
    monkeypatch.setattr(PM.gemma4_prefill, "PREFILL_CHUNK_SIZE", 64)

    with PM.use_tuned_prefill_moe(_model()):
        assert contextual_builder(32, 192) == ("original", 32, 192, 1)


def test_tuned_prefill_moe_resets_context_after_error(monkeypatch, contextual_builder, expect_error):
    monkeypatch.setenv(PM.FLAG, "1")

    with expect_error(RuntimeError, match="stop"):
        with PM.use_tuned_prefill_moe(_model()):
            raise RuntimeError("stop")
    assert contextual_builder(32, 192) == ("original", 32, 192, 1)


def test_tuned_prefill_moe_does_not_leak_across_threads(monkeypatch, contextual_builder):
    monkeypatch.setenv(PM.FLAG, "1")
    entered = Barrier(2)
    completed = Barrier(2)

    def tuned_call():
        with PM.use_tuned_prefill_moe(_model()):
            entered.wait()
            result = contextual_builder(32, 192)
            completed.wait()
            return result

    def stock_call():
        entered.wait()
        result = contextual_builder(32, 192)
        completed.wait()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuned = executor.submit(tuned_call)
        stock = executor.submit(stock_call)

    assert tuned.result()["compute_with_storage_grid_size"] == (6, 1)
    assert stock.result() == ("original", 32, 192, 1)


def test_one_tile_causal_sdpa_fallback_is_diffusion_gemma_context_local(monkeypatch):
    calls = []
    monkeypatch.setattr(
        PM,
        "_original_sdpa",
        lambda query, key, value, *args, **kwargs: (
            calls.append(("stock", query.shape[-2], kwargs.get("is_causal"), kwargs.get("attn_mask"))) or "stock"
        ),
    )
    monkeypatch.setattr(
        PM,
        "_manual_one_tile_causal_attention",
        lambda query, key, value: calls.append(("manual", query.shape[-2])) or "manual",
    )
    monkeypatch.setattr(
        PM,
        "_manual_chunked_causal_attention",
        lambda query, key, value, sliding_window: calls.append(("manual-chunked", query.shape[-2], sliding_window))
        or "manual-chunked",
    )
    one_tile = SimpleNamespace(shape=(1, 4, ttnn.TILE_SIZE, 256))
    two_tiles = SimpleNamespace(shape=(1, 4, 2 * ttnn.TILE_SIZE, 256))

    assert PM._contextual_sdpa(one_tile, one_tile, one_tile, is_causal=True, scale=1.0) == "stock"
    token = PM._dg_ccl_active.set(True)
    try:
        assert PM._contextual_sdpa(one_tile, one_tile, one_tile, is_causal=True, scale=1.0) == "manual"
        assert (
            PM._contextual_sdpa(
                two_tiles,
                two_tiles,
                two_tiles,
                is_causal=True,
                scale=1.0,
                sliding_window_size=1024,
            )
            == "manual-chunked"
        )
        assert PM._contextual_sdpa(one_tile, one_tile, one_tile, is_causal=False, scale=1.0) == "stock"
    finally:
        PM._dg_ccl_active.reset(token)

    assert calls == [
        ("stock", 32, True, None),
        ("manual", 32),
        ("manual-chunked", 64, 1024),
        ("stock", 32, False, None),
    ]


def test_use_ragged_for_window_follows_long_flag(monkeypatch):
    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "0")
    # Long off: original 1 < S <= RAGGED_PREFILL_CHUNK window.
    assert not PM._use_ragged_for(1)
    assert PM._use_ragged_for(2)
    assert PM._use_ragged_for(PM.RAGGED_PREFILL_CHUNK)
    assert not PM._use_ragged_for(PM.RAGGED_PREFILL_CHUNK + 32)

    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "1")
    # Long on: any multi-token prefill is ragged (chunked wrapper handles S beyond one chunk).
    assert not PM._use_ragged_for(1)
    assert PM._use_ragged_for(2)
    assert PM._use_ragged_for(PM.RAGGED_PREFILL_CHUNK)
    assert PM._use_ragged_for(PM.RAGGED_PREFILL_CHUNK * 4 + 32)


def test_prefill_dispatch_routes_long_prompts_to_chunked(monkeypatch, fake_ttnn):
    monkeypatch.setenv(PM.FLAG, "0")
    monkeypatch.setenv(PM.RAGGED_FLAG, "1")
    monkeypatch.setattr(PM, "_original_prefill_forward", lambda *a, **k: "dense")
    monkeypatch.setattr(PM, "ragged_sparse_prefill_forward", lambda *a, **k: "ragged")
    monkeypatch.setattr(PM, "chunked_ragged_sparse_prefill_forward", lambda *a, **k: "chunked")

    def hidden(seq_len):
        return SimpleNamespace(shape=(1, 1, seq_len, 2816))

    over = PM.RAGGED_PREFILL_CHUNK * 4  # 16384 — the cliff case

    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "0")
    with PM.use_tuned_prefill_moe(_model()):
        # Long off: <= chunk -> ragged, > chunk -> shared dense (the pre-extension behavior).
        assert PM._contextual_prefill_forward(hidden_states=hidden(128)) == "ragged"
        assert PM._contextual_prefill_forward(hidden_states=hidden(PM.RAGGED_PREFILL_CHUNK)) == "ragged"
        assert PM._contextual_prefill_forward(hidden_states=hidden(over)) == "dense"
        assert PM._contextual_prefill_forward(hidden_states=hidden(1)) == "dense"

    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "1")
    with PM.use_tuned_prefill_moe(_model()):
        # Long on: every multi-token prefill flows through the chunked wrapper.
        assert PM._contextual_prefill_forward(hidden_states=hidden(128)) == "chunked"
        assert PM._contextual_prefill_forward(hidden_states=hidden(over)) == "chunked"
        assert PM._contextual_prefill_forward(hidden_states=hidden(1)) == "dense"

    # Context-local: outside the prefill context, always the shared dense path.
    assert PM._contextual_prefill_forward(hidden_states=hidden(over)) == "dense"


def test_router_dispatch_moves_with_prefill_gate(monkeypatch, fake_ttnn):
    # The ragged router emits a RaggedRouting only the ragged prefill can consume, so the router
    # and prefill gates MUST open on the same window.
    monkeypatch.setenv(PM.RAGGED_FLAG, "1")
    monkeypatch.setattr(PM, "_original_router_forward", lambda router, hs: "dense")
    monkeypatch.setattr(PM, "ragged_router_forward", lambda router, hs: "ragged")
    router = object()

    def hidden(seq_len):
        return SimpleNamespace(shape=(1, 1, seq_len, 2816))

    over = PM.RAGGED_PREFILL_CHUNK * 4

    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "0")
    with PM.use_tuned_prefill_moe(_model()):
        assert PM._contextual_router_forward(router, hidden(PM.RAGGED_PREFILL_CHUNK)) == "ragged"
        assert PM._contextual_router_forward(router, hidden(over)) == "dense"

    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "1")
    with PM.use_tuned_prefill_moe(_model()):
        assert PM._contextual_router_forward(router, hidden(over)) == "ragged"

    assert PM._contextual_router_forward(router, hidden(PM.RAGGED_PREFILL_CHUNK)) == "dense"


# --- ragged assignment packer ------------------------------------------------------------------


def _assert_packer_zero_drop_bijection(expert_index, num_experts, max_m_blocks):
    """Exhaustively verify the ragged packer's zero-drop round-trip invariants.

    Every routed (token, k) pair must land on a UNIQUE packed row, that row must round-trip
    back to the token via ``slot_token`` and carry the BF16-1.0 valid bit, and the row must
    fall inside exactly one m-block group whose ``group_experts`` entry matches the routed
    expert. Returns the raw packer outputs for extra per-test asserts.
    """
    sequence_length, top_k = expert_index.shape
    (
        slot_token,
        slot_valid,
        token_slot,
        group_counts,
        group_experts,
        group_start,
    ) = SM._pack_ragged_assignments(expert_index, num_experts, max_m_blocks)

    # Every routed (token, k) maps to a distinct packed row -> zero-drop bijection.
    assert token_slot.shape == (sequence_length, top_k)
    assert len(np.unique(token_slot)) == expert_index.size

    for token in range(sequence_length):
        for k_index in range(top_k):
            packed_row = token_slot[token, k_index]
            assert slot_token[packed_row] == token
            assert slot_valid[packed_row] == 0x3F80  # BF16 1.0 valid bit
            for m_blocks in range(1, max_m_blocks + 1):
                start = group_start[m_blocks - 1]
                end = start + group_counts[m_blocks - 1] * m_blocks * SM.TILE
                if start <= packed_row < end:
                    local_group = (packed_row - start) // (m_blocks * SM.TILE)
                    assert group_experts[m_blocks - 1, local_group] == expert_index[token, k_index]
                    break
            else:
                raise AssertionError(f"packed row {packed_row} was not assigned to a group")

    return slot_token, slot_valid, token_slot, group_counts, group_experts, group_start


def test_ragged_packer_multi_segment_and_multi_group():
    if SM._pack_ragged_assignments is None:
        pytest.skip("Numba acceleration is optional")
    max_m_blocks = 4
    capacity_rows = max_m_blocks * SM.TILE  # 128
    num_experts = 8
    sequence_length = 200
    top_k = 2

    # Column 0 -> expert 0 for EVERY token: 200 assignments > capacity_rows (128), so expert 0
    # splits across 2 segments: segment 0 = 128 rows (4 m-blocks), segment 1 = 72 rows (3 m-blocks).
    # Column 1 cycles experts 1..5, 40 assignments each -> 2 m-blocks each. So the run touches the
    # m_blocks = 2, 3 and 4 groups (>= 2 distinct groups) and expert 0 is multi-segment.
    expert_index = np.empty((sequence_length, top_k), dtype=np.int64)
    expert_index[:, 0] = 0
    expert_index[:, 1] = 1 + (np.arange(sequence_length) % 5)

    (
        slot_token,
        slot_valid,
        token_slot,
        group_counts,
        group_experts,
        group_start,
    ) = _assert_packer_zero_drop_bijection(expert_index, num_experts, max_m_blocks)

    # Sanity-check the intended geometry actually materialized.
    assert group_counts[3] == 1  # one 4-m-block segment (expert 0, segment 0: 128 rows)
    assert group_counts[2] == 1  # one 3-m-block segment (expert 0, segment 1: 72 rows)
    assert group_counts[1] == 5  # five 2-m-block segments (experts 1..5: 40 rows each)
    # expert 0 owns a 4-block AND a 3-block segment -> it genuinely spans >= 2 segments/groups.
    assert group_experts[3, 0] == 0
    assert group_experts[2, 0] == 0
    assert set(group_experts[1, :5].tolist()) == {1, 2, 3, 4, 5}


def test_ragged_packer_numba_matches_torch_fallback():
    # torch-vs-numba equivalence is asserted here at the ROUND-TRIP-SEMANTICS level, not by
    # comparing internal layouts. The two implementations pack in different orders (the numba
    # path counts per expert; the torch fallback in SM._ragged_metadata_host sorts assignments),
    # but both MUST agree on the observable contract: a zero-drop bijection token -> packed row ->
    # token with per-(token,k) valid bits and correct group membership. The torch fallback is
    # embedded in _ragged_metadata_host and only reachable with an on-device routing tensor, so we
    # instead assert the numba packer's OUTPUT invariants exhaustively on a randomized medium case
    # (the same invariants the torch fallback is constructed to satisfy).
    if SM._pack_ragged_assignments is None:
        pytest.skip("Numba acceleration is optional")
    num_experts = 128
    top_k = 8
    max_m_blocks = 4
    rng = np.random.default_rng(0)
    sequence_length = int(rng.integers(64, 513))  # S in [64, 512]

    # Distinct experts per token, mirroring the router's top-k contract.
    expert_index = np.empty((sequence_length, top_k), dtype=np.int64)
    for token in range(sequence_length):
        expert_index[token] = rng.choice(num_experts, size=top_k, replace=False)

    _assert_packer_zero_drop_bijection(expert_index, num_experts, max_m_blocks)


def test_ragged_packer_single_expert_all_tokens():
    if SM._pack_ragged_assignments is None:
        pytest.skip("Numba acceleration is optional")
    num_experts = 128
    top_k = 4
    max_m_blocks = 4
    capacity_rows = max_m_blocks * SM.TILE  # 128
    sequence_length = 150  # > capacity_rows so the hot expert must span >= 2 segments

    # Every token routes all top_k to DISTINCT experts, but expert 0 is in every token (hot).
    # Expert 0 therefore collects `sequence_length` assignments -> 2 segments (128 + 22 rows).
    expert_index = np.empty((sequence_length, top_k), dtype=np.int64)
    expert_index[:, 0] = 0
    for token in range(sequence_length):
        # 1..127 consecutive-offset experts are distinct within a token and never collide with 0.
        expert_index[token, 1:] = 1 + (token * 3 + np.arange(top_k - 1)) % 127

    (
        slot_token,
        slot_valid,
        token_slot,
        group_counts,
        group_experts,
        group_start,
    ) = _assert_packer_zero_drop_bijection(expert_index, num_experts, max_m_blocks)

    # expert 0 (150 assignments) splits into a 4-m-block segment (128 rows) and a
    # 1-m-block segment (22 rows -> ceil(22/32) = 1 block).
    assert 0 in group_experts[3, : group_counts[3]].tolist()
    assert 0 in group_experts[0, : group_counts[0]].tolist()


# --- ragged group ladder -----------------------------------------------------------------------
#
# Each group goes to ``ttnn.sparse_matmul`` as [1, group_size, m_blocks*TILE, H] with nnz=group_size,
# and ``group_size`` is the number of expert-segments that happen to have that m_blocks —
# routing-dependent, so a new prompt is a new program built ON THE HOST. Measured on QB2 2026-07-31
# with DG_PREFILL_CPU_PROBE: 4.0-7.98 s per prefill at cache_len 128 and 5.0-17.4 s at 2048,
# thread_cpu_frac 0.947-0.982, with py-spy landing in ``ragged_sparse_prefill_forward``'s
# sparse_matmul.
#
# Host-only. ``_ragged_metadata_host`` consumes a ttnn tensor, so these build the same unpadded group
# layout it produces and exercise ``_quantize_ragged_groups`` directly. What must hold:
#   * every (token, k) still resolves to a row holding that token, marked valid, on the same expert;
#   * padded rows are inert (slot_token 0, slot_valid 0);
#   * sparsity keeps exactly one non-zero per group row — nnz != count_nonzero(sparsity) HANGS the
#     kernel on device (matmul_nanobind.cpp:1053), and nnz is passed as the padded group_size;
#   * group sizes land on the ladder and the shape space collapses across different routings;
#   * packed_rows equals the padded total, since it sizes the concat the gather indexes into.

_LADDER_E = 128


def _build_groups(layout, seed=0):
    """Build an unpadded (groups, token_slot, packed_rows) triple.

    ``layout`` is [(m_blocks, [rows_used_per_segment, ...]), ...] in ascending m_blocks, matching
    the real builder: segments are expert-homogeneous, padded to a tile multiple inside the
    segment, and groups are concatenated in m_blocks order.
    """
    generator = torch.Generator().manual_seed(seed)
    groups, assignments = [], {}
    offset, token = 0, 0
    for m_blocks, segment_rows in layout:
        group_size = len(segment_rows)
        rows_per_segment = m_blocks * SM.TILE
        total_rows = group_size * rows_per_segment
        slot_token = torch.zeros(total_rows, dtype=torch.int32)
        slot_valid = torch.zeros((total_rows, 1), dtype=torch.bfloat16)
        experts = torch.randperm(_LADDER_E, generator=generator)[:group_size]
        sparsity = torch.zeros((1, 1, group_size, _LADDER_E), dtype=torch.bfloat16)
        sparsity[0, 0, torch.arange(group_size), experts] = 1
        for segment, used in enumerate(segment_rows):
            assert used <= rows_per_segment
            for row in range(used):
                absolute = offset + segment * rows_per_segment + row
                slot_token[segment * rows_per_segment + row] = token
                slot_valid[segment * rows_per_segment + row] = 1
                assignments[token] = (absolute, int(experts[segment]))
                token += 1
        groups.append((m_blocks, group_size, slot_token, slot_valid, sparsity))
        offset += total_rows
    token_slot = torch.zeros((token, 1), dtype=torch.int32)
    for tok, (absolute, _) in assignments.items():
        token_slot[tok, 0] = absolute
    return groups, token_slot, offset, assignments


def _resolve(groups, token_slot, num_tokens):
    """Reconstruct {token -> expert} through the packed layout, asserting row identity."""
    slot_token = torch.cat([g[2].reshape(-1) for g in groups])
    slot_valid = torch.cat([g[3].reshape(-1) for g in groups])
    expert_of_row = torch.cat([g[4][0, 0].argmax(dim=-1).repeat_interleave(g[0] * SM.TILE) for g in groups])
    out = {}
    for tok in range(num_tokens):
        row = int(token_slot[tok, 0])
        assert int(slot_token[row]) == tok, f"row {row} holds {int(slot_token[row])}, not token {tok}"
        assert float(slot_valid[row]) == 1.0, f"row {row} for token {tok} is not valid"
        out[tok] = int(expert_of_row[row])
    return out


_LAYOUTS = [
    [(1, [32, 17, 3])],
    [(1, [32] * 5), (2, [64, 40])],
    [(1, [1]), (2, [33, 64, 12]), (3, [96] * 9), (4, [128, 5])],
]


def test_ladder_rounds_up_and_is_monotone():
    for n in range(1, 600):
        step = SM._ladder_group_size(n)
        assert step >= n
        assert step in SM._GROUP_LADDER or step > SM._GROUP_LADDER[-1]
    assert SM._ladder_group_size(1) == 1
    assert SM._ladder_group_size(5) == 8
    assert SM._ladder_group_size(9) == 12
    assert SM._ladder_group_size(33) == 48
    # beyond the table it keeps doubling rather than raising
    assert SM._ladder_group_size(700) == 1024


def test_padding_preserves_every_real_assignment():
    for seed, layout in enumerate(_LAYOUTS):
        groups, token_slot, packed_rows, assignments = _build_groups(layout, seed=seed)
        num_tokens = len(assignments)
        before = _resolve(groups, token_slot, num_tokens)
        padded, padded_slot, padded_rows = SM._quantize_ragged_groups(groups, token_slot, packed_rows)
        assert _resolve(padded, padded_slot, num_tokens) == before
        assert padded_rows >= packed_rows


def test_padded_groups_are_on_ladder_inert_and_one_hot():
    for seed, layout in enumerate(_LAYOUTS):
        groups, token_slot, packed_rows, _ = _build_groups(layout, seed=seed)
        padded, _, padded_rows = SM._quantize_ragged_groups(groups, token_slot, packed_rows)
        total = 0
        for m_blocks, group_size, slot_token, slot_valid, sparsity in padded:
            assert 1 <= m_blocks <= SM.RAGGED_MAX_M_BLOCKS
            assert SM._ladder_group_size(group_size) == group_size, f"off-ladder group_size {group_size}"
            rows = group_size * m_blocks * SM.TILE
            assert slot_token.numel() == rows
            assert slot_valid.numel() == rows
            assert tuple(sparsity.shape) == (1, 1, group_size, _LADDER_E)
            # nnz is passed as group_size; a mismatch deadlocks the kernel on device.
            assert int((sparsity != 0).sum()) == group_size
            assert torch.equal((sparsity != 0).sum(dim=-1)[0, 0], torch.ones(group_size, dtype=torch.int64))
            invalid = slot_valid.reshape(-1) == 0
            assert torch.all(slot_token.reshape(-1)[invalid] == 0), "padded rows must gather row 0"
            total += rows
        assert padded_rows == total


def test_geometry_collapses_across_routings():
    """The point of the change: different routings must reuse one small set of shapes."""
    raw_shapes, padded_shapes = set(), set()
    for seed in range(12):
        generator = torch.Generator().manual_seed(seed)
        layout = [
            (
                m_blocks,
                [
                    int(x)
                    for x in torch.randint(
                        1,
                        m_blocks * SM.TILE,
                        (int(torch.randint(1, 20, (1,), generator=generator)),),
                        generator=generator,
                    )
                ],
            )
            for m_blocks in range(1, SM.RAGGED_MAX_M_BLOCKS + 1)
        ]
        groups, token_slot, packed_rows, _ = _build_groups(layout, seed=seed)
        raw_shapes.update((g[0], g[1]) for g in groups)
        padded, _, _ = SM._quantize_ragged_groups(groups, token_slot, packed_rows)
        padded_shapes.update((g[0], g[1]) for g in padded)
    assert len(padded_shapes) < len(raw_shapes), f"no collapse: {len(raw_shapes)} -> {len(padded_shapes)}"
    assert len(padded_shapes) <= SM.RAGGED_MAX_M_BLOCKS * len(SM._GROUP_LADDER)


# --- long-prompt chunked ragged prefill --------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in that records shape and deallocation for the chunk-loop plumbing test."""

    def __init__(self, shape, tag=""):
        self.shape = tuple(shape)
        self.tag = tag
        self.deallocated = False

    def deallocate(self, force=True):
        self.deallocated = True


def _install_fake_chunk_ops(monkeypatch):
    slice_calls = []
    concat_calls = []
    ragged_calls = []

    def fake_slice(tensor, start, end):
        result = _FakeTensor(tuple(e - s for s, e in zip(start, end)), tag="slice")
        slice_calls.append((start[2], end[2], result))
        return result

    def fake_concat(tensors, dim):
        concat_calls.append((list(tensors), dim))
        total = sum(t.shape[2] for t in tensors)
        return _FakeTensor((1, 1, total, tensors[0].shape[3]), tag="concat")

    def fake_ragged(hidden, routing, weights, config, sparsity, mesh_config=None, mesh_device=None, ccl_manager=None):
        ragged_calls.append(SimpleNamespace(seq=hidden.shape[2], routing=routing, mesh_config=mesh_config))
        return _FakeTensor((1, 1, hidden.shape[2], hidden.shape[3]), tag="out")

    monkeypatch.setattr(SM, "ttnn", SimpleNamespace(slice=fake_slice, concat=fake_concat))
    monkeypatch.setattr(SM, "ragged_sparse_prefill_forward", fake_ragged)
    return slice_calls, concat_calls, ragged_calls


def test_chunked_ragged_prefill_slices_tail_and_concats(monkeypatch):
    monkeypatch.setenv("DG_PREFILL_RAGGED_CHUNK", "64")
    slice_calls, concat_calls, ragged_calls = _install_fake_chunk_ops(monkeypatch)

    seq_len, hidden_size, top_k = 160, 2816, 8  # 64 + 64 + 32 (tail) -> 3 chunks
    hidden = _FakeTensor((1, 1, seq_len, hidden_size))
    scale = object()
    values = _FakeTensor((1, 1, seq_len, top_k))
    indices = _FakeTensor((1, 1, seq_len, top_k))
    routing = SM.RaggedRouting(values, indices, scale)
    mesh_cfg = SimpleNamespace(tp=4)

    out = SM.chunked_ragged_sparse_prefill_forward(
        hidden, routing, "weights", "config", "sparsity", mesh_config=mesh_cfg, mesh_device="mesh"
    )

    # One ragged call per chunk, with the exact per-chunk sequence lengths.
    assert [c.seq for c in ragged_calls] == [64, 64, 32]
    # Each chunk gets its own RaggedRouting slice sharing the (unsliced) per-expert scale, and the
    # per-chunk TP all-reduce is preserved (mesh_config threaded through unchanged).
    for call, expected in zip(ragged_calls, [64, 64, 32]):
        assert isinstance(call.routing, SM.RaggedRouting)
        assert call.routing.per_expert_scale is scale
        assert call.routing.values.shape[2] == expected
        assert call.mesh_config is mesh_cfg
    # Sequence sliced at chunk-aligned boundaries with a 32-row tail — for hidden + values + indices.
    hidden_ranges = [(s, e) for (s, e, r) in slice_calls if r.shape[3] == hidden_size]
    assert hidden_ranges == [(0, 64), (64, 128), (128, 160)]
    # Concatenated once on the token dim back to the full length.
    assert len(concat_calls) == 1 and concat_calls[0][1] == 2
    assert out.shape[2] == seq_len
    # Parent routing tensors are freed by the wrapper (their per-chunk slices are freed downstream).
    assert values.deallocated and indices.deallocated


def test_chunked_ragged_prefill_single_chunk_is_passthrough(monkeypatch):
    monkeypatch.setenv("DG_PREFILL_RAGGED_CHUNK", "4096")
    slice_calls, concat_calls, ragged_calls = _install_fake_chunk_ops(monkeypatch)

    hidden = _FakeTensor((1, 1, 2048, 2816))  # <= chunk -> single direct call, no slice/concat
    routing = SM.RaggedRouting(_FakeTensor((1, 1, 2048, 8)), _FakeTensor((1, 1, 2048, 8)), object())

    out = SM.chunked_ragged_sparse_prefill_forward(hidden, routing, "w", "c", "s", mesh_config=None)

    assert len(ragged_calls) == 1 and ragged_calls[0].seq == 2048
    assert slice_calls == [] and concat_calls == []
    assert out.shape[2] == 2048


def test_chunked_ragged_prefill_passes_dense_routing_straight_through(monkeypatch):
    # A non-RaggedRouting argument (e.g. a dense routing tensor) must delegate unchanged, never slice.
    monkeypatch.setenv("DG_PREFILL_RAGGED_CHUNK", "64")
    slice_calls, concat_calls, ragged_calls = _install_fake_chunk_ops(monkeypatch)

    hidden = _FakeTensor((1, 1, 160, 2816))  # > chunk, but routing is not RaggedRouting
    dense_routing = _FakeTensor((1, 1, 160, 128))

    out = SM.chunked_ragged_sparse_prefill_forward(hidden, dense_routing, "w", "c", "s")

    assert len(ragged_calls) == 1 and ragged_calls[0].seq == 160
    assert slice_calls == [] and concat_calls == []
    assert out.shape[2] == 160


# --- concat-experts denoise MoE, numerics on device --------------------------------------------
#
# The concat path (``tt/concat_moe.py``) is the only denoise MoE since 2026-07-29. It folds the
# routing weights into the GeGLU output so the down projection is one wide matmul:
#
#     out = (geglu(x @ gate_cat, x @ up_cat) * (routing @ expand)) @ down_cat
#
# instead of the per-expert form the reference and the retired token-gather path compute:
#
#     out = sum_e routing_e * (geglu(x @ W_gate_e, x @ W_up_e) @ W_down_e)
#
# Those are equal by linearity of the down projection, but "equal on paper" is not the claim that
# matters — the device path applies the routing weight to a bf16 GeGLU output *before* a single
# 24576-long reduction, where the per-expert form reduces 192 at a time and applies the routing
# weight afterwards. These tests measure what that costs, on device, against a torch oracle, and pin
# the two contracts the fold silently depends on:
#
# * **the routing tensor must be exactly zero for unselected experts** — the fold has no other way to
#   exclude them, so a router that returned a full softmax would leak all 128 experts into the output;
# * **the padded intermediate columns must contribute zero** — ``weights.py`` pads ``I/tp`` up to a
#   tile, and the concat matmul computes over the pad, so a nonzero pad would be summed in by
#   ``down_cat``.
#
# Run on QB2::
#
#     DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_moe.py -s

_requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device",
)
# one device open/teardown for the whole module — avoid QB2 erisc cycling
_module_device = pytest.mark.use_module_device

# Small but structurally faithful: E a multiple of 32, I and H tile-aligned, S = a real canvas row
# count. The shipped shape is E=128, H=2816, I_dev=192, S=256; this keeps the same ratios at 1/8 the
# expert count so the test runs in seconds.
_E, _H, _I, _S, _TOPK = 16, 256, 64, 256, 4


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _make_weights(seed=3):
    """Torch expert weights in the layout ``gemma4`` produces."""
    return SimpleNamespace(
        gate_proj=_rand(1, _E, _H, _I, seed=seed),  # [1,E,H,I] column-parallel
        up_proj=_rand(1, _E, _H, _I, seed=seed + 1),
        down_proj=_rand(1, _E, _I, _H, seed=seed + 2),  # [1,E,I,H] row-parallel
        intermediate_size_per_device=_I,
    )


def _make_routing(seed=9, zero_unselected=True):
    """Dense ``[1,1,S,E]`` routing, top-k masked — the contract ``concat_experts_forward`` assumes."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, 1, _S, _E, generator=g)
    probs = torch.softmax(logits, dim=-1)
    if not zero_unselected:
        return probs
    topk = torch.topk(probs, _TOPK, dim=-1).indices
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk, 1.0)
    return probs * mask


def _torch_oracle(x, w, routing):
    """Per-expert reference: ``sum_e routing_e * (geglu(x@gate_e, x@up_e) @ down_e)``.

    Shapes come from the weights, not from module constants, so a padded-intermediate variant works
    unchanged.
    """
    num_experts, hidden = w.gate_proj.shape[1], w.gate_proj.shape[2]
    seq = x.shape[-2]
    out = torch.zeros(1, 1, seq, hidden, dtype=torch.float32)
    xf = x.float().reshape(seq, hidden)
    for e in range(num_experts):
        gate = xf @ w.gate_proj[0, e].float()
        up = xf @ w.up_proj[0, e].float()
        # tanh GeLU — DiffusionGemma's configured variant (tt/expert_operations.py:apply_gelu)
        act = torch.nn.functional.gelu(gate, approximate="tanh") * up
        out[0, 0] += (act @ w.down_proj[0, e].float()) * routing[0, 0, :, e : e + 1].float()
    return out


def _to_dev(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _fake_experts(w, device):
    """Minimal stand-in for ``Gemma4Experts``: concat_experts_forward reads only these."""
    return SimpleNamespace(
        weights=SimpleNamespace(
            gate_proj=_to_dev(w.gate_proj, device),
            up_proj=_to_dev(w.up_proj, device),
            down_proj=_to_dev(w.down_proj, device),
            intermediate_size_per_device=int(w.gate_proj.shape[3]),
        ),
        mesh_config=None,  # single device -> no all-reduce, isolating the fold itself
        ccl_manager=None,
    )


def _run_concat(device, w, routing, x):
    experts = _fake_experts(w, device)
    tt_x = _to_dev(x, device)
    tt_routing = _to_dev(routing, device)
    try:
        out = concat_moe.concat_experts_forward(experts, tt_x, tt_routing)
        hidden, seq = w.gate_proj.shape[2], x.shape[-2]
        return ttnn.to_torch(out).float().reshape(1, 1, seq, hidden)
    finally:
        # Order matters: release the concat weights FIRST. ``down_cat`` is a view of
        # ``weights.down_proj``, so freeing the root first and then touching the view would read
        # DRAM the allocator has already reclaimed. ``ConcatExpertWeights.deallocate`` uses
        # ``deallocate(False)`` and so correctly skips the aliasing view.
        cached = getattr(experts, "_dg_concat_weights", None)
        if cached is not None:
            cached.deallocate()
        for t in (tt_x, tt_routing, experts.weights.gate_proj, experts.weights.up_proj, experts.weights.down_proj):
            t.deallocate(True)


@_requires_device
@_module_device
def test_concat_matches_per_expert_oracle(device):
    """The headline: does the fold reproduce the per-expert MoE on device?"""

    w = _make_weights()
    routing = _make_routing()
    x = _rand(1, 1, _S, _H, seed=21) * 0.1  # keep the GeGLU in a sane range for bf16

    got = _run_concat(device, w, routing, x)
    expected = _torch_oracle(x, w, routing)

    passing, pcc = comp_pcc(expected, got, 0.99)
    rel = ((got - expected).abs().max() / expected.abs().max()).item()
    print(f"[concat vs per-expert oracle] {pcc}  max_rel_err={rel:.4f}")
    assert passing, f"concat MoE disagrees with the per-expert oracle: {pcc}"


@_requires_device
@_module_device
def test_fold_requires_zero_for_unselected_experts(device):
    """Pin the contract: a router that does NOT zero unselected experts breaks the fold.

    This is not a bug report against the router — ``_denoise_router_forward`` does mask. It exists so
    that if anyone changes the router to return an unmasked distribution, this fails loudly here
    rather than silently degrading generation quality, which is the failure mode that would be
    hardest to attribute.
    """

    w = _make_weights()
    x = _rand(1, 1, _S, _H, seed=22) * 0.1
    masked = _make_routing(zero_unselected=True)
    unmasked = _make_routing(zero_unselected=False)

    got_masked = _run_concat(device, w, masked, x)
    got_unmasked = _run_concat(device, w, unmasked, x)

    # Against the SAME oracle (the masked, top-k one), the unmasked routing must be visibly wrong.
    expected = _torch_oracle(x, w, masked)
    _, pcc_masked = comp_pcc(expected, got_masked, 0.99)
    diff = (got_unmasked - got_masked).abs().max().item()
    print(f"[fold contract] masked {pcc_masked}  |unmasked - masked|_max={diff:.4f}")
    assert diff > 1e-3, (
        "unselected experts made no difference — either the router mask is being applied somewhere "
        "else, or this test is not exercising the fold"
    )


@_requires_device
@_module_device
def test_padded_intermediate_columns_contribute_zero(device):
    """``weights.py`` pads I/tp up to a tile; the concat matmul computes over the pad.

    Zero-pad the intermediate on BOTH gate/up and down and check the result is unchanged. If the pad
    ever carried garbage instead of zeros, ``down_cat`` would sum it into every token.
    """

    w = _make_weights()
    routing = _make_routing()
    x = _rand(1, 1, _S, _H, seed=23) * 0.1
    baseline = _run_concat(device, w, routing, x)

    pad = 32
    padded = SimpleNamespace(
        gate_proj=torch.nn.functional.pad(w.gate_proj, (0, pad)),  # [1,E,H,I+pad]
        up_proj=torch.nn.functional.pad(w.up_proj, (0, pad)),
        down_proj=torch.nn.functional.pad(w.down_proj, (0, 0, 0, pad)),  # [1,E,I+pad,H]
        intermediate_size_per_device=_I + pad,
    )
    got = _run_concat(device, padded, routing, x)

    _, pcc = comp_pcc(baseline, got, 0.999)
    print(f"[zero pad invariance] {pcc}")
    assert pcc, f"zero-padded intermediate changed the result: {pcc}"


@_requires_device
@_module_device
def test_down_concat_is_a_pure_reshape(device):
    """The memory budget rests on ``[1,E,I,H] -> [1,1,E*I,H]`` being free at bf16 TILE."""

    w = _make_weights()
    source = _to_dev(w.down_proj, device)
    try:
        info = concat_moe.verify_down_concat_is_free(SimpleNamespace(down_proj=source))
        print(f"[down concat] {info}")
        assert info["values_match"], f"down concat is not byte-order preserving: {info}"
    finally:
        source.deallocate(True)


@_requires_device
@_module_device
def test_deallocate_does_not_free_the_aliased_down_weights(device):
    """``down_cat`` is a view of ``weights.down_proj``; releasing it must not free the root.

    ``deallocate(True)`` bypasses the not-sole-owner guard and reaches the root holder, so a
    force-free here would release the live row-parallel down weights that prefill and the sparse
    path still read — and the crash would surface inside prefill, far from this module.
    """

    w = _make_weights()
    experts = _fake_experts(w, device)
    down = experts.weights.down_proj
    try:
        concat = concat_moe.concat_weights_for(experts)
        assert concat.down_cat.buffer_address() == down.buffer_address(), (
            "down_cat is no longer a view of down_proj — the 7.7 GiB memory budget in "
            "concat_moe.py assumes it is; re-derive it before changing this"
        )
        concat.deallocate()
        assert down.is_allocated(), "ConcatExpertWeights.deallocate freed the shared down weights"
        # And the root must still be readable, not merely flagged allocated.
        assert torch.isfinite(ttnn.to_torch(down).float()).all()
    finally:
        for t in (experts.weights.gate_proj, experts.weights.up_proj, down):
            t.deallocate(True)
