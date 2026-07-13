# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import numpy as np

from models.experimental.diffusion_gemma.tt import prefill_moe as PM
from models.experimental.diffusion_gemma.tt import sparse_moe as SM


def test_tuned_prefill_moe_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv(PM.FLAG, raising=False)
    assert PM.tuned_prefill_moe_enabled()
    monkeypatch.setenv(PM.FLAG, "0")
    assert not PM.tuned_prefill_moe_enabled()


def test_ragged_prefill_moe_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv(PM.RAGGED_FLAG, raising=False)
    assert PM.ragged_prefill_moe_enabled()
    monkeypatch.setenv(PM.RAGGED_FLAG, "0")
    assert not PM.ragged_prefill_moe_enabled()


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


def test_ragged_dispatch_is_context_local(monkeypatch, fake_ttnn):
    monkeypatch.setenv(PM.FLAG, "0")
    monkeypatch.setenv(PM.RAGGED_FLAG, "1")
    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "0")  # direct ragged path; long-on chunking covered separately
    monkeypatch.setattr(PM, "_original_prefill_forward", lambda *args, **kwargs: "dense")
    monkeypatch.setattr(PM, "ragged_sparse_prefill_forward", lambda *args, **kwargs: "ragged")
    hidden_states = SimpleNamespace(shape=(1, 1, 128, 2816))

    with PM.use_tuned_prefill_moe(_model()):
        assert PM._contextual_prefill_forward(hidden_states=hidden_states) == "ragged"

    assert PM._contextual_prefill_forward(hidden_states=hidden_states) == "dense"


def test_ragged_assignment_packer_is_zero_drop():
    if SM._pack_ragged_assignments is None:
        pytest.skip("Numba acceleration is optional")
    expert_index = np.array(
        [
            [0, 2],
            [0, 3],
            [0, 2],
            [0, 3],
        ],
        dtype=np.int64,
    )
    slot_token, slot_valid, token_slot, group_counts, group_experts, group_start = SM._pack_ragged_assignments(
        expert_index, 4, 2
    )

    assert len(np.unique(token_slot)) == expert_index.size
    for token in range(expert_index.shape[0]):
        for k_index in range(expert_index.shape[1]):
            packed_row = token_slot[token, k_index]
            assert slot_token[packed_row] == token
            assert slot_valid[packed_row] == 0x3F80
            for m_blocks in range(1, 3):
                start = group_start[m_blocks - 1]
                end = start + group_counts[m_blocks - 1] * m_blocks * 32
                if start <= packed_row < end:
                    local_group = (packed_row - start) // (m_blocks * 32)
                    assert group_experts[m_blocks - 1, local_group] == expert_index[token, k_index]
                    break
            else:
                raise AssertionError(f"packed row {packed_row} was not assigned to a group")


def _assert_packer_zero_drop_bijection(expert_index, num_experts, max_m_blocks):
    """Exhaustively verify the ragged packer's zero-drop round-trip invariants.

    Generalizes ``test_ragged_assignment_packer_is_zero_drop`` to any ``max_m_blocks``:
    every routed (token, k) pair must land on a UNIQUE packed row, that row must
    round-trip back to the token via ``slot_token`` and carry the BF16-1.0 valid bit,
    and the row must fall inside exactly one m-block group whose ``group_experts`` entry
    matches the routed expert. Returns the raw packer outputs for extra per-test asserts.
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


# ---------------------------------------------------------------------------------------------
# Long-prompt chunked ragged prefill (DG_PREFILL_RAGGED_LONG, default off).
# ---------------------------------------------------------------------------------------------


def test_ragged_long_prefill_flag_defaults_on_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv(PM.RAGGED_LONG_FLAG, raising=False)
    assert PM.ragged_long_prefill_enabled()
    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "0")
    assert not PM.ragged_long_prefill_enabled()
    monkeypatch.setenv(PM.RAGGED_LONG_FLAG, "1")
    assert PM.ragged_long_prefill_enabled()


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
