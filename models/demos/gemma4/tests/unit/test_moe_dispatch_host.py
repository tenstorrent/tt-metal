# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the dispatch-prefill chunk sizing (tt/experts/dispatch_prefill.py): with
GEMMA4_MOE_DISPATCH_FIT_CHUNK=1 a call shorter than the full chunk runs on a chunk of its own tile-aligned length
(a 128-token prompt on a 128-token chunk instead of 4096 padded rows). No device.

    pytest models/demos/gemma4/tests/unit/test_dispatch_fit_chunk_host.py --noconftest -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import compute_constants
from models.demos.gemma4.custom_ops.expert_gather_small import ExpertGatherSmall
from models.demos.gemma4.custom_ops.router_topk import RouterTopk
from models.demos.gemma4.tt.experts import dispatch_prefill as dp
from models.demos.gemma4.tt.experts import local_routed_prefill as lrp
from models.demos.gemma4.tt.experts import moe_decode as md
from models.demos.gemma4.tt.experts.dispatch_prefill import (
    FABRIC_DEFAULT_PAYLOAD_BYTES,
    FABRIC_MAX_PAYLOAD_BYTES_BLACKHOLE,
    RING_FWD_EXTRA_BYTES,
    Gemma4DispatchExperts,
    combine2d_extra_kwargs,
    combine_fwd_reduction,
    ring_fabric_payload_bytes,
    ring_fabric_router_config,
    ring_payload_required_bytes,
    ring_token_bytes,
    ring_transport_reason,
)
from models.demos.gemma4.tt.experts.local_routed_index import (
    aligned_layout,
    index_pipeline,
    local_expert_mask,
    local_routed_golden,
)
from models.demos.gemma4.tt.experts.moe_decode import kernel_decode_frame, moe_compute_impl, tile_tail_enabled
from models.demos.gemma4.tt.moe import MoEBlock


@pytest.fixture()
def _clean_env(monkeypatch):
    monkeypatch.delenv(dp.FIT_CHUNK_ENV, raising=False)


@pytest.mark.usefixtures("_clean_env")
def test_fit_chunk_default_follows_the_dispatch_default():
    from models.demos.gemma4.tt.mesh_orientation import fit_chunk_default

    assert fit_chunk_default((4, 1), blackhole=True)  # the QB2 column mesh: the validated configuration
    assert not fit_chunk_default((1, 4), blackhole=True)
    assert not fit_chunk_default((4, 1), blackhole=False)
    assert not fit_chunk_default((8, 4), blackhole=True)
    assert dp.fit_chunk_tokens(128, 4, 4096, fit=False) == 4096
    assert dp.fit_chunk_tokens(4096, 4, 4096, fit=True) == 4096


@pytest.mark.usefixtures("_clean_env")
def test_fit_chunk_env_overrides_the_default(monkeypatch):
    from models.demos.gemma4.tt.mesh_orientation import fit_chunk_default

    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "0")
    assert not fit_chunk_default((4, 1), blackhole=True)
    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "1")
    assert fit_chunk_default((1, 4), blackhole=True)
    assert fit_chunk_default((8, 4), blackhole=False)


@pytest.mark.usefixtures("_clean_env")
def test_fit_chunk_rounds_to_the_dispatch_group_tile(monkeypatch):
    monkeypatch.setenv(dp.FIT_CHUNK_ENV, "1")
    assert dp.fit_chunk_enabled((1, 4))
    assert dp.fit_chunk_tokens(128, 4, 4096) == 256  # 64 rows per chip of the group (masked_bincount's grid)
    assert dp.fit_chunk_tokens(32, 4, 4096) == 256  # never below 64 rows per chip
    assert dp.fit_chunk_tokens(256, 4, 4096) == 256
    assert dp.fit_chunk_tokens(1024, 4, 4096) == 1024
    assert dp.fit_chunk_tokens(2065, 4, 4096) == 2304  # padded to the next 256
    assert dp.fit_chunk_tokens(4096, 4, 4096) == 4096
    assert dp.fit_chunk_tokens(4224, 4, 4096) == 4096  # the loop takes a full chunk, then the 128 remainder
    assert dp.fit_chunk_tokens(128, 8, 4096) == 512  # an 8-chip group: 64 rows per chip
    assert dp.fit_chunk_tokens(128, 4, 4096, fit=False) == 4096
    assert dp.MIN_ROWS_PER_CHIP % 32 == 0


@pytest.mark.usefixtures("_clean_env")
def test_small_fit_rounds_to_the_dispatch_group_tile_without_the_dispatch_floor():
    from models.demos.gemma4.tt.experts.dispatch_prefill import small_fit_chunk_tokens

    # the small-row path has no dispatch histogram: a 128-token prompt on the 4-chip group runs 128 rows (32/chip)
    assert small_fit_chunk_tokens(1, 4) == 128
    assert small_fit_chunk_tokens(128, 4) == 128
    assert small_fit_chunk_tokens(129, 4) == 256
    assert small_fit_chunk_tokens(256, 4) == 256
    assert small_fit_chunk_tokens(64, 8) == 256  # eight chips: one tile per chip


@pytest.mark.usefixtures("_clean_env")
def test_small_row_max_tokens_default_env_and_cap(monkeypatch):
    from models.demos.gemma4.custom_ops.expert_gather_small import ExpertGatherSmall
    from models.demos.gemma4.tt.experts.dispatch_prefill import small_row_max_tokens

    monkeypatch.delenv("GEMMA4_SMALL_ROW_MAX_TOKENS", raising=False)
    assert small_row_max_tokens() == 128
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MAX_TOKENS", "256")
    assert small_row_max_tokens() == 256
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MAX_TOKENS", "4096")
    assert small_row_max_tokens() == ExpertGatherSmall.MAX_TOKENS


@pytest.mark.usefixtures("_clean_env")
def test_small_row_bounds_keep_the_cached_prefix_continuations_on_the_dispatch_path(monkeypatch):
    from models.demos.gemma4.tt.experts.dispatch_prefill import small_row_min_tokens, small_rows_admit

    monkeypatch.delenv("GEMMA4_SMALL_ROW_MIN_TOKENS", raising=False)
    lo = small_row_min_tokens()
    assert lo == 32
    for rows, ok in (
        (1, False),
        (21, False),
        (29, False),
        (31, False),
        (32, True),
        (64, True),
        (128, True),
        (129, False),
    ):
        assert small_rows_admit(rows, lo, 128) == ok, rows
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MIN_TOKENS", "1")
    assert small_row_min_tokens() == 1 and small_rows_admit(8, small_row_min_tokens(), 128)


# ===== merged from tests/unit/test_dispatch_chunk_capacity_host.py: Host-only: the dispatch chunk's per-expert capacity contract (``_DispatchChunkPlan``: the GeGLU op's max_tokens is
E, K = 128, 8


def _capacity(seq_per_chip, dgs=4):
    _, _, max_buf, max_tok = compute_constants(seq_per_chip, E, K, dgs, dgs, K)
    assert max_buf >= dgs * seq_per_chip * K, "drop-free buffer contract"
    return max_tok


def _ids(n, skew_to=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(n)])
    if skew_to is not None:
        ids[:, 0] = skew_to  # every real token's first slot on one expert, the other slots distinct
        for r in range(n):
            if (ids[r, 1:] == skew_to).any():
                ids[r, 1:] = torch.tensor([e for e in range(E) if e != skew_to][: K - 1])
    return ids


def test_capacity_is_the_chunk_rows():
    assert _capacity(64) == 256 and _capacity(256) == 1024 and _capacity(1024) == 4096


def test_ci_shape_zero_pad_overflowed_sentinel_pad_does_not():
    n, chunk = 128, 256  # the fitted chunk of a 128-token call on the (4,1) column: 64 rows/chip
    cap = _capacity(chunk // 4)
    ids = _ids(n)
    zero = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "zero")
    assert int(zero[0]) > cap and int(zero[0]) >= (chunk - n) * K, int(zero[0])  # the CI assert
    sentinel = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "sentinel")
    assert int(sentinel.max()) <= cap and int(sentinel.sum()) == n * K
    dense = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "dense")
    assert int(dense.max()) <= cap  # the dense twin passed on CI: +pad per expert 0..K-1


def test_skewed_routing_stays_within_capacity_for_every_pad():
    for n, chunk in ((129, 256), (300, 512), (2065, 2304), (4096, 4096)):
        cap = _capacity(chunk // 4)
        ids = _ids(n, skew_to=5)
        sentinel = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "sentinel")
        assert int(sentinel[5]) == n and int(sentinel.max()) <= cap, (n, chunk, int(sentinel.max()), cap)
        dense = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "dense")
        assert int(dense[5]) == n + (chunk - n) and int(dense.max()) <= cap, (n, chunk, int(dense.max()), cap)
        if chunk > n:
            zero = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "zero")
            assert int(zero[0]) == int(torch.bincount(ids.reshape(-1), minlength=E)[0]) + (chunk - n) * K


def test_served_shapes_old_pad_overflow_boundary():
    """The old zero pad exceeded the capacity when the real rows filled less than ~7/8 of the fitted chunk
    (K x pad + real_0 > chunk): a 300-token prompt on its 512-row chunk did, a 2065-token prompt on its 2304-row
    chunk did not; the sentinel pad never does."""
    for n, chunk, overflowed in ((300, 512, True), (2065, 2304, False), (129, 256, True)):
        cap = _capacity(chunk // 4)
        ids = _ids(n)
        zero = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "zero")
        assert (int(zero[0]) > cap) == overflowed, (n, chunk, int(zero[0]), cap)
        sentinel = Gemma4DispatchExperts.chunk_expert_counts(ids, chunk - n, E, "sentinel")
        assert int(sentinel.max()) <= cap


# ===== merged from tests/unit/test_dispatch_ring_payload_host.py: Host-only: the ring transport's fabric payload contract (dispatch_fabric2d sends one whole bf16 token page plus
HIDDEN_26B = 2816  # google/gemma-4-26B-A4B-it text hidden_size


def test_26b_token_needs_more_than_the_fabric_default_on_both_trees():
    assert ring_token_bytes(HIDDEN_26B) == 5632 and RING_FWD_EXTRA_BYTES == 64
    assert ring_payload_required_bytes(HIDDEN_26B) == 5696
    assert FABRIC_DEFAULT_PAYLOAD_BYTES == 4352  # the PR base and main: 4 bfp8 tiles, static_assert in the builder
    reason = ring_transport_reason(HIDDEN_26B, FABRIC_DEFAULT_PAYLOAD_BYTES)
    assert reason is not None and "5696" in reason and "4352" in reason and "6528" in reason, reason


def test_router_config_for_the_26b_is_the_smallest_fitting_unit():
    assert ring_fabric_payload_bytes(HIDDEN_26B) == 6528  # 6 x 1088 >= 5696
    assert ring_fabric_payload_bytes(HIDDEN_26B) % 64 == 0  # the fabric's L1 alignment check
    assert ring_fabric_payload_bytes(HIDDEN_26B) <= FABRIC_MAX_PAYLOAD_BYTES_BLACKHOLE
    assert ring_transport_reason(HIDDEN_26B, ring_fabric_payload_bytes(HIDDEN_26B)) is None
    cfg = ring_fabric_router_config(HIDDEN_26B)
    assert cfg.max_packet_payload_size_bytes == 6528


def test_small_tokens_keep_the_default():
    assert ring_fabric_payload_bytes(1152) == FABRIC_DEFAULT_PAYLOAD_BYTES  # 2304 + 64 fits
    assert ring_transport_reason(2144, FABRIC_DEFAULT_PAYLOAD_BYTES) is None  # 4288 + 64 = 4352 exactly
    assert ring_transport_reason(2176, FABRIC_DEFAULT_PAYLOAD_BYTES) is not None  # one tile over


# ===== merged from tests/unit/test_local_routed_index_host.py: Host-only checks of the local routed prefill's index algebra (tt/experts/local_routed_index.py): the sorted pair
def _routing(S, E, K, seed=0, skew=None):
    g = torch.Generator().manual_seed(seed)
    indices = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(S)])
    if skew is not None:
        # most tokens take the same few experts: large regions for a handful of experts, empty ones elsewhere
        hot = torch.arange(K) + skew
        n_hot = int(0.9 * S)
        indices[:n_hot] = hot
    values = torch.rand(S, K, generator=g) + 0.1
    values = values / values.sum(-1, keepdim=True)
    return values.to(torch.bfloat16), indices.to(torch.int64)


def test_index_pipeline_properties():
    S, E, K = 256, 128, 8
    _, indices = _routing(S, E, K)
    pos, counts, offsets, inv = index_pipeline(indices, E)
    N = S * K
    gid = indices.reshape(-1)
    assert pos.shape == (N,) and inv.shape == (N,)
    # sorted by expert id, ties by pair index (stable)
    sorted_gid = gid[pos]
    assert torch.all(sorted_gid[1:] >= sorted_gid[:-1])
    same = sorted_gid[1:] == sorted_gid[:-1]
    assert torch.all(pos[1:][same] > pos[:-1][same])
    # counts / exclusive offsets delimit every expert's region in the sorted order
    assert int(counts.sum()) == N
    assert torch.equal(offsets, torch.cumsum(counts, 0) - counts)
    for g_ in (0, 5, E - 1):
        s, n = int(offsets[g_]), int(counts[g_])
        assert torch.all(sorted_gid[s : s + n] == g_)
    # inverse permutation: every pair's buffer row lies inside its expert's region
    assert torch.equal(inv[pos], torch.arange(N))
    rows = inv  # row of pair p = t*K + k
    assert torch.all(rows >= offsets[gid]) and torch.all(rows < offsets[gid] + counts[gid])


def test_aligned_layout_regions():
    S, E, K = 256, 128, 8
    for seed, skew in ((0, None), (1, 40)):
        _, indices = _routing(S, E, K, seed=seed, skew=skew)
        pos, counts, _, _ = index_pipeline(indices, E)
        aligned_offsets, dest, M = aligned_layout(indices, E)
        gid = indices.reshape(-1).to(torch.int64)
        assert M == S * K + 32 * E
        # every region starts on a tile row, holds its expert's rows contiguously and ends inside the buffer
        assert torch.all(aligned_offsets % 32 == 0)
        assert int(aligned_offsets[-1] + counts[-1]) <= M
        assert torch.unique(dest).numel() == dest.numel()
        for g in (0, 40, 41, E - 1):
            s, n = int(aligned_offsets[g]), int(counts[g])
            rows = dest[gid[pos] == g]
            assert torch.equal(rows, torch.arange(s, s + n))
        # the buffer row of pair p is dest[inv[p]]
        inv = torch.empty_like(pos)
        inv[pos] = torch.arange(pos.numel())
        rows_of_pair = dest[inv]
        assert torch.all(rows_of_pair >= aligned_offsets[gid])
        assert torch.all(rows_of_pair < aligned_offsets[gid] + counts[gid])


def test_local_expert_mask():
    m = local_expert_mask(128, 64, 32)
    assert m.shape == (129,) and m.dtype == torch.int32
    assert torch.all(m[64:96] == 0) and torch.all(m[:64] == -1) and torch.all(m[96:] == -1)


def _dense_reference(x, values, indices, base, num_local, gate, up, down):
    """sum over the local slots of w * expert(x) in token order (what the device chip must produce)."""
    S, K = indices.shape
    out = torch.zeros(S, x.shape[-1])
    xf = x.float()
    for t in range(S):
        for k in range(K):
            g = int(indices[t, k])
            if base <= g < base + num_local:
                e = g - base
                h = F.gelu(xf[t] @ gate[e].float(), approximate="tanh") * (xf[t] @ up[e].float())
                out[t] += float(values[t, k]) * (h @ down[e].float())
    return out


def test_local_routed_golden_matches_dense_sum():
    S, E, K, H, I = 128, 32, 4, 64, 48
    num_local = 8
    for base, skew in ((0, None), (8, 8), (24, None)):
        values, indices = _routing(S, E, K, seed=base, skew=skew)
        x = torch.randn(S, H)
        gate = [torch.randn(H, I) * 0.1 for _ in range(num_local)]
        up = [torch.randn(H, I) * 0.1 for _ in range(num_local)]
        down = [torch.randn(I, H) * 0.1 for _ in range(num_local)]
        got = local_routed_golden(x, values, indices, base, num_local, gate, up, down)
        ref = _dense_reference(x, values, indices, base, num_local, gate, up, down)
        assert torch.allclose(got, ref, atol=1e-4, rtol=1e-4), (base, skew, (got - ref).abs().max())
        if skew is not None:
            # the hot experts of this chip carry 90% of the tokens; the others of this chip see only the random tail
            _, counts, _, _ = index_pipeline(indices, E)
            assert int(counts[base]) >= int(0.9 * S)


# ===== merged from tests/unit/test_fused_moe_planner_host.py: Host-only: the custom MoE programs are selected, never asserted into, on configurations they cannot take.
def test_router_program_contract_over_expert_counts():
    assert [RouterTopk.supports(e, 8) for e in (4, 32, 128, 256, 288)] == [False, True, True, True, False]
    assert RouterTopk.supports(128, 32) and not RouterTopk.supports(128, 33) and not RouterTopk.supports(128, 0)
    # packed rows of 2k bf16 must be 32-byte aligned: k a multiple of 8
    assert RouterTopk.supports(128, 8, packed=True) and not RouterTopk.supports(128, 2, packed=True)
    assert not RouterTopk.supports(4, 8, packed=True)


def test_small_gather_contract():
    assert ExpertGatherSmall.supports(128, 32, 8, 2688)
    assert not ExpertGatherSmall.supports(4, 1, 2, 2688)  # the CI reduced-expert count
    assert not ExpertGatherSmall.supports(128, 32, 8, 2690)  # hidden not a tile multiple
    assert not ExpertGatherSmall.supports(128, 32, 8, 8192)  # a row must fit the L1 staging (2H <= 8192)
    assert not ExpertGatherSmall.supports(128, 256, 8, 2688)  # more local experts than experts


def _moe(num_experts, top_k, fused=True, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("GEMMA4_MOE_FUSED", "1" if fused else "0")
    # the block keeps the expert count and top-k on its router (MoEBlock has no num_experts of its own)
    return SimpleNamespace(
        mesh_config=None,
        experts=SimpleNamespace(ep=1, moe_decode=None),
        router=SimpleNamespace(per_expert_scale=object(), num_experts=num_experts, top_k=top_k),
    )


def test_fused_decode_falls_back_below_the_tile_multiple(monkeypatch):
    ok = [MoEBlock.fused_decode_ok(_moe(e, 8, monkeypatch=monkeypatch)) for e in (4, 32, 128, 256, 288)]
    assert ok == [False, True, True, True, False], ok
    assert MoEBlock.fused_packed_routing_ok(_moe(128, 8, monkeypatch=monkeypatch))
    assert not MoEBlock.fused_packed_routing_ok(_moe(128, 2, monkeypatch=monkeypatch))
    assert not MoEBlock.fused_packed_routing_ok(_moe(4, 8, monkeypatch=monkeypatch))
    # the flag off, lanes, EP > 1, the decode kernel or no per-expert scale keep the shared chain as before
    assert not MoEBlock.fused_decode_ok(_moe(128, 8, fused=False, monkeypatch=monkeypatch))
    m = _moe(128, 8, monkeypatch=monkeypatch)
    m.experts.ep = 2
    assert not MoEBlock.fused_decode_ok(m)
    m = _moe(128, 8, monkeypatch=monkeypatch)
    m.router.per_expert_scale = None
    assert not MoEBlock.fused_decode_ok(m) and not MoEBlock.fused_packed_routing_ok(m)


def test_ring_combine_passes_fwd_reduction_only_when_the_signature_has_it(monkeypatch):
    class WithoutIt:
        def __init__(self, mesh_device, experts_per_chip, num_experts_per_tok, seq_len_per_chip, cluster_axis=0):
            pass

    class WithIt:
        def __init__(
            self,
            mesh_device,
            experts_per_chip,
            num_experts_per_tok,
            seq_len_per_chip,
            fwd_reduction: int,
            cluster_axis=0,
        ):
            pass

    monkeypatch.delenv("GEMMA4_MOE_COMBINE_FWD_REDUCTION", raising=False)
    assert combine2d_extra_kwargs(WithoutIt) == {}
    assert combine2d_extra_kwargs(WithIt) == {"fwd_reduction": 1}
    monkeypatch.setenv("GEMMA4_MOE_COMBINE_FWD_REDUCTION", "4")
    assert combine_fwd_reduction() == 4 and combine2d_extra_kwargs(WithIt) == {"fwd_reduction": 4}
    monkeypatch.setenv("GEMMA4_MOE_COMBINE_FWD_REDUCTION", "0")
    assert combine_fwd_reduction() == 1  # the op requires >= 1


# ===== merged from tests/unit/test_moe_decode_registry_host.py: The decode MoE module's registry rule (``Gemma4MoEDecode``): one shared module per mesh, built once every layer of
CONFIG = SimpleNamespace(hidden_size=2560, top_k=2, num_experts=8, moe_intermediate_size=2560)


@pytest.fixture
def mesh(monkeypatch):
    """A stand-in mesh (the registry keys on its ``id()``, the mesh's own id as ``MeshDevice.id()``) whose module build
    is recorded, not run."""
    device = SimpleNamespace(shape=(8, 4), id=lambda: 10**9 + 1)  # far above any real mesh counter in this process
    builds = []

    def _build(self, config):
        builds.append(self.layer_idx)
        self._reg["module"] = f"module for {self.num_layers} layers"

    monkeypatch.setattr(md.Gemma4MoEDecode, "_build", _build)
    yield device, builds
    md._SHARED.pop(device.id(), None)


def _register(device, layer_idx, num_layers):
    return md.Gemma4MoEDecode(
        device, CONFIG, object(), object(), layer_idx=layer_idx, num_layers=num_layers, frame_rows=256
    )


def test_a_one_layer_unit_registers_with_num_layers_one(mesh):
    device, builds = mesh
    layer = _register(device, 0, 1)
    assert builds == [0] and layer.module == "module for 1 layers"


def test_one_registered_layer_of_a_thirty_layer_config_never_builds(mesh, expect_error):
    device, builds = mesh
    layer = _register(device, 0, 30)
    assert builds == []
    with expect_error(AssertionError, "not every layer registered"):
        layer.module


def test_the_module_is_built_once_when_the_last_layer_registers(mesh):
    device, builds = mesh
    first = _register(device, 0, 3)
    second = _register(device, 1, 3)
    assert builds == []
    third = _register(device, 2, 3)
    assert builds == [2]
    assert first.module is second.module is third.module
    _register(device, 2, 3)  # a re-registration does not rebuild
    assert builds == [2]


def test_a_reopened_mesh_never_inherits_the_closed_mesh_module(mesh):
    """The registry keys on the mesh's own id (never reused), not the Python address: the next pytest case's mesh,
    wherever its object lands, starts with no registered layers and builds its own module."""
    device, builds = mesh
    old = _register(device, 0, 1)
    assert builds == [0]
    reopened = SimpleNamespace(shape=(8, 4), id=lambda: device.id() + 1)
    try:
        new = _register(reopened, 0, 1)
        assert builds == [0, 0] and new._reg is not old._reg
        assert set(new._reg["layers"]) == {0} and old._reg["module"] is not None
    finally:
        md._SHARED.pop(reopened.id(), None)


# ===== merged from tests/unit/test_moe_decode_scope_host.py: The decode MoE kernel wrapper's scoping of the custom moe_compute op and the TILE tail to decode frames.
def _lanes(owner, frame_rows):
    return SimpleNamespace(lane_sharded=True, lanes=4, _g4_single_lane_prefill_owner=owner, frame_rows=frame_rows)


@pytest.mark.parametrize("frame_rows", [128, 256], ids=["block_frame_128", "galaxy_frame_256"])
def test_lane_engine_frames_are_told_apart_by_the_owner_not_the_rows(frame_rows):
    # the 128-user decode step and the 128-token owner prefill both hand the wrapper 128 rows
    assert kernel_decode_frame(_lanes(None, frame_rows), 128) is True
    for owner in range(4):
        assert kernel_decode_frame(_lanes(owner, frame_rows), 128) is False
    assert kernel_decode_frame(_lanes(None, frame_rows), 32) is True


def test_tp_only_shapes_use_the_row_count():
    tp_only = SimpleNamespace(lane_sharded=False)
    assert kernel_decode_frame(tp_only, 1) and kernel_decode_frame(tp_only, 32)
    assert not kernel_decode_frame(tp_only, 64) and not kernel_decode_frame(tp_only, 128)
    assert kernel_decode_frame(None, 8) and not kernel_decode_frame(None, 128)


@pytest.mark.parametrize("custom_enabled", [False, True], ids=["custom_off", "custom_on"])
def test_custom_op_only_on_decode_frames(custom_enabled):
    assert moe_compute_impl(True, custom_enabled) == ("custom" if custom_enabled else "shared")
    assert moe_compute_impl(False, custom_enabled) == "shared"


@pytest.mark.parametrize("batch_per_device", [32, 64], ids=["galaxy_32_per_chip", "block_64_per_chip"])
def test_tile_tail_only_on_decode_frames(batch_per_device):
    assert tile_tail_enabled(True, True, True, batch_per_device) is True
    assert tile_tail_enabled(False, True, True, batch_per_device) is False  # prefill rows: ROW_MAJOR tail
    assert tile_tail_enabled(True, False, True, batch_per_device) is False
    assert tile_tail_enabled(True, True, False, batch_per_device) is False
    assert tile_tail_enabled(True, True, True, 8) is False  # a partial row tile per chip


# ===== the local routed prefill's shared expert-output buffer (``local_routed_prefill._out_buffer``, the ``output`` of the
# GeGLU routed-expert generic op): keyed by the mesh's own id and owned by the module cache. The 2026-10-25 CI flake
# (test_experts_local_routed_chunked after test_experts_local_routed[prefill_128], both on the 5120-row buffer): the
# key was ``id(mesh_device)``, the reopened mesh reused the closed one's Python address, and the op was enqueued on a
# buffer whose mesh was gone ("Can't get device from mesh buffer, already deallocated").


class _Mesh:
    """A stand-in MeshDevice: only ``id()`` (the process-wide mesh id) matters to the cache."""

    def __init__(self, mesh_id):
        self._mesh_id = mesh_id

    def id(self):
        return self._mesh_id


@pytest.fixture
def out_buffer_cache(monkeypatch):
    """``_out_buffer`` with its device calls stubbed: zeros / typecast hand out tagged objects, deallocate marks them;
    the entries the test adds are dropped afterwards."""
    made = []

    class _Buf:
        def __init__(self, tag):
            self.tag, self.alive = tag, True

    monkeypatch.setattr(lrp.ttnn, "zeros", lambda *a, **k: _Buf("bf16"))
    monkeypatch.setattr(lrp.ttnn, "typecast", lambda t, dtype: made.append(_Buf("bf8")) or made[-1])
    monkeypatch.setattr(lrp.ttnn, "deallocate", lambda t: setattr(t, "alive", False))
    keys_before = set(lrp._OUT_BUFFERS)
    yield made
    for k in set(lrp._OUT_BUFFERS) - keys_before:
        lrp._OUT_BUFFERS.pop(k, None)


def _experts_on(mesh, hidden=2816):
    inst = object.__new__(lrp.Gemma4LocalRoutedExperts)
    inst.mesh_device, inst.H = mesh, hidden
    return inst


def test_local_routed_out_buffer_key_is_the_mesh_id():
    """Equal mesh ids share a key whatever the Python objects are; a new mesh id, row count or width is a new key."""
    key = lrp._out_buffer_key
    assert key(_Mesh(7), 5120, 2816) == key(_Mesh(7), 5120, 2816) == (7, 5120, 2816)
    assert key(_Mesh(7), 5120, 2816) != key(_Mesh(8), 5120, 2816)
    assert key(_Mesh(7), 5120, 2816) != key(_Mesh(7), 20480, 2816)
    assert key(_Mesh(7), 5120, 2816) != key(_Mesh(7), 5120, 2560)


def test_local_routed_out_buffer_owned_per_mesh(out_buffer_cache):
    """Layers on one mesh share the buffer; a reopened mesh (new mesh id, the same or any Python address) gets its
    own; the cache keeps every buffer it handed out alive (only the bf16 seed is freed)."""
    made = out_buffer_cache
    old = _Mesh(100)
    layer0, layer1 = _experts_on(old), _experts_on(old)
    first = layer0._out_buffer(5120)
    assert layer1._out_buffer(5120) is first and len(made) == 1
    del old, layer0, layer1  # the previous test case closes and frees its mesh ...
    new = _Mesh(101)  # ... and the next one opens a new mesh (same shape, possibly the same Python address)
    second = _experts_on(new)._out_buffer(5120)
    assert second is not first and len(made) == 2
    assert _experts_on(new)._out_buffer(20480) is not second and len(made) == 3
    assert all(b.alive for b in made), "the cache owns the buffers it handed to the op"
    assert lrp._OUT_BUFFERS[lrp._out_buffer_key(_Mesh(100), 5120, 2816)] is first
