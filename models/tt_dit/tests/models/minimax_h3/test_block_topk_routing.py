# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""BlockTopKRouting (models/tt_dit/layers/block_topk_routing.py) on device vs its torch mirror, on VSA geometries.

`test_pool`: block means vs the mirror. `test_route`: the block mask against tie-robust top-k bounds, with exact
ties in the data, `k = k_max`, a cached second call, and in-place request swaps. `test_coarse`: the coarse output vs
the mirror.

Run (Galaxy): pytest models/tt_dit/tests/models/minimax_h3/test_block_topk_routing.py -k 4x8 -s --tb=short
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from loguru import logger

import ttnn

from ....layers.block_topk_routing import BlockTopKRouting, RoutingRequest, torch_reference
from ....models.transformers.minimax_h3.vsa_prepass import _round_up, compute_topk, vsa_routing_request
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.vsa_geometry import VSA_TILE_TOKENS, build_vsa_geometry
from ....utils.tensor import from_torch
from .common import GALAXY_RING

HEADS = 14
HEAD_DIM = 128
SPARSITY = 0.9
SWAPPED_IN = ((300, 0, 200), (8, 32, 32))
TAU_OF_ROW_SCALE = 2.0**-10

QUICK = [
    pytest.param((512, 0, 0), (8, 16, 16), 18944, id="quick-aligned"),
    pytest.param((70, 5, 130), (9, 10, 13), 20224, id="quick-ragged-straggler"),
]
LARGE = [
    pytest.param((39, 0, 414), (37, 24, 42), 45312, id="5s-1344x768"),
    pytest.param((39, 0, 1206), (107, 24, 42), 120320, id="15s-1344x768"),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


class _Case:
    """One geometry on the mesh: its routing request, sharded Q/K/V on host and device, and readback helpers."""

    def __init__(self, mesh_device, sp_axis, tp_axis, geometry, *, seed: int, tie: bool) -> None:
        self.mesh_device, self.sp_axis, self.tp_axis = mesh_device, sp_axis, tp_axis
        self.geometry = geometry
        sp = geometry.sp_factor
        self.rows = geometry.rows_per_shard
        self.nb_pad = _round_up(geometry.blocks_per_shard, 32)
        self.k = compute_topk(SPARSITY, int(geometry.is_candidate.sum()))
        self.k_max = compute_topk(SPARSITY, geometry.n_blocks)
        self.request = vsa_routing_request(geometry, self.nb_pad, self.k)

        torch.manual_seed(seed)
        packed = [torch.randn(geometry.seq_len, HEADS, HEAD_DIM).bfloat16().float() for _ in range(3)]
        q, k, v = (geometry.permute_metadata(x).permute(1, 0, 2) for x in packed)
        if tie:
            # Two full candidate blocks with identical keys: their columns score identically in every row.
            full = torch.nonzero(geometry.is_candidate & (geometry.valid_counts == VSA_TILE_TOKENS)).reshape(-1)
            a, b = (int(geometry.block_starts[i]) for i in (full[0], full[len(full) // 2]))
            k[:, b : b + VSA_TILE_TOKENS] = k[:, a : a + VSA_TILE_TOKENS]
        self.shards = [x.reshape(HEADS, sp, self.rows, HEAD_DIM).permute(1, 0, 2, 3).contiguous() for x in (q, k, v)]
        self.device = [
            from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None])
            for x in self.shards
        ]

    def to_host(self, t: ttnn.Tensor) -> torch.Tensor:
        """Per-device [1, ...] -> [S, TP, ...], replicas along TP checked identical, first replica returned."""
        dims = [None, None]
        dims[self.sp_axis] = 0
        dims[self.tp_axis] = 1
        out = ttnn.to_torch(
            t,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, dims=dims, mesh_shape=tuple(self.mesh_device.shape)
            ),
        )
        tp = tuple(self.mesh_device.shape)[self.tp_axis]
        out = out.reshape(out.shape[0], tp, out.shape[1] // tp, *out.shape[2:])
        assert all(torch.equal(out[:, 0], out[:, i]) for i in range(1, tp)), "TP replicas differ"
        return out[:, 0]


def _routing(case: _Case, ccl_manager) -> BlockTopKRouting:
    return BlockTopKRouting(
        mesh_device=case.mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=case.sp_axis,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        rows=case.rows,
        k_max=case.k_max,
    )


def _mirror_scores(case: _Case, routing: BlockTopKRouting) -> torch.Tensor:
    """fp32 scores from the device's own block means: [S, H, nb_pad, S * nb_pad]."""
    q_bar, kv_bar = routing.pool(*case.device)
    qb, kvb = case.to_host(q_bar).float(), case.to_host(kv_bar).float()
    ttnn.deallocate(q_bar)
    ttnn.deallocate(kv_bar)
    k_glob = kvb[:, :HEADS].permute(1, 0, 2, 3).reshape(HEADS, -1, HEAD_DIM)
    return torch.einsum("shbd,hcd->shbc", qb, k_glob)


def _routed_mask(case: _Case, routing: BlockTopKRouting) -> torch.Tensor:
    out = routing(*case.device, coarse=False)
    mask = case.to_host(out["block_mask"]).long()
    ttnn.deallocate(out["block_mask"])
    return mask


def _check_mask(mask: torch.Tensor, scores: torch.Tensor, request: RoutingRequest, label: str) -> None:
    """Exact rules outside the candidate columns of scored rows; tie-robust top-k bounds inside them."""
    assert set(mask.unique().tolist()) <= {0, 1}, mask.unique()
    kept = mask != 0
    cand = request.candidate.reshape(-1)
    exempt = request.exempt.reshape(-1)
    scored = request.scored[:, None, :, None]
    expected_rest = (exempt & ~cand) & scored
    assert torch.equal(kept[..., ~cand], expected_rest.expand_as(kept)[..., ~cand]), f"{label}: non-candidate columns"
    assert not kept[..., cand][~scored.expand_as(kept)[..., cand]].any(), f"{label}: an unscored row keeps a candidate"

    k = request.k
    rows = request.scored[:, None, :].expand(scores.shape[:3])
    sc = scores[..., cand][rows]
    kc = kept[..., cand][rows]
    v = sc.topk(k, dim=-1).values[:, -1:]
    row_scale = sc.abs().amax(dim=-1, keepdim=True)
    # The device's dot products differ from torch's by up to ~5e-4 of the row's largest |score| (measured as the
    # largest dropped-above-kept inversion of the mask on these data), so the band is twice that.
    tau = TAU_OF_ROW_SCALE * row_scale
    inverted = torch.where(~kc, sc, -torch.inf).amax(-1) - torch.where(kc, sc, torch.inf).amin(-1)
    worst_inversion = float((inverted.clamp(min=0) / row_scale[:, 0]).max())
    above, below = sc > v + tau, sc < v - tau
    free = ~above & ~below
    n_kept, n_free = kc.sum(-1), free.sum(-1)
    assert bool((kc | ~above).all()), f"{label}: a candidate clearly above the k-th score is dropped"
    assert bool((~kc | ~below).all()), f"{label}: a candidate clearly below the k-th score is kept"
    assert bool((n_kept >= k).all() and (n_kept <= k + n_free).all()), f"{label}: kept count out of bounds"
    exact = n_free == 1
    assert bool((n_kept[exact] == k).all()), f"{label}: rows without a tie must keep exactly k"
    logger.info(
        f"{label}: k={k}, {rows.sum().item()} scored rows, {int((n_free > 1).sum())} with a free zone, "
        f"{int((n_kept > k).sum())} keep more than k (max {int(n_kept.max())}), "
        f"worst kept/dropped inversion {worst_inversion:.2e} of the row scale"
    )


@GALAXY_RING
@pytest.mark.parametrize(("prefix_segments", "grid", "rung"), QUICK)
def test_pool(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, prefix_segments, grid, rung, reset_seeds):
    sp = tuple(mesh_device.shape)[sp_axis]
    case = _Case(
        mesh_device,
        sp_axis,
        tp_axis,
        build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung),
        seed=0,
        tie=False,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    routing = _routing(case, ccl_manager)
    routing.update(case.request, traced=False)
    ref = torch_reference(*case.shards, case.request)

    q_bar, kv_bar = routing.pool(*case.device)
    qb, kvb = case.to_host(q_bar).float(), case.to_host(kv_bar).float()
    pad = case.request.counts == 0
    for name, dev, expected in (
        ("q", qb, ref["q_bar"]),
        ("k", kvb[:, :HEADS], ref["k_bar"]),
        ("v", kvb[:, HEADS:], ref["v_bar"]),
    ):
        torch.testing.assert_close(dev, expected, atol=3e-2, rtol=3e-2, msg=lambda m: f"{name}_bar: {m}")
        assert torch.equal(dev.permute(0, 2, 1, 3)[pad], torch.zeros_like(dev.permute(0, 2, 1, 3)[pad])), name
        logger.info(
            f"{name}_bar: max abs err {float((dev - expected).abs().max()):.3e}, bit-equal {float((dev == expected).float().mean()):.4f}"
        )


@GALAXY_RING
@pytest.mark.parametrize(("prefix_segments", "grid", "rung"), QUICK + LARGE)
def test_route(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, prefix_segments, grid, rung, reset_seeds):
    sp = tuple(mesh_device.shape)[sp_axis]
    case = _Case(
        mesh_device,
        sp_axis,
        tp_axis,
        build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung),
        seed=0,
        tie=True,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    routing = _routing(case, ccl_manager)
    logger.info(f"rung {rung}: nb_pad {case.nb_pad}, k {case.k}, k_max {case.k_max} ({routing.kt} sorted tiles)")

    routing.update(case.request, traced=False)
    scores = _mirror_scores(case, routing)
    first = _routed_mask(case, routing)
    _check_mask(first, scores, case.request, "request")
    assert torch.equal(
        _routed_mask(case, routing), first
    ), "the cached program (rebound buffers) must reproduce the first call"

    at_k_max = dataclasses.replace(case.request, k=min(case.k_max, int(case.request.candidate.sum())))
    routing.update(at_k_max, traced=True)
    _check_mask(_routed_mask(case, routing), scores, at_k_max, "k = k_max")

    other = _Case(
        mesh_device, sp_axis, tp_axis, build_vsa_geometry(*SWAPPED_IN, sp_factor=sp, padded_len=rung), seed=1, tie=False
    )
    assert other.k != case.k, "the swapped-in request must change k"
    routing.update(other.request, traced=True)
    other_mask = _routed_mask(other, routing)
    _check_mask(other_mask, _mirror_scores(other, routing), other.request, "swapped in")
    fresh = _routing(other, ccl_manager)
    fresh.update(other.request, traced=False)
    assert torch.equal(_routed_mask(other, fresh), other_mask), "an in-place update must route as a fresh instance"

    routing.update(case.request, traced=True)
    assert torch.equal(_routed_mask(case, routing), first), "updating back in place must restore the routing"


@GALAXY_RING
@pytest.mark.parametrize(("prefix_segments", "grid", "rung"), QUICK + LARGE)
def test_coarse(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, prefix_segments, grid, rung, reset_seeds):
    sp = tuple(mesh_device.shape)[sp_axis]
    geometry = build_vsa_geometry(prefix_segments, grid, sp_factor=sp, padded_len=rung)
    case = _Case(mesh_device, sp_axis, tp_axis, geometry, seed=0, tie=False)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    routing = _routing(case, ccl_manager)
    routing.update(case.request, traced=False)
    ref = torch_reference(*case.shards, case.request)

    out = routing(*case.device, coarse=True)
    ttnn.deallocate(out["block_mask"])
    rows = case.to_host(out["coarse_out"])[:, 0].float().reshape(sp, case.rows, HEADS, HEAD_DIM)
    ttnn.deallocate(out["coarse_out"])
    # Every row of a block carries the block's coarse output: compare all rows of the real blocks.
    block = torch.arange(case.rows) // VSA_TILE_TOKENS
    real = case.request.counts[:, block] > 0
    a = rows[real]
    b = ref["coarse"].permute(0, 2, 1, 3)[:, block][real]
    pcc = _pcc(a, b)
    logger.info(f"coarse vs mirror: PCC {pcc:.6f}, max abs err {float((a - b).abs().max()):.3e}")
    assert pcc >= 0.999, pcc
    torch.testing.assert_close(a, b, atol=3e-2, rtol=3e-2)
