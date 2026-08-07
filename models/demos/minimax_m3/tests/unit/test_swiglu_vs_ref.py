# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for MiniMax-M3 clamped "swigluoai" SwiGLU vs a hand-written torch reference.

M3 uses the gpt-oss SwiGLU variant (hidden_act="swigluoai", swiglu_alpha=1.702, swiglu_limit=7.0):
    gate = clamp(gate, max=limit); up = clamp(up, -limit, limit)
    out  = (up + 1) * (gate * sigmoid(alpha * gate))
vs a plain SiLU SwiGLU (silu(gate) * up). Reference anchor: transformers
modeling_gpt_oss.py:119-122.

Inputs are scaled past ±limit so the clamp path is actually exercised. Depends ONLY on torch
(no HuggingFace / checkpoint), random inputs — runs on a single Wormhole/Blackhole card.
"""

import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.tt.moe.activation import apply_swiglu, apply_swiglu_fused

from ..test_factory import parametrize_mesh_with_fabric


def _torch_swiglu(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """gpt-oss clamped swigluoai reference (fp32)."""
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(alpha * gate)
    return (up + 1.0) * glu


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("alpha, limit", [(1.702, 7.0)], ids=["a1.702_l7"])
@pytest.mark.parametrize(
    "m, width",
    [
        (128, 3072),  # expert / shared intermediate_size
        (128, 12288),  # dense_intermediate_size (layers 0-2)
        (32, 3072),  # single tile of tokens
    ],
    ids=["i3072", "i12288", "m32"],
)
def test_swiglu_vs_ref(mesh_device, device_params, alpha, limit, m, width, reset_seeds):
    """apply_swiglu (clamped swigluoai) vs torch reference, random inputs spanning past ±limit."""
    # Scale past the ±7 clamp so both clamp branches are exercised.
    gate = torch.randn(1, 1, m, width) * 3.0
    up = torch.randn(1, 1, m, width) * 3.0

    ref = _torch_swiglu(gate, up, alpha, limit)

    config = SimpleNamespace(swiglu_limit=limit, alpha=alpha)

    def _to_tt(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    out_tt = apply_swiglu(_to_tt(gate), _to_tt(up), config)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, 1, m, width)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"swiglu m={m} width={width} alpha={alpha} limit={limit}: {pcc}")
    assert passing, f"PCC fail (width={width}): {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("alpha, limit", [(1.702, 7.0)], ids=["a1.702_l7"])
@pytest.mark.parametrize(
    "m, width",
    [
        (640, 768),  # shared expert at the real prefill shape: 640 tok/chip, 3072/TP=4 per device
        (128, 3072),  # full shared/expert intermediate (TP=1)
        (32, 3072),  # single tile of tokens
    ],
    ids=["shared640x768", "i3072", "m32"],
)
def test_swiglu_fused_vs_chain(mesh_device, device_params, alpha, limit, m, width, reset_seeds):
    """``apply_swiglu_fused`` (ONE multiply with activation spans) vs the 7-op ``apply_swiglu`` chain
    vs an fp32 torch reference.

    This is the activation identity the shared-expert/dispatch overlap depends on: the chain's two
    ``ttnn.clamp`` calls accept no ``sub_core_grids``, so they cannot be confined to a sub-device,
    while the fused form can. Verifying it standalone — at the real per-device shape — comes BEFORE
    wiring any of the overlap, because an op-level mistake here is far cheaper to find now.

    Expect the fused form to match the fp32 reference at least as well as the chain: it keeps the
    intermediates in the SFPU instead of round-tripping each of 7 steps through bf16 DRAM.
    """
    # Scale past the +-7 clamp so both clamp branches are exercised.
    gate = torch.randn(1, 1, m, width) * 3.0
    up = torch.randn(1, 1, m, width) * 3.0
    ref = _torch_swiglu(gate, up, alpha, limit)
    config = SimpleNamespace(swiglu_limit=limit, alpha=alpha)

    def _to_tt(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def _to_torch(tt):
        return ttnn.to_torch(ttnn.get_device_tensors(tt)[0]).reshape(1, 1, m, width).float()

    fused = _to_torch(apply_swiglu_fused(_to_tt(gate), _to_tt(up), config))
    chain = _to_torch(apply_swiglu(_to_tt(gate), _to_tt(up), config))

    ok_ref, pcc_ref = comp_pcc(ref, fused, 0.999)
    ok_chain, pcc_chain = comp_pcc(chain, fused, 0.999)
    _, pcc_chain_ref = comp_pcc(ref, chain, 0.99)
    logger.info(
        f"swiglu fused m={m} width={width}: vs_ref={pcc_ref} vs_chain={pcc_chain} (chain vs_ref={pcc_chain_ref})"
    )
    assert ok_ref, f"fused swiglu vs fp32 reference PCC fail: {pcc_ref}"
    assert ok_chain, f"fused swiglu vs the 7-op chain PCC fail: {pcc_chain}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_swiglu_fused_is_confinable(mesh_device, device_params, reset_seeds):
    """PROVE that ``sub_core_grids`` actually confines the fused activation's program.

    This matters because the shared-expert/dispatch overlap is only real if the two ops land on
    DISJOINT cores, and the failure mode is SILENT: a program spanning both sub-devices still passes
    the sub-device assert (its cores are covered by the union) and still returns the right answer — it
    is merely tracked on both and stops overlapping. That is what sank the previous attempt.

    Two things are checked, under a manager whose only sub-device is rows 1..N (row 0 excluded):

      1. correctness — the confined activation still matches an fp32 reference;
      2. that the argument CHANGES THE PROGRAM — the same tensors and math confined to ONE core must
         be dramatically slower than on the full sub-grid.

    Runtime is the discriminator rather than an expected exception, because neither correctness nor
    "it didn't crash" separates the cases: ttnn eltwise auto-restricts to the active sub-device, so an
    unconfined multiply runs fine here too (measured — an earlier version of this test wrongly assumed
    it would fault). The rigorous alternative is the profiler's CORE COUNT column; this is the cheap
    version that can live in a unit test.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    assert grid.y >= 2, f"need >= 2 worker rows to exclude row 0, got {grid.y}"
    confined_cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )  # rows 1..N-1 — row 0 deliberately left OUT of every sub-device

    m, width = 640, 768
    alpha, limit = 1.702, 7.0
    config = SimpleNamespace(swiglu_limit=limit, alpha=alpha)
    gate = torch.randn(1, 1, m, width) * 3.0
    up = torch.randn(1, 1, m, width) * 3.0
    ref = _torch_swiglu(gate, up, alpha, limit)

    def _to_tt(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    mgr = mesh_device.create_sub_device_manager([ttnn.SubDevice([confined_cores])], 0)
    try:
        mesh_device.load_sub_device_manager(mgr)

        # (1) CONFINED — must run and be correct.
        out_tt = apply_swiglu_fused(_to_tt(gate), _to_tt(up), config, sub_core_grids=confined_cores)
        out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, 1, m, width).float()
        passing, pcc = comp_pcc(ref, out, 0.999)
        logger.info(f"fused swiglu CONFINED to rows 1..{grid.y - 1}: pcc={pcc}")
        assert passing, f"confined fused swiglu PCC fail: {pcc}"

        # (2) Does sub_core_grids actually change the PROGRAM, or is it accepted and ignored?
        # "It didn't crash" proves nothing: an op that ignores the argument still returns the right
        # answer, and ttnn eltwise auto-restricts to the active sub-device anyway. So compare RUNTIME
        # between the full shared sub-grid and a deliberately tiny one-core grid. Same math, same
        # tensors; if the argument is honoured, one core must be dramatically slower than 13x9=117.
        one_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1))})
        g_tt, u_tt = _to_tt(gate), _to_tt(up)

        def _timed(cores, iters=20):
            apply_swiglu_fused(u_tt, g_tt, config, sub_core_grids=cores)  # warm up / compile
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(iters):
                apply_swiglu_fused(u_tt, g_tt, config, sub_core_grids=cores)
            ttnn.synchronize_device(mesh_device)
            return (time.perf_counter() - t0) / iters * 1e6  # us/call

        many_us = _timed(confined_cores)
        one_us = _timed(one_core)
        ratio = one_us / many_us
        logger.info(
            f"fused swiglu: {grid.x * (grid.y - 1)}-core sub-grid {many_us:.1f} us/call vs "
            f"1-core sub-grid {one_us:.1f} us/call -> {ratio:.1f}x"
        )
        assert ratio > 5.0, (
            f"confining to ONE core was only {ratio:.1f}x slower than the full {grid.x * (grid.y - 1)}-core "
            f"sub-grid, so sub_core_grids looks accepted-but-ignored. The overlap cannot rely on it; "
            f"verify placement with the profiler CORE COUNT column before proceeding."
        )
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(mgr)
