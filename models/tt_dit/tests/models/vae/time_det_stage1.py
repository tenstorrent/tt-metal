"""Time ONE deterministic NABlock of stage 1 -- the replicated stage none of the fast paths reach.

Stage 1 carries ~2.2x the per-chip matmul FLOPs of stages 2-4 combined (tokens x dim^2 x depth:
12,240 x 2048^2 x 4 against their 95G), because it is the widest stage and the only one that runs
on every chip: W=60 does not divide the size-8 mesh axis, so it gets no SP, and the decoder denies
it tp_axis in the same condition.

Standalone like ``time_det_nablock.py``, and for the same reason -- the pytest fixture's numbers are
~1.5x higher, so an A/B has to stay inside one harness.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.tt_dit.layers.na3d import build_device_plan, plan_na3d
from models.tt_dit.models.vae.diffvae_ltx import NABlock, default_rope_dim_split, rope_tables
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.vae.time_det_nablock import fill

HEAD_DIM = 64
ITERS = int(os.environ.get("ITERS", 10))

# Stage 1 of the s34x60 decode: dims are stage 2's (6,68,120) divided by upsamples[0]'s (1,2,2).
LABEL, DIM, KERNEL, DIMS, DEPTH = "stage1", 2048, (3, 7, 7), (6, 34, 60), 4


def time_stage1(mesh) -> float:
    t, h, w = DIMS
    tokens = t * h * w  # replicated: every chip holds the whole volume

    # The plan takes the ccl_manager even though the block does not -- the gather backend
    # query-shards across the mesh, which is the only parallelism stage 1 gets.
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    plan = build_device_plan(plan_na3d(DIMS, KERNEL), mesh_device=mesh, ccl_manager=ccl)

    block = NABlock(
        DIM,
        KERNEL,
        head_dim=HEAD_DIM,
        mesh_device=mesh,
        na3d_backend="gather",
        ccl_manager=None,
        sp_axis=None,
        tp_axis=None,
    )
    fill(block)

    # Reachability, not configuration: a run that believes it enabled a flag the stage cannot reach
    # is silently measuring the unflagged path. colpar and flat_seq stay out of reach here -- the
    # first needs a TP axis to shard the weight over, the second exists only in the W-sharded
    # attention -- while fused qkv, fused RoPE and fused SwiGLU all apply to a replicated stage.
    want_qkv = os.environ.get("DIFFVAE_DET_FUSED_QKV") == "1"
    want_rope = want_qkv and os.environ.get("DIFFVAE_DET_FUSED_ROPE") == "1"
    want_fused = os.environ.get("DIFFVAE_DET_FUSED_SWIGLU") == "1" or os.environ.get("DIFFVAE_DET_TP_MLP") == "1"
    assert block.attn.fused_qkv is want_qkv, f"fused_qkv={block.attn.fused_qkv} != {want_qkv}"
    assert block.attn.fused_rope is want_rope, f"fused_rope={block.attn.fused_rope} != {want_rope}"
    assert block.attn.colpar_qkv is False, "colpar_qkv needs a tp_axis, which stage 1 has none"
    assert block.attn.flat_seq is False, "flat_seq is W-sharded only; stage 1 runs the gather backend"
    assert block.mlp.fused is want_fused, f"mlp.fused={block.mlp.fused} != {want_fused}"
    assert block.attn.tp == 1, f"stage 1 must be replicated; tp={block.attn.tp}"

    cos, sin = rope_tables(DIMS, default_rope_dim_split(HEAD_DIM), mesh_device=mesh)
    x = ttnn.from_torch(
        torch.randn(tokens, DIM, generator=torch.Generator().manual_seed(3)),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    for _ in range(2):  # warm the program cache
        x = block(x, dims=DIMS, cos=cos, sin=sin, device_plan=plan)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        x = block(x, dims=DIMS, cos=cos, sin=sin, device_plan=plan)
    ttnn.synchronize_device(mesh)
    per_block = (time.perf_counter() - t0) / ITERS * 1000

    print(
        f"[{LABEL:7s}] dim={DIM:5d} heads={DIM // HEAD_DIM:3d} dims={DIMS} replicated "
        f"tokens/chip={tokens:7d} | {per_block:8.2f} ms/block  x{DEPTH} = {per_block * DEPTH:8.1f} ms",
        flush=True,
    )
    ttnn.deallocate(x)
    return per_block


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        on = [
            n
            for n, e in (
                ("qkv", "DIFFVAE_DET_FUSED_QKV"),
                ("rope", "DIFFVAE_DET_FUSED_ROPE"),
                ("swiglu", "DIFFVAE_DET_FUSED_SWIGLU"),
            )
            if os.environ.get(e) == "1"
        ]
        arm = "+".join(on) if on else "baseline"
        print(f"\n=== det stage-1 NABlock timing · {arm} · {ITERS} iters ===", flush=True)
        per_block = time_stage1(mesh)
        print(f"\n[TOTAL stage-1 det blocks] {per_block * DEPTH:8.1f} ms  ({arm})\n", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
