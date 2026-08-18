"""Time ONE deterministic NABlock per W-sharded stage, in the production 2-D SP x TP config.

Standalone (``python <this file>``), not pytest: it opens the mesh itself rather than taking the
conftest fixture. That is not cosmetic -- the same arms timed through the fixture come out ~1.5x
slower, growing with tokens per chip, and the cause is not yet known. Numbers from the two paths
are not comparable; keep an A/B inside one of them.

Geometry is the s34x60 decode's, stage by stage. Arms are the DIFFVAE_DET_* flags, read at
construction.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import NABlock, default_rope_dim_split, rope_tables
from models.tt_dit.parallel.manager import CCLManager

HEAD_DIM = 64
SP_AXIS, TP_AXIS = 1, 0
ITERS = int(os.environ.get("ITERS", 10))

# (label, dim, kernel, dims, blocks_in_stage) for the three W-sharded det stages of the
# s34x60 decode. Stage 1 is excluded: W=60 does not divide the size-8 axis, so it runs
# replicated on the gather backend and this change does not reach it.
STAGES = [
    ("stage2", 1024, (3, 7, 7), (6, 68, 120), 6),
    ("stage3", 512, (3, 5, 5), (11, 68, 120), 4),
    ("stage4", 512, (3, 5, 5), (21, 136, 240), 2),
]


def fill(module, prefix: str = "", g: torch.Generator | None = None) -> None:
    g = g or torch.Generator().manual_seed(7)
    for name, param in module.named_parameters():
        shape = tuple(param.total_shape)
        path = f"{prefix}{name}"
        if "norm" in path:
            t = 1.0 + 0.05 * torch.randn(*shape, generator=g)
        else:
            t = torch.randn(*shape, generator=g) * (shape[-2] ** -0.5 if len(shape) > 1 else 0.02)
        param.load_torch_tensor(t.to(torch.float32))
    for name, child in module.named_children():
        fill(child, f"{prefix}{name}.", g)


def time_stage(mesh, label, dim, kernel, dims, depth) -> tuple[float, int]:
    sp = int(list(mesh.shape)[SP_AXIS])
    t, h, w = dims
    assert w % sp == 0, f"{label}: W={w} not divisible by sp={sp}"
    w_local = w // sp
    tokens = t * h * w_local

    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)
    block = NABlock(
        dim,
        kernel,
        head_dim=HEAD_DIM,
        mesh_device=mesh,
        na3d_backend="op_sp_w_sharded",
        ccl_manager=ccl,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
    )
    fill(block)
    want_colpar = os.environ.get("DIFFVAE_DET_COLPAR_QKV") == "1"
    want_qkv = want_colpar or os.environ.get("DIFFVAE_DET_FUSED_QKV") == "1"
    want_tp = os.environ.get("DIFFVAE_DET_TP_MLP") == "1"
    want_fused = want_tp or os.environ.get("DIFFVAE_DET_FUSED_SWIGLU") == "1"
    assert block.attn.fused_qkv is want_qkv, f"{label}: attn.fused_qkv={block.attn.fused_qkv} != {want_qkv}"
    assert block.attn.colpar_qkv is want_colpar, f"{label}: colpar={block.attn.colpar_qkv} != {want_colpar}"
    want_rope = os.environ.get("DIFFVAE_DET_FUSED_ROPE") == "1"
    assert block.attn.fused_rope is want_rope, f"{label}: fused_rope={block.attn.fused_rope} != {want_rope}"
    assert block.mlp.fused is want_fused, f"{label}: mlp.fused={block.mlp.fused} != {want_fused}"
    assert block.mlp.tp_mlp is want_tp, f"{label}: mlp.tp_mlp={block.mlp.tp_mlp} != {want_tp}"

    cos, sin = rope_tables(dims, default_rope_dim_split(HEAD_DIM), mesh_device=mesh)
    cos = ttnn.mesh_partition(cos, dim=3, cluster_axis=SP_AXIS)
    sin = ttnn.mesh_partition(sin, dim=3, cluster_axis=SP_AXIS)

    x = ttnn.from_torch(
        torch.randn(tokens, dim, generator=torch.Generator().manual_seed(3)),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    local = (t, h, w_local)
    for _ in range(2):  # warm the program cache
        x = block(x, dims=local, cos=cos, sin=sin, device_plan=None)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        x = block(x, dims=local, cos=cos, sin=sin, device_plan=None)
    ttnn.synchronize_device(mesh)
    per_block = (time.perf_counter() - t0) / ITERS * 1000

    heads = dim // HEAD_DIM
    print(
        f"[{label:7s}] dim={dim:5d} heads={heads:3d} dims={dims} w_local={w_local:3d} "
        f"tokens/chip={tokens:7d} | {per_block:8.2f} ms/block  x{depth} = {per_block * depth:8.1f} ms",
        flush=True,
    )
    ttnn.deallocate(x)
    return per_block, depth


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        on = [
            n
            for n, e in (
                ("qkv", "DIFFVAE_DET_FUSED_QKV"),
                ("colpar", "DIFFVAE_DET_COLPAR_QKV"),
                ("rope", "DIFFVAE_DET_FUSED_ROPE"),
                ("swiglu", "DIFFVAE_DET_FUSED_SWIGLU"),
                ("tp_mlp", "DIFFVAE_DET_TP_MLP"),
            )
            if os.environ.get(e) == "1"
        ]
        arm = "+".join(on) if on else "baseline"
        print(f"\n=== det NABlock timing · {arm} · {ITERS} iters ===", flush=True)
        total = 0.0
        for label, dim, kernel, dims, depth in STAGES:
            per_block, depth = time_stage(mesh, label, dim, kernel, dims, depth)
            total += per_block * depth
        print(f"\n[TOTAL W-sharded det blocks] {total:8.1f} ms  ({arm})\n", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
