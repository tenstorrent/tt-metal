"""Sweep MinimalMatmul blocking for the stage-5 projection/MLP shapes that have no table entry.

``get_matmul_config`` warns "No known best blocking for (M, K, N) ... using default 8x8x8" for every
stage-5 projection, so all of them run on a blocking nobody chose. In the shipped GNA config the
proj/RoPE/norm residual is the LARGEST item in the attention block (191 ms of 618 ms, against 56 ms of
fused SDPA), and it is identical for NA and every GNA stride -- so this is worth more than the attention
kernel and it cannot regress output quality, only speed.

Runs on the MESH, not a single device: the model's ops see an 11x10 storage grid while one device
reports 12x10, and the blocking table is keyed on the grid. Tuning on the wrong grid writes entries that
never match.

Reports, per shape, the best blocking found and its margin over the (8,8,8) default.

TWO THINGS TO FIX BEFORE TRUSTING A WINNER end-to-end:
  * This builds MinimalMatmulConfig with NO compute_kernel_config, so it tunes at a different fidelity
    and accumulation mode than the model's Linear layers use, and optimal blocking depends on both.
    Winners measured here (1.12-1.34x in isolation) came out as 0.1% -- noise -- in situ.
  * The table is keyed on EXACT M, and M moves with banding, latent length and resolution: the same
    stage-5 projection is M=2366400 unbanded but 1191360/1175040/1272960/1256640 across bands. Point
    entries are whack-a-mole; the fallback heuristic in get_matmul_config is the thing worth changing.

Env: ``ITERS`` (default 3), ``SHAPES`` to select by index (comma-separated).
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import time

import torch

import ttnn
from models.tt_dit.utils.matmul import get_matmul_core_grid

ITERS = int(os.environ.get("ITERS", 3))

#: (M, K, N) triples get_matmul_config warned about on the 11x10 grid during a 6s decode. M is the
#: per-device stage-5 token count; the 22400/32640/261120 rows are the smaller det stages.
SHAPES = [
    (2366400, 256, 256),
    (2366400, 64, 256),
    (2366400, 256, 64),
    (2366400, 1024, 256),
    (2366400, 256, 2048),
]


def candidates(K_tiles: int, N_tiles: int, M_tiles: int):
    """Blockings worth trying. subblock_h*subblock_w is capped at 4: the device rejects more with
    "subblock_h * subblock_w must be <= max_dest_volume", and each reject still costs a JIT compile."""
    out = []
    for mb in (2, 4, 8, 16):
        if mb > M_tiles:
            continue
        for kb in (1, 2, 4, 8, 32):
            if kb > K_tiles:
                continue
            for nb in (2, 4, 8, 16):
                if nb > N_tiles:
                    continue
                for sh, sw in ((1, 4), (2, 2), (2, 4), (4, 2)):
                    if sh * sw > 4 or sh > mb or sw > nb:
                        continue
                    out.append((mb, kb, nb, (sh, sw)))
    return out


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        grid = get_matmul_core_grid(mesh)  # the clamp the model's own ops get (11x10 on BH Galaxy)
        print(f"\n=== matmul blocking sweep · storage grid {grid.x}x{grid.y} · {ITERS} iters ===", flush=True)
        want = {int(v) for v in os.environ.get("SHAPES", "").split(",") if v.strip()}
        g = torch.Generator().manual_seed(0)
        for idx, (M, K, N) in enumerate(SHAPES):
            if want and idx not in want:
                continue
            Mt, Kt, Nt = math.ceil(M / 32), math.ceil(K / 32), math.ceil(N / 32)
            try:
                a = ttnn.from_torch(
                    torch.randn(1, 1, M, K, generator=g).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                )
                b = ttnn.from_torch(
                    torch.randn(1, 1, K, N, generator=g).to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                )
            except Exception as exc:  # a shape whose operands do not fit is not a tuning target
                print(f"[{idx}] (M,K,N)=({M},{K},{N})  SKIP alloc: {type(exc).__name__}", flush=True)
                continue

            def timed(mb, kb, nb, sub) -> float | None:
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sub[0],
                    subblock_w=sub[1],
                    compute_with_storage_grid_size=grid,
                )

                def mm():
                    return ttnn.experimental.minimal_matmul(
                        input_tensor=a, weight_tensor=b, bias_tensor=None, config=cfg
                    )

                try:
                    for _ in range(2):
                        ttnn.deallocate(mm())
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        ttnn.deallocate(mm())
                    ttnn.synchronize_device(mesh)
                    return (time.perf_counter() - t0) / ITERS * 1000
                except Exception:
                    return None  # illegal blocking / L1 overflow: not a candidate

            base = timed(8, min(8, Kt), 8, (1, 4))
            best, best_cfg, tried, ok = base, (8, min(8, Kt), 8, (1, 4)), 0, 0
            for mb, kb, nb, sub in candidates(Kt, Nt, Mt):
                tried += 1
                ms = timed(mb, kb, nb, sub)
                if ms is None:
                    continue
                ok += 1
                if best is None or ms < best:
                    best, best_cfg = ms, (mb, kb, nb, sub)
            if base is None or best is None:
                print(f"[{idx}] (M,K,N)=({M},{K},{N})  no legal blocking measured", flush=True)
            else:
                mb, kb, nb, sub = best_cfg
                print(
                    f"[{idx}] (M,K,N)=({M},{K},{N})  default {base:8.2f} ms  best {best:8.2f} ms "
                    f"({base / best:.2f}x)  -> ({mb}, {kb}, {nb}, {sub})   [{ok}/{tried} legal]",
                    flush=True,
                )
            ttnn.deallocate(a)
            ttnn.deallocate(b)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
