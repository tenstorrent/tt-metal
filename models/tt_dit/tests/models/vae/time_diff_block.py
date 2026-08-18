"""Time ONE stage-5 DiffusionNABlock in the production 2-D SP x TP config.

Stage 5 is the largest block in the decoder -- 2,973,696 sites at the 1080p grid against stage 4's
685,440 -- and it is the only one with no committed harness: the parity tests need an LTX-2 checkout
for their torch reference and run a toy grid, so there has been nothing to A/B an optimization
against. Weights are filled rather than loaded for exactly that reason: this measures the block, not
the checkpoint.

Standalone like ``time_det_nablock.py`` and for the same reason -- the pytest fixture's numbers run
~1.5x higher, so a comparison has to stay inside one harness.

``GRID_T``/``GRID_H``/``GRID_W`` override the grid. The default is the shipped 1080p 25-frame
geometry, whose host-side context and noise tensors are several GB; shrink T first when validating.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config, Grid
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.vae.time_det_nablock import fill

SP_AXIS, TP_AXIS = 1, 0
ITERS = int(os.environ.get("ITERS", 10))
GRID = Grid(
    batch=1,
    t=int(os.environ.get("GRID_T", 121)),
    h=int(os.environ.get("GRID_H", 128)),
    w=int(os.environ.get("GRID_W", 192)),
)


def _flat(x: torch.Tensor, channels: int) -> torch.Tensor:
    """``(B, T, H, W, C)`` -> the module's ``(1, B, sites, C)`` layout."""
    return x.reshape(1, x.shape[0], -1, channels)


def main() -> None:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        cfg = DiffVAEStage5Config()
        sp = int(list(mesh.shape)[SP_AXIS])
        assert GRID.w % sp == 0, f"W={GRID.w} not divisible by sp={sp}"

        ccl = CCLManager(mesh, num_links=int(os.environ.get("LINKS", 1)), topology=ttnn.Topology.Linear)
        model = DiffVAEStage5(
            cfg,
            mesh_device=mesh,
            dtype=ttnn.bfloat16,
            ccl_manager=ccl,
            na3d_backend="op_sp_w_sharded",
            sp_axis=SP_AXIS,
            tp_axis=TP_AXIS,
        )
        fill(model)

        g = torch.Generator().manual_seed(99)
        context = torch.randn(GRID.batch, GRID.t, GRID.h, GRID.w, cfg.context_channels, generator=g)
        x_t = torch.randn(
            GRID.batch, cfg.out_channels, GRID.t, GRID.h * cfg.patch_size, GRID.w * cfg.patch_size, generator=g
        )

        # Same construction order as forward / forward_diff_step, so the block sees exactly what it
        # sees in a decode: context W-sharded to this chip's band, modulation from the shared AdaLN,
        # and one RoPE table set per band.
        tt_context = ttnn.from_torch(
            _flat(context, cfg.context_channels).contiguous(), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_context = model._wshard_context(tt_context, GRID)
        bands = model.bands(GRID)
        x_bands = model.embed_x_t(x_t, bands)
        timestep = ttnn.from_torch(
            torch.tensor([0.7] * GRID.batch).reshape(1, 1, -1, 1),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
        )
        modulation = model.shared_adaln(
            model.t_embedder(ttnn.multiply(timestep, cfg.timestep_scale_multiplier)), GRID.batch
        )
        tables = model.rope_tables(GRID)
        band_tables = tuple(tables.frames(band.pad_lo, band.pad_hi) for band in bands)

        block = model.diff_blocks[0]
        sites_local = GRID.t * GRID.h * (GRID.w // sp)
        if os.environ.get("CHECK") == "1":
            # The only correctness signal available without an LTX-2 checkout: run the arm against
            # the unflagged path in the same process. ``fill`` is seeded, so both models hold
            # identical weights and any divergence is the arm's.
            out = block(list(x_bands), tt_context, modulation, GRID, bands, band_tables)
            got = ttnn.to_torch(ttnn.get_device_tensors(out[0] if isinstance(out, list) else out)[0]).float()
            torch.save(got, os.environ.get("CHECK_OUT", "/tmp/s5_out.pt"))
            print(f"[check] wrote {tuple(got.shape)} to {os.environ.get('CHECK_OUT', '/tmp/s5_out.pt')}", flush=True)
            return
        print(
            f"\n=== stage-5 DiffusionNABlock · grid {(GRID.t, GRID.h, GRID.w)} · {GRID.sites} sites "
            f"({sites_local}/chip) · {len(bands)} band(s) · {ITERS} iters ===",
            flush=True,
        )

        for _ in range(2):  # warm the program cache
            x_bands = block(x_bands, tt_context, modulation, GRID, bands, band_tables)
        ttnn.synchronize_device(mesh)

        t0 = time.perf_counter()
        for _ in range(ITERS):
            x_bands = block(x_bands, tt_context, modulation, GRID, bands, band_tables)
        ttnn.synchronize_device(mesh)
        per_block = (time.perf_counter() - t0) / ITERS * 1000

        print(
            f"[stage5 ] dim={cfg.dim} heads={cfg.dim // cfg.head_dim} sites/chip={sites_local} "
            f"| {per_block:8.2f} ms/block  x{cfg.num_blocks} = {per_block * cfg.num_blocks:8.1f} ms",
            flush=True,
        )
        print(f"\n[TOTAL stage-5 blocks] {per_block * cfg.num_blocks:8.1f} ms\n", flush=True)

        # The module's own section counters. forward_diff_step prints these; a single-block harness
        # has to read them itself. Values accumulate over every call, so divide by the calls made.
        if os.environ.get("DIFFVAE_STAGE_TIMING", "") not in ("", "0"):
            from models.tt_dit.layers.na3d import SP_W_PROF
            from models.tt_dit.models.vae.diffvae_ltx_stage5 import _BLOCK_PROF

            calls = ITERS + 2
            merged = {**_BLOCK_PROF, **SP_W_PROF}
            total = sum(merged.values()) or 1.0
            print(f"{'section':44s} {'ms/block':>10} {'share':>7}")
            for key, ms in sorted(merged.items(), key=lambda kv: -kv[1]):
                print(f"{key:44s} {ms / calls:10.2f} {100 * ms / total:6.1f}%", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
