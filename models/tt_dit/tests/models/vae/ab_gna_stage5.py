"""A/B one stage-5 DiffusionNABlock: standard neighborhood attention vs Generalized NA (stride > 1).

Answers the two questions a GNA rollout turns on, in one process so weights and inputs are shared:

1. What does it cost in quality? Under a stride the queries in a group stop being centered on their
   own window and share the group leader's, which is not the attention the network was trained for.
   Measured on the bare attention op (``attention_quality``) with NA as the reference, NOT on a block
   forward: ``fill``'s random weights drive a block output to ~1e38, where PCC reports that blowup
   instead of the stride. Isolating the op also isolates the one thing GNA actually changes.
2. What does it buy in time? Per-block ms for each arm, on the real stage-5 block.

The stride is taken from the Q block by default (``DIFFVAE_GNA=1``), the setting that collapses each
chunk's neighborhood box to a single shared window and makes the chunk perfectly block-sparse.
``GNA_STRIDE="t,h,w"`` A/Bs an explicit stride instead.

Both arms are pinned to the production fast path (``DIFFVAE_BLOCK`` + ``DIFFVAE_SP_FUSED``), which is
also what makes the comparison legal: the block-permuted fused kernel is the only path that derives a
stride from the Q block, so without it the GNA arm would quietly fall back to stride 1 and the A/B would
be measuring NA against itself. ``_STRIDES_SEEN`` observes what actually reached the op so that failure
mode is an assertion rather than a 1.00x result that looks like a finding.

Env: ``ITERS``, ``GRID_T``/``GRID_H``/``GRID_W`` (as ``time_diff_block``), ``GNA_STRIDE``.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.tt_dit.layers.na3d import SP_W_PROF
from models.tt_dit.models.vae.diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config, Grid
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.vae.time_det_nablock import fill

SP_AXIS, TP_AXIS = 1, 0
ITERS = int(os.environ.get("ITERS", 10))

# Shared by both arms: the block-permuted fused kernel. Set before the model is built so neither arm can
# race a default. Only DIFFVAE_GNA / DIFFVAE_GNA_STRIDE differ between arms.
os.environ.setdefault("DIFFVAE_BLOCK", "1")
os.environ.setdefault("DIFFVAE_SP_FUSED", "1")

# The shipped 1080p 6s grid. GNA's speedup is not scale-invariant, so a reduced grid does not report a
# smaller version of the same number -- it reports a different one. Its win comes from a block of queries
# sharing one kernel-sized window, and the block must divide every axis, so the queries-per-window this
# geometry admits (T = 145 = 5*29 forces bt = 5) is nothing like what a rounder T admits.
GRID = Grid(
    batch=1,
    t=int(os.environ.get("GRID_T", 145)),
    h=int(os.environ.get("GRID_H", 272)),
    w=int(os.environ.get("GRID_W", 480)),
)


def _flat(x: torch.Tensor, channels: int) -> torch.Tensor:
    return x.reshape(1, x.shape[0], -1, channels)


_STRIDES_SEEN: set[tuple[int, int, int] | None] = set()


def _install_sdpa_probe() -> None:
    """Record the ``neighborhood_stride`` each SDPA call actually receives.

    na3d resolves the stride from env at call time and can legitimately return None (no legal block for
    the geometry), which would make the GNA arm identical to NA. Observing the kwarg at the op boundary
    is the only check that cannot drift from that logic, so the arms are compared on what ran.
    """
    inner = ttnn.transformer.scaled_dot_product_attention

    def probed(*args, **kwargs):
        stride = kwargs.get("neighborhood_stride")
        _STRIDES_SEEN.add(tuple(stride) if stride is not None else None)
        return inner(*args, **kwargs)

    ttnn.transformer.scaled_dot_product_attention = probed


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.flatten().double(), b.flatten().double()
    x, y = x - x.mean(), y - y.mean()
    denom = (x.norm() * y.norm()).item()
    return 1.0 if denom == 0.0 else (x @ y).item() / denom


def _correlated(shape: tuple[int, ...], corr: int, g: torch.Generator) -> torch.Tensor:
    """Random (1, T, H, W, NH, HD) with a spatial correlation length of ``corr`` tokens.

    White noise is the wrong input for judging a stride. Attention over iid keys is close to a plain
    mean over the window, so PCC between two windows degenerates to their overlap fraction and reports
    the geometry rather than any property of the data. Real latents at this depth are smooth over
    several tokens, which is the regime that decides whether sharing a window is survivable.

    corr == 1 is white noise; larger values upsample coarse noise trilinearly.
    """
    b, t, h, w, nh, hd = shape
    if corr <= 1:
        return torch.randn(*shape, generator=g)
    coarse = torch.randn(1, nh * hd, max(t // corr, 1), max(h // corr, 1), max(w // corr, 1), generator=g)
    fine = torch.nn.functional.interpolate(coarse, size=(t, h, w), mode="trilinear", align_corners=False)
    # Upsampling averages independent samples, shrinking variance; renormalize so rms matches white
    # noise and the PCCs stay comparable across correlation lengths.
    fine = fine / fine.std()
    return fine.reshape(1, nh, hd, t, h, w).permute(0, 3, 4, 5, 1, 2).contiguous()


def attention_quality(mesh, kernel: tuple[int, int, int], heads: int = 1, corr: int = 1) -> None:
    """PCC between the NA and GNA attention outputs at the production geometry.

    Deliberately measures the bare attention op rather than a block forward. GNA changes exactly one
    thing -- which keys a query sees -- so comparing attention outputs on the same unit-scale Q/K/V is
    the cleanest read on what it costs. A block forward under ``fill``'s random weights reaches ~1e38
    and its PCC is dominated by that blowup, not by the stride.

    Both arms are the same op on the same inputs, so bf16 rounding is common-mode and the residual is
    the semantic change: under a stride a query attends to its group leader's window instead of its own,
    displaced by up to half the group extent.
    """
    from models.tt_dit.layers.na3d import neighborhood_attention_3d_op_sp_w_sharded
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import from_torch
    from models.tt_dit.utils.tensor import to_torch as to_torch_replicated

    T, H, W = GRID.t, GRID.h, GRID.w
    g = torch.Generator().manual_seed(0)
    q, k, v = (_correlated((1, T, H, W, heads, 64), corr, g) for _ in range(3))

    shard_axes = [None] * 6
    shard_axes[3] = SP_AXIS
    q_tt, k_tt, v_tt = (
        from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=shard_axes)
        for x in (q, k, v)
    )
    ccl = CCLManager(mesh, num_links=1, topology=ttnn.Topology.Linear)

    def run(gna_stride, from_block: bool = False) -> torch.Tensor:
        os.environ["DIFFVAE_GNA"] = "1" if from_block else "0"
        _STRIDES_SEEN.clear()
        out = neighborhood_attention_3d_op_sp_w_sharded(
            q_tt,
            k_tt,
            v_tt,
            dims=(T, H, W),
            kernel_size=kernel,
            sp_axis=SP_AXIS,
            ccl_manager=ccl,
            scale=1.0,
            gna_stride=gna_stride,
        )
        return to_torch_replicated(out, mesh_axes=[None, None, None, SP_AXIS, None]).float()

    na = run(None)
    rms = na.pow(2).mean().sqrt().item()

    # A stride only shares a window where it is > 1, so the sweep walks one axis at a time and then
    # combines, to show which axis the quality actually pays for. Every entry must divide its axis
    # (121 = 11*11, 128, 192/8 = 24 per shard) and stay <= the kernel, or the op rejects it host-side.
    sweep: list[tuple[str, tuple[int, int, int] | None, bool]] = [
        ("stride (1,1,2)  w only", (1, 1, 2), False),
        ("stride (1,2,1)  h only", (1, 2, 1), False),
        ("stride (11,1,1) t only", (11, 1, 1), False),
        ("stride (1,2,2)", (1, 2, 2), False),
        ("stride (1,4,4)", (1, 4, 4), False),
        ("stride (11,4,8) = Q block", None, True),
    ]
    print(
        f"\n[quality] attention-only, kernel {kernel}, grid {(T, H, W)}, {heads} head(s), "
        f"correlation length {corr} token(s), rms(NA) = {rms:.4f}\n"
        f"[quality] NA is the reference; rel-rms is the error relative to the signal (1.0 = error as big\n"
        f"[quality] as the output itself). Physical (t,h,w) stride.\n"
        f"  {'stride':28s} {'PCC':>9} {'rel-rms':>9} {'max|d|':>9}",
        flush=True,
    )
    for label, stride, from_block in sweep:
        got = run(stride, from_block=from_block)
        seen = sorted(_STRIDES_SEEN, key=str)
        delta = na - got
        print(
            f"  {label:28s} {_pcc(na, got):9.4f} {delta.pow(2).mean().sqrt().item() / rms:9.4f} "
            f"{delta.abs().max().item():9.4f}   op saw {seen}",
            flush=True,
        )


def main() -> None:
    ring = os.environ.get("DIFFVAE_TOPOLOGY", "linear").lower() == "ring"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if ring else ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
    try:
        cfg = DiffVAEStage5Config()
        sp = int(list(mesh.shape)[SP_AXIS])
        assert GRID.w % sp == 0, f"W={GRID.w} not divisible by sp={sp}"

        ccl = CCLManager(
            mesh,
            num_links=int(os.environ.get("LINKS", 1)),
            topology=ttnn.Topology.Ring if ring else ttnn.Topology.Linear,
        )
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

        tt_context = ttnn.from_torch(
            _flat(context, cfg.context_channels).contiguous(), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_context = model._wshard_context(tt_context, GRID)
        bands = model.bands(GRID)
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

        stride_env = os.environ.get("GNA_STRIDE")
        arms = [("NA (stride 1)", {"DIFFVAE_GNA": "0", "DIFFVAE_GNA_STRIDE": ""})]
        if stride_env:
            arms.append((f"GNA (stride {stride_env})", {"DIFFVAE_GNA": "0", "DIFFVAE_GNA_STRIDE": stride_env}))
        else:
            arms.append(("GNA (stride = Q block)", {"DIFFVAE_GNA": "1", "DIFFVAE_GNA_STRIDE": ""}))

        print(
            f"\n=== stage-5 GNA A/B · grid {(GRID.t, GRID.h, GRID.w)} · {GRID.sites} sites "
            f"({sites_local}/chip) · {len(bands)} band(s) · {ITERS} iters ===",
            flush=True,
        )

        _install_sdpa_probe()

        results = []
        for name, env in arms:
            for key, val in env.items():
                if val == "":
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

            # Timing only. Quality is measured separately on the bare attention op (attention_quality):
            # under fill()'s random weights a block forward reaches ~1e38, so its PCC would report the
            # blowup rather than the stride.
            _STRIDES_SEEN.clear()
            x_bands = model.embed_x_t(x_t, bands)
            for _ in range(2):  # warm the program cache (stride is part of the cache key)
                x_bands = block(x_bands, tt_context, modulation, GRID, bands, band_tables)
            ttnn.synchronize_device(mesh)
            strides = sorted(_STRIDES_SEEN, key=lambda s: (s is not None, s))
            print(f"[{name:24s}] neighborhood_stride reaching the op: {strides}", flush=True)

            # SP_W_PROF accumulates across calls, so it has to be zeroed per arm or the second arm
            # reports both. Only populated when DIFFVAE_STAGE_TIMING was set before na3d was imported.
            SP_W_PROF.clear()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                x_bands = block(x_bands, tt_context, modulation, GRID, bands, band_tables)
            ttnn.synchronize_device(mesh)
            per_block = (time.perf_counter() - t0) / ITERS * 1000
            sections = {k: v / ITERS for k, v in SP_W_PROF.items()}
            for key, ms in sorted(sections.items(), key=lambda kv: -kv[1]):
                print(
                    f"[{name:24s}]   {key:16s} {ms:8.2f} ms/block ({100 * ms / per_block:4.1f}% of block)", flush=True
                )

            results.append((name, per_block, strides, sections))
            print(
                f"[{name:24s}] {per_block:8.2f} ms/block  x{cfg.num_blocks} = {per_block * cfg.num_blocks:8.1f} ms",
                flush=True,
            )

        (na_name, na_ms, na_strides, na_sec), (gna_name, gna_ms, gna_strides, gna_sec) = results
        assert na_strides == [None], f"NA arm should run at no stride, saw {na_strides}"
        assert gna_strides != [None], (
            "GNA arm ran at stride 1 -- no legal block for this geometry, so this would compare NA to "
            "itself. Pass GNA_STRIDE=t,h,w explicitly or pick a divisible grid."
        )

        print(f"\n{'':26s} {'ms/block':>10} {'x' + str(cfg.num_blocks):>10} {'speedup':>9}")
        print(f"{na_name:26s} {na_ms:10.2f} {na_ms * cfg.num_blocks:10.1f} {1.0:8.2f}x")
        print(f"{gna_name:26s} {gna_ms:10.2f} {gna_ms * cfg.num_blocks:10.1f} {na_ms / gna_ms:8.2f}x")

        # Per-section speedup, and the Amdahl ceiling the untouched sections impose: a stride only
        # changes the SDPA, so everything else in the block is a floor on what any stride can win.
        if na_sec:
            print(f"\n{'section':18s} {'NA ms':>9} {'GNA ms':>9} {'speedup':>9}")
            for key in sorted(set(na_sec) | set(gna_sec), key=lambda k: -na_sec.get(k, 0.0)):
                a, b = na_sec.get(key, 0.0), gna_sec.get(key, 0.0)
                print(f"{key:18s} {a:9.2f} {b:9.2f} {(a / b if b else float('nan')):8.2f}x")
            na_attn = na_sec.get("fused-sdpa", na_sec.get("op-sdpa", 0.0))
            rest = na_ms - sum(na_sec.values())
            print(
                f"{'block rest':18s} {rest:9.2f} {gna_ms - sum(gna_sec.values()):9.2f}        1.00x  "
                f"(projections/FFN/RoPE/norms -- stride cannot touch these)\n"
                f"[amdahl] SDPA is {100 * na_attn / na_ms:.1f}% of the NA block, so a free SDPA would cap the "
                f"block at {na_ms / max(na_ms - na_attn, 1e-9):.2f}x; measured {na_ms / gna_ms:.2f}x"
            )

        for corr in (int(c) for c in os.environ.get("GNA_CORR", "1,4,8").split(",")):
            attention_quality(mesh, kernel=cfg.kernel_size, corr=corr)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
