# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Time one DiffVAE module outside pytest.

    python -m models.tt_dit.tools.time_module det_nablock [--arm recommended] [--iters 10]
    python -m models.tt_dit.tools.time_module det_stage1  [--arm qkv_rope]
    python -m models.tt_dit.tools.time_module diff_block  [--grid 121x128x192]
    python -m models.tt_dit.tools.time_module decoder     [--latent-t 19] [--output float,yuv] [--trace]
    python -m models.tt_dit.tools.time_module det_stages  --trace [--latent-t 4]
    python -m models.tt_dit.tools.time_module trace_check [--latent-t 4]

Each target is the twin of a pytest case in ``tests/models/vae/test_diffvae_ltx.py`` and calls the
same ``diffvae_bench`` functions, so the numbers are comparable: the only difference is that the
mesh is opened here rather than by the ``mesh_device`` fixture. Options that the pytest cases read
from the environment (``ITERS``, ``DIFFVAE_LATENT_T``, ``GRID_T``...) are set into the environment
by the matching flag before anything is built, so ``--iters 20`` and ``ITERS=20`` are the same run.

    target       pytest twin
    det_nablock  test_det_nablock_arm_timing      one W-sharded deterministic NABlock per stage 2-4
    det_stage1   test_det_stage1_arm_timing       the replicated stage-1 NABlock
    diff_block   test_diff_block_timing           one stage-5 DiffusionNABlock
    decoder      test_decode_wsp_timing           whole decode in the runner's configuration
                 test_decode_tail_timing          with --output float,yuv
                 test_decode_trace_timing[decode] with --trace
    det_stages   test_decode_trace_timing[det_context]  (--trace is required)
    trace_check  test_trace_reexecutes            proves a captured trace re-executes

The decoder targets need the shipped checkpoint (``DIFFVAE_CHECKPOINT``) and read the same
``DIFFVAE_*`` knobs as the runner scripts; the block targets fill seeded weights and need nothing.
The trace targets additionally need ``TT_DIT_STAGE_TIMING`` and ``DIFFVAE_STAGES_WSP`` unset: the
first syncs inside the capture, the second hangs it (see ``diffvae_bench.trace_replay``).
"""

from __future__ import annotations

import argparse
import os
import sys

import ttnn
from models.tt_dit.tools import diffvae_bench as bench

TARGETS = ("det_nablock", "det_stage1", "diff_block", "decoder", "det_stages", "trace_check")

FABRICS = {"1d": ttnn.FabricConfig.FABRIC_1D, "ring": ttnn.FabricConfig.FABRIC_1D_RING}


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="time_module", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("target", choices=TARGETS)
    p.add_argument("--iters", type=int, help="timed iterations after 2 warm-ups (ITERS)")
    p.add_argument(
        "--arm",
        default="baseline",
        help="det_nablock: baseline|" + "|".join(bench.ARMS) + "; det_stage1: baseline|" + "|".join(bench.STAGE1_ARMS),
    )
    p.add_argument(
        "--grid", help="diff_block: TxHxW stage-5 grid (GRID_T/GRID_H/GRID_W); default the 1080p 25-frame 121x128x192"
    )
    p.add_argument(
        "--latent-t", type=int, help="decoder targets: latent frames (DIFFVAE_LATENT_T); output frames are 8T-7"
    )
    p.add_argument("--output", default="float", help="decoder: comma-separated output types to time, e.g. float,yuv")
    p.add_argument("--trace", action="store_true", help="decoder / det_stages: capture a trace and time the replay")
    p.add_argument("--mesh", default="4x8", help="mesh shape RxC")
    p.add_argument(
        "--fabric",
        choices=list(FABRICS),
        help="override the target's fabric (1d for the block targets, the DIFFVAE_TOPOLOGY-matched one for decoder targets)",
    )
    return p.parse_args(argv)


def _export(name: str, value) -> None:
    if value is not None:
        os.environ[name] = str(value)


def _block_targets(args, mesh) -> None:
    iters = bench.iters_from_env()
    if args.target == "det_nablock":
        arm = bench.arm_flags(args.arm)
        print(f"\n=== det NABlock timing · {args.arm} · {iters} iters ===", flush=True)
        total = 0.0
        for stage in bench.STAGES:
            b = bench.det_block_bench(mesh, stage, arm)
            ms = bench.timed(b, mesh, iters)
            total += ms * b.depth
            print(bench.block_line(b, ms), flush=True)
            b.close()
        print(f"\n[TOTAL W-sharded det blocks] {total:8.1f} ms  ({args.arm})\n", flush=True)
    elif args.target == "det_stage1":
        arm = bench.arm_flags(args.arm, bench.STAGE1_ARMS)
        print(f"\n=== det stage-1 NABlock timing · {args.arm} · {iters} iters ===", flush=True)
        b = bench.stage1_bench(mesh, arm)
        ms = bench.timed(b, mesh, iters)
        print(bench.block_line(b, ms), flush=True)
        print(f"\n[TOTAL stage-1 det blocks] {ms * b.depth:8.1f} ms  ({args.arm})\n", flush=True)
        b.close()
    elif args.target == "diff_block":
        grid = bench.grid_from_env()
        b = bench.diff_block_bench(mesh, grid)
        print(f"\n=== stage-5 DiffusionNABlock · grid {(grid.t, grid.h, grid.w)} · {iters} iters ===", flush=True)
        with bench.timing_tree.span(mesh, "time_module", root=True):
            ms = bench.timed(b, mesh, iters)
        print(bench.block_line(b, ms), flush=True)
        print(f"\n[TOTAL stage-5 blocks] {ms * b.depth:8.1f} ms\n", flush=True)
        if sections := bench.block_sections(iters + 2):
            print(bench.render_sections(sections), flush=True)


def _decoder_targets(args, mesh) -> int:
    with bench.heartbeat():
        dec, config = bench.loaded_production_decoder(mesh)
        if args.target == "trace_check":
            t_lat = bench.latent_t_from_env(4)
            result = bench.trace_validate(
                dec, mesh, bench.latent(config, t_lat, seed=1), bench.latent(config, t_lat, seed=2)
            )
            ok = (
                result.eager_reproducible
                and result.replay_matches_a
                and result.replay_follows_input
                and result.replay_matches_b
            )
            print(f"\n[trace_check] {'PASS' if ok else 'FAIL'}  replay {result.replay_ms:8.1f} ms\n", flush=True)
            return 0 if ok else 1

        if args.trace:
            region = "decode" if args.target == "decoder" else "det_context"
            latent = bench.latent(config, bench.latent_t_from_env(4))
            print(
                f"[setup] latent T={latent.shape[2]} ({8 * latent.shape[2] - 7} frames), tracing {region}", flush=True
            )
            raw = bench.upload_latent(dec, latent, mesh)
            run = bench.trace_region(dec, region, latent, raw)
            report = bench.trace_replay(mesh, run, bench.iters_from_env(3), probe_dispatch=region == "decode")
            return 0 if report.identical else 1

        t_lat = bench.latent_t_from_env()
        latent = bench.latent(config, t_lat, seed=3)
        print(f"[setup] {8 * t_lat - 7} frames, {bench.describe_production_decoder()}", flush=True)
        for kind in args.output.split(","):
            for i in range(bench.iters_from_env(1)):
                out, dt = bench.timed_decode(dec, latent, mesh, output_type=kind)
                print(f"[{kind:5s} {i}] {dt:9.1f} ms  out={tuple(out.shape)}", flush=True)
                del out
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.target == "det_stages" and not args.trace:
        raise SystemExit("det_stages only has a traced instrument; pass --trace (or use `decoder` for an eager decode)")
    _export("ITERS", args.iters)
    _export("DIFFVAE_LATENT_T", args.latent_t)
    if args.grid:
        t, h, w = (int(v) for v in args.grid.lower().split("x"))
        _export("GRID_T", t), _export("GRID_H", h), _export("GRID_W", w)

    block_target = args.target in ("det_nablock", "det_stage1", "diff_block")
    fabric = (
        FABRICS[args.fabric]
        if args.fabric
        else (ttnn.FabricConfig.FABRIC_1D if block_target else bench.fabric_for_topology())
    )
    trace_region_size = bench.TRACE_REGION_SIZE if (args.trace or args.target == "trace_check") else None
    shape = tuple(int(v) for v in args.mesh.lower().split("x"))

    with bench.open_mesh(shape, fabric=fabric, trace_region_size=trace_region_size) as mesh:
        if block_target:
            _block_targets(args, mesh)
            return 0
        return _decoder_targets(args, mesh)


if __name__ == "__main__":
    sys.exit(main())
