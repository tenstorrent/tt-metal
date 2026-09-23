# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rank SDPA program configurations by device kernel time.

The encoder calls one SDPA per layer, and at B8/S512 that op holds 24% of the
forward. `_sdpa_program_config` in tt/attention.py carries six knobs, and the
notes beside it were measured when Q, K and V were bfloat16. They now arrive in
bfloat8_b, so the operands are half the size and the earlier optimum no longer
follows.

This sweep builds Q, K and V exactly as the encoder hands them to SDPA, launches
one configuration at a time, and reads the device kernel duration the profiler
records. It reports the same quantity as a Tracy capture: last kernel end minus
first kernel start over the op's cores. It does not run the model and it does
not check PCC, so a winner still has to be applied and gated.

    TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 \
    python_env/bin/python models/demos/wormhole/bge_m3/tests/perf/sdpa_sweep.py --batch 8

Knobs, and why each is swept:

  q_chunk_size, k_chunk_size  how the kernel cuts the score matrix. The cut
                              decides both the L1 working set and how much
                              parallel work exists.
  compute_with_storage_grid_size  the core rectangle. More cores is not always
                              better: B1 measured 8x8 above 11x10.
  max_cores_per_head_batch    cores one (batch, head) pair may take. Caps
                              fan-out so several pairs run at once.
  exp_approx_mode             the softmax exponential. Measured numerically
                              inert at S512, so it is a speed knob here.
  fidelity                    HiFi2 against HiFi4 on the score matmuls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

import ttnn

HEAD_DIM = 64
NUM_HEADS = 16
WARMUP_LAUNCHES = 2
MEASURED_LAUNCHES = 5
# Configurations per device open. The profiler's host buffer holds a bounded
# number of zones, so a run that launches thousands of times before reading
# loses the tail: 1502 configurations returned 3834 durations for 10514
# launches, and a fixed-stride slice then labels every row with another row's
# time. A small chunk keeps the read inside the buffer.
CHUNK_SIZE = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--dtype", default="bfloat8_b", choices=["bfloat8_b", "bfloat16"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--budget-s", type=float, default=900.0, help="stop enumerating after this long")
    p.add_argument("--top", type=int, default=12, help="rows to print")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def legal_configs(seq_len, grid_x_max, grid_y_max):
    """Every configuration worth launching at this sequence length.

    A chunk must divide the sequence. The kernel works in tiles, so 32 is the
    smallest chunk, but a chunk that small leaves one tile of work per step and
    only ever measured dispatch, so the range starts at 64.

    Grids are the rectangles that fit the board, widest first: the model's own
    notes record 11x10 beating 8x8 at this batch, and a budget that runs out
    should have covered the wide ones.
    """
    chunks = [c for c in (64, 128, 256, 512) if c <= seq_len and seq_len % c == 0]
    grids = [(grid_x_max, grid_y_max), (11, 10), (10, 10), (8, 8)]
    grids = [g for g in grids if g[0] <= grid_x_max and g[1] <= grid_y_max]
    out = []
    for q_chunk in chunks:
        for k_chunk in chunks:
            for grid in grids:
                for max_cores in (None, 4, 8, 16):
                    for exp_approx in (False, True):
                        out.append(
                            {
                                "q_chunk_size": q_chunk,
                                "k_chunk_size": k_chunk,
                                "grid": grid,
                                "max_cores_per_head_batch": max_cores,
                                "exp_approx_mode": exp_approx,
                            }
                        )
    return out


def build_program_config(cfg):
    kwargs = {
        "compute_with_storage_grid_size": ttnn.CoreCoord(cfg["grid"][0], cfg["grid"][1]),
        "q_chunk_size": cfg["q_chunk_size"],
        "k_chunk_size": cfg["k_chunk_size"],
        "exp_approx_mode": cfg["exp_approx_mode"],
    }
    if cfg["max_cores_per_head_batch"] is not None:
        kwargs["max_cores_per_head_batch"] = cfg["max_cores_per_head_batch"]
    return ttnn.SDPAProgramConfig(**kwargs)


DEVICE_LOG = os.path.join(
    os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs", "profile_log_device.csv"
)


def launch_durations_us():
    """Per-launch device kernel duration, in microseconds.

    Reads the profiler's per-op list, not the aggregated per-device analysis:
    the aggregate blends the compile, the warm-ups and the measured launches
    into one average, so a number read from it would include compilation.

    This is the quantity a Tracy capture reports as DEVICE KERNEL DURATION.
    """
    from tracy.device_post_proc_config import default_setup
    from tracy.process_device_log import import_log_run_stats

    # default_setup() already resolves deviceInputLog to an absolute path under
    # the tt-metal tree. Overriding it with a relative one breaks as soon as the
    # process runs from anywhere else.
    setup = default_setup()
    if not os.path.exists(setup.deviceInputLog):
        return []
    data = import_log_run_stats(setup)
    # A chunk whose every configuration was refused launches nothing, so the log
    # carries no device. Report no durations rather than raising.
    devices = data.get("devices") or {}
    if 0 not in devices:
        return []
    freq = data["deviceInfo"]["freq"]
    risc = devices[0]["cores"]["DEVICE"]["riscs"]["TENSIX"]
    out = []
    for op in risc.get("ops", []):
        analysis = op.get("analysis", {}).get("device_kernel_duration")
        if analysis:
            out.append(analysis["stats"]["Max"] / freq)
    return out


def main():
    args = parse_args()
    if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "1":
        raise SystemExit("TT_METAL_DEVICE_PROFILER=1 is required: the sweep reads device kernel time.")

    dtype = ttnn.bfloat8_b if args.dtype == "bfloat8_b" else ttnn.bfloat16
    batch, seq_len = args.batch, args.seq_len
    scale = HEAD_DIM**-0.5

    # The encoder passes a compute kernel config, and it is part of the op: with
    # fp32_dest_acc_en the kernel holds fp32 intermediates, which doubles the
    # circular buffers. A sweep that left it unset ranked q512/k512 first and the
    # model then refused it, because the real op needs 2290816 B of circular
    # buffers against 1572864 B of L1.
    from models.demos.wormhole.bge_m3.tt.optimizations import sdpa_compute_kernel_config

    probe = ttnn.open_device(device_id=args.device_id)
    grid = probe.compute_with_storage_grid_size()
    grid_x, grid_y = int(grid.x), int(grid.y)
    compute_kernel_config = sdpa_compute_kernel_config(probe, max_seq_len=seq_len, max_batch_size=batch, dtype=dtype)
    ttnn.close_device(probe)
    print(f"device grid {grid_x}x{grid_y}  B{batch} S{seq_len} heads {NUM_HEADS} head_dim {HEAD_DIM} {args.dtype}")
    print(
        "compute kernel: %s fp32_dest_acc_en=%s packer_l1_acc=%s"
        % (
            compute_kernel_config.math_fidelity,
            compute_kernel_config.fp32_dest_acc_en,
            compute_kernel_config.packer_l1_acc,
        )
    )

    shape = (batch, NUM_HEADS, seq_len, HEAD_DIM)
    configs = legal_configs(seq_len, grid_x, grid_y)
    print(f"enumerated {len(configs)} configurations, {CHUNK_SIZE} per device open, budget {args.budget_s:.0f} s\n")

    per_config = WARMUP_LAUNCHES + MEASURED_LAUNCHES
    results = []
    started = time.time()

    for base in range(0, len(configs), CHUNK_SIZE):
        if time.time() - started > args.budget_s:
            print(f"budget spent after {base} of {len(configs)} configurations")
            break
        chunk = configs[base : base + CHUNK_SIZE]

        # A fresh log per chunk, so the durations read back belong to this chunk
        # alone and position maps to configuration.
        if os.path.exists(DEVICE_LOG):
            os.remove(DEVICE_LOG)

        device = ttnn.open_device(device_id=args.device_id)
        # Rebuilt per device open: the object is bound to the device it came from.
        chunk_compute_kernel_config = sdpa_compute_kernel_config(
            device, max_seq_len=seq_len, max_batch_size=batch, dtype=dtype
        )
        torch.manual_seed(0)
        tensors = {
            name: ttnn.from_torch(
                torch.randn(shape, dtype=torch.bfloat16),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for name in ("q", "k", "v")
        }

        launched = []
        for cfg in chunk:
            try:
                program_config = build_program_config(cfg)
                for _ in range(per_config):
                    out = ttnn.transformer.scaled_dot_product_attention(
                        tensors["q"],
                        tensors["k"],
                        tensors["v"],
                        is_causal=False,
                        scale=scale,
                        program_config=program_config,
                        compute_kernel_config=chunk_compute_kernel_config,
                    )
                    ttnn.deallocate(out)
                ttnn.synchronize_device(device)
                launched.append(cfg)
            except Exception as exc:  # an illegal configuration is an answer, not a stop
                first = str(exc).split("\n")[0][:160]
                results.append({**cfg, "us": None, "error": first})
                if len([r for r in results if r.get("error")]) <= 3:
                    print(f"  refused: {first}")

        # The profiler writes its CSV when the device closes, not when
        # ReadDeviceProfiler runs, so the read has to come after the close.
        ttnn.close_device(device)

        durations = launch_durations_us()
        expected = len(launched) * per_config
        if launched and len(durations) < expected:
            print(f"  chunk at {base}: {len(durations)} durations for {expected} launches")
        for position, cfg in enumerate(launched):
            window = durations[position * per_config : (position + 1) * per_config]
            samples = [s for s in window[WARMUP_LAUNCHES:] if not math.isnan(s)]
            if not samples:
                results.append({**cfg, "us": None, "error": "profiler recorded no launch"})
                continue
            results.append({**cfg, "us": min(samples), "cores": cfg["grid"][0] * cfg["grid"][1]})

        done = min(base + CHUNK_SIZE, len(configs))
        measured = len([r for r in results if r.get("us")])
        print(f"  {done}/{len(configs)} configurations, {measured} measured, {time.time() - started:.0f} s")

    ranked = sorted((r for r in results if r.get("us")), key=lambda r: r["us"])
    failed = [r for r in results if not r.get("us")]
    print(f"measured {len(ranked)}  refused {len(failed)}\n")
    print("  %8s  %6s %6s %8s %10s %8s" % ("us", "q", "k", "grid", "max_cores", "exp"))
    for row in ranked[: args.top]:
        print(
            "  %8.1f  %6d %6d %8s %10s %8s"
            % (
                row["us"],
                row["q_chunk_size"],
                row["k_chunk_size"],
                "%dx%d" % row["grid"],
                row["max_cores_per_head_batch"],
                row["exp_approx_mode"],
            )
        )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump({"batch": batch, "seq_len": seq_len, "dtype": args.dtype, "results": results}, handle, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
