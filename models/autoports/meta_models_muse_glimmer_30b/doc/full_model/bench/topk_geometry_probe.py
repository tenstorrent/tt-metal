# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which ``ttnn.topk`` widths reach the multi-core factory, and what does each cost?

``topk_device_operation.cpp:select_program_factory`` takes the multi-core factory only
when **all** of these hold:

* ``width >= multi_core_min_width`` (8192);
* ``width < 65535`` (multi-core indices are UInt16);
* ``is_power_of_two(width)`` (bitonic sort);
* ``k <= 64``.

This port's per-device vocab shard is **50688** -- not a power of two, so single core.
Padding it to the next power of two gives **65536**, which fails the ``< 65535`` bound,
so it is *still* single core and merely wider.  That is the whole of the
``pad_logits_to_power_of_2`` A/B: both arms ran the same single-core kernel and the ratio
is the width ratio.  Neither arm ever reached a fast path.

The way in is to make each call's width a power of two *below* 65535 -- pad 50688 to
65536 and split it into 2 x 32768.  This measures the op directly, at the real decode
payload ``[1, 1, 32, W]`` bf16 and the real ``k = 32``, so the split is justified by a
measurement rather than by reading the admission rules:

* 50688 -- the shipped shard (single core);
* 65536 -- the padded shard (single core, wider);
* 32768, 16384, 8192 -- power-of-two widths under the uint16 bound (multi-core);
* 4096 -- under ``multi_core_min_width``, so single core again. This is the control that
  matters: it is *smaller* than 32768 and slower, which is how you know the boundary is
  the factory rule and not the width.

Also reported: two 32768 calls back to back, which is what a padded split actually costs,
against one 50688 call.

**No timings are quoted in this docstring on purpose.** They live in
``topk_geometry_probe.json``, which this script writes; quoting them here meant editing
the script every time the numbers were re-measured, which put the script's mtime after
its own artifact and kept re-opening the ordering question.

Usage::

    python doc/full_model/bench/topk_geometry_probe.py [--replays 20]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

ROWS = 32  # the decode payload: DECODE_ROWS
K = 32  # the shipped max_top_k


def say(*args) -> None:
    print(*args, flush=True)


def stage(mesh, width: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """``(values, indices)`` inputs of the real decode shape, replicated."""
    torch.manual_seed(23)
    values = torch.randn(1, 1, ROWS, width, dtype=torch.bfloat16)
    indices = torch.arange(width, dtype=torch.int32).reshape(1, 1, 1, width).repeat(1, 1, ROWS, 1)
    tt_values = ttnn.from_torch(
        values,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    tt_indices = ttnn.from_torch(
        indices,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return tt_values, tt_indices


def time_topk(mesh, width: int, replays: int, *, calls: int = 1) -> dict:
    """Median-of-rounds latency of ``calls`` back-to-back topk calls at ``width``."""
    tt_values, tt_indices = stage(mesh, width)
    stable = ttnn.device.is_wormhole_b0(mesh) or ttnn.device.is_blackhole(mesh)

    def once():
        outs = []
        for _ in range(calls):
            outs.append(ttnn.topk(tt_values, k=K, dim=-1, indices_tensor=tt_indices, stable=stable))
        for values, indices in outs:
            ttnn.deallocate(values)
            ttnn.deallocate(indices)

    once()  # compile
    ttnn.synchronize_device(mesh)
    rounds = []
    for _ in range(3):
        started = time.perf_counter()
        for _ in range(replays):
            once()
        ttnn.synchronize_device(mesh)
        rounds.append((time.perf_counter() - started) / replays * 1e3)
    ttnn.deallocate(tt_values)
    ttnn.deallocate(tt_indices)
    row = {
        "width": width,
        "calls": calls,
        "total_width": width * calls,
        "ms": round(min(rounds), 4),
        "power_of_two": width & (width - 1) == 0,
        "under_uint16": width < 65535,
        "at_least_min_width": width >= 8192,
        "multicore_eligible": (width & (width - 1) == 0) and width < 65535 and width >= 8192 and K <= 64,
    }
    say(f"TOPK {json.dumps(row)}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", default="50688,65536,32768,16384,8192,4096")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--out", default="topk_geometry_probe.json")
    args = parser.parse_args()

    mesh = open_multichip_mesh()
    results = []
    try:
        for width in [int(w) for w in args.widths.split(",")]:
            try:
                results.append(time_topk(mesh, width, args.replays))
            except Exception as exc:  # noqa: BLE001
                say(f"TOPK width={width} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
                results.append({"width": width, "error": str(exc)[:400]})
        # What a padded split would actually cost: two 32768 calls for one 50688 shard.
        try:
            results.append(time_topk(mesh, 32768, args.replays, calls=2))
        except Exception as exc:  # noqa: BLE001
            say(f"TOPK split FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
        out = ROOT / "doc/full_model" / args.out
        out.write_text(json.dumps(results, indent=2) + "\n")
        say(f"TOPK wrote {out}")
        say("TOPK_OK")
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
