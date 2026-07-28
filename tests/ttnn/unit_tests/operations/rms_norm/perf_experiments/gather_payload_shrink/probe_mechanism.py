# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""MECHANISM PROBE for the `gather_payload_shrink` idea (no perf, pure discovery).

Answers one question on real silicon: **where does a `REDUCE_ROW` result land as
a function of which position of the scaler tile carries the 1.0?**  If the
scaler's ROW index selects the output COLUMN then `ht` tile-rows' row-sums can be
column-PACKED into ONE tile and the cross-core gather payload drops `ht`x.

Also probes `reduce_init`'s documented packer edge mask (which zeroes every datum
outside column 0) and whether `reduce_uninit()` clears it in time for the pack.

    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/test_gather_payload_shrink.py -k probe

(The pytest DRIVER has to live under tests/ because pytest.ini uses
--import-mode=importlib, which derives a module name from the rootdir-relative
path: a test file under ttnn/ttnn/... would be imported as `ttnn.ttnn....` and
re-register every C++ op. All experiment code lives here.)
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNELS = Path(__file__).parent / "kernels"

NUM_IN = 8
NUM_SC = 8
NUM_OUT = 16


def _one_core_shard(device, shape, dtype):
    from eval.sharding import shard_config

    return shard_config(
        [shape[-2], shape[-1]],
        (1, 1),
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
    )


def run_probe(device):
    in_shape = (1, 1, 32, 32 * NUM_IN)
    out_shape = (1, 1, 32, 32 * NUM_OUT)

    x = torch.ones(in_shape, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_one_core_shard(device, in_shape, ttnn.float32),
    )
    tt_y = ttnn.from_torch(
        torch.full(out_shape, -7.0, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_one_core_shard(device, out_shape, ttnn.float32),
    )

    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    ct = [NUM_IN, NUM_SC, NUM_OUT]

    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, tt_x),
        ttnn.CBDescriptor(
            total_size=4096 * NUM_SC,
            core_ranges=core,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.float32, page_size=4096)],
        ),
        ttnn.cb_descriptor_from_sharded_tensor(16, tt_y),
    ]

    rt = ttnn.RuntimeArgs()
    rt[0][0] = [0]

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False

    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "probe_reader.cpp"),
                core_ranges=core,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "probe_compute.cpp"),
                core_ranges=core,
                compile_time_args=ct,
                runtime_args=rt,
                config=cfg,
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )

    ttnn.generic_op([tt_x, tt_y], prog)
    out = ttnn.to_torch(tt_y).to(torch.float32)[0, 0]  # [32, 32*NUM_OUT]

    names = {
        0: "sc0 canonical: row0 of every face",
        1: "sc1: FACE-ROW 1 of every face",
        2: "sc2: FACE-ROW 3 of every face",
        3: "sc3: one-hot (f0,r0,c0)",
        4: "sc4: (f0,r0,c5)+(f2,r0,c5)",
        5: "sc5: (f1,r0,c2)+(f3,r0,c2)",
        6: "sc6: row0 of faces 0,1 only",
        7: "sc7: COL 0 of faces 0,1 (all rows)",
        8: "accum sc0+sc1+sc2, packer mask ON",
        9: "accum sc0+sc1+sc2, mask CLEARED",
        10: "canonical over 4 input tiles (mask ON)",
        11: "mul_tiles_bcast_cols(sc7, in0)",
    }

    print("\n===== reduce scaler-position probe (input = all ones) =====")
    for t in range(12):
        tile = out[:, t * 32 : (t + 1) * 32]
        nz = (tile.abs() > 1e-9).nonzero()
        rows = sorted(set(int(r) for r, _ in nz))
        cols = sorted(set(int(c) for _, c in nz))
        vals = sorted(set(round(float(v), 4) for v in tile[tile.abs() > 1e-9]))
        print(f"[{t:2d}] {names[t]:42s} rows={_rng(rows)} cols={_rng(cols)} vals={vals[:6]}")
        if t in (1, 2, 9):
            # show the first 4 rows x 8 cols so the exact placement is legible
            print("      head=", [[round(float(tile[r, c]), 2) for c in range(8)] for r in range(3)])
    print("===== end probe =====\n")


def _rng(v):
    if not v:
        return "[]"
    if v == list(range(v[0], v[-1] + 1)):
        return f"{v[0]}..{v[-1]}"
    return str(v)
