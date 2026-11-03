# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of ``row_gather2`` for 1..8 live rows at the verify call sites' shapes and memory configs: the rows 3..8
run the barrier + CB landing program (``kernels/transport8.cpp``); rows 1 and 2 keep the barrier-free program.

Cases (call site -> input rows S, live rows R, gathered memory):
  * ``moe``  : S = R logical rows (MoE ``_all_gather_hidden``), five-core L1 width shard (32 x 512) -- no ``live_rows``;
  * ``gdn``  : S = 32 with R live rows, rows >= R zero in the input (GDN ``_all_gather_rows`` / QSA verify hidden), eight-core
               L1 width shard (32 x 320), ``live_rows=R``;
  * ``gdn_dirty``: as ``gdn`` but rows >= R of the INPUT hold random bits: the gathered rows >= R must still be zero;
  * ``dram`` : S = R logical rows, DRAM interleaved output.
Source memory: DRAM interleaved or L1 interleaved.  Per case and seed the fused gather must equal ``ttnn.all_gather`` on the live
rows 0..R-1 bit for bit on every device (eager, 4 repeats, a 3-call trace replayed 3 times); for S = 32 and R >= 3 the gathered
rows >= R must be exact zeros (and, for the clean input, the whole tile equal to the stock gather).  A rows-1/2 or S = 32 /
R <= 2 call goes through ``row_gather_or_all_gather`` and reports which path it took.  Logical S < 32 outputs expose only
their S rows to the host, so the zero check runs on the S = 32 cases (same kernel, same page layout).  One trace with every
case interleaved checks that the 1-/2-row program and the rows program share a mesh.  Traced time per call (slope of a 12-op
and a 4-op trace): stock ``ttnn.all_gather`` vs the fused call.  One ROW_GATHER_ROWS_RESULT json line per case.

Run:  flock -w 7200 /home/ttuser/atupe/device.lock.g3 timeout 3000 python -m pytest \
          models/demos/blackhole/qwen38_flash_next/tests/test_fused_row_gather_rows_device.py -s -p no:cacheprovider -o addopts=
"""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import test_fused_row_reduce_device as base
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_row_reduce_device import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather2 as rg2

ROWS = [1, 2, 3, 5, 8]
SHARD = 320  # hidden 2560 / TP 8
HIDDEN = 2560
SEEDS = [11, 22]
SITES = ["moe", "gdn", "gdn_dirty", "dram"]
SOURCES = ["dram", "l1"]
RESULTS = os.environ.get("ROW_GATHER_RESULTS")


def _out_config(site):
    if site == "moe":
        return ttnn.create_sharded_memory_config(
            (ttnn.TILE_SIZE, HIDDEN // 5),
            ttnn.CoreGrid(x=5, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    if site in ("gdn", "gdn_dirty"):
        return ttnn.create_sharded_memory_config(
            (ttnn.TILE_SIZE, HIDDEN // 8),
            ttnn.CoreGrid(x=8, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    return ttnn.DRAM_MEMORY_CONFIG


def _shard(mesh, site, rows, source, seed):
    """The per-device shard ``[1, 1, S, 320]`` bf16 TILE (device d holds block d) and its host copy ``[8, 1, S, 320]``."""

    g = torch.Generator().manual_seed(seed)
    s_rows = 32 if site.startswith("gdn") else rows
    x = torch.randn(8, 1, s_rows, SHARD, generator=g).to(torch.bfloat16)
    if site.startswith("gdn"):
        if site == "gdn":
            x[:, :, rows:, :] = 0
        else:
            x[:, :, rows:, :] = torch.randn(8, 1, 32 - rows, SHARD, generator=g).to(torch.bfloat16) * 100
    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG if source == "dram" else ttnn.L1_MEMORY_CONFIG,
    )
    return t, x


def _live_arg(site, rows):
    return rows if site.startswith("gdn") else None


def _host(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _same_bits(a, b):
    return int((a.to(torch.bfloat16).view(torch.int16) != b.to(torch.bfloat16).view(torch.int16)).sum())


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [base.DEVICE_PARAMS], indirect=True)
def test_row_gather_rows_is_bitwise_the_stock_all_gather(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    os.environ.pop(rg2.ROWS_ENV, None)
    assert rg2.rows_max() == 8
    failures, reports, kept = [], [], []
    for site in SITES:
        cfg = _out_config(site)
        for source in SOURCES:
            for rows in ROWS:
                live = _live_arg(site, rows)
                report = {"site": site, "source": source, "rows": rows, "mismatch": {}}
                try:
                    for seed in SEEDS:
                        x, host = _shard(mesh, site, rows, source, seed * 100 + rows)
                        fused_path = rg2.admits(x, cluster_axis=1, memory_config=cfg, live_rows=live)
                        report["path"] = "rows" if fused_path and rows > 2 else ("packed" if fused_path else "stock")

                        def stock_call():
                            return rg2.composed(x, cluster_axis=1, memory_config=cfg)

                        def new_call():
                            return rg2.row_gather_or_all_gather(x, cluster_axis=1, memory_config=cfg, live_rows=live)

                        want = _host(mesh, stock_call())[:, :, :rows, :]
                        got_t = new_call()
                        got_full = _host(mesh, got_t)
                        got = got_full[:, :, :rows, :]
                        # torch reference of the gather: every device holds [x_0 | ... | x_7] on the live rows
                        ref = torch.cat([host[d : d + 1, :, :rows, :] for d in range(8)], dim=3).repeat(8, 1, 1, 1)
                        key = f"s{seed}"
                        report["mismatch"][key + "_vs_stock"] = _same_bits(got, want)
                        report["mismatch"][key + "_vs_torch"] = _same_bits(got, ref)
                        if site.startswith("gdn") and rows > 2:
                            tail = got_full[:, :, rows:, :]
                            report["mismatch"][key + "_dead_rows_nonzero"] = int((tail.float() != 0).sum())
                            if site == "gdn":
                                stock_full = _host(mesh, stock_call())
                                report["mismatch"][key + "_whole_tile_vs_stock"] = _same_bits(got_full, stock_full)
                        report["mismatch"][key + "_repeat"] = sum(
                            _same_bits(_host(mesh, new_call())[:, :, :rows, :], want) for _ in range(4)
                        )
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        outs = [new_call() for _ in range(3)]
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        bad = 0
                        for _ in range(3):
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                            for o in outs:
                                full = _host(mesh, o)
                                bad += _same_bits(full[:, :, :rows, :], want)
                                if site.startswith("gdn") and rows > 2:
                                    bad += int((full[:, :, rows:, :].float() != 0).sum())
                        ttnn.release_trace(mesh, tid)
                        report["mismatch"][key + "_trace"] = bad
                    us = {
                        "stock_all_gather": base._traced_us(
                            mesh, lambda: rg2.composed(x, cluster_axis=1, memory_config=cfg)
                        ),
                        "fused": base._traced_us(
                            mesh,
                            lambda: rg2.row_gather_or_all_gather(x, cluster_axis=1, memory_config=cfg, live_rows=live),
                        ),
                    }
                    report["us_per_op_slope_and_mean12"] = us
                    kept.append((site, source, rows, x, cfg, live, want))
                except Exception as e:  # noqa: BLE001 (report the case, run the others)
                    report["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                ok = "error" not in report and all(v == 0 for v in report["mismatch"].values())
                report["pass"] = ok
                print("ROW_GATHER_ROWS_RESULT " + json.dumps(report), flush=True)
                reports.append(report)
                if not ok:
                    failures.append((site, source, rows))
    # every case interleaved in one trace: the 1-/2-row program (persistent landing buffer, parity counter) and the rows
    # program (barrier semaphores, CB landing) alternate on the same cores
    if kept and not failures:
        # the moe / gdn outputs are 32 KB L1 width shards per core: every one of the 40 cases at once would fill the cores' L1 up to
        # the programs' CBs, so one case per (site, rows) shape alternates the two programs, three L1 outputs in all
        order = [i for i in (0, 31, 3, 33, 14, 32, 34) if i < len(kept)]
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = [
            (
                i,
                rg2.row_gather_or_all_gather(
                    kept[i][3], cluster_axis=1, memory_config=kept[i][4], live_rows=kept[i][5]
                ),
            )
            for i in order
        ]
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        bad = 0
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            for i, o in outs:
                site, _, rows, _, _, _, want = kept[i]
                full = _host(mesh, o)
                bad += _same_bits(full[:, :, :rows, :], want)
                if site.startswith("gdn") and rows > 2:
                    bad += int((full[:, :, rows:, :].float() != 0).sum())
        ttnn.release_trace(mesh, tid)
        print("ROW_GATHER_ROWS_INTERLEAVED_MISMATCH", bad, flush=True)
        if bad:
            failures.append(("interleaved", bad))
    if RESULTS:
        with open(RESULTS, "a") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
    assert not failures, failures
