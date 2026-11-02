# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of ``row_reduce`` for 1..8 valid rows (bf16 DRAM, bf16 L1 width sharded, fp32 DRAM): the rows above the
packed multicast's limits (bf16 3..8, fp32 2..8) run the unicast slice mode (kernels/transport4.cpp, transport4l.cpp).

Per case (dtype, memory, rows in 1, 2, 3, 5, 8) and per seed, 1 link and 2 links: ``row_reduce`` must equal on the live
rows 0..rows-1, bit for bit,
  * ``ttnn.reduce_scatter(dim=3, cluster_axis=1, topology=Linear)`` (gdn / qsa), and
  * ``tt_all_reduce`` on the 1x8 mesh = ``reduce_scatter_minimal_async`` (moe),
including 4 repeated calls and a 3-call trace replayed 3 times.  Traced time per call (slope of a 12-op and a 4-op trace):
stock, the moe minimal async op, row_reduce with 1 link and with 2 links.  The test runs every case, prints one
ROW_REDUCE_ROWS_RESULT json line per case and asserts at the end that no case had a mismatch.

Run:  flock -w 10800 /home/ttuser/atupe/device.lock timeout 1800 python -m pytest \
          models/demos/blackhole/qwen38_flash_next/tests/test_fused_row_reduce_rows_device.py -s -p no:cacheprovider -o addopts=
"""

import json
import os

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import test_fused_row_reduce_device as base
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_row_reduce_device import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_reduce as rr

ROWS = [1, 2, 3, 5, 8]
MEMS = [("bf16", False), ("bf16", True), ("fp32", False)]  # (dtype, L1 width sharded: 16 cores as the GDN partial)
SEEDS = [11, 22]
RESULTS = os.environ.get("ROW_REDUCE_RESULTS")


def _call(partial):
    return rr.row_reduce(partial, cluster_axis=1)


def _set_links(links):
    os.environ[rr.LINKS_ENV] = str(links)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [base.DEVICE_PARAMS], indirect=True)
def test_row_reduce_rows_is_bitwise_the_stock_reduce_scatter(mesh_device):
    from models.tt_transformers.tt.ccl import TT_CCL, tt_all_reduce

    mesh = mesh_device
    mesh.enable_program_cache()
    os.environ[rr.ENV] = "2"
    rr.prepare(mesh)
    tt_ccl = TT_CCL(mesh)

    def minimal(src):
        return ttnn.experimental.reduce_scatter_minimal_async(
            src,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    failures, reports, kept = [], [], []
    for dtype, sharded in MEMS:
        for rows in ROWS:
            report = {"dtype": dtype, "l1_sharded": sharded, "rows": rows, "mismatch": {}}
            try:
                for seed in SEEDS:
                    x = base._partials(dtype, rows, seed=seed * 100 + rows)
                    partial = base._to_mesh(mesh, x, dtype, sharded)
                    assert rr.admits(partial), "the contract must admit this partial"
                    stock = base._from_mesh(mesh, rr.composed(partial), rows)
                    try:
                        helper = base._from_mesh(
                            mesh,
                            tt_all_reduce(
                                base._to_mesh(mesh, x, dtype),
                                mesh,
                                tt_ccl,
                                cluster_axis=0,
                                dim=3,
                                num_reduce_scatter_links=1,
                                num_all_gather_links=1,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                topology=ttnn.Topology.Linear,
                            ),
                            rows,
                        )
                        report["mismatch"][f"s{seed}_stock_rs_vs_tt_all_reduce"] = base._mismatches(
                            stock, helper, dtype
                        )
                    except Exception as e:  # noqa: BLE001 (the moe helper may not take fp32)
                        helper = None
                        report["tt_all_reduce_error"] = type(e).__name__
                    for links in (1, 2):
                        _set_links(links)
                        got = base._from_mesh(mesh, _call(partial), rows)
                        key = f"s{seed}_l{links}"
                        report["mismatch"][key + "_vs_stock_rs"] = base._mismatches(got, stock, dtype)
                        if helper is not None:
                            report["mismatch"][key + "_vs_tt_all_reduce"] = base._mismatches(got, helper, dtype)
                        report["mismatch"][key + "_repeat"] = sum(
                            base._mismatches(base._from_mesh(mesh, _call(partial), rows), stock, dtype)
                            for _ in range(4)
                        )
                        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                        outs = [_call(partial) for _ in range(3)]
                        ttnn.end_trace_capture(mesh, tid, cq_id=0)
                        bad = 0
                        for _ in range(3):
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                            bad += sum(base._mismatches(base._from_mesh(mesh, o, rows), stock, dtype) for o in outs)
                        ttnn.release_trace(mesh, tid)
                        report["mismatch"][key + "_trace"] = bad
                    _set_links(2)
                us = {"stock_reduce_scatter": base._traced_us(mesh, lambda: rr.composed(partial))}
                if dtype == "bf16":
                    us["moe_minimal_async"] = base._traced_us(mesh, lambda: minimal(partial))
                for links in (1, 2):
                    _set_links(links)
                    us[f"row_reduce_{links}link"] = base._traced_us(mesh, lambda: _call(partial))
                _set_links(2)
                report["us_per_op_slope_and_mean12"] = us
                kept.append((dtype, rows, partial, stock))
            except Exception as e:  # noqa: BLE001 (report the case, run the others)
                report["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            os.environ.pop(rr.LINKS_ENV, None)
            ok = "error" not in report and all(v == 0 for v in report["mismatch"].values())
            report["pass"] = ok
            print("ROW_REDUCE_ROWS_RESULT " + json.dumps(report), flush=True)
            reports.append(report)
            if not ok:
                failures.append((dtype, sharded, rows))
    # every case interleaved in one trace (the one landing buffer, counter and semaphores serve every shape and mode)
    if kept and not failures:
        _set_links(2)
        order = list(range(len(kept))) + list(range(len(kept) - 1, -1, -1)) + [0, 5, 7]
        for dtype, rows, partial, stock in kept:
            ttnn.deallocate(_call(partial))
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = [(i, _call(kept[i][2])) for i in order]
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        bad = 0
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            for i, o in outs:
                dtype, rows, _, stock = kept[i]
                bad += base._mismatches(base._from_mesh(mesh, o, rows), stock, dtype)
        ttnn.release_trace(mesh, tid)
        print("ROW_REDUCE_ROWS_INTERLEAVED_MISMATCH", bad, flush=True)
        if bad:
            failures.append(("interleaved", bad))
    if RESULTS:
        with open(RESULTS, "a") as f:
            for r in reports:
                f.write(json.dumps(r) + "\n")
    assert not failures, failures
