#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test for the perf pipeline on fabricated data.

Usage::

    python -m scripts.tt_hw_planner.perf._smoke [/tmp/some/dir]

Exercises: join -> cluster -> regions -> all 8 charts -> HTML report ->
status board -> diagnose -> apply (writes patch) -> finalize.
Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


CSV = """OP CODE,OP TYPE,GLOBAL CALL COUNT,DEVICE ID,ATTRIBUTES,MATH FIDELITY,CORE COUNT,HOST DURATION [ns],DEVICE FW DURATION [ns],DEVICE KERNEL DURATION [ns],OP TO OP LATENCY [ns],DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC KERNEL DURATION [ns],DEVICE COMPUTE CB WAIT FRONT [ns],INPUTS,OUTPUTS,METAL TRACE ID,COMPUTE KERNEL SOURCE,COMPUTE KERNEL HASH,DATA MOVEMENT KERNEL SOURCE,DATA MOVEMENT KERNEL HASH,PROGRAM HASH,PROGRAM CACHE HIT,PM IDEAL [ns],PM COMPUTE [ns],PM BANDWIDTH [ns],PM REQ I BW,PM REQ O BW,PM FPU UTIL (%),NOC UTIL (%),MULTICAST NOC UTIL (%),DRAM BW UTIL (%),ETH BW UTIL (%),INPUT_0_W,INPUT_0_Z,INPUT_0_Y,INPUT_0_X,INPUT_0_LAYOUT,INPUT_0_DATATYPE
signpost,signpost,0,0,`TT_SIGNPOST: block_start decoder.layer_0`,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
ttnn.matmul,tt_dnn_device,1,0,,HiFi2,64,5000,4500,4000,200,500,500,1000,1000,1000,0,0,1x1x4096x4096,1x1x4096x4096,0,matmul.cpp,abc123,dm.cpp,def456,p1,1,3500,3500,3000,200,200,90.0,15.0,0,15.0,0,1,1,4096,4096,TILE,bf16
ttnn.add,tt_dnn_device,2,0,,HiFi2,64,2000,1800,1500,100,300,300,300,300,300,0,0,1x1x4096x4096,1x1x4096x4096,0,add.cpp,fff,dm.cpp,def456,p2,1,800,500,800,150,150,5.0,80.0,0,80.0,0,1,1,4096,4096,TILE,bf16
ttnn.matmul,tt_dnn_device,3,0,,HiFi4,64,8000,7800,7000,150,1000,1000,1500,1500,1500,0,0,1x1x4096x4096,1x1x4096x4096,0,matmul.cpp,xyz789,dm.cpp,def456,p3,1,6000,6000,5000,200,200,89.0,10.0,0,10.0,0,1,1,4096,4096,TILE,bf16
signpost,signpost,0,0,`TT_SIGNPOST: block_end decoder.layer_0`,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
signpost,signpost,0,0,`TT_SIGNPOST: block_start decoder.layer_1`,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
ttnn.matmul,tt_dnn_device,4,0,,HiFi2,64,5000,4500,4000,15000,500,500,1000,1000,1000,0,0,1x1x4096x4096,1x1x4096x4096,0,matmul.cpp,abc123,dm.cpp,def456,p1,1,3500,3500,3000,200,200,5.0,3.0,0,3.0,0,1,1,4096,4096,TILE,bf16
ttnn.layer_norm,tt_dnn_device,5,0,,HiFi2,64,2500,2300,2000,200,400,400,400,400,400,0,0,1x1x32x4096,1x1x32x4096,0,ln.cpp,lll,dm.cpp,def456,p4,1,1700,1500,1700,180,180,40.0,30.0,0,30.0,0,1,1,32,4096,TILE,bf16
signpost,signpost,0,0,`TT_SIGNPOST: block_end decoder.layer_1`,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"""


TRACER = {
    "version": 1,
    "operations": {
        "ttnn.matmul": {
            "configurations": [
                {
                    "config_id": "cfg-1",
                    "arguments": {
                        "memory_config": "DRAM_INTERLEAVED",
                        "math_fidelity": "HiFi2",
                        "shape": [1, 1, 4096, 4096],
                    },
                    "executions": [{"counter": 2}],
                },
                {
                    "config_id": "cfg-2",
                    "arguments": {
                        "memory_config": "DRAM_INTERLEAVED",
                        "math_fidelity": "HiFi4",
                        "shape": [1, 1, 4096, 4096],
                    },
                    "executions": [{"counter": 1}],
                },
            ],
        },
        "ttnn.add": {
            "configurations": [
                {"arguments": {"memory_config": "DRAM_INTERLEAVED"}, "executions": [{"counter": 1}]},
            ]
        },
        "ttnn.layer_norm": {
            "configurations": [
                {"arguments": {"memory_config": "DRAM_INTERLEAVED"}, "executions": [{"counter": 1}]},
            ]
        },
    },
}


def main(argv: list[str]) -> int:
    root = Path(argv[0]) if argv else Path("/tmp/tt_perf_smoke")
    if root.exists():
        shutil.rmtree(root)
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / run_id
    run_dir.mkdir(parents=True)

    (run_dir / "ops_perf_results.csv").write_text(CSV)
    (run_dir / "ttnn_operations_master.json").write_text(json.dumps(TRACER, indent=2))
    meta = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": "synthetic/smoke-test",
        "box": "QB2",
        "arch": "Blackhole",
        "mesh_shape": [1, 4],
        "mesh_device": "P150x4",
        "dtype": "bf16",
        "git_sha": "synthetic",
        "baseline_run_id": None,
        "pytest_argv": ["pytest", "models/tt_transformers/demo/simple_text_demo.py"],
        "env": {},
        "test_path": "models/tt_transformers/demo/simple_text_demo.py",
        "notes": ["synthetic smoke test"],
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[smoke] synthetic run: {run_id}")
    print(f"[smoke] run_dir:       {run_dir}")

    from scripts.tt_hw_planner.perf.report import build_report

    report = build_report(run_id, run_dir_root=root)
    print(f"[smoke] report.html:   {report}  ({report.stat().st_size:,} B)")

    from scripts.tt_hw_planner.perf.status_board import render_status_board

    print()
    print(render_status_board(run_id, run_dir_root=root))

    from scripts.tt_hw_planner.perf.runner import apply_block, diagnose_all

    findings = diagnose_all(run_id, run_dir_root=root)
    print("\n[smoke] findings per block:")
    for blk, fs in findings.items():
        if fs:
            print(f"  {blk}: {len(fs)} findings")

    res = apply_block("trace_capturer", run_id=run_id, run_dir_root=root)
    print(f"\n[smoke] applied trace_capturer -> {res.patch_path}")

    from scripts.tt_hw_planner.perf.finalize import finalize_run

    fr = finalize_run(run_id, run_dir_root=root)
    print()
    print(fr.summary)
    print(f"\n[smoke] OK. Open {report} in a browser to view the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
