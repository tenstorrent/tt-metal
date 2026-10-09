# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.all_gather vs ttnn.bringup.fabric_all_gather vs ttnn.experimental.high_bw_all_gather at the model's own
gather shapes: every all-gather group in <repo>/generated/<model>/op_report.json (tests/test_op_report.py; input shape,
dtype, cluster axis, gather dim -2). Per shape: the three ops on per-chip-distinct data, the two fabric ops' outputs
checked bit-identical to ttnn.all_gather's, and each op's device time per call (median over ITERS, slowest chip) from
the profiler's per-call records. fabric / high_bw get a fresh output per call (ttnn.empty) and no semaphores (the op
makes and caches its own), as a drop-in replacement would.

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 BRINGUP_SPEC=... \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_all_gather_bench.py -s
BRINGUP_AGB_ITERS (default 10), BRINGUP_AGB_LINKS (fabric / high_bw num_links, default 2)."""

import json
import os
import statistics

import torch

from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
ITERS = int(os.environ.get("BRINGUP_AGB_ITERS", "10"))
LINKS = int(os.environ.get("BRINGUP_AGB_LINKS", "2"))
DT = {"BFLOAT16": "bfloat16", "FLOAT32": "float32", "UINT32": "uint32", "UINT16": "uint16", "BFLOAT8_B": "bfloat8_b"}


def _shapes():
    rep = json.loads((S.repo / "generated" / S.model / "op_report.json").read_text())
    seen = {}
    for r in rep["rows"]:
        if r["op"] != "all_gather" or "ccl" not in r:
            continue
        ins = r["shape"].split(" -> ")[0].split(" · ")[0]  # "640x4096 bf16"
        dims, dt = ins.split(" ")
        key = (dims, dt, r["ccl"]["axis"])
        seen.setdefault(key, 0)
        seen[key] += r["calls"]
    return [(tuple(int(x) for x in k[0].split("x")), k[1], int(k[2]), calls) for k, calls in seen.items()]


@mesh_parametrize
def test_all_gather_bench(mesh_device):
    import ttnn

    for k, v in profiler.PROFILER_ENV.items():
        assert os.environ.get(k) == v, f"set {k}={v}"
    short = {
        "bf16": ttnn.bfloat16,
        "fp32": ttnn.float32,
        "u32": ttnn.uint32,
        "u16": ttnn.uint16,
        "bfp8": ttnn.bfloat8_b,
    }
    rows, cols = tuple(mesh_device.shape)
    results = []
    for dims, dt_s, axis, calls in _shapes():
        dt = short[dt_s]
        shp = (1,) * (4 - len(dims)) + dims  # per-chip [.., R, W]
        G = rows if axis == 0 else cols
        gathered = list(shp)
        gathered[-2] *= G
        # per-chip-distinct data: chip (r, c) holds value block r * cols + c
        host = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols, 1, 1, 1, 1) + torch.rand(
            rows, cols, *shp
        )
        if dt in (ttnn.uint32, ttnn.uint16):
            host = (host * 1000).floor()
        x = ttnn.from_torch(
            host.reshape(rows * shp[0], cols * shp[1], *shp[2:]),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
        )
        ops = {
            "ttnn.all_gather": lambda: ttnn.all_gather(
                x, dim=2, cluster_axis=axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
        }
        for name, fn in (
            ("fabric_all_gather", ttnn.bringup.fabric_all_gather),
            ("high_bw_all_gather", ttnn.experimental.high_bw_all_gather),
        ):

            def run(fn=fn):
                out = ttnn.empty(
                    gathered,
                    dtype=dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                fn(x, dim=2, output_tensor=out, cluster_axis=axis, num_links=LINKS)
                return out

            ops[name] = run
        ref = None
        line = {"shape": "x".join(map(str, dims)) + " " + dt_s, "axis": axis, "G": G, "model_calls": calls}
        for name, fn in ops.items():
            try:
                out = fn()  # compile + correctness
                o = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)])
                ttnn.deallocate(out)
                if ref is None:
                    ref = o
                    same = True
                else:
                    same = torch.equal(o, ref)
                profiler.enable(mesh_device, ops=True, calls=True)
                profiler.set_layer(0)
                profiler.signpost("bench")
                try:
                    for _ in range(ITERS):
                        ttnn.deallocate(fn())
                    profiler.signpost("end")
                    recs = profiler.result()["calls"]
                finally:
                    profiler.disable()
                times = [max(c["ns_dev"].values()) / 1e3 for c in recs if "gather" in c["op"]]
                line[name] = (statistics.median(times) if times else None, same)
            except Exception as ex:
                line[name] = (None, f"FAILED {str(ex).splitlines()[0][:70]}")
        ttnn.deallocate(x)
        results.append(line)
        cells = "  ".join(
            f"{n.split('.')[-1]:>18}: {('%.1f us' % v[0]) if v[0] else '   -   ':>9} {'ok' if v[1] is True else v[1]}"
            for n, v in ((k, line[k]) for k in ops)
        )
        print(f"[agb] {line['shape']:>16} axis {axis} (G {G}, {calls:4d} calls/chunk)  {cells}", flush=True)
    out = S.repo / "generated" / S.model / "all_gather_bench.json"
    out.write_text(json.dumps(results, indent=1, default=str))
    print(f"wrote {out}", flush=True)
