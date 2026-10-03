# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op table of ONE decode layer: ordered ttnn calls (recorder) + device profiler digest + graph capture.

Env: DSV41_OPT_LAYER (default 2), DSV41_OPT_CHAIN (default /mnt/tt-data/ssinghal/dsv4-chain-m), DSV41_OPT_OUT.
Run with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DIR=<out>/prof for the device data;
without them only the trace timing and the recorded call list are produced.
Outputs <out>/raw.json, <out>/eager_cpp.csv, <out>/trace_cpp.csv (copies of the profiler digest after each phase).
Offline: tests/op_table_build.py.
"""

import gc
import json
import os
import shutil
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder, dev_id
from models.demos.blackhole.deepseek_v41_flash.tt.loader import layer_meta
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain

LAYER = int(os.environ.get("DSV41_OPT_LAYER", "2"))
CHAIN = os.environ.get("DSV41_OPT_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-m")
OUT = os.environ.get("DSV41_OPT_OUT", f"/mnt/tt-data/ssinghal/dsv4-logs/optable_L{LAYER}")


class MarkDict(dict):
    """layer.forward(profile=...) stores the wall time of each section under its name when the section ends."""

    def __init__(self, rec):
        super().__init__()
        self.rec = rec

    def __setitem__(self, k, v):
        if not k.startswith("_"):
            self.rec.mark(k)
        super().__setitem__(k, v)


def grab_csv(tag):
    prof = os.environ.get("TT_METAL_PROFILER_DIR")
    if not prof:
        return
    src = os.path.join(prof, ".logs", "cpp_device_perf_report.csv")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, f"{tag}_cpp.csv"))
    else:
        print(f"OPTABLE no digest at {src}", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(3000)
@torch.no_grad()
def test_layer_op_table(mesh_device):
    os.makedirs(OUT, exist_ok=True)
    md = mesh_device
    toks = torch.load(os.path.join(CHAIN, "tokens.pt"))
    S = toks["prefill_tokens"].shape[1]
    chain = DSV41DecodeChain(md, log=lambda m: print(m, flush=True))
    meta = layer_meta(LAYER)
    if meta["ratio"] and not meta["is_kv_source"]:  # attention reads the compressed cache of its kv source layer
        src = meta["kv_source"]
        ref_s = torch.load(os.path.join(CHAIN, f"layer_{src}.pt"))
        ref_s["S"] = S
        chain.build_layer(src, ref_s)
    ref = torch.load(os.path.join(CHAIN, f"layer_{LAYER}.pt"))
    ref["S"] = S
    layer, attn = chain.build_layer(LAYER, ref)
    B = chain.B
    st = attn.step_inputs(torch.full((B,), S))
    x, pre = chain.to_dev(ref["dec_in"].reshape(B, -1), 4 * 5120), chain.to_dev(ref["pre_in"].reshape(B, -1), 4)

    def fwd(profile=None):
        o, n = layer.forward(x, pre, st, profile=profile)
        ttnn.synchronize_device(md)
        ttnn.deallocate(o)
        ttnn.deallocate(n)

    for _ in range(3):
        fwd()
        gc.collect()
    result = dict(
        layer=LAYER,
        S=S,
        B=B,
        meta={k: v for k, v in meta.items() if k != "args"},
        profiler=bool(os.environ.get("TT_METAL_DEVICE_PROFILER")),
    )

    # ---- trace timing
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    t_out, t_nxt = layer.forward(x, pre, st)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    n = 50
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    result["traced_ms"] = (time.perf_counter() - t) / n * 1e3
    ttnn.release_trace(md, tid)
    print(f"OPTABLE layer {LAYER} traced {result['traced_ms']:.4f} ms (profiler {result['profiler']})", flush=True)

    rec = OpRecorder()
    rec.install()

    # ---- eager pass with the recorder + (if enabled) device profiler
    ttnn.synchronize_device(md)
    if result["profiler"]:
        ttnn.ReadDeviceProfiler(md)  # drain everything before the measured forward
    rec.enabled = True
    op0 = dev_id()
    fwd(MarkDict(rec))
    op1 = dev_id()
    rec.enabled = False
    result.update(op0=op0, op1=op1, eager_rows=rec.rows, eager_marks=rec.marks, eager_zero=rec.zero)
    print(
        f"OPTABLE eager: {len(rec.rows)} recorded calls with device programs, ids {op0}..{op1} ({op1 - op0} programs)",
        flush=True,
    )
    if result["profiler"]:
        ttnn.ReadDeviceProfiler(md)
        grab_csv("eager")

    # ---- trace pass (recorder on during capture gives the id -> call map of the replayed programs)
    rec.rows, rec.zero, rec.marks = [], {}, []
    rec.enabled = True
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    c0 = dev_id()
    t_out, t_nxt = layer.forward(x, pre, st)
    c1 = dev_id()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    rec.enabled = False
    ttnn.synchronize_device(md)
    result.update(cap0=c0, cap1=c1, trace_rows=rec.rows)
    if result["profiler"]:
        ttnn.ReadDeviceProfiler(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
        grab_csv("trace")
    ttnn.release_trace(md, tid)

    # ---- graph capture (names of the device programs behind composite calls)
    try:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        fwd()
        g = ttnn.graph.end_graph_capture()
        json.dump(g, open(os.path.join(OUT, "graph.json"), "w"))
        print(f"OPTABLE graph nodes: {len(g)}", flush=True)
    except Exception as e:  # noqa
        print(f"OPTABLE graph capture failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    json.dump(result, open(os.path.join(OUT, "raw.json"), "w"), indent=1)
    print(f"OPTABLE wrote {OUT}/raw.json", flush=True)
