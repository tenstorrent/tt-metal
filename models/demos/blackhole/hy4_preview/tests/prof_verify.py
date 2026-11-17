# SPDX-License-Identifier: Apache-2.0
"""verify-round track P2: per-op device-profiler replay of ONE round on the TP32 product stack, NL = HY4_SERVE_LAYERS layers (+ MTP layer in spec mode), synthetic FILL-token context (topk mode, like combo/pfc2).
MODE=spec: Hy4Engine K=3 round trace (derive + S=4 verify + greedy + accept + MTP tail); MODE=plain: Hy4Engine K=0 plain S=1 step (DeviceLoop trace body: dec.step). Run under TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1.
Writes /rout/<TAGNAME>_ops.json (OpRecorder rows) and replays the trace twice under the profiler. Env: MODE, FILL (65536), MAXLEN, TAGNAME, REPS (timing replays)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

MODE = os.environ.get("MODE", "spec")
FILL = int(os.environ.get("FILL", "65536"))
MAXLEN = int(os.environ.get("MAXLEN", str(FILL)))
R = 4


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def tms(md, fn, n=10, reps=5):
    v = []
    for _ in range(reps):
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(n):
            fn()
        ttnn.synchronize_device(md)
        v.append((time.perf_counter() - t) / n * 1e3)
    v.sort()
    return v[len(v) // 2]


def main():
    from models.demos.blackhole.hy4_preview.tt.hy4_serve_engine import Hy4Engine

    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "1900e6"))))
    try:
        K = 3 if MODE == "spec" else 0
        eng = Hy4Engine(md, R, MAXLEN, k=K)
        eng.warm_done = True  # no warm prefill (the profiler build of the PF sparse_sdpa kernel exceeds the kernel config buffer); lazies are created by the eager round body below
        res(event="warm_done", pages=eng.pages, K=K)
        dec = eng.dec
        dec.fill_synthetic(FILL // 512 * 512, torch.Generator().manual_seed(11))
        ttnn.synchronize_device(md)
        res(event="filled", fill=FILL)
        rec = OpRecorder()
        rec.install()
        if (
            os.environ.get("B28_NV", "0") == "1"
        ):  # B28: per-chip valid-key counts of every split-SDPA call (eager round only)
            import importlib

            hs = importlib.import_module("models.demos.blackhole.hy4_preview.tt.hy4_split")
            _orig = hs.sparse_sdpa_split
            _cnt = [0]
            res(event="b28_devids", ids=[int(i) for i in md.get_device_ids()])

            def _wrap(q, kv, idx, *a, **k):
                if hs._b28_on:
                    try:
                        ttnn.synchronize_device(md)
                        nvs = []
                        for t in ttnn.get_device_tensors(idx):
                            x = ttnn.to_torch(t).reshape(-1, int(idx.shape[3])).to(torch.int64)
                            nvs.append([int((x[r] != 0xFFFFFFFF).sum()) for r in range(min(4, x.shape[0]))])
                        res(event="b28_nv", call=_cnt[0], S=int(idx.shape[2]), nv=nvs)
                    except Exception as e:
                        res(event="b28_nv_err", call=_cnt[0], err=repr(e)[:200])
                _cnt[0] += 1
                return _orig(q, kv, idx, *a, **k)

            hs.sparse_sdpa_split = _wrap
            hs._b28_on = True
        if MODE == "spec":
            rn = eng.runner
            rn.load([1000, 2000, 3000, 4000], [FILL - 8] * R, [[1, 2, 3]] * R, cap=[3] * R)
            rn.set_park([0] * R)
            ttnn.synchronize_device(md)
            rn.body("G")  # eager (records nothing: rec disabled), compiles at the filled bucket
            ttnn.synchronize_device(md)
            if os.environ.get("B28_NV", "0") == "1":
                hs._b28_on = False
            rn.load([1000, 2000, 3000, 4000], [FILL - 8] * R, [[1, 2, 3]] * R, cap=[3] * R)
            rn.set_park([0] * R)
            tr = ttnn.begin_trace_capture(md, cq_id=0)
            rec.enabled = True
            rn.body("G")
            rec.enabled = False
            ttnn.end_trace_capture(md, tr, cq_id=0)
        else:
            dec.set_inputs([1000, 2000, 3000, 4000], [FILL - 8] * R)
            dec.release()
            lg = dec.step()
            ttnn.synchronize_device(md)
            ttnn.deallocate(lg)
            tr = ttnn.begin_trace_capture(md, cq_id=0)
            rec.enabled = True
            out = dec.step()
            rec.enabled = False
            ttnn.end_trace_capture(md, tr, cq_id=0)
        ttnn.synchronize_device(md)
        tag = os.environ.get("TAGNAME", "pv")
        json.dump(dict(rows=rec.rows, zero=rec.zero, mode=MODE), open(f"/rout/{tag}_ops.json", "w"))
        res(event="recorded", nops=len(rec.rows))
        ms = tms(md, lambda: ttnn.execute_trace(md, tr, cq_id=0, blocking=False))
        res(event="trace_ms", mode=MODE, nl=int(os.environ.get("HY4_SERVE_LAYERS", "78")), fill=FILL, ms=round(ms, 3))
        if os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "1":
            ttnn.execute_trace(md, tr, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
            for _ in range(2):
                ttnn.execute_trace(md, tr, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(md)
            res(event="prof_done")
        res(event="finished")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
