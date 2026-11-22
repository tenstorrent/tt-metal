# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
# M2 model A/B in ONE process (one 128K load): baseline, then tt/mk/overlap.py variants.
# Per variant: trace capture, 6 traced positions' logits (bit-equal vs baseline, replay), median ms of 3 x 10 steps at
# pos 2048 and 122880, overlap stats. DRAM-busy fraction = GB per step (P3: 4.864 / 7.230) / (ms * 443 GB/s).
# usage: python models/demos/inkling2_mini/tests/m2_model_ab.py <out.jsonl> [variant ...]   variant = weights:split:staged_kb:parts:max_cores
#   e.g. w_all:3:0:4:1 or w_all+w_o:3:0:4:1 (staged_kb 0 = all)
import json
import os
import statistics
import sys
import time

import torch

import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.mk.overlap import install
from models.demos.inkling2_mini.tt.model import Inkling2Model

OUT = sys.argv[1]
VARIANTS = sys.argv[2:] or ["w_all:3:0:4:1"]
GB = {2048: 4.864, 122880: 7.230}
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20)


def run(m, tag, ov=None):
    g = Generator(m)
    g.capture()
    toks = [200005, 17, 4000, 123, 9, 77]

    def seq():
        m.reset()
        return torch.stack([ttnn.to_torch(g.decode(t, p)["logits"]).float().clone() for p, t in enumerate(toks)])

    a1, a2 = seq(), seq()
    res = {"tag": tag, "replay_bit_equal": bool(torch.equal(a1, a2)), "ms": {}, "dram_busy": {}}
    for pos in (2048, 122880):
        g.decode(1, pos)
        ttnn.synchronize_device(dev)
        ms = []
        for _ in range(3):
            t1 = time.perf_counter()
            for i in range(10):
                g.decode(1, pos + i)
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t1) / 10 * 1e3)
        res["ms"][pos] = round(statistics.median(ms), 2)
        res["dram_busy"][pos] = round(GB[pos] / (res["ms"][pos] * 1e-3) / 443, 3)
    if ov is not None:
        res["overlap"] = ov.stats()
    g.release()
    return res, a1


try:
    m = Inkling2Model(dev, max_ctx=131072, log=lambda s: None)
    m.set_inputs(0, 0)
    m.step()
    ttnn.synchronize_device(dev)
    if os.environ.get("M2_EAGER_ONLY"):  # mockwarm: compile every program of an eager step per variant, no traces
        for v in VARIANTS:
            f = v.split(":")
            xr = tuple(range(int(f[5]) + 1)) if len(f) > 5 else (0,)
            import models.demos.inkling2_mini.tt.mk.stage as STG

            STG.FILL = tuple(int(t) for t in f[6].split("/")) if len(f) > 6 else (16384, 4)
            ov = install(
                m,
                weights=tuple(f[0].split("+")),
                split=int(f[1]),
                staged_kb=int(f[2]) or None,
                parts=int(f[3]),
                max_cores=int(f[4]),
                exclude_rows=xr,
            )
            m.step()
            m.step()
            print("M2EAGER", v, json.dumps(ov.stats()), flush=True)
            ov.uninstall(m)
        sys.exit(0)
    base, ref = run(m, "base")
    print("M2MODEL", json.dumps(base), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(base) + "\n")
    for v in VARIANTS:
        f = v.split(":")
        ws, split, kb, parts, mc = f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4])
        xr = tuple(range(int(f[5]) + 1)) if len(f) > 5 else (0,)
        import models.demos.inkling2_mini.tt.mk.stage as STG

        STG.FILL = tuple(int(t) for t in f[6].split("/")) if len(f) > 6 else (16384, 4)
        ov = install(
            m,
            weights=tuple(ws.split("+")),
            split=split,
            staged_kb=kb or None,
            parts=parts,
            max_cores=mc,
            exclude_rows=xr,
        )
        try:
            m.step()  # two eager steps: the attach pattern is periodic from the 2nd step on (capture must not see new
            m.step()  # host + fill programs)
            ttnn.synchronize_device(dev)
            res, got = run(m, f"overlap {v}", ov)
            res["logits_bit_equal_vs_base"] = bool(torch.equal(got, ref))
            res["logits_max_abs_vs_base"] = float((got - ref).abs().max())
            for pos in res["ms"]:
                res.setdefault("saved_ms", {})[pos] = round(base["ms"][pos] - res["ms"][pos], 2)
        finally:
            ov.uninstall(m)
        print("M2MODEL", json.dumps(res), flush=True)
        with open(OUT, "a") as f:
            f.write(json.dumps(res) + "\n")
finally:
    ttnn.close_device(dev)
