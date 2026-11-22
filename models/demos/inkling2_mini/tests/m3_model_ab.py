# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
# M3 model A/B in ONE process (one 128K load): baseline without staging, then variants.
# A variant is JSON: {"tag": str, "ov": {tt/mk/overlap.install kwargs} or null (no staging), "pipe":
# {tt/mk/pipe_stage.PipeOverlap kwargs} (staging by the DRAM cores), "norm": bool (fused MoE
# norm, tt/mk/moe_norm.py)}. Per variant: trace capture, 6 traced positions' logits vs baseline (bit-equal, max abs,
# PCC, argmax), replay bit-equal, median ms of 3 x 10 steps at pos 2048 and 122880, overlap stats. DRAM-busy fraction
# = GB per step (P3: 4.864 / 7.230) / (ms * 443 GB/s).
# usage: python models/demos/inkling2_mini/tests/m3_model_ab.py <out.jsonl> '<variant json>' ...
# M3_EAGER_ONLY=1 (mockwarm): two eager steps per variant, no traces, no numbers.
import json
import os
import statistics
import sys
import time

import torch

import models.demos.inkling2_mini.tt.mk.moe_norm as MN
import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.mk.overlap import install
from models.demos.inkling2_mini.tt.mk.pipe_stage import PipeOverlap
from models.demos.inkling2_mini.tt.model import Inkling2Model

OUT = sys.argv[1]
VARIANTS = [json.loads(v) for v in sys.argv[2:]]
GB = {2048: 4.864, 122880: 7.230}
EAGER = bool(os.environ.get("M3_EAGER_ONLY"))
dev = ttnn.open_device(device_id=0, trace_region_size=512 << 20)


def run(m, tag, ov=None):
    g = Generator(m, overlap=False)  # this script installs its own variants
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


def emit(res):
    print("M3MODEL", json.dumps(res), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(res) + "\n")


try:
    m = Inkling2Model(dev, max_ctx=4096 if EAGER else 131072, log=lambda s: None)
    m.set_inputs(0, 2048 if EAGER else 0)
    m.step()
    ttnn.synchronize_device(dev)
    base, ref = (None, None) if EAGER else run(m, "base")
    if base:
        emit(base)
    for v in VARIANTS:
        (MN.install if v.get("norm") else MN.uninstall)(m)
        ov = install(m, **v["ov"]) if v.get("ov") is not None else None
        if v.get("pipe") is not None:
            assert ov is None, "pipe and tile-fill staging both wrap w_all"
            ov = PipeOverlap(m, **v["pipe"])
        try:
            m.step()  # two eager steps: the attach pattern is periodic from the 2nd step on (the capture must not
            m.step()  # meet new host + fill programs)
            ttnn.synchronize_device(dev)
            if EAGER:
                print("M3EAGER", v["tag"], json.dumps(ov.stats() if ov else None), flush=True)
                continue
            res, got = run(m, v["tag"], ov)
        finally:
            if ov is not None:
                ov.uninstall(m)
        res["variant"] = v
        res["logits_bit_equal_vs_base"] = bool(torch.equal(got, ref))
        res["logits_max_abs_vs_base"] = float((got - ref).abs().max())
        g2, r2 = got.reshape(len(got), -1).double(), ref.reshape(len(ref), -1).double()
        res["logits_pcc_min"] = round(min(torch.corrcoef(torch.stack([a, b]))[0, 1].item() for a, b in zip(g2, r2)), 6)
        res["argmax_equal"] = f"{int((g2.argmax(-1) == r2.argmax(-1)).sum())}/{len(g2)}"
        res["saved_ms"] = {p: round(base["ms"][p] - res["ms"][p], 2) for p in res["ms"]}
        emit(res)
    MN.uninstall(m)
finally:
    ttnn.close_device(dev)
