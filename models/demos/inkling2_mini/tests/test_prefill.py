# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill vs the CPU golden on code1500 (1478 tokens: crosses the 1024 window).

1. Teacher-forced prefill (each application gets its golden input): logits PCC over all positions >= 0.9999.
2. prompt0: 16 greedy tokens after prefill equal the golden greedy tokens.
3. code1500: 16 decode steps after prefill, fed the decode-only path's tokens: top-1 agrees on >= 14 of 16 steps.
Logits are streamed per 128 rows (host RSS < 3 GB).

    flock /tmp/tt-card.lock pytest -s models/demos/inkling2_mini/tests/test_prefill.py
"""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.generator import Generator
from models.demos.inkling2_mini.tt.model import Inkling2Model
from models.demos.inkling2_mini.tt.prefill_model import Inkling2Prefill

GOLDEN = Path("/work/inkling-bringup/golden")


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0, trace_region_size=256 << 20, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def test_prefill_code1500(device):
    d = GOLDEN / "code1500"
    ids = torch.load(d / "input_ids.pt")[0].tolist()
    T = len(ids)
    model = Inkling2Model(device, max_ctx=2048, log=lambda s: None)
    pf = Inkling2Prefill(model)
    ref = torch.load(d / "logits_bf16.pt")[0][:T]  # bf16

    inputs = {}

    def force(a, s, t):
        if a not in inputs:
            inputs.clear()
            inputs[a] = torch.load(d / f"in_{a:02d}.pt")[0].float()
        g = inputs[a][s : s + t]
        return torch.cat([g, torch.zeros(t - g.shape[0], g.shape[1])]) if g.shape[0] < t else g

    acc = torch.zeros(5, dtype=torch.float64)

    def cb(p0, blk):  # streamed PCC sums
        a, b = blk.double(), ref[p0 : p0 + blk.shape[0]].double()
        acc[0] += a.sum()
        acc[1] += b.sum()
        acc[2] += (a * b).sum()
        acc[3] += (a * a).sum()
        acc[4] += (b * b).sum()

    pf.logits_cb = cb
    pf.prefill(ids, logits_all=True, force=force)
    pf.logits_cb = None
    n = ref.numel()
    lp = float((acc[2] - acc[0] * acc[1] / n) / ((acc[3] - acc[0] ** 2 / n) * (acc[4] - acc[1] ** 2 / n)).sqrt())
    print(f"teacher-forced logits PCC {lp:.6f}")
    assert lp >= 0.9999

    # decode after prefill. Prefill each length class before capture (its cached consts must exist first).
    p0 = torch.load(GOLDEN / "prompt0" / "input_ids.pt")[0].tolist()
    pf.prefill(p0)
    pf.prefill(ids)
    gen = Generator(model)
    gen.capture()
    last = pf.prefill(p0)
    toks = [int(last[-1].argmax())]
    for i in range(15):
        toks.append(gen.read_token(gen.decode(toks[-1], len(p0) + i)))
    gold = json.load(open(GOLDEN / "greedy.json"))["new_ids"]
    print(f"prompt0 after prefill {toks}\ngolden                {gold}")
    assert toks == gold

    ref_toks, _ = gen.generate(ids, 16)  # decode-only path
    model.reset()
    for i, tk in enumerate(ids):
        o = gen.decode(tk, i)
    dec = [int(ttnn.to_torch(o["logits"]).float().reshape(-1)[:200058].argmax())]
    for i, tk in enumerate(ref_toks[:15]):
        dec.append(gen.read_token(gen.decode(tk, T + i)))
    last = pf.prefill(ids)
    after = [int(last[-1].argmax())]
    for i, tk in enumerate(ref_toks[:15]):
        after.append(gen.read_token(gen.decode(tk, T + i)))
    gen.release()
    agree = sum(int(a == b) for a, b in zip(after, dec))
    print(f"code1500 decode after prefill {after}\ndecode only                   {dec}\nagree {agree}/16")
    assert agree >= 14
