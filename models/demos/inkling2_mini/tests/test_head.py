# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused decode logits tail (tt/fused/head.py) vs the ttnn op chain it replaces (tt/model.py head):
logits bit-equal over the real vocab, same argmax token (planted exact ties, saturated softcap, a dominant
masked column), eager and traced, plus replay time of both. Random BFP8 unembed with the model's shapes.
"""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.dram_sharded import DramShardedLinear
from models.demos.inkling2_mini.tt.fused.head import FusedLogitsTail
from models.demos.inkling2_mini.tt.model import HIDDEN, SOFTCAP, UNEMBED_CHUNK, VOCAB_PAD, VOCAB_REAL

TRACE_PARAMS = [{"trace_region_size": 16 * 1024 * 1024}]


def old_tail(dev, unembed, mask, x):
    logits = unembed(x)
    logits = ttnn.multiply(ttnn.tanh(ttnn.multiply(logits, 1.0 / SOFTCAP)), SOFTCAP)
    logits = ttnn.add(logits, mask)
    return logits, ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)


def host(lg, tok):
    return ttnn.to_torch(lg).float().reshape(-1)[:VOCAB_REAL], int(ttnn.to_torch(tok).reshape(-1)[0])


def replay_us(dev, tid, reps=50):
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e6


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_head_tail(device):
    dev = device
    torch.manual_seed(0)
    x = torch.randn(HIDDEN)
    w = torch.randn(HIDDEN, VOCAB_PAD) * 0.02
    # planted exact tie: two identical 32-column blocks (same BFP8 blocks), in different chunks and cores,
    # with column 7 of each block dominant -> the lower one must win
    lo, hi = 32 * 160, 32 * 6000
    w[:, lo : lo + 32] = w[:, hi : hi + 32]
    w[:, lo + 7] = w[:, hi + 7] = x / x.norm() * 2.0
    w[:, VOCAB_REAL + 2] = x / x.norm() * 4.0  # dominant masked column: never the token
    cc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    unembed = DramShardedLinear(
        dev, w, dtype=ttnn.bfloat8_b, chunk=UNEMBED_CHUNK, workers=3, out_dtype=ttnn.float32, compute_kernel_config=cc
    )
    del w
    mask = torch.zeros(1, 1, 1, VOCAB_PAD)
    mask[..., VOCAB_REAL:] = -1.0e4
    mask = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tail = FusedLogitsTail(dev, unembed, VOCAB_PAD, VOCAB_REAL, SOFTCAP)

    def put(v):
        return ttnn.from_torch(v.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    cases = {"planted_tie": x, "saturated": x * 400.0, "random": torch.randn(HIDDEN)}
    for name, v in cases.items():
        xt = put(v)
        ref_lg, ref_tok = host(*old_tail(dev, unembed, mask, xt))
        lg, tok = host(*tail(xt))
        n_max = int((ref_lg == ref_lg.max()).sum())
        print(f"{name}: token {tok} ref {ref_tok} ties_at_max {n_max} max {float(ref_lg.max())}")
        assert torch.equal(lg, ref_lg), (name, float((lg - ref_lg).abs().max()), int((lg != ref_lg).sum()))
        assert tok == ref_tok, (name, tok, ref_tok)
        if name == "planted_tie":
            assert tok == lo + 7, tok
        if name == "saturated":
            assert n_max > 1, n_max  # ties really exercised

    # traced: two replays bit-equal and equal to eager; replay time vs the op chain
    xt = put(cases["saturated"])
    e_lg, e_tok = host(*tail(xt))
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    out = tail(xt)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    reps = []
    for _ in range(2):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        reps.append(host(*out))
    assert torch.equal(reps[0][0], reps[1][0]) and torch.equal(reps[0][0], e_lg)
    assert reps[0][1] == reps[1][1] == e_tok
    new_us = replay_us(dev, tid)
    ttnn.release_trace(dev, tid)
    old_tail(dev, unembed, mask, xt)  # compile before capture
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    old_tail(dev, unembed, mask, xt)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    old_us = replay_us(dev, tid)
    ttnn.release_trace(dev, tid)
    print(f"HEAD_TAIL_US old {old_us:.1f} new {new_us:.1f}")
