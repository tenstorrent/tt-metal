# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused global near softmax (tt/fused/near_softmax.py) vs the op chain in attention_fast._near_ops (u = 1):
p_near PCC and max diff, lse max diff, on random scores with masked slots and a sink logit; replays bit-equal.
"""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.near_softmax import NearSoftmax

TRACE_PARAMS = [{"trace_region_size": 8 * 1024 * 1024}]


def ops(near, bias4, sink_blk, ones_col):
    logits = ttnn.concat([ttnn.add(near, bias4), sink_blk], dim=-1)
    m = ttnn.max(logits, dim=-1, keepdim=True)
    e = ttnn.exp(ttnn.subtract(logits, m))
    z = ttnn.sum(e, dim=-1, keepdim=True)
    p = ttnn.multiply(ttnn.slice(e, [0, 0, 0, 0], [1, 8, 3, 1024]), ttnn.reciprocal(z), dtype=ttnn.bfloat16)
    return p, ttnn.add(m, ttnn.log(z))


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("device_params", TRACE_PARAMS, indirect=True)
def test_near_softmax(device):
    dev = device
    g = torch.Generator().manual_seed(0)
    put = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    ones_col = put(torch.ones(1, 8, 1056, 1), ttnn.float32)
    sm = NearSoftmax(dev)
    worst = []
    for case in range(4):
        near_t = torch.randn(1, 8, 3, 1024, generator=g) * (4.0 if case % 2 else 1.0)
        mask_t = torch.randn(1, 1, 24, 1024, generator=g) * 0.5
        valid = 1024 if case < 2 else 37  # few valid slots (early positions)
        mask_t[..., valid:] = -1.0e9
        sink_t = torch.full((1, 8, 3, 32), -1.0e9)
        sink_t[..., 0] = torch.randn(1, 8, 3, generator=g) * 3
        near, mask, sink = put(near_t, ttnn.float32), put(mask_t, ttnn.bfloat16), put(sink_t, ttnn.float32)
        rp, rl = ops(near, ttnn.reshape(mask, [1, 8, 3, 1024]), sink, ones_col)
        fp, fl = sm(near, mask, sink)
        rp, fp = ttnn.to_torch(rp).float(), ttnn.to_torch(fp).float()
        rl = ttnn.to_torch(rl).float()[..., 0].reshape(-1)
        fl = ttnn.to_torch(fl).float()[..., 0].reshape(-1)
        x = torch.cat([near_t.double() + ttnn.to_torch(mask).double().reshape(1, 8, 3, 1024), sink_t.double()], -1)
        tl = torch.logsumexp(x, -1).reshape(-1)
        tp = torch.softmax(x, -1)[..., :1024]
        res = dict(
            case=case,
            lse_err_fused=float((fl.double() - tl).abs().max()),
            lse_err_ops=float((rl.double() - tl).abs().max()),
            p_err_fused=float((fp.double() - tp).abs().max()),
            p_err_ops=float((rp.double() - tp).abs().max()),
            p_pcc=pcc(fp, rp),
            p_maxdiff=float((fp - rp).abs().max()),
            lse_maxdiff=float((fl - rl).abs().max()),
        )
        print("NEAR_SM", res)
        worst.append(res)
    assert all(r["p_pcc"] > 0.99999 and r["lse_err_fused"] <= 2 * r["lse_err_ops"] + 1e-3 for r in worst), worst

    near, mask, sink = put(near_t, ttnn.float32), put(mask_t, ttnn.bfloat16), put(sink_t, ttnn.float32)
    sm(near, mask, sink)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    p, l = sm(near, mask, sink)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    outs = []
    for _ in range(2):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        outs.append((ttnn.to_torch(p).clone(), ttnn.to_torch(l).clone()))
    ttnn.release_trace(dev, tid)
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
