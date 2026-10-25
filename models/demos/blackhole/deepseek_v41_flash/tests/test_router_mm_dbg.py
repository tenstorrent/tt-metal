import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_router_mm import (
    PARAMS,
    bits_eq,
    make_gate,
    old_chain,
    router_mm,
    rows,
    up,
)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@torch.no_grad()
def test_dbg(mesh_device):
    md = mesh_device
    gate, _ = make_gate(md, 2, 8)
    x = torch.randn(8, 5120) * 3
    h = up(md, x)
    wk, bt = gate._mm_consts(1)
    ref = x.bfloat16().float() @ ttnn.to_torch(ttnn.get_device_tensors(gate._w_exact)[0]).reshape(5120, 384).float()
    ref64 = x.bfloat16().double() @ ttnn.to_torch(ttnn.get_device_tensors(gate._w_exact)[0]).reshape(5120, 384).double()
    lg = ttnn.matmul(
        h, gate._w_exact, compute_kernel_config=gate._ckc_exact, dtype=ttnn.float32, core_grid=ttnn.CoreGrid(y=1, x=8)
    )
    print("DBG old logits err", float((rows(lg, 8).double() - ref64).abs().max()), flush=True)
    for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4):
        rn, sn = router_mm(h, wk, bt, 1, mode=3, fid=fid)
        print("DBG fid", fid, "err", float((rows(rn, 8).double() - ref64).abs().max()), flush=True)
    for md_ in (3, 1):
        rn, sn = router_mm(h, wk, bt, 1, mode=md_)
        print("DBG mode", md_, "rank", rows(rn, 8)[0, :4].tolist(), "ref", ref[0, :4].tolist(), flush=True)
    for S in (1, 4):
        wk, bt = gate._mm_consts(S)
        ro, so = old_chain(gate, h)
        rn, sn = router_mm(h, wk, bt, S)
        a, b = rows(ro, 8), rows(rn, 8)
        print("DBG S", S, "old rank", a[0, :6].tolist(), "new rank", b[0, :6].tolist(), flush=True)
        a, b = rows(so, 8), rows(sn, 8)
        print("DBG S", S, "old score", a[0, :6].tolist(), "new score", b[0, :6].tolist(), flush=True)
        print(
            "DBG ref logits",
            (
                x.bfloat16().float()
                @ ttnn.to_torch(ttnn.get_device_tensors(gate._w_exact)[0]).reshape(5120, 384).float()
            )[0, :6].tolist(),
            flush=True,
        )
        print(
            "DBG nan/inf",
            torch.isnan(b).sum().item(),
            torch.isinf(b).sum().item(),
            "diffbits",
            bits_eq(a, b),
            (a - b).abs().max().item(),
            flush=True,
        )

    for ap in (False, True):
        rn, sn = router_mm(h, wk, bt, 1, approx=ap)
        ro, so = old_chain(gate, h)
        print(
            "DBG approx",
            ap,
            "rank bits differ",
            bits_eq(rows(ro, 8), rows(rn, 8)),
            "score",
            bits_eq(rows(so, 8), rows(sn, 8)),
            "maxd",
            float((rows(so, 8) - rows(sn, 8)).abs().max()),
            flush=True,
        )

    lgt = rows(router_mm(h, wk, bt, 1, mode=3)[1], 8).double()
    sp = rows(router_mm(h, wk, bt, 1, mode=4)[1], 8).double()
    ref = torch.nn.functional.softplus(lgt)
    print("DBG softplus maxrel", float(((sp - ref).abs() / ref).max()), flush=True)
    sq = rows(router_mm(h, wk, bt, 1, mode=5)[1], 8).double()
    m = lgt > 0
    print("DBG sqrt maxrel", float(((sq - lgt.sqrt()).abs() / lgt.sqrt())[m].max()), flush=True)
    so = rows(old_chain(gate, h)[1], 8).double()
    print("DBG old score maxrel", float(((so - ref.sqrt()).abs() / ref.sqrt()).max()), flush=True)

    so = rows(old_chain(gate, h)[1], 8).double()
    for ap in (False, True):
        sn = rows(router_mm(h, wk, bt, 1, approx=ap)[1], 8).double()
        for lo, hi in ((-30, -8), (-8, -2), (-2, 0), (0, 2), (2, 8), (8, 40)):
            m = (lgt >= lo) & (lgt < hi)
            if m.any():
                print(
                    "DBG bin",
                    ap,
                    lo,
                    hi,
                    int(m.sum()),
                    "new-vs-old maxrel",
                    float(((sn - so).abs() / so)[m].max()),
                    "old-vs-true",
                    float(((so - ref.sqrt()).abs() / ref.sqrt())[m].max()),
                    "new-vs-true",
                    float(((sn - ref.sqrt()).abs() / ref.sqrt())[m].max()),
                    flush=True,
                )
