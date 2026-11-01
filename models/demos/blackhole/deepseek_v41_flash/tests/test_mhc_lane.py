"""mHC lane-parallel prototype (DSV41_MHC_LANEPAR): production-shaped per-op timing of one layer's four mHC programs, baseline vs switch, in one process.

test_mhc_lane_us : traced us per call of the production calls of one decode layer (front attn = mixes_collapse, expand attn (X2LITE: lite rows),
                   front ffn (emit_rm), expand ffn (no pool: EXPAND_VALID is off in production)), at T in DSV41_MHC_PAR_T (default 4,8,12,16), for the arms in
                   DSV41_MHC_ARMS (default "base,lane": env sets below). Prints MHCLANE lines and the layer sum.
"""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_mhc_par import DEV, D, Ts, mk, traced_us
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

PROD = {
    "DSV41_MHC_PRE16": "1",
    "DSV41_MHC_X2LITE": "1",
    "DSV41_MHC_EXPAND_FAST": "1",
    "DSV41_MHC_PAR_SINKHORN": "1",
}
ARMS = {
    "base": {},
    "ff": {"DSV41_MHC_FRONT_FAST": "1"},
    "lane": {"DSV41_MHC_LANEPAR": "1"},
    "lanefuse": {"DSV41_MHC_LANEPAR": "1", "DSV41_MHC_LANEFUSE": "1"},
}
ALLK = {k for a in ARMS.values() for k in a} | set(PROD)


def set_arm(name):
    for k in ALLK:
        os.environ.pop(k, None)
    os.environ.update(PROD)
    os.environ.update(ARMS[name])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_lane_us(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    up = mk(md)
    set_arm("base")
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    for T in Ts("4,8,12,16"):
        x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
        pre_in = up(torch.rand(T, 1, 1, 4))
        a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
        for arm in os.environ.get("DSV41_MHC_ARMS", "base,lane").split(","):
            set_arm(arm)
            f1 = traced_us(md, lambda: m.mixes_collapse(x, pre_in, w, 1e-20))
            f2 = traced_us(md, lambda: m.mixes_collapse(x, pre_in, w, 1e-20, emit_rm=True))
            e1 = traced_us(md, lambda: m.expand(a, x, post, comb, None, lite=True))
            e2 = traced_us(md, lambda: m.expand(a, x, post, comb, None))
            print(
                f"MHCLANE {arm} T={T} front_attn={f1:.1f} expand_attn={e1:.1f} front_ffn_rm={f2:.1f} expand_ffn={e2:.1f} sum={f1 + e1 + f2 + e2:.1f}",
                flush=True,
            )


def _host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_lane_bits(mesh_device):
    """fused expand + front (DSV41_MHC_LANEPAR=1: DSV41MHC.expand_front) == expand (valid rows, pooled) + mixes_collapse(emit_rm), bit for bit: x2 (valid pieces and the
    zero padding of the pooled buffer), pre / post / comb, h, rm; T in DSV41_MHC_PAR_T (default 1..16), 3 seeds"""
    md = mesh_device
    up = mk(md)
    set_arm("base")
    os.environ["DSV41_MHC_EXPAND_VALID"] = "1"
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    bad = tot = 0
    for T in Ts(",".join(str(t) for t in range(1, 17))):
        for seed in range(3):
            torch.manual_seed(seed * 1000 + T)
            x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
            pre_in = up(torch.rand(T, 1, 1, 4))
            a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
            post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
            set_arm("base")
            os.environ["DSV41_MHC_EXPAND_VALID"] = "1"
            x2 = m.expand(a, x, post, comb, None, pool=("ref", "attn"))
            (pr, po, cb), h, rm = m.mixes_collapse(x2, pre_in, w, 1e-20, emit_rm=True)
            ref = [_host(t).reshape(-1) for t in (x2, pr, po, cb, h, rm)]
            set_arm("lanefuse")
            f = m.expand_front(a, x, post, comb, pre_in, w, 1e-20, pool=("lane", "attn"))
            assert f is not None, T
            got = [_host(f[0]).reshape(-1)] + [_host(t).reshape(-1) for t in (*f[1], f[2], f[3])]
            for nm, g, r in zip(("x2", "pre", "post", "comb", "h", "rm"), got, ref):
                tot += 1
                if not torch.equal(g, r):
                    bad += 1
                    print(f"LANEDIFF T={T} seed={seed} {nm} maxabs={float((g - r).abs().max()):.3e}", flush=True)
        print(f"LANEBITS T={T} done bad={bad}/{tot}", flush=True)
    assert bad == 0
