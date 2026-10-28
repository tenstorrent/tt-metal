"""mHC at the larger T of B=8 decode (T = 8 k=3 verify, T = 12/16 k=5 verify padded to 16): traced microbench of the layer-level mHC chain
(the calls DSV41Layer.forward makes, attention / MoE removed) for several VARIANTS selected by env switches, and a correctness check of each variant against the base path.

Env: DSV41_MHC_B8_T (comma list, default 8,16), DSV41_MHC_B8_MODES (comma list of mode names below, default base), DSV41_MHC_B8_CHECK=1 (compare every non-base mode with base).
Modes: base = production defaults; pre16 = DSV41_MHC_PRE16=1 (fused front for T=12/16); exp<N> = expand with N cores."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

D = 5120
MODES = {
    "base": {},
    "pre16": {"DSV41_MHC_PRE16": "1"},
    "exp32": {"DSV41_MHC_EXP_CORES": "32"},
    "exp40": {"DSV41_MHC_EXP_CORES": "40"},
    "exp64": {"DSV41_MHC_EXP_CORES": "64"},
    "exp80": {"DSV41_MHC_EXP_CORES": "80"},
    "exp160": {"DSV41_MHC_EXP_CORES": "160"},
    "w1": {"DSV41_MHC_EXP_WMODE": "1"},
    "w2": {"DSV41_MHC_EXP_WMODE": "2"},
    "w1c64": {"DSV41_MHC_EXP_WMODE": "1", "DSV41_MHC_EXP_CORES": "64"},
    "w2c64": {"DSV41_MHC_EXP_WMODE": "2", "DSV41_MHC_EXP_CORES": "64"},
    "w2c80": {"DSV41_MHC_EXP_WMODE": "2", "DSV41_MHC_EXP_CORES": "80"},
    "w2c40": {"DSV41_MHC_EXP_WMODE": "2", "DSV41_MHC_EXP_CORES": "40"},
    "pre16w2": {"DSV41_MHC_PRE16": "1", "DSV41_MHC_EXP_WMODE": "2"},
    "lite": {"DSV41_MHC_X2LITE": "1"},
    "pre16lite": {"DSV41_MHC_PRE16": "1", "DSV41_MHC_X2LITE": "1"},
    "pre16exp32": {"DSV41_MHC_PRE16": "1", "DSV41_MHC_EXP_CORES": "32"},
    "pre16exp40": {"DSV41_MHC_PRE16": "1", "DSV41_MHC_EXP_CORES": "40"},
    "pre16exp64": {"DSV41_MHC_PRE16": "1", "DSV41_MHC_EXP_CORES": "64"},
}
KEYS = ("DSV41_MHC_PRE16", "DSV41_MHC_EXP_CORES", "DSV41_MHC_EXP_WMODE", "DSV41_MHC_X2LITE")


def set_mode(name):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update(MODES[name])


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def traced_us(md, fn, n=20, reps=5):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    ttnn.release_trace(md, tid)
    return best


def chain(ma, mf, x, pre_in, a, m, sh, wa, wf):
    """the mHC calls of DSV41Layer.forward (decode path, T-aware): returns every tensor the layer consumes"""
    T = int(x.shape[0])
    if ma.pre_fused_ok(T):
        (ap, apo, ac), h = ma.mixes_collapse(x, pre_in, wa, 1e-20)
        x2 = ma.expand(a, x, apo, ac, None, lite=os.environ.get("DSV41_MHC_X2LITE", "0") == "1")
        (fp, fpo, fc), h2, h_tok = mf.mixes_collapse(x2, ap, wf, 1e-20, emit_rm=True)
    else:
        ap, apo, ac = ma.mixes(x)
        h = ma.collapse_norm(x, pre_in, wa, 1e-20)
        x2, (fp, fpo, fc) = ma.expand_mixes(a, x, apo, ac, None, mf)
        h2, h_tok = mf.collapse_norm_rm(x2, ap, wf, 1e-20)
    x3 = mf.expand(m, x2, fpo, fc, sh)
    return dict(ap=ap, apo=apo, ac=ac, h=h, x2=x2, fp=fp, fpo=fpo, fc=fc, h2=h2, h_tok=h_tok, x3=x3)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


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
@torch.no_grad()
def test_mhc_b8(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t.contiguous(), device=md, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    mk = lambda: DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    ma, mf = mk(), mk()
    wa, wf = up(torch.rand(1, 1, 1, D) + 0.5), up(torch.rand(1, 1, 1, D) + 0.5)
    Ts = [int(v) for v in os.environ.get("DSV41_MHC_B8_T", "8,16").split(",")]
    modes = os.environ.get("DSV41_MHC_B8_MODES", "base").split(",")
    check = os.environ.get("DSV41_MHC_B8_CHECK", "0") == "1"
    for T in Ts:
        x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
        pre = up(torch.rand(T, 1, 1, 4))
        a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        m = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        sh = up(torch.randn(1, 1, T, D))
        args = (ma, mf, x, pre, a, m, sh, wa, wf)
        ref = None
        for mode in modes:
            set_mode(mode)
            # warm-up outside the trace for every program / weight class of this mode
            out = chain(*args)
            ttnn.synchronize_device(md)
            us = traced_us(md, lambda: chain(*args))
            fused = ma.pre_fused_ok(T)
            line = f"MHCB8 T={T} mode={mode} fused_front={int(fused)} chain_us={us:.1f}"
            if os.environ.get("DSV41_MHC_B8_PARTS", "0") == "1":
                post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
                if fused:
                    f1 = traced_us(md, lambda: ma.mixes_collapse(x, pre, wa, 1e-20))
                    f2 = traced_us(md, lambda: mf.mixes_collapse(x, pre, wf, 1e-20, emit_rm=True))
                else:
                    f1 = traced_us(md, lambda: (ma.mixes(x), ma.collapse_norm(x, pre, wa, 1e-20)))
                    f2 = traced_us(md, lambda: (mf.mixes(x), mf.collapse_norm_rm(x, pre, wf, 1e-20)))
                e1 = traced_us(md, lambda: ma.expand(a, x, post, comb, None))
                e2 = traced_us(md, lambda: mf.expand(m, x, post, comb, sh))
                line += f" front_attn={f1:.1f} front_ffn_rm={f2:.1f} expand_a={e1:.1f} expand_f={e2:.1f}"
            if check:
                o = {k: host(v) for k, v in out.items()}
                if mode == "base" or ref is None:
                    ref = o
                else:
                    for k in o:
                        lead = o[k].shape
                        r = ref[k]
                        line += f"\n   CHK {k:6s} shape={tuple(lead)} equal={torch.equal(o[k], r)} maxabs={(o[k] - r).abs().max().item():.3e} pcc={pcc(o[k], r):.8f}"
            print(line, flush=True)
        set_mode("base")


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
@torch.no_grad()
def test_mhc_b8_groups(mesh_device):
    """DSV41_MHC_PRE16=1: the T=12 / 16 fused front is bit-identical, per group of 4 tokens, to the T=4 program (what plain decode runs)."""
    md = mesh_device
    torch.manual_seed(7)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t: ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    os.environ["DSV41_MHC_PRE16"] = "1"
    bad = tot = 0
    for trial in range(2):
        for T in (8, 9, 10, 11, 12, 13, 14, 15, 16):
            x_h = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
            x_h[..., :4] *= 300.0
            pre_h = torch.rand(T, 1, 1, 4)
            for emit_rm in (False, True):
                out = m.mixes_collapse(up(x_h), up(pre_h), w, 1e-20, emit_rm=emit_rm)
                got = [host(o).reshape(T, -1) for o in out[0]] + [host(out[1])[0, 0, :T]]
                if emit_rm:
                    got.append(host(out[2]).reshape(T, -1))
                for g in range(-(-T // 4)):
                    n4 = min(4, T - 4 * g)
                    o4 = m.mixes_collapse(
                        up(x_h[4 * g : 4 * g + n4]), up(pre_h[4 * g : 4 * g + n4]), w, 1e-20, emit_rm=emit_rm
                    )
                    ref = [host(o).reshape(n4, -1) for o in o4[0]] + [host(o4[1])[0, 0, :n4]]
                    if emit_rm:
                        ref.append(host(o4[2]).reshape(n4, -1))
                    for name, a_, b_ in zip(("pre", "post", "comb", "h", "tok"), got, ref):
                        tot += 1
                        if not torch.equal(a_[4 * g : 4 * g + n4], b_):
                            bad += 1
                            print(
                                "DIFF T",
                                T,
                                "group",
                                g,
                                "rm",
                                emit_rm,
                                name,
                                (a_[4 * g : 4 * g + n4] - b_).abs().max().item(),
                                flush=True,
                            )
    os.environ.pop("DSV41_MHC_PRE16", None)
    print(f"GROUPS_TOTAL differing {bad} of {tot}", flush=True)
    assert bad == 0
