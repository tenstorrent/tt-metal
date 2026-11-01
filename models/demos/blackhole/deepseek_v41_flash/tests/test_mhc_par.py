"""mHC parallel-Sinkhorn / fast-expand work (DSV41_MHC_PAR_SINKHORN, DSV41_MHC_EXPAND_FAST): zone profile driver, per-op microbench and bit-exactness checks.

test_mhc_par_prof : eager calls of mixes_collapse (emit_rm) and the valid-row expand for T in DSV41_MHC_PAR_T (default 12,24) under the device profiler
                    (run with TT_METAL_DEVICE_PROFILER=1); zones are analysed by tests/mhc_zone_sum.py
test_mhc_par_bits : flag-on == flag-off bit-identity of mixes_collapse (T=1..32 incl. padding rows, 3 seeds, drafter pre_max shapes) and of the expand (T-aware / valid-row)
test_mhc_par_us   : traced per-op us of mixes_collapse (attn / ffn_rm) and expand (attn / ffn) at DSV41_MHC_PAR_T, for the env of the run
"""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC

D = 5120
DEV = dict(l1_small_size=16384, fabric_config=ttnn.FabricConfig.FABRIC_1D_RING, trace_region_size=100_000_000)
Ts = lambda default: [int(v) for v in os.environ.get("DSV41_MHC_PAR_T", default).split(",")]


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def mk(md):
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    return up


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


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_par_prof(mesh_device):
    md = mesh_device
    os.environ["DSV41_MHC_PRE16"] = os.environ["DSV41_MHC_PRE24"] = "1"
    os.environ.setdefault("DSV41_MHC_EXPAND_VALID", "1")
    torch.manual_seed(3)
    up = mk(md)
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    for T in Ts("12,24"):
        x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
        pre_in = up(torch.rand(T, 1, 1, 4))
        a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        sh = up(torch.randn(1, 1, T, D))
        post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
        (pp, ppo, pc), h, rm = m.mixes_collapse(x, pre_in, w, 1e-20, emit_rm=True)
        m.expand(a, x, ppo, pc, sh, pool=("prof", "ffn"))
        ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)  # drop the warm-up
        for _ in range(2):
            m.mixes_collapse(x, pre_in, w, 1e-20, emit_rm=False)
            ttnn.synchronize_device(md)
            m.mixes_collapse(x, pre_in, w, 1e-20, emit_rm=True)
            ttnn.synchronize_device(md)
            m.expand(a, x, ppo, pc, None, pool=("prof", "attn"))
            ttnn.synchronize_device(md)
            m.expand(a, x, ppo, pc, sh, pool=("prof", "ffn"))
            ttnn.synchronize_device(md)
        ttnn.ReadDeviceProfiler(md)
        print(f"PROF_DONE T={T}", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_par_us(mesh_device):
    """traced per-op us (front attn / front ffn+rm / expand attn / expand ffn); the env of the run selects the variant"""
    md = mesh_device
    os.environ["DSV41_MHC_PRE16"] = os.environ["DSV41_MHC_PRE24"] = "1"
    os.environ.setdefault("DSV41_MHC_EXPAND_VALID", "1")
    torch.manual_seed(1)
    up = mk(md)
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    tag = os.environ.get("DSV41_MHC_PAR_TAG", "run")
    for T in Ts("6,12,24"):
        x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
        pre_in = up(torch.rand(T, 1, 1, 4))
        a = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
        sh = up(torch.randn(1, 1, T, D))
        post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
        f1 = traced_us(md, lambda: m.mixes_collapse(x, pre_in, w, 1e-20))
        f2 = traced_us(md, lambda: m.mixes_collapse(x, pre_in, w, 1e-20, emit_rm=True))
        e1 = traced_us(md, lambda: m.expand(a, x, post, comb, None, pool=("us", "attn")))
        e2 = traced_us(md, lambda: m.expand(a, x, post, comb, sh, pool=("us", "ffn")))
        print(
            f"MHCPAR {tag} T={T} front_attn={f1:.1f} front_ffn_rm={f2:.1f} expand_attn={e1:.1f} expand_ffn={e2:.1f}",
            flush=True,
        )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_par_probe(mesh_device):
    """x-gather probe (mhc_kernels/probe_read.cpp): the profile zones PB_ALL / PB_issued per mode (DSV41_MHC_PROBE_MODES, default 0,1,2,3,4) and T"""
    from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _kernel

    md = mesh_device
    up = mk(md)
    nb = int(md.dram_grid_size().x) if hasattr(md, "dram_grid_size") else 8
    print("PROBE dram_grid", md.dram_grid_size(), flush=True)
    NT, S, TPJ = D // 32, 40, 4
    for T in Ts("12,24"):
        x = up(torch.randn(T, 1, 4, D))
        dummy = up(torch.zeros(1, 1, 32, 32))
        g = md.compute_with_storage_grid_size()
        cores = [(cx, cy) for cy in range(int(g.y)) for cx in range(4, int(g.x))][:S]
        cs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
        for mode in [int(v) for v in os.environ.get("DSV41_MHC_PROBE_MODES", "0,1,2,3,4").split(",")]:
            rt = ttnn.RuntimeArgs()
            for r, (cx, cy) in enumerate(cores):
                rt[cx][cy] = [r]
            k = _kernel(
                "probe_read.cpp",
                cs,
                [0, T, NT, TPJ, mode, int(os.environ.get("DSV41_MHC_PROBE_NB", "8"))] + _acc(x),
                rt,
                None,
                ttnn.ReaderConfigDescriptor(),
                common=[x.buffer_address()],
            )
            prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[_cb(cs, 0, T * TPJ + 1, 4096, ttnn.float32)])
            prog.custom_program_hash = (0x7F1 << 40) | (hash((T, mode)) & ((1 << 40) - 1))
            ttnn.generic_op([x, dummy], prog)
            ttnn.synchronize_device(md)
            ttnn.generic_op([x, dummy], prog)
            ttnn.synchronize_device(md)
            ttnn.ReadDeviceProfiler(md)
            print(f"PROBE_DONE T={T} mode={mode}", flush=True)


def _front(m, md, up, w, T, seed, emit_rm):
    torch.manual_seed(seed * 1000 + T)
    x_h = torch.randn(T, 1, 1, D) + 0.3 * torch.randn(T, 1, 4, D)
    x_h[..., :4] *= 300.0
    pre_h = torch.rand(T, 1, 1, 4)
    out = m.mixes_collapse(up(x_h), up(pre_h), w, 1e-20, emit_rm=emit_rm)
    res = [host(o).reshape(T, -1) for o in out[0]] + [host(out[1])[0, 0, :T]]
    if emit_rm:
        res.append(host(out[2]).reshape(T, -1))
    return res


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_par_bits(mesh_device):
    """front: every flag variant in DSV41_MHC_PAR_VARIANTS (default 'par' = DSV41_MHC_PAR_SINKHORN=1; also e.g. 'par+ff' with DSV41_MHC_FRONT_FAST=1) is bit-identical to the unmodified
    fused front for T = 1..32 (padding rows included: the compared outputs are all T valid rows, the 32-row-padded shapes are covered by T not a multiple of 4), 3 seeds, emit_rm False/True.
    """
    md = mesh_device
    os.environ["DSV41_MHC_PRE16"] = os.environ["DSV41_MHC_PRE24"] = "1"
    up = mk(md)
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    m.pre_max = int(os.environ.get("DSV41_MHC_PAR_TMAX", "32"))  # the drafter's pre_max path
    w = up(torch.rand(1, 1, 1, D) + 0.5)
    variants = os.environ.get("DSV41_MHC_PAR_VARIANTS", "par").split(",")
    flags = {
        "par": {"DSV41_MHC_PAR_SINKHORN": "1"},
        "ff": {"DSV41_MHC_FRONT_FAST": "1"},
        "lane": {"DSV41_MHC_LANEPAR": "1"},
    }
    allk = {k for v in flags.values() for k in v}
    bad = tot = 0
    for T in Ts(",".join(str(t) for t in range(1, 33))):
        for seed in range(3):
            for emit_rm in (False, True):
                for k in allk:
                    os.environ.pop(k, None)
                ref = _front(m, md, up, w, T, seed, emit_rm)
                for var in variants:
                    for k in allk:
                        os.environ.pop(k, None)
                    for part in var.split("+"):
                        os.environ.update(flags[part])
                    got = _front(m, md, up, w, T, seed, emit_rm)
                    for i, (a, b) in enumerate(zip(got, ref)):
                        tot += 1
                        if not torch.equal(a, b):
                            bad += 1
                            print(
                                "DIFF",
                                var,
                                "T",
                                T,
                                "seed",
                                seed,
                                "rm",
                                emit_rm,
                                "out",
                                i,
                                (a - b).abs().max().item(),
                                flush=True,
                            )
        print(f"BITS T={T} done bad={bad}/{tot}", flush=True)
    print(f"PAR_BITS_TOTAL differing {bad} of {tot}", flush=True)
    assert bad == 0


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DEV, id="ring")], indirect=True)
@torch.no_grad()
def test_mhc_par_expand_bits(mesh_device):
    """DSV41_MHC_EXPAND_FAST=1 expand (valid-row, stream pool) is bit-identical to the unmodified valid-row expand, T = 1..32, 3 seeds, y fp32 / bf16 rows, with / without y2:
    the whole output buffer (valid rows AND the zero padding) is compared."""
    md = mesh_device
    os.environ["DSV41_MHC_EXPAND_VALID"] = "1"
    up = mk(md)
    m = DSV41MHC(md, torch.randn(24, 4 * D) * 0.02, torch.randn(24) * 0.3, torch.tensor([0.4, 0.3, 0.5]))
    bad = tot = 0
    for T in Ts(",".join(str(t) for t in range(1, 33))):
        if T % 2:  # odd T goes through the padded path of DSV41MHC.expand (rows padded to 8): not the valid-row kernel
            pass
        for seed in range(3):
            torch.manual_seed(seed * 100 + T)
            x = up(torch.randn(T, 1, 4, D) + 0.3 * torch.randn(T, 1, 1, D))
            post, comb = up(torch.rand(T, 1, 4, 1)), up(torch.rand(T, 1, 4, 4))
            for ydt in (ttnn.bfloat16, ttnn.float32):
                y = up(torch.randn(1, 1, T, D), ydt)
                for use_y2 in (False, True):
                    y2 = up(torch.randn(1, 1, T, D)) if use_y2 else None
                    outs = {}
                    for fast in ("0", "1"):
                        os.environ["DSV41_MHC_EXPAND_FAST"] = fast
                        o = m.expand(y, x, post, comb, y2, pool=("ebits", f"{fast}{ydt}{use_y2}"))
                        # the pool buffer: compare the FULL padded tile data (host copy of the tensor's tile pages)
                        outs[fast] = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().clone()
                    tot += 1
                    if not torch.equal(outs["0"], outs["1"]):
                        bad += 1
                        print(
                            "DIFF T",
                            T,
                            "seed",
                            seed,
                            ydt,
                            "y2",
                            use_y2,
                            (outs["0"] - outs["1"]).abs().max().item(),
                            flush=True,
                        )
        print(f"EBITS T={T} done bad={bad}/{tot}", flush=True)
    os.environ.pop("DSV41_MHC_EXPAND_FAST", None)
    print(f"EXPAND_BITS_TOTAL differing {bad} of {tot}", flush=True)
    assert bad == 0
