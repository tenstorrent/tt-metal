# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""kwalk_head_ramp whole-op A/B: shrink the K-walk head (compute starts on a small first K delivery).

`base` runs genify_matmul as-is. Every other variant monkeypatches the op's descriptor builder to swap the
reader + compute kernel sources for perf_experiments/kwalk_head_ramp/kernels/kh_{reader,compute}.cpp (the op's
kernels with the K-delivery schedule of kh_schedule.hpp) and adds that variant's defines. Host blocking, CBs,
split and every other kernel are the op's own. Each variant: DEVICE KERNEL DURATION [ns] of the op, PCC vs torch,
max |variant - base|.

    KH_CASE=focus KH_VARIANTS=base,rf_c scripts/run_safe_pytest.sh <this file>
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import importlib
import math
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.genify_matmul import genify_matmul
from eval.golden_tests.genify_matmul.helpers import create_ttnn_input_tensor

gm_mod = importlib.import_module("ttnn.operations.genify_matmul.genify_matmul")

ROOT = Path(__file__).resolve().parents[7]
KDIR = ROOT / "ttnn/ttnn/operations/genify_matmul/perf_experiments/kwalk_head_ramp/kernels"
_ORIG = gm_mod.create_program_descriptor

C = {"KH_COMPACT_A": "1", "KH_COMPACT_B": "1"}
VARIANTS = {
    "base": None,
    "copy": {},  # the copied kernels, op schedule: must match base
    "rf": {"KH_RAGGED_FIRST": "1"},  # ragged first, nominal-size multicasts
    "rf_c": {"KH_RAGGED_FIRST": "1", **C},  # ragged first, valid-bytes multicasts
    "c": dict(C),  # op schedule, valid-bytes multicasts (only the tail delivery changes)
    "r48": {"KH_H0": "4", "KH_H1": "8", **C},  # [4, 8, bk.., rem]
    "r448": {"KH_H0": "4", "KH_H1": "4", "KH_H2": "8", **C},
    "r248": {"KH_H0": "2", "KH_H1": "4", "KH_H2": "8", **C},
    "r36": {"KH_H0": "3", "KH_H1": "6", **C},
    "r48_rf": {"KH_H0": "4", "KH_H1": "8", "KH_RAGGED_FIRST": "1", **C},
    "r6": {"KH_H0": "6", **C},
    "r46810": {"KH_H0": "4", "KH_H1": "6", "KH_H2": "8", "KH_H3": "10", **C},  # [4, 6, 8, 10, 12 x 11]
    "r358": {"KH_H0": "3", "KH_H1": "5", "KH_H2": "8", **C},  # [3, 5, 8, 12 x 12]
    "r2468": {"KH_H0": "2", "KH_H1": "4", "KH_H2": "6", "KH_H3": "8", **C},  # [2, 4, 6, 8, 12 x 11, 8]
    "r4689": {"KH_H0": "4", "KH_H1": "6", "KH_H2": "8", "KH_H3": "9", **C},
    "auto_c": {"KH_RAMP_AUTO": "1", **C},  # ramp round(bk*{2,3,4,5}/6), A+B valid-bytes multicast
    "auto_cb": {"KH_RAMP_AUTO": "1", "KH_COMPACT_B": "1"},  # same, B-only compaction (helper K-walk)
    # the graduation candidate: auto ramp; compact A wherever the raw K-walk runs (epilogue, sb_h == 1, no
    # residency), else B only
    "auto": "AUTO",
    "bad": {"KH_H0": "13"},  # sanity: must fail to compile (head > block_k)
    # B-only compaction (the helper path cannot stride A by kv)
    "rf_cb": {"KH_RAGGED_FIRST": "1", "KH_COMPACT_B": "1"},
    "r48_cb": {"KH_H0": "4", "KH_H1": "8", "KH_COMPACT_B": "1"},
}

# name: (m, k, n, A dtype, B dtype, fidelity, fp32_dest_acc_en, epilogue)
CASES = {
    "focus": (2048, 5120, 8704, "bfloat16", "bfloat4_b", "LoFi", True, "swiglu"),
    "none4160": (2048, 5120, 4160, "bfloat16", "bfloat4_b", "LoFi", True, "none"),
    "addcmul": (2048, 6144, 3072, "bfloat16", "bfloat16", "HiFi2", True, "addcmul"),
    "swiglu4160": (2048, 5120, 4160, "bfloat16", "bfloat4_b", "LoFi", True, "swiglu"),
    "hifi4_4096": (4096, 4096, 4096, "bfloat16", "bfloat16", "HiFi4", True, "none"),
    "r1_64x16384": (64, 16384, 256, "bfloat16", "bfloat16", "HiFi4", False, "none"),
    "r1_256x6144x9216": (256, 6144, 9216, "bfloat16", "bfloat16", "HiFi2", True, "none"),
    "swiglu_k4128": (2048, 4128, 8704, "bfloat16", "bfloat4_b", "LoFi", True, "swiglu"),
    "swiglu_k5024": (2048, 5024, 8704, "bfloat16", "bfloat4_b", "LoFi", True, "swiglu"),  # ragged last = 1 tile
    "r1_tallM": (8192, 4096, 64, "bfloat16", "bfloat16", "HiFi2", True, "none"),
    "r1_tallM_bias": (8192, 4096, 64, "bfloat16", "bfloat16", "HiFi2", True, "bias"),
    "none4160_acc16": (2048, 5120, 4160, "bfloat16", "bfloat4_b", "LoFi", False, "none"),  # 16-bit DEST partials
    "swiglu4160_acc16": (2048, 5120, 4160, "bfloat16", "bfloat4_b", "LoFi", False, "swiglu"),
    "bias384": (384, 1536, 1024, "bfloat16", "bfloat16", "HiFi4", True, "bias"),
}

PLAN = {}


def _patched_for(defines):
    def _patched(*args, **kwargs):
        d = _ORIG(*args, **kwargs)
        defs = defines
        if defines == "AUTO":
            ct = [
                list(k.compile_time_args)
                for k in d.kernels
                if str(k.kernel_source).endswith("genify_matmul_compute.cpp")
            ][0]
            raw_walk = ct[13] != 0 and ct[7] == 1 and ct[22] == 0 and ct[23] == 0
            defs = {"KH_RAMP_AUTO": "1", "KH_COMPACT_B": "1", **({"KH_COMPACT_A": "1"} if raw_walk else {})}
            if (ct[10] == 1 and ct[13] == 0) or ct[22] or ct[23]:
                defs = {}  # inexpressible as-is: no cb_partials (one-K-block, epilogue none) / residency walks
            PLAN["auto_defs"] = defs
        n = 0
        for k in d.kernels:
            src = str(k.kernel_source)
            if src.endswith("genify_matmul_reader.cpp") or src.endswith("genify_matmul_compute.cpp"):
                k.kernel_source = str(KDIR / ("kh_reader.cpp" if "reader" in src else "kh_compute.cpp"))
                k.defines = list(k.defines) + list(defs.items())
                n += 1
            if src.endswith("genify_matmul_compute.cpp"):
                ct = list(k.compile_time_args)
                PLAN["compute"] = dict(
                    block_m=ct[16], block_n=ct[12], sb_h=ct[7], sb_w=ct[8], block_k=ct[9], nkb=ct[10], last=ct[11]
                )
            if src.endswith("genify_matmul_reader.cpp"):
                PLAN["reader_mode"] = list(k.compile_time_args)[22]
        assert n >= 2, n
        return d

    return _patched


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    vals = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            e = (getattr(program, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                vals.append(float(e.duration))
    return vals


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def test_kh(device):
    case = os.environ.get("KH_CASE", "focus")
    m, k, n, a_dt, w_dt, fid, acc, epi = CASES[case]
    torch.manual_seed(0)
    A = torch.randn(1, m, k)
    B = torch.randn(k, n) / math.sqrt(k)
    ta = create_ttnn_input_tensor(A, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT)
    tb = create_ttnn_input_tensor(B, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
    kwargs = {}
    Aq, Bq = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
    y = Aq @ Bq
    if epi == "swiglu":
        kwargs["activation"] = "swiglu"
        g = y.reshape(1, m, n // 64, 2, 32)
        ref = (torch.nn.functional.silu(g[..., 0, :]) * g[..., 1, :]).reshape(1, m, n // 2)
    elif epi == "addcmul":
        R = torch.randn(1, m, n)
        G = torch.randn(1, n)
        tr = create_ttnn_input_tensor(R, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT)
        tg = create_ttnn_input_tensor(G, device, dtype=getattr(ttnn, a_dt), layout=ttnn.TILE_LAYOUT)
        kwargs["residual"], kwargs["gate"] = tr, tg
        ref = ttnn.to_torch(tr).float() + ttnn.to_torch(tg).float() * y
    elif epi == "bias":
        Bi = torch.randn(1, n)
        tbias = create_ttnn_input_tensor(Bi, device, dtype=getattr(ttnn, w_dt), layout=ttnn.TILE_LAYOUT)
        kwargs["bias"] = tbias
        ref = y + ttnn.to_torch(tbias).float()
    else:
        ref = y
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=acc, math_approx_mode=False
    )
    _read_kernel_ns(device)
    res, outs = {}, {}
    order = os.environ.get("KH_VARIANTS", "base,rf_c").split(",")
    for v in order:
        device.clear_program_cache()
        defs = VARIANTS[v]
        gm_mod.create_program_descriptor = _ORIG if defs is None else _patched_for(defs)
        try:
            out = genify_matmul(ta, tb, compute_kernel_config=cfg, **kwargs)
            ttnn.synchronize_device(device)
        finally:
            gm_mod.create_program_descriptor = _ORIG
        ns = _read_kernel_ns(device)
        got = ttnn.to_torch(out).float()
        outs[v] = got
        pcc = _pcc(got, ref)
        rms = ((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        res[v] = (ns, pcc, rms)
    lines = [f"KH case={case} plan={PLAN}"]
    base = outs.get("base")
    for v, r in res.items():
        extra = ""
        if base is not None and v != "base":
            d = (outs[v] - base).abs()
            extra = (
                f" max|d_base| {d.max().item():.3e} n_diff {(d > 0).sum().item()} pcc_base {_pcc(outs[v], base):.7f}"
            )
        lines.append(f"KH_RES case={case} v={v:<7} ns {r[0]} pcc {r[1]:.6f} rel_rms {r[2]:.5f}{extra}")
    logger.info("\n" + "\n".join(lines))
    for v, r in res.items():
        assert r[1] > 0.9995, (v, r)
