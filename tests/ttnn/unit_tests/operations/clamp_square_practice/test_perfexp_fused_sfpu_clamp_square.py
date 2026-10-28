# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf experiment `fused_sfpu_clamp_square` for clamp_square_practice (NOT the op's test suite).

Bake-off: Clamp->Square as two SFPU passes (baseline, the op's kernel) vs ONE fused SFPU pass.
Every compute variant is swapped into the REAL op descriptor (same CBs, block_tiles, work split, precision
config bf16 / fp32_dest_acc_en=False). DM modes:
  full - the op's real reader/writer (whole-op ns + correctness)
  stub - reader/writer NoC payload stubbed, CB sync kept (compute-stage-isolated ns)
Kernels: ttnn/ttnn/operations/clamp_square_practice/perf_experiments/fused_sfpu_clamp_square/kernels/

Env filters (run_safe_pytest word-splits -k):  FCS_VARIANTS=a,b  FCS_CASES=focus,...  FCS_MODES=stub,full
Correctness/precision is the only assertion (variants flagged precision-option are reported, not gated
on bit-identity).
"""
import os
from pathlib import Path

_HERE_EXP = (
    Path(__file__).resolve().parents[5]
    / "ttnn/ttnn/operations/clamp_square_practice/perf_experiments/fused_sfpu_clamp_square"
)
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_PROFILER_DIR", str(_HERE_EXP / "profiler"))
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics

import torch
import ttnn
from loguru import logger

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = _HERE_EXP / "kernels"
GEN = _HERE_EXP / "gen_kernels"
SRC = Path(pd.KERNEL_DIR)
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_CSV = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "profile_log_device.csv"

# name -> (kernel file, defines, must_be_bit_identical_to_baseline)
C = "compute.cpp"
VARIANTS = {
    "baseline": (C, {"CS_VARIANT": 0}, True),
    "fused_sfpi": (C, {"CS_VARIANT": 1}, True),
    "fused_creg": (C, {"CS_VARIANT": 2}, True),
    "fused_creg_mask": (C, {"CS_VARIANT": 5}, True),
    "fused_stoch": (C, {"CS_VARIANT": 3}, False),
    "fused_trunc": (C, {"CS_VARIANT": 4}, False),
    "raw_rne": (C, {"CS_VARIANT": 6}, True),
    "raw_stoch": (C, {"CS_VARIANT": 7}, False),
    "lm_rne": (C, {"CS_VARIANT": 8}, True),
    # graduation candidates (standalone kernel, same code paths as raw_rne / lm_rne)
    "final_raw": ("fused_compute.cpp", {"CS_LOADMACRO": 0}, True),
    "final_lm": ("fused_compute.cpp", {"CS_LOADMACRO": 1}, True),
    "abl_nosfpu": (C, {"CS_VARIANT": 9}, None),  # timing floor only; output = x (wrong by construction)
}

# case -> (shape, core_grid or None=full grid)
CASES = {
    "focus": ((1, 1, 4096, 4096), None),
    "b8_1k": ((8, 1, 1024, 1024), None),
    "small": ((2, 4, 256, 512), None),
    "one": ((1, 1, 32, 32), None),
    "1c_137": ((1, 1, 32, 32 * 137), (1, 1)),
    "1c_8": ((1, 1, 32, 32 * 8), (1, 1)),
    "1c_1024": ((1, 1, 1024, 1024), (1, 1)),
}
BOUNDS = [(-1.0, 1.0), (0.0, 2.5), (0.5, 0.5), (-100.0, 100.0), (-0.3, 0.7)]


def _sel(env, allowed):
    v = os.environ.get(env)
    return list(allowed) if not v else [x for x in v.split(",") if x in allowed]


def _gen_dm():
    GEN.mkdir(exist_ok=True)
    r = (SRC / "clamp_square_practice_reader.cpp").read_text()
    r = r.replace(
        "noc_async_read_page(page0 + i, input_accessor, l1_write_addr);",
        "if constexpr (!ABLATE_READ) { noc_async_read_page(page0 + i, input_accessor, l1_write_addr); }",
    )
    r = r.replace("#include <cstdint>", "#include <cstdint>\n#ifndef ABLATE_READ\n#define ABLATE_READ 0\n#endif", 1)
    w = (SRC / "clamp_square_practice_writer.cpp").read_text()
    w = w.replace(
        "noc_async_write_page(page0 + i, output_accessor, l1_read_addr);",
        "if constexpr (!ABLATE_WRITE) { noc_async_write_page(page0 + i, output_accessor, l1_read_addr); }",
    )
    w = w.replace("#include <cstdint>", "#include <cstdint>\n#ifndef ABLATE_WRITE\n#define ABLATE_WRITE 0\n#endif", 1)
    assert "ABLATE_READ) {" in r and "ABLATE_WRITE) {" in w
    (GEN / "reader.cpp").write_text(r)
    (GEN / "writer.cpp").write_text(w)


def _run(device, x, out, variant, mode, grid, lo, hi):
    kfile, defs, _ = VARIANTS[variant]
    cg = None if grid is None else ttnn.CoreCoord(*grid)
    desc = pd.create_program_descriptor(x, out, lo, hi, core_grid=cg)
    rk, wk, ck = desc.kernels
    stub = "1" if mode == "stub" else "0"
    rk.kernel_source = str(GEN / "reader.cpp")
    wk.kernel_source = str(GEN / "writer.cpp")
    rk.defines = [("ABLATE_READ", stub)]
    wk.defines = [("ABLATE_WRITE", stub)]
    ck.kernel_source = str(KDIR / kfile)
    ck.defines = [(k, str(v)) for k, v in defs.items()]
    desc.kernels = [rk, wk, ck]
    ttnn.generic_op([x, out], desc)


def _kernel_ns():
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            e = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if e is not None:
                total += float(e.duration)
                found = True
    return total if found else None


def _csv_size():
    return _CSV.stat().st_size if _CSV.exists() else 0


def _zone_trisc(offset, zone="compute_chain"):
    """Median-over-cores ns of `zone` per TRISC, rows appended after byte offset."""
    if not _CSV.exists():
        return {}
    with open(_CSV, "rb") as f:
        header = f.readline().decode()
        f.readline()
        if f.tell() < offset:
            f.seek(offset)
        lines = f.read().decode().splitlines()
    freq = 1000.0
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            freq = float(part.split(":")[1])
    st, en = {}, {}
    for ln in lines:
        r = [c.strip() for c in ln.split(",")]
        if len(r) < 12 or r[10] != zone:
            continue
        key = (r[3], r[1], r[2])
        (st if r[11] == "ZONE_START" else en)[key] = int(r[5])
    per = {}
    for k, s in st.items():
        if k in en:
            per.setdefault(k[0], []).append((en[k] - s) * 1000.0 / freq)
    lab = {"TRISC_0": "unp", "TRISC_1": "math", "TRISC_2": "pack"}
    return {lab.get(k, k): statistics.median(v) for k, v in per.items() if k.startswith("TRISC")}


def _alloc(device, shape):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    ok = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[ok], b[ok]
    if torch.equal(a, b):
        return 1.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _ord(t):
    i = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -(i & 0x7FFF), i)


def _ulp(a, b):
    """max / count of ULP diffs (bf16), NaN positions must agree."""
    na, nb = torch.isnan(a.float()), torch.isnan(b.float())
    nan_mismatch = int((na ^ nb).sum())
    ok = ~(na | nb)
    d = (_ord(a[ok]) - _ord(b[ok])).abs()
    return int(d.max()) if d.numel() else 0, int((d > 0).sum()), nan_mismatch


def _all_bf16(device):
    """Every bf16 bit pattern exactly once: 65536 values in (1,1,256,256)."""
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    t = bits.view(torch.bfloat16).reshape(1, 1, 256, 256)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _randn(device, shape, seed=7):
    torch.manual_seed(seed)
    t = (torch.randn(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def test_fcs_precision(device):
    """Exhaustive bf16 input (+ randn focus-ish) per bounds pair: bit-identity vs baseline, ULP/PCC vs torch."""
    _gen_dm()
    variants = [v for v in _sel("FCS_VARIANTS", VARIANTS) if VARIANTS[v][2] is not None]
    inputs = {"all_bf16": _all_bf16(device), "randn_b8": _randn(device, (8, 1, 1024, 1024))}
    for iname, (t, x) in inputs.items():
        shape = tuple(t.shape)
        finite = torch.isfinite(t.float())
        for lo, hi in BOUNDS:
            # torch references: fp32 math rounded once to bf16 (RNE), and pure-bf16 torch op chain.
            ref32 = (torch.clamp(t.float(), lo, hi) ** 2).to(torch.bfloat16)
            refbf = torch.clamp(t, lo, hi) ** 2
            outs = {}
            for v in ["baseline"] + [v for v in variants if v != "baseline"]:
                out = _alloc(device, shape)
                _run(device, x, out, v, "full", None, lo, hi)
                outs[v] = ttnn.to_torch(out)
            for v, o in outs.items():
                nbase = int((o.view(torch.int16) != outs["baseline"].view(torch.int16)).sum())
                u32, c32, n32 = _ulp(o[finite], ref32[finite])
                ubf, cbf, _ = _ulp(o[finite], refbf[finite])
                pcc = _pcc(o.float(), ref32.float())
                logger.info(
                    f"PREC in={iname} bounds=({lo},{hi}) {v:16s} diff_vs_baseline={nbase} "
                    f"| vs torch_fp32->bf16: maxULP={u32} nULP={c32} nanmis={n32} | vs torch_bf16: maxULP={ubf} "
                    f"nULP={cbf} | pcc={pcc:.6f}"
                )
                assert pcc > 0.999, f"{v} {iname} ({lo},{hi}) pcc {pcc}"
                bf16_exact = all(float(torch.tensor(b).to(torch.bfloat16)) == b for b in (lo, hi))
                if VARIANTS[v][2] and bf16_exact:
                    assert nbase == 0, f"{v} {iname} ({lo},{hi}) differs from baseline in {nbase} elements"
                if VARIANTS[v][2] and not bf16_exact and v != "baseline":
                    # baseline squares the bf16-TRUNCATED bound (Clamp stores to 16-bit DEST first); the fused
                    # pass squares the fp32 bound. Gate: no worse than baseline vs the op's fp32 reference.
                    assert c32 <= base_c32, f"{v} {iname} ({lo},{hi}) worse than baseline vs fp32 ref"
                if v == "baseline":
                    base_c32 = c32
            ttnn.ReadDeviceProfiler(device)


def test_fcs_perf(device):
    """One measured run per (variant, case, mode), after a JIT/warm run."""
    _gen_dm()
    rows = []
    for mode in _sel("FCS_MODES", ["stub", "full"]):
        for case in _sel("FCS_CASES", CASES):
            shape, grid = CASES[case]
            t, x = _randn(device, shape)
            ref32 = (torch.clamp(t.float(), -1.0, 1.0) ** 2).to(torch.bfloat16)
            out = _alloc(device, shape)
            for v in _sel("FCS_VARIANTS", VARIANTS):
                _run(device, x, out, v, mode, grid, -1.0, 1.0)
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
                off = _csv_size()
                _run(device, x, out, v, mode, grid, -1.0, 1.0)
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
                ns = _kernel_ns()
                z = _zone_trisc(off)
                if mode == "full" and VARIANTS[v][2]:
                    # contract-exact variants: bit-identical to the op's fp32 reference on this input
                    o = ttnn.to_torch(out)
                    nd = int((o.view(torch.int16) != ref32.view(torch.int16)).sum())
                    assert nd == 0, f"{v} {case}: {nd} elements differ from fp32 reference"
                rows.append((mode, case, v, ns, z))
    logger.info("==== SUMMARY ====")
    base = {(m, c): ns for m, c, v, ns, _ in rows if v == "baseline"}
    for m, c, v, ns, z in rows:
        b = base.get((m, c))
        sp = f"{b / ns:.3f}x" if b and ns else "-"
        zs = " ".join(f"{k}={z[k]:.0f}" for k in sorted(z))
        logger.info(f"BENCH {m:5s} {c:8s} {v:16s} {ns:10.0f} ns  vs_base={sp}  zone[{zs}]")
