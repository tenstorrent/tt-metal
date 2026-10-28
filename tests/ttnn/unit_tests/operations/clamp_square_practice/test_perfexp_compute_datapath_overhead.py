# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf experiment `compute_datapath_overhead` for clamp_square_practice (NOT the op's test suite).

Bake-off of the compute stage's non-SFPU-math overhead: copy path, SFPU bracket/init structure, pack
path. Every kernel variant is swapped into the REAL op descriptor (same CBs, block_tiles, work split,
precision config: bf16, fp32_dest_acc_en=False). Two DM modes:
  full  - the op's real reader/writer (whole-op ns + correctness)
  stub  - reader/writer NoC payload stubbed, CB sync kept (compute-stage-isolated ns)
Measured in-process (DEVICE KERNEL DURATION [ns]) + per-TRISC "compute_chain" zone from the device CSV.
Correctness (bit-identical vs the helper baseline + PCC vs torch) is the only assertion.

Env filters (run_safe_pytest word-splits -k):
  CDO_VARIANTS=a,b   CDO_CASES=focus,...   CDO_MODES=stub,full
"""
import os
from pathlib import Path

_HERE_EXP = (
    Path(__file__).resolve().parents[5]
    / "ttnn/ttnn/operations/clamp_square_practice/perf_experiments/compute_datapath_overhead"
)
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_PROFILER_DIR", str(_HERE_EXP / "profiler"))
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.clamp_square_practice import clamp_square_practice_program_descriptor as pd

KDIR = _HERE_EXP / "kernels"
GEN = _HERE_EXP / "gen_kernels"
SRC = Path(pd.KERNEL_DIR)
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_CSV = Path(os.environ["TT_METAL_PROFILER_DIR"]) / ".logs" / "profile_log_device.csv"

# name -> (kernel file, defines, is_ablation)
R = "raw_compute.cpp"
VARIANTS = {
    "helper": ("baseline_helper_compute.cpp", {}, False),
    "raw_base": (R, {}, False),
    # ---- ablations (timing only; output is wrong by construction) ----
    "abl_nosfpu": (R, {"CDO_SFPU_MODE": 3}, True),
    "abl_clamp_only": (R, {"CDO_DO_SQUARE": 0}, True),
    "abl_square_only": (R, {"CDO_DO_CLAMP": 0}, True),
    "abl_nocopy": (R, {"CDO_COPY_MODE": 2}, True),
    "abl_nocopy_nosfpu": (R, {"CDO_COPY_MODE": 2, "CDO_SFPU_MODE": 3}, True),
    # ---- candidates ----
    "copyblk": (R, {"CDO_COPY_MODE": 1}, False),
    "packblk": (R, {"CDO_PACK_MODE": 1}, False),
    "inithoist": (R, {"CDO_SFPU_INIT_PER_BLOCK": 0}, False),
    "bracket": (R, {"CDO_SFPU_MODE": 1}, False),
    "fusedwalk": (R, {"CDO_SFPU_MODE": 2}, False),
    "all_bracket": (R, {"CDO_COPY_MODE": 1, "CDO_SFPU_MODE": 1, "CDO_PACK_MODE": 1}, False),
    "all_fusedwalk": (R, {"CDO_COPY_MODE": 1, "CDO_SFPU_MODE": 2, "CDO_PACK_MODE": 1}, False),
    "helper_hoistclamp": ("helper_hoistclamp_compute.cpp", {}, False),
    "hoistclamp": (R, {"CDO_SFPU_MODE": 4}, False),
    "hoistclamp_nop": (R, {"CDO_SFPU_MODE": 4, "CDO_SWAP_NOP": 1}, False),
    "abl_hoistclamp_only": (R, {"CDO_SFPU_MODE": 4, "CDO_DO_SQUARE": 0}, True),
    "all_hoistclamp": (R, {"CDO_COPY_MODE": 1, "CDO_SFPU_MODE": 4, "CDO_PACK_MODE": 1}, False),
}

# case -> (shape, core_grid or None=full grid)
CASES = {
    "focus": ((1, 1, 4096, 4096), None),
    "b8_1k": ((8, 1, 1024, 1024), None),
    "small": ((2, 4, 256, 512), None),
    "one": ((1, 1, 32, 32), None),
    "1c_137": ((1, 1, 32, 32 * 137), (1, 1)),
    "1c_8": ((1, 1, 32, 32 * 8), (1, 1)),
    "1c_13": ((1, 1, 32, 32 * 13), (1, 1)),
    "1c_1024": ((1, 1, 1024, 1024), (1, 1)),
}
BOUNDS = [(-1.0, 1.0), (-0.5, 3.0), (-0.3, 0.7), (0.1000001, 0.1000002), (-2.7182817, -0.33333334)]


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
        key = (r[3], r[1], r[2])  # risc, core_x, core_y
        (st if r[11] == "ZONE_START" else en)[key] = int(r[5])
    per = {}
    for k, s in st.items():
        if k in en:
            per.setdefault(k[0], []).append((en[k] - s) * 1000.0 / freq)
    lab = {"TRISC_0": "unp", "TRISC_1": "math", "TRISC_2": "pack"}
    return {lab.get(k, k): statistics.median(v) for k, v in per.items() if k.startswith("TRISC")}


def _inputs(device, shape, seed=7):
    torch.manual_seed(seed)
    t = (torch.randn(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    # special values sprinkled in: +-inf, exact bounds, +-0, huge/tiny magnitudes
    flat = t.view(-1)
    sp = torch.tensor([float("inf"), -float("inf"), -1.0, 1.0, -0.5, 3.0, 0.0, -0.0, 1e30, -1e30, 1e-30, -1e-30])
    idx = torch.randperm(flat.numel())[: min(flat.numel(), 4 * sp.numel())]
    flat[idx] = sp.repeat(4)[: idx.numel()].to(torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return t, x


def _alloc(device, shape):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if torch.equal(a, b):
        return 1.0
    if a.std() == 0 or b.std() == 0:  # constant output (lo ~= hi): fall back to closeness
        return 1.0 if torch.allclose(a, b, rtol=1e-2, atol=1e-3) else 0.0
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def test_correctness(device):
    """Every non-ablation variant: bit-identical to the helper baseline, PCC vs torch."""
    _gen_dm()
    variants = [v for v in _sel("CDO_VARIANTS", VARIANTS) if not VARIANTS[v][2]]
    for case in _sel("CDO_CASES", CASES):
        shape, grid = CASES[case]
        t, x = _inputs(device, shape)
        for lo, hi in BOUNDS:
            ref = torch.clamp(t.float(), lo, hi) ** 2
            outs = {}
            for v in ["helper"] + [v for v in variants if v != "helper"]:
                out = _alloc(device, shape)
                _run(device, x, out, v, "full", grid, lo, hi)
                outs[v] = ttnn.to_torch(out)
            fin = torch.isfinite(ref)
            for v, ob in outs.items():
                bit = torch.equal(ob.view(torch.int16), outs["helper"].view(torch.int16))
                o = ob.float()
                pcc = _pcc(o[fin], ref[fin])
                maxerr = (o[fin] - ref[fin]).abs().max().item()
                logger.info(
                    f"CORR case={case} bounds=({lo},{hi}) {v}: bit_identical={bit} pcc={pcc:.6f} maxabs={maxerr:.3g}"
                )
                assert pcc > 0.999, f"{v} {case} pcc {pcc}"
                assert bit, f"{v} {case} ({lo},{hi}) not bit-identical to helper baseline"
        ttnn.ReadDeviceProfiler(device)  # keep profiler buffers drained


def test_perf(device):
    """One fresh run per (variant, case, mode); compile/warm run first, then the measured run."""
    _gen_dm()
    rows = []
    for mode in _sel("CDO_MODES", ["stub", "full"]):
        for case in _sel("CDO_CASES", CASES):
            shape, grid = CASES[case]
            _, x = _inputs(device, shape)
            out = _alloc(device, shape)
            for v in _sel("CDO_VARIANTS", VARIANTS):
                _run(device, x, out, v, mode, grid, -1.0, 1.0)  # JIT + warm
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
                off = _csv_size()
                _run(device, x, out, v, mode, grid, -1.0, 1.0)
                ttnn.synchronize_device(device)
                ttnn.ReadDeviceProfiler(device)
                ns = _kernel_ns()
                z = _zone_trisc(off)
                rows.append((mode, case, v, ns, z))
                zs = " ".join(f"{k}={z[k]:.0f}" for k in sorted(z))
                logger.info(f"PERF mode={mode} case={case} variant={v} kernel_ns={ns:.0f} zone[{zs}]")
    logger.info("==== SUMMARY ====")
    base = {(m, c): ns for m, c, v, ns, _ in rows if v == "helper"}
    for m, c, v, ns, z in rows:
        b = base.get((m, c))
        sp = f"{b / ns:.3f}x" if b and ns else "-"
        zs = " ".join(f"{k}={z[k]:.0f}" for k in sorted(z))
        logger.info(f"BENCH {m:5s} {c:8s} {v:18s} {ns:10.0f} ns  vs_helper={sp}  zone[{zs}]")
