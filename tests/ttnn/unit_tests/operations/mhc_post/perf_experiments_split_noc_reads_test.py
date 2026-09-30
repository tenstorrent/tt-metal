# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""split_noc_reads bake-off (perf_experiments/split_noc_reads). Baseline = the op's kernels (copied verbatim);
candidates = reads / writes shared across both DM RISCs. Run:

  scripts/run_safe_pytest.sh --device 1 --run-all <this file> -k correct
  SPLIT_PERF_SET=focus scripts/run_safe_pytest.sh --device 1 --profile --run-all <this file> -k perf

Every perf test launches exactly ONE device op; its id is appended (in order) to
perf_experiments/split_noc_reads/run_order.jsonl so report.py can match the ops_perf_results rows.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

REPO = Path(__file__).resolve().parents[5]
EXP_DIR = REPO / "ttnn/ttnn/operations/mhc_post/perf_experiments/split_noc_reads"
_spec = importlib.util.spec_from_file_location("split_pd", EXP_DIR / "split_program_descriptor.py")
spd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spd)
SplitConfig = spd.SplitConfig
HelperConfig = spd.HelperConfig

ORDER_LOG = EXP_DIR / "run_order.jsonl"

# name -> SplitConfig (None = the op's own kernels, verbatim)
VARIANTS = {
    "orig": None,
    "ubase": SplitConfig(),
    "F_b": SplitConfig(f_on="brisc"),
    "X1_b": SplitConfig(x_brisc=1),
    "X2_b": SplitConfig(x_brisc=2),
    "FX1_b": SplitConfig(f_on="brisc", x_brisc=1),
    "W1_n": SplitConfig(out_ncrisc=1),
    "W2_n": SplitConfig(out_ncrisc=2),
    # ---- round 2: dynamic-NoC mode, NoC choice decoupled from RISC choice ----
    "dyn_base": SplitConfig(dynamic=True),  # baseline traffic, dynamic-NoC mode (mode overhead)
    "nX1_alt": SplitConfig(ncrisc_x_alt=1),  # NCRISC reads X_{n-1} on NoC1 (reader splits its own reads)
    "nF_alt": SplitConfig(ncrisc_f_alt=True),  # NCRISC reads F on NoC1
    "bX1_n0": SplitConfig(x_brisc=1, brisc_x_alt=1),  # BRISC reads X_{n-1}, on NoC0
    "bF_n0": SplitConfig(f_on="brisc", brisc_f_alt=True),  # BRISC reads F, on NoC0
    "W1_n_n1": SplitConfig(out_ncrisc=1, ncrisc_w_alt=1),  # NCRISC writes X'_{n-1}, on NoC1
    "bW1_alt": SplitConfig(brisc_w_alt=1),  # BRISC writes X'_{n-1} on NoC0
    # ---- round 3: event-loop schedule on the RISC that does both reads and writes ----
    "W1_n_ev1": SplitConfig(out_ncrisc=1, sched=1),
    "W1_n_ev2": SplitConfig(out_ncrisc=1, sched=2),
    "W2_n_ev1": SplitConfig(out_ncrisc=2, sched=1),
    "bX1_n0_ev1": SplitConfig(x_brisc=1, brisc_x_alt=1, sched=1),
    "bX1_n0_ev2": SplitConfig(x_brisc=1, brisc_x_alt=1, sched=2),
    "X1_b_ev1": SplitConfig(x_brisc=1, sched=1),
    "sym_ev1": SplitConfig(x_brisc=1, brisc_x_alt=1, out_ncrisc=1, sched=1),
    # ---- round 4: helper DMA (op's CBs + compute kernel unchanged) ----
    "h_base": HelperConfig(),  # no help: event-loop DM kernels in the baseline split
    "hW1": HelperConfig(w_help=1),  # NCRISC writes X'_{n-1} (NoC0)
    "hW1_n1": HelperConfig(w_help=1, ncrisc_help_alt=True),  # NCRISC writes X'_{n-1} on NoC1
    "hW2": HelperConfig(w_help=2),
    "hX1_n0": HelperConfig(x_help=1, brisc_help_alt=True),  # BRISC reads X_{n-1} on NoC0
    "hX1": HelperConfig(x_help=1),  # BRISC reads X_{n-1} on NoC1
    "hF_n0": HelperConfig(f_help=True, brisc_help_alt=True),  # BRISC reads F on NoC0
    "hW1X1_n0": HelperConfig(w_help=1, x_help=1, brisc_help_alt=True),
    # ---- round 5: coefficient-load scheduling on the helping BRISC ----
    "h_inc": HelperConfig(coef_incremental=True),
    "hX1_n0_hf1": HelperConfig(x_help=1, brisc_help_alt=True, help_from=1),
    "hX1_n0_inc": HelperConfig(x_help=1, brisc_help_alt=True, coef_incremental=True),
    "hX1_n0_hf1_inc": HelperConfig(x_help=1, brisc_help_alt=True, help_from=1, coef_incremental=True),
    "hF_n0_hf1_inc": HelperConfig(f_help=True, brisc_help_alt=True, help_from=1, coef_incremental=True),
    "hX2_n0_hf1_inc": HelperConfig(x_help=2, brisc_help_alt=True, help_from=1, coef_incremental=True),
    "hW1n1_inc": HelperConfig(w_help=1, ncrisc_help_alt=True, coef_incremental=True),
    "hW1n1_X1n0_hf1_inc": HelperConfig(
        w_help=1, ncrisc_help_alt=True, x_help=1, brisc_help_alt=True, help_from=1, coef_incremental=True
    ),
}
extra = os.environ.get("SPLIT_VARIANTS_EXTRA")  # "name:f_on,x_brisc,out_ncrisc,ncrisc_noc,brisc_noc;..."
if extra:
    for item in extra.split(";"):
        name, spec = item.split(":")
        f_on, xb, on, nn, bn = spec.split(",")
        VARIANTS[name] = SplitConfig(
            f_on=f_on, x_brisc=int(xb), out_ncrisc=int(on), ncrisc_noc=int(nn), brisc_noc=int(bn)
        )

bf, f32 = ttnn.bfloat16, ttnn.float32
# id -> (lead, T, C, n, x_dtype, f_dtype)
SHAPES = {
    "T640_C7168_bf16": ((1, 1), 640, 7168, 4, bf, bf),
    "T640_C1792_bf16": ((1, 1), 640, 1792, 4, bf, bf),
    "T1280_C4096_bf16": ((1, 1), 1280, 4096, 4, bf, bf),
    "T640_C7168_fp32": ((1, 1), 640, 7168, 4, f32, f32),
    "T640_C1792_fp32": ((1, 1), 640, 1792, 4, f32, f32),
    "T640_C7168_xf32_fbf16": ((1, 1), 640, 7168, 4, f32, bf),
    "T1000_C1792_bf16": ((1, 1), 1000, 1792, 4, bf, bf),
    "T1000_C7168_bf16": ((1, 1), 1000, 7168, 4, bf, bf),
    "T32_C32_bf16": ((1, 1), 32, 32, 4, bf, bf),
    "T17_C128_fp32": ((1, 1), 17, 128, 4, f32, f32),
    "T640_C1792_n1_bf16": ((1, 1), 640, 1792, 1, bf, bf),
    # n=2 at B >= 2 does not build in the op itself (compute SFPU spill, "cannot write SFPU object to memory"
    # in WeightedSumSfpu::row, reproduced on the real op at T640 C1792): n=2 only at a B=1 shape here.
    "T64_C224_n2_bf16": ((1, 1), 64, 224, 2, bf, bf),
    "T640_C1792_n3_bf16": ((1, 1), 640, 1792, 3, bf, bf),
    "T640_C1792_n5_bf16": ((1, 1), 640, 1792, 5, bf, bf),
    # row-straddling: 3 token rows x 37 cols = 111 units over 110 cores; 2x5 rows x 19 cols batch
    "T96_C1184_straddle_bf16": ((1, 1), 96, 1184, 4, bf, bf),
    "B2_T160_C608_straddle_fp32": ((2, 1), 160, 608, 4, f32, bf),
    # ---- coordinator domain sweep (LOOSE perf dims x SP token counts) ----
    "T256_C1792_bf16": ((1, 1), 256, 1792, 4, bf, bf),
    "T1280_C1792_bf16": ((1, 1), 1280, 1792, 4, bf, bf),
    "T2048_C1792_bf16": ((1, 1), 2048, 1792, 4, bf, bf),
    "T640_C2560_bf16": ((1, 1), 640, 2560, 4, bf, bf),
    "T640_C4096_bf16": ((1, 1), 640, 4096, 4, bf, bf),
    "T256_C7168_bf16": ((1, 1), 256, 7168, 4, bf, bf),
    "T1024_C5120_bf16": ((1, 1), 1024, 5120, 4, bf, bf),
    "T2048_C7168_bf16": ((1, 1), 2048, 7168, 4, bf, bf),
    "T1280_C4096_fp32": ((1, 1), 1280, 4096, 4, f32, f32),
    "T640_C1792_xf32_fbf16": ((1, 1), 640, 1792, 4, f32, bf),
    "T1000_C1792_fp32": ((1, 1), 1000, 1792, 4, f32, f32),
    "T1000_C1792_xf32_fbf16": ((1, 1), 1000, 1792, 4, f32, bf),
    "T1000_C7168_fp32": ((1, 1), 1000, 7168, 4, f32, f32),
    "T1000_C7168_xf32_fbf16": ((1, 1), 1000, 7168, 4, f32, bf),
    "T1024_C7168_bf16": ((1, 1), 1024, 7168, 4, bf, bf),
    "T2048_C4096_bf16": ((1, 1), 2048, 4096, 4, bf, bf),
    "T4096_C1792_bf16": ((1, 1), 4096, 1792, 4, bf, bf),
    "T1280_C6144_bf16": ((1, 1), 1280, 6144, 4, bf, bf),
    "T1024_C4096_bf16": ((1, 1), 1024, 4096, 4, bf, bf),
    "T512_C5120_bf16": ((1, 1), 512, 5120, 4, bf, bf),
    "T1024_C2560_bf16": ((1, 1), 1024, 2560, 4, bf, bf),
    "T4096_C2560_bf16": ((1, 1), 4096, 2560, 4, bf, bf),
}

FOCUS = ["T640_C7168_bf16", "T640_C1792_bf16", "T1280_C4096_bf16"]


def _inputs(device, lead, T, C, n, x_dtype, f_dtype, seed=0):
    torch.manual_seed(seed)
    f = torch.randn(*lead, T, C)
    x = torch.randn(*lead, T, n * C)
    post = torch.rand(*lead, T, n) * 2
    comb = torch.rand(*lead, T, n * n)

    def dev(t, dt):
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tensors = (dev(f, f_dtype), dev(x, x_dtype), dev(post, f32), dev(comb, f32))
    fr = f.bfloat16().float() if f_dtype == bf else f
    xr = x.bfloat16().float() if x_dtype == bf else x
    ref = post.reshape(-1, n, 1) * fr.reshape(-1, 1, C) + torch.einsum(
        "tij,tic->tjc", comb.reshape(-1, n, n), xr.reshape(-1, n, C)
    )
    return tensors, ref.reshape(*lead, T, n * C)


def _run(device, tensors, variant, skip_compute=False, skip_expand=False):
    f, x, post, comb = tensors
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x.shape)), x.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    split = VARIANTS[variant]
    if split is None:
        pd = spd.create_program_descriptor(f, x, post, comb, out, cfg, skip_compute, skip_expand)
    elif isinstance(split, HelperConfig):
        pd = spd.create_helper_program_descriptor(f, x, post, comb, out, cfg, split, skip_compute, skip_expand)
    else:
        pd = spd.create_split_program_descriptor(f, x, post, comb, out, cfg, split, skip_compute, skip_expand)
    return ttnn.generic_op([f, x, post, comb, out], pd)


def _check_ref(got, ref, x_dtype):
    tol = 1e-5 if x_dtype == f32 else 2e-2
    assert torch.allclose(got, ref, rtol=tol, atol=tol), f"max abs err {(got - ref).abs().max()}"


# ---------------------------------------------------------------- correctness
CORRECT_SHAPES = list(SHAPES)
CORRECT_VARIANTS = [v for v in VARIANTS if v != "orig"]


@pytest.mark.parametrize("variant", CORRECT_VARIANTS)
@pytest.mark.parametrize("shape_id", os.environ.get("SPLIT_CORRECT_SHAPES", ",".join(CORRECT_SHAPES)).split(","))
def test_correct(device, shape_id, variant):
    lead, T, C, n, x_dtype, f_dtype = SHAPES[shape_id]
    tensors, ref = _inputs(device, lead, T, C, n, x_dtype, f_dtype)
    base = ttnn.to_torch(_run(device, tensors, "orig")).float()
    got = ttnn.to_torch(_run(device, tensors, variant)).float()
    _check_ref(base, ref, x_dtype)
    assert torch.equal(got, base), f"{variant}: not bit-exact vs baseline, max diff {(got - base).abs().max()}"


# ---------------------------------------------------------------- perf
PERF_SETS = {
    "focus": FOCUS,
    "sweep": [s for s in SHAPES if s not in FOCUS],
    "all": list(SHAPES),
}
PERF_SHAPES = os.environ.get("SPLIT_PERF_SHAPES", ",".join(PERF_SETS[os.environ.get("SPLIT_PERF_SET", "focus")])).split(
    ","
)
PERF_VARIANTS = os.environ.get("SPLIT_PERF_VARIANTS", ",".join(VARIANTS)).split(",")
PERF_MODES = os.environ.get("SPLIT_PERF_MODES", "full,dmfloor").split(",")


@pytest.mark.parametrize("variant", PERF_VARIANTS)
@pytest.mark.parametrize("mode", PERF_MODES)
@pytest.mark.parametrize("shape_id", PERF_SHAPES)
def test_perf(device, shape_id, mode, variant):
    lead, T, C, n, x_dtype, f_dtype = SHAPES[shape_id]
    tensors, ref = _inputs(device, lead, T, C, n, x_dtype, f_dtype)
    stub = mode == "dmfloor"  # compute AND coefficient expansion stubbed (CB handshakes kept)
    spd.SKIP_NOC = mode == "compute"  # data NoC transfers stubbed (compute alone, CB handshakes kept)
    with open(ORDER_LOG, "a") as fh:
        fh.write(json.dumps({"shape": shape_id, "mode": mode, "variant": variant, "pid": os.getpid()}) + "\n")
    out = _run(device, tensors, variant, skip_compute=stub, skip_expand=stub)
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    spd.SKIP_NOC = False
    if mode == "full":
        _check_ref(ttnn.to_torch(out).float(), ref, x_dtype)
