#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Precision analysis for rms_norm kernel.

Evaluates 4 configurations:
  A) bf16 input, bf16 intermediate CBs, fp32_dest_acc_en=False
  B) bf16 input, bf16 intermediate CBs, fp32_dest_acc_en=True
  C) fp32 input, fp32 intermediate CBs, fp32_dest_acc_en=True
  D) bf16 input, fp32 intermediate CBs, fp32_dest_acc_en=True

Metrics per configuration:
  - PCC (Pearson Correlation Coefficient)
  - Abs diff: p50, p90, p99, max
  - ULP diff: p50, p90, p99, max
  - RMS error
  - Where do the biggest errors occur (value magnitudes)
"""

import struct
import math
import json
import numpy as np
import torch
import ttnn

from pathlib import Path

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def pytorch_rms_norm(x, *, gamma=None, epsilon=1e-6):
    """Reference in float64 for maximum precision."""
    x64 = x.to(torch.float64)
    rms = torch.sqrt(torch.mean(x64**2, dim=-1, keepdim=True) + epsilon)
    out = x64 / rms
    if gamma is not None:
        out = out * gamma.to(torch.float64)
    return out


def pcc(actual, expected):
    """Pearson correlation coefficient."""
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    a_centered = a - a.mean()
    e_centered = e - e.mean()
    num = (a_centered * e_centered).sum()
    den = a_centered.norm() * e_centered.norm()
    if den < 1e-30:
        return 1.0 if num.abs() < 1e-30 else 0.0
    return (num / den).item()


def bf16_ulp(value):
    """Return the ULP (unit in the last place) for a bfloat16 value."""
    # bf16: 8 exponent bits, 7 mantissa bits
    # ULP = 2^(exponent - 7) for normal numbers
    f = abs(float(value))
    if f == 0.0:
        return 2.0 ** (-133)  # smallest bf16 subnormal
    exp = math.floor(math.log2(f)) if f > 0 else -126
    exp = max(exp, -126)  # clamp to bf16 min exponent
    return 2.0 ** (exp - 7)


def fp32_ulp(value):
    """Return the ULP for a float32 value."""
    f = abs(float(value))
    if f == 0.0:
        return 2.0 ** (-149)
    exp = math.floor(math.log2(f)) if f > 0 else -126
    exp = max(exp, -126)
    return 2.0 ** (exp - 23)


def ulp_diff(actual, expected, dtype_str):
    """Compute ULP distance for each element."""
    ulp_fn = bf16_ulp if dtype_str == "bf16" else fp32_ulp
    diffs = []
    a_flat = actual.flatten()
    e_flat = expected.flatten()
    for i in range(len(a_flat)):
        u = ulp_fn(e_flat[i].item())
        diffs.append(abs(a_flat[i].item() - e_flat[i].item()) / u if u > 0 else 0.0)
    return torch.tensor(diffs, dtype=torch.float64)


def percentiles(t, ps=[50, 90, 99]):
    """Return dict of percentile values + max."""
    t_np = t.numpy()
    result = {}
    for p in ps:
        result[f"p{p}"] = float(np.percentile(t_np, p))
    result["max"] = float(t_np.max())
    return result


def rms_error(actual, expected):
    return ((actual.to(torch.float64) - expected.to(torch.float64)) ** 2).mean().sqrt().item()


def analyze_worst_elements(actual, expected, abs_diffs, n=10):
    """Find the n elements with largest abs diff, report their values."""
    flat_actual = actual.flatten()
    flat_expected = expected.flatten()
    flat_diffs = abs_diffs.flatten()

    _, indices = flat_diffs.topk(min(n, len(flat_diffs)))
    records = []
    for idx in indices:
        i = idx.item()
        records.append(
            {
                "actual": float(flat_actual[i]),
                "expected": float(flat_expected[i]),
                "abs_diff": float(flat_diffs[i]),
                "abs_expected": abs(float(flat_expected[i])),
            }
        )
    return records


# ─────────────────────────────────────────────
# Patched rms_norm that accepts config overrides
# ─────────────────────────────────────────────


def run_rms_norm_with_config(
    input_tensor, gamma_tensor, epsilon, device, *, input_dtype, intermed_dtype, fp32_dest_acc_en
):
    """
    Run rms_norm with specific dtype/accumulation configuration.

    Returns the output tensor (on device).
    """
    from ttnn.operations.rms_norm.rms_norm_program_descriptor import (
        create_program_descriptor,
        KERNEL_DIR,
        _float_to_uint32,
    )
    import ttnn as _ttnn

    # Prepare input in requested dtype
    x_torch = ttnn.to_torch(input_tensor)
    if input_dtype == ttnn.float32:
        x_tt = ttnn.from_torch(
            x_torch.float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        x_tt = ttnn.from_torch(
            x_torch.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Prepare gamma in same dtype
    gamma_tiled = None
    if gamma_tensor is not None:
        g_torch = ttnn.to_torch(gamma_tensor)
        if input_dtype == ttnn.float32:
            g_tt = ttnn.from_torch(
                g_torch.float(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            g_tt = ttnn.from_torch(
                g_torch.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        gamma_tiled = g_tt

    # Now build program descriptor with overrides
    has_gamma = gamma_tiled is not None
    is_float32_input = input_dtype == ttnn.float32

    shape = x_tt.shape
    rank = len(shape)
    TILE_H, TILE_W = 32, 32
    H, W = shape[rank - 2], shape[rank - 1]
    Ht, Wt = H // TILE_H, W // TILE_W
    NC = 1
    for i in range(rank - 2):
        NC *= shape[i]
    total_rows = NC * Ht

    tile_size = x_tt.buffer_page_size()
    scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)

    # Intermediate tile size based on intermed_dtype
    if intermed_dtype == ttnn.float32:
        i_tile_size = ttnn.tile_size(ttnn.float32)
    else:
        i_tile_size = tile_size

    # Core grid
    compute_grid = device.compute_with_storage_grid_size()
    max_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    (num_cores, core_grid, cg1, cg2, rpc1, rpc2) = ttnn.split_work_to_cores(all_cores, total_rows)

    # CBs
    cbs = []
    # c_0: input
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=input_dtype, page_size=tile_size)],
        )
    )
    # c_1: reduce scaler (always bf16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=scaler_tile_size)
            ],
        )
    )
    # c_2: epsilon (always bf16)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=scaler_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=2, data_format=ttnn.bfloat16, page_size=scaler_tile_size)
            ],
        )
    )
    # c_3: gamma
    if has_gamma:
        g_ts = gamma_tiled.buffer_page_size()
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * g_ts,
                core_ranges=core_grid,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=3, data_format=input_dtype, page_size=g_ts)],
            )
        )
    # c_16: output
    out_ts = tile_size  # same as input dtype
    cbs.append(
        ttnn.CBDescriptor(
            total_size=2 * out_ts,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=16, data_format=input_dtype, page_size=out_ts)],
        )
    )
    # c_24: x_sq (intermediate)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * i_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=24, data_format=intermed_dtype, page_size=i_tile_size)
            ],
        )
    )
    # c_25: reduce_out (intermediate)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=i_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=25, data_format=intermed_dtype, page_size=i_tile_size)
            ],
        )
    )
    # c_26: rsqrt (intermediate)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=i_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=26, data_format=intermed_dtype, page_size=i_tile_size)
            ],
        )
    )
    # c_27: normed (intermediate)
    cb27_pages = Wt if has_gamma else 1
    cbs.append(
        ttnn.CBDescriptor(
            total_size=cb27_pages * i_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=27, data_format=intermed_dtype, page_size=i_tile_size)
            ],
        )
    )

    # Reader kernel
    reader_ct = [1 if has_gamma else 0]
    reader_ct.extend(ttnn.TensorAccessorArgs(x_tt).get_compile_time_args())
    if has_gamma:
        reader_ct.extend(ttnn.TensorAccessorArgs(gamma_tiled).get_compile_time_args())
    else:
        reader_ct.append(0)

    epsilon_u32 = _float_to_uint32(epsilon)
    reader_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in cg1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                reader_rt[x][y] = [
                    x_tt.buffer_address(),
                    gamma_tiled.buffer_address() if has_gamma else 0,
                    cur,
                    rpc1,
                    Wt,
                    epsilon_u32,
                ]
                cur += rpc1
    if rpc2 > 0:
        for cr in cg2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    reader_rt[x][y] = [
                        x_tt.buffer_address(),
                        gamma_tiled.buffer_address() if has_gamma else 0,
                        cur,
                        rpc2,
                        Wt,
                        epsilon_u32,
                    ]
                    cur += rpc2
    for xi in range(compute_grid.x):
        for yi in range(compute_grid.y):
            try:
                _ = reader_rt[xi][yi]
            except:
                reader_rt[xi][yi] = []

    reader_k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "reader_rms_norm.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Output tensor
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), input_dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # Writer kernel
    writer_ct = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    ct = 0
    for cr in cg1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                nt = rpc1 * Wt
                writer_rt[x][y] = [output_tensor.buffer_address(), ct, nt]
                ct += nt
    if rpc2 > 0:
        for cr in cg2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    nt = rpc2 * Wt
                    writer_rt[x][y] = [output_tensor.buffer_address(), ct, nt]
                    ct += nt
    for xi in range(compute_grid.x):
        for yi in range(compute_grid.y):
            try:
                _ = writer_rt[xi][yi]
            except:
                writer_rt[xi][yi] = []

    writer_k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "writer_rms_norm.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel
    max_rows = max(rpc1, rpc2) if rpc2 > 0 else rpc1
    compute_ct = [max_rows, Wt, 1 if has_gamma else 0]
    compute_rt = ttnn.RuntimeArgs()
    cur = 0
    for cr in cg1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                compute_rt[x][y] = [rpc1]
                cur += rpc1
    if rpc2 > 0:
        for cr in cg2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    compute_rt[x][y] = [rpc2]
                    cur += rpc2
    for xi in range(compute_grid.x):
        for yi in range(compute_grid.y):
            try:
                _ = compute_rt[xi][yi]
            except:
                compute_rt[xi][yi] = []

    compute_k = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "compute_rms_norm.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=False
        ),
    )

    pd = ttnn.ProgramDescriptor(kernels=[reader_k, writer_k, compute_k], semaphores=[], cbs=cbs)

    io = [x_tt]
    if has_gamma:
        io.append(gamma_tiled)
    io.append(output_tensor)

    return ttnn.generic_op(io, pd)


# ─────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────

CONFIGS = {
    "A_bf16_noacc": dict(input_dtype=ttnn.bfloat16, intermed_dtype=ttnn.bfloat16, fp32_dest_acc_en=False),
    "B_bf16_fp32acc": dict(input_dtype=ttnn.bfloat16, intermed_dtype=ttnn.bfloat16, fp32_dest_acc_en=True),
    "C_fp32_fp32acc": dict(input_dtype=ttnn.float32, intermed_dtype=ttnn.float32, fp32_dest_acc_en=True),
    "D_bf16_fp32cb_fp32acc": dict(input_dtype=ttnn.bfloat16, intermed_dtype=ttnn.float32, fp32_dest_acc_en=True),
}

# Test shapes: small, medium, wide, tall, batched
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 32, 128),
    (1, 1, 128, 256),
    (1, 1, 32, 1024),
    (2, 3, 64, 128),
    (1, 1, 512, 512),
]

GAMMA_MODES = ["no_gamma", "with_gamma"]


def run_analysis():
    device = ttnn.open_device(device_id=0)
    all_results = {}

    for shape in SHAPES:
        shape_str = "x".join(str(s) for s in shape)
        torch.manual_seed(42)
        N, C, H, W = shape

        x_torch = torch.randn(N, C, H, W, dtype=torch.bfloat16)
        gamma_torch = torch.randn(1, 1, 1, W, dtype=torch.bfloat16)

        # Pre-create device tensors for reuse (bf16 baseline)
        x_tt = ttnn.from_torch(
            x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gamma_tt = ttnn.from_torch(
            gamma_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for gmode in GAMMA_MODES:
            gamma_arg = gamma_tt if gmode == "with_gamma" else None
            gamma_torch_arg = gamma_torch if gmode == "with_gamma" else None

            # Reference: float64
            ref_f64 = pytorch_rms_norm(
                x_torch.float(), gamma=gamma_torch_arg.float() if gamma_torch_arg is not None else None, epsilon=1e-6
            )

            for cfg_name, cfg in CONFIGS.items():
                key = f"{shape_str}__{gmode}__{cfg_name}"
                print(f"Running {key} ...", flush=True)

                try:
                    out_tt = run_rms_norm_with_config(x_tt, gamma_arg, 1e-6, device, **cfg)
                    out_torch = ttnn.to_torch(out_tt).to(torch.float64)
                except Exception as e:
                    all_results[key] = {"error": str(e)}
                    print(f"  ERROR: {e}")
                    continue

                # Determine output dtype for ULP calculation
                out_dtype_str = "fp32" if cfg["input_dtype"] == ttnn.float32 else "bf16"

                # Metrics
                abs_diffs = (out_torch - ref_f64).abs()
                pcc_val = pcc(out_torch, ref_f64)
                rms_err = rms_error(out_torch, ref_f64)
                abs_pcts = percentiles(abs_diffs.flatten())
                ulp_diffs = ulp_diff(out_torch, ref_f64, out_dtype_str)
                ulp_pcts = percentiles(ulp_diffs)

                # Worst elements analysis
                worst = analyze_worst_elements(out_torch, ref_f64, abs_diffs, n=5)

                # Value magnitude buckets for error analysis
                ref_abs = ref_f64.abs().flatten()
                ad_flat = abs_diffs.flatten()
                buckets = {}
                for bname, lo, hi in [("<0.1", 0, 0.1), ("0.1-1", 0.1, 1.0), ("1-10", 1.0, 10.0), (">10", 10.0, 1e30)]:
                    mask = (ref_abs >= lo) & (ref_abs < hi)
                    if mask.sum() > 0:
                        buckets[bname] = {
                            "count": int(mask.sum()),
                            "mean_abs_diff": float(ad_flat[mask].mean()),
                            "max_abs_diff": float(ad_flat[mask].max()),
                        }

                all_results[key] = {
                    "shape": shape_str,
                    "gamma": gmode,
                    "config": cfg_name,
                    "pcc": pcc_val,
                    "rms_error": rms_err,
                    "abs_diff": abs_pcts,
                    "ulp_diff": ulp_pcts,
                    "worst_elements": worst,
                    "error_by_magnitude": buckets,
                }
                print(f"  PCC={pcc_val:.8f} max_abs={abs_pcts['max']:.6f} max_ulp={ulp_pcts['max']:.1f}")

    ttnn.close_device(device)
    return all_results


def format_report(results):
    """Generate structured markdown report."""
    lines = []
    lines.append("# RMS Norm Precision Analysis\n")
    lines.append("## Configurations\n")
    lines.append("| Config | Input dtype | Intermediate CB dtype | fp32_dest_acc_en |")
    lines.append("|--------|------------|----------------------|-----------------|")
    lines.append("| A | bfloat16 | bfloat16 | False |")
    lines.append("| B | bfloat16 | bfloat16 | True |")
    lines.append("| C | float32 | float32 | True |")
    lines.append("| D | bfloat16 | float32 | True |")
    lines.append("")

    # Group by shape+gamma
    from collections import defaultdict

    groups = defaultdict(dict)
    for key, data in results.items():
        if "error" in data:
            continue
        group_key = f"{data['shape']}__{data['gamma']}"
        groups[group_key][data["config"]] = data

    for group_key in sorted(groups.keys()):
        shape_str, gmode = group_key.split("__")
        lines.append(f"## Shape: {shape_str}, Gamma: {gmode}\n")

        # Summary table
        lines.append("| Metric | A (bf16, no acc) | B (bf16, fp32 acc) | C (fp32, fp32 acc) | D (bf16 in, fp32 cb) |")
        lines.append("|--------|-----------------|-------------------|-------------------|---------------------|")

        configs_ordered = ["A_bf16_noacc", "B_bf16_fp32acc", "C_fp32_fp32acc", "D_bf16_fp32cb_fp32acc"]
        cfg_data = [groups[group_key].get(c) for c in configs_ordered]

        def val(d, path, fmt=".6f"):
            if d is None:
                return "ERR"
            obj = d
            for p in path.split("."):
                obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
            return f"{obj:{fmt}}"

        lines.append(
            f"| PCC | {val(cfg_data[0], 'pcc', '.8f')} | {val(cfg_data[1], 'pcc', '.8f')} | {val(cfg_data[2], 'pcc', '.8f')} | {val(cfg_data[3], 'pcc', '.8f')} |"
        )
        lines.append(
            f"| RMS error | {val(cfg_data[0], 'rms_error')} | {val(cfg_data[1], 'rms_error')} | {val(cfg_data[2], 'rms_error')} | {val(cfg_data[3], 'rms_error')} |"
        )

        for metric, label in [("abs_diff", "Abs diff"), ("ulp_diff", "ULP diff")]:
            for pct in ["p50", "p90", "p99", "max"]:
                row = f"| {label} {pct} |"
                for d in cfg_data:
                    if d is None:
                        row += " ERR |"
                    else:
                        v = d[metric][pct]
                        if metric == "ulp_diff":
                            row += f" {v:.1f} |"
                        else:
                            row += f" {v:.6f} |"
                lines.append(row)

        lines.append("")

        # Worst elements for each config
        lines.append("### Worst elements (top 5 by abs diff)\n")
        for cfg_name, d in zip(configs_ordered, cfg_data):
            if d is None:
                continue
            lines.append(f"**{cfg_name}**:\n")
            lines.append("| actual | expected | abs_diff | |expected| |")
            lines.append("|--------|----------|----------|-----------|")
            for w in d["worst_elements"]:
                lines.append(
                    f"| {w['actual']:.6f} | {w['expected']:.6f} | {w['abs_diff']:.6f} | {w['abs_expected']:.4f} |"
                )
            lines.append("")

        # Error by value magnitude
        lines.append("### Error by output magnitude\n")
        for cfg_name, d in zip(configs_ordered, cfg_data):
            if d is None:
                continue
            lines.append(f"**{cfg_name}**:\n")
            lines.append("| Magnitude bucket | Count | Mean abs diff | Max abs diff |")
            lines.append("|-----------------|-------|---------------|--------------|")
            for bname, bdata in d.get("error_by_magnitude", {}).items():
                lines.append(
                    f"| {bname} | {bdata['count']} | {bdata['mean_abs_diff']:.6f} | {bdata['max_abs_diff']:.6f} |"
                )
            lines.append("")

    # ─── Error analysis section ───
    lines.append("## Analysis: Are These Errors Expected or Anomalous?\n")
    lines.append("### Quantization step budget\n")
    lines.append("The kernel pipeline has these CB pack/unpack boundaries (each introduces bf16 quantization):\n")
    lines.append("1. `x * x` → cb_x_sq (pack to bf16)")
    lines.append("2. `reduce_sum(x_sq)` → cb_reduce_out (pack to bf16)")
    lines.append("3. `add_eps + rsqrt` → cb_rsqrt (pack to bf16)")
    lines.append("4. `x * rsqrt` → cb_output or cb_normed (pack to bf16)")
    lines.append("5. (if gamma) `normed * gamma` → cb_output (pack to bf16)\n")
    lines.append("Each bf16 pack introduces up to 1 ULP of error at that value's magnitude.")
    lines.append("With 4-5 steps, the accumulated ULP error of 4-5 is **expected** for bf16.\n")
    lines.append("### Expected vs observed\n")
    lines.append(
        "- **Config A** (bf16, no fp32 acc): Errors should be highest — accumulation in bf16 loses precision during reduce."
    )
    lines.append(
        "- **Config B** (bf16, fp32 acc): Reduces reduce-sum error but CB quantization remains. ~4-5 ULP expected."
    )
    lines.append(
        "- **Config C** (fp32 everywhere): Only error source is rsqrt approximation + epsilon tile (bf16). Should be near-zero ULP in fp32 terms."
    )
    lines.append(
        "- **Config D** (bf16 input, fp32 CBs): Input/output still bf16, but intermediates preserve precision. Should help reduce error vs B.\n"
    )

    lines.append("### Conclusion placeholder")
    lines.append("*(filled after data collection)*\n")

    return "\n".join(lines)


if __name__ == "__main__":
    results = run_analysis()

    # Save raw JSON
    out_dir = Path(__file__).parent
    with open(out_dir / "precision_raw.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save report
    report = format_report(results)
    report_path = out_dir / "precision_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nReport written to {report_path}")
    print(f"Raw data written to {out_dir / 'precision_raw.json'}")
