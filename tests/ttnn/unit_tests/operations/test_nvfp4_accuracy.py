# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible NVFP4 error survey; NVFP4_ACCURACY_REPORT selects the JSON artifact."""

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="NVFP4 decoder requires Blackhole")

SFPU_ACCURACY_GATE = {
    "max_bf16_ulp": 2.0,
    "mean_bf16_ulp": 1.0,
    "median_bf16_ulp": 1.0,
    "snr_db_exclusive_min": 46.0,
}


def _accuracy_gate_failures(metrics):
    failures = []
    for key in ("max_bf16_ulp", "mean_bf16_ulp", "median_bf16_ulp"):
        value, limit = metrics[key], SFPU_ACCURACY_GATE[key]
        if not math.isfinite(value) or value > limit:
            failures.append(f"{key}={value} must be <= {limit}")
    snr = metrics["snr_db"]
    # An all-zero reference has undefined SNR and uses the ULP limits above.
    if snr is not None and snr != "inf" and not snr > SFPU_ACCURACY_GATE["snr_db_exclusive_min"]:
        failures.append(f"snr_db={snr} must be > {SFPU_ACCURACY_GATE['snr_db_exclusive_min']}")
    return failures


def _flush_subnormals(values):
    # FP32 and BF16 have the same minimum normal magnitude. This concerns
    # reconstructed operands/results, not FP8 subnormal scale encodings,
    # which decode to ordinary normal BF16 values and remain supported.
    return torch.where(values.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(values), values)


def _corpus():
    generator = torch.Generator().manual_seed(20261027)
    # 256 complete blocks per scale; 127 valid scales plus one padding scale.
    # Each block is 16 horizontal values, matching both NVFP4 and BFP groups.
    codes = torch.randint(0, 16, (128, 256, 16), generator=generator, dtype=torch.int32)
    for code in range(16):
        codes[:, code, :] = code
        codes[:, 16 + code, :] = (torch.arange(16) + code) % 16
    # Explicit outlier, sign, zero, smallest nonzero and maximum-magnitude pairs.
    for code in range(16):
        codes[:, 32 + code, :] = code
        codes[:, 32 + code, 0] = 7
        codes[:, 48 + code, :] = 7
        codes[:, 48 + code, 0] = code
    # Exhaust every (target code, block maximum magnitude) combination. This
    # covers all BFP numerical contexts for aligned, same-scale groups.
    block = 64
    for maximum in range(8):
        for code in range(16):
            if (code & 7) <= maximum:
                codes[:, block, :] = maximum
                codes[:, block, 0] = code
                block += 1
    assert block == 136, "Expected 72 target/maximum combinations"
    scales = torch.arange(128).clamp_max(126).repeat_interleave(256).to(torch.int32)
    return codes.reshape(4096, 128), scales.reshape(4096, 8)


def _reference(codes, scales, global_scale, *, flush_subnormals=False):
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
    values = lut[codes.long() & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    exponent, mantissa = scales >> 3, scales & 7
    scale = torch.where(
        exponent == 0, mantissa.double() * 2**-9, torch.ldexp(1 + mantissa.double() / 8, exponent - 7)
    )
    # The API accepts FP32 tensor scales. Products need at most 30 significant
    # bits here, so FP64 reconstructs them exactly without intermediate rounding.
    tensor_scale = float(torch.tensor(global_scale, dtype=torch.float32))
    if flush_subnormals and abs(tensor_scale) < torch.finfo(torch.float32).tiny:
        tensor_scale = 0.0
    result = values * scale.repeat_interleave(16, dim=-1) * tensor_scale
    return _flush_subnormals(result) if flush_subnormals else result


def _bf16_quantum(reference):
    _, exponent = torch.frexp(reference.abs())
    exponent = torch.maximum(exponent - 8, torch.full_like(exponent, -133))
    exponent = torch.where(reference == 0, torch.full_like(exponent, -133), exponent)
    return torch.ldexp(torch.ones_like(reference), exponent)


def _round_bf16_exact(reference):
    # Round exact FP64 products directly, avoiding FP64 -> FP32 -> BF16
    # double rounding when an FP32 intermediate lands on a BF16 midpoint.
    quantum = _bf16_quantum(reference)
    rounded = torch.round(reference / quantum) * quantum
    largest = float(torch.finfo(torch.bfloat16).max)
    return torch.where(rounded.abs() > largest, torch.copysign(torch.full_like(rounded, math.inf), rounded), rounded)


def _bf16_codes(values):
    bits = (values.float().contiguous().view(torch.int32).long() >> 16) & 65535
    # Monotone ordering; collapse the two zero signs to the same code.
    return torch.where((bits & 32768) != 0, 32768 - (bits & 32767), 32768 + bits)


def _metrics(actual, reference):
    actual, reference = actual.double().flatten(), reference.double().flatten()
    delta = actual - reference
    absolute = delta.abs()
    nonzero = reference != 0
    # BF16 ULP at the ideal magnitude, including zero/subnormal spacing.
    quantum = _bf16_quantum(reference)
    ulp = absolute / quantum
    signal, noise = float(reference.square().sum()), float(delta.square().sum())
    snr = 10 * math.log10(signal / noise) if signal and noise else (None if not signal else "inf")
    relative = absolute[nonzero] / reference[nonzero].abs()
    nz_error = nonzero & (absolute != 0)
    per_value_snr = 20 * torch.log10(reference[nz_error].abs() / absolute[nz_error])
    code_distance = (_bf16_codes(actual) - _bf16_codes(_round_bf16_exact(reference))).abs()
    group_signal = reference.reshape(-1, 16).square().sum(dim=-1)
    group_noise = delta.reshape(-1, 16).square().sum(dim=-1)
    noisy_groups = (group_signal > 0) & (group_noise > 0)
    group_snr = 10 * torch.log10(group_signal[noisy_groups] / group_noise[noisy_groups])
    return {
        "max_bf16_code_distance_from_exact_rne": int(code_distance.max()),
        "mean_bf16_code_distance_from_exact_rne": float(code_distance.double().mean()),
        "minimum_nonexact_nonzero_group_snr_db": float(group_snr.min()) if group_snr.numel() else None,
        "max_bf16_ulp": float(ulp.max()),
        "mean_bf16_ulp": float(ulp.mean()),
        "median_bf16_ulp": float(ulp.median()),
        "p99_bf16_ulp": float(torch.quantile(ulp, 0.99)),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "snr_db": snr,
        "max_relative_error_nonzero": float(relative.max()) if relative.numel() else None,
        "median_relative_error_nonzero": float(relative.median()) if relative.numel() else None,
        "p99_relative_error_nonzero": float(torch.quantile(relative, 0.99)) if relative.numel() else None,
        "per_value_snr_infinite_exact_nonzero_count": int((nonzero & (absolute == 0)).sum()),
        "per_value_snr_undefined_zero_signal_count": int((~nonzero).sum()),
        "median_per_value_snr_db_nonexact_nonzero": float(per_value_snr.median()) if per_value_snr.numel() else None,
        "exact_fraction": float((absolute == 0).double().mean()),
        "nonzero_to_zero": int(((actual == 0) & nonzero).sum()),
        "zero_reference_nonzero_output": int(((reference == 0) & (actual != 0)).sum()),
        "minimum_per_value_snr_db_nonexact_nonzero": float(per_value_snr.min()) if per_value_snr.numel() else None,
        "worst_flat_index": int(ulp.argmax()),
    }


def _read_and_release(tensor):
    result = ttnn.to_torch(tensor).float()
    ttnn.deallocate(tensor)
    return result


def test_nvfp4_accuracy_survey(device, tmp_path):
    torch.set_num_threads(4)
    codes, scales = _corpus()
    # Exclude the final padding scale from every reported statistic.
    valid_count = 127 * 256 * 16
    precise_packed, precise_scales = ttnn.experimental.nvfp4.pack(codes, scales)
    fast_packed, fast_scales = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True)
    uploaded = []
    for host, dtype, layout in [
        (precise_packed, ttnn.uint16, ttnn.TILE_LAYOUT),
        (precise_scales, ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT),
        (fast_packed, ttnn.uint16, ttnn.TILE_LAYOUT),
        (fast_scales, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ]:
        uploaded.append(ttnn.from_torch(host, dtype=dtype, layout=layout, device=device))
    kernel = (
        Path(__file__).resolve().parents[4]
        / "ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/device/kernels/compute/nvfp4_sfpu.hpp"
    )
    report = {
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "reference": "Exact FP64 reconstruction of E2M1 x UE4M3 x original FP32 tensor scale with operand/result FTZ",
        "storage_baseline": "TTNN host packer from FP32-staged reconstruction, aligned horizontal groups of 16",
        "ulp_unit": "BF16 spacing at exact magnitude, with BF16 subnormal spacing at zero",
        "seed": 20261027,
        "valid_values_per_scale_case": valid_count,
        "blocks_per_scale": 256,
        "structured_blocks_per_scale": 136,
        "random_blocks_per_scale": 120,
        "accuracy_gate": {"path": "fast_sfpu_scale", "limits": SFPU_ACCURACY_GATE, "subnormals": "flush to zero"},
        "cases": [],
    }
    # Include exact BF16 scales, FP32 -> BF16 halfway cases and a non-dyadic
    # decimal specified through the FP32 API, plus widely separated exponents.
    tensor_scales = [
        0.0,
        1.0,
        0.8125,
        0.7314453125,
        0.1,
        1.00390625,
        1.01171875,
        1.0234375,
        1.02734375,
        1.03125,
        2**-40,
        2**40,
    ]
    gate_failures = []
    try:
        for tensor_scale in tensor_scales:
            reference = _reference(codes, scales, tensor_scale, flush_subnormals=True)
            selected = reference.flatten()[:valid_count]
            outputs = {"bf16_exact_rne": _round_bf16_exact(selected).float()}
            # Storage-format baselines use TTNN's actual host quantizer. FP32
            # input staging is explicit; these are not device-packer results.
            for name, dtype in [
                ("bf16_storage", ttnn.bfloat16),
                ("bfp8_storage", ttnn.bfloat8_b),
                ("bfp4_storage", ttnn.bfloat4_b),
            ]:
                quantized = ttnn.from_torch(reference.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
                outputs[name] = _read_and_release(quantized).flatten()[:valid_count]
            for name, fast, fpu in [
                ("precise", False, True),
                ("fast_sfpu_scale", True, False),
                ("fast_fpu_scale", True, True),
            ]:
                data, block = uploaded[2:4] if fast else uploaded[:2]
                decoded = ttnn.experimental.dequantize_nvfp4(
                    data, block, global_scale=tensor_scale, experimental_fast=fast, fpu_tensor_scale=fpu
                )
                outputs[name] = _read_and_release(decoded).flatten()[:valid_count]
            baseline = outputs["bfp8_storage"].double()
            baseline_noise = float((baseline - selected).square().sum())
            entries = {}
            for name, output in outputs.items():
                assert torch.isfinite(output).all(), f"Nonfinite {name} output for scale {tensor_scale}"
                metrics = _metrics(output, selected)
                error = output.double() - selected
                noise = float(error.square().sum())
                metrics["noise_energy_vs_bfp8"] = (
                    noise / baseline_noise if baseline_noise else (0.0 if not noise else None)
                )
                metrics["fraction_worse_than_bfp8"] = float((error.abs() > (baseline - selected).abs()).double().mean())
                metrics["fraction_different_from_bfp8"] = float((output.double() != baseline).double().mean())
                group_baseline_noise = (baseline - selected).reshape(-1, 16).square().sum(dim=-1)
                group_noise = error.reshape(-1, 16).square().sum(dim=-1)
                nonexact_baseline = group_baseline_noise != 0
                ratios = group_noise[nonexact_baseline] / group_baseline_noise[nonexact_baseline]
                metrics["max_group_noise_energy_vs_nonexact_bfp8"] = float(ratios.max()) if ratios.numel() else None
                metrics["groups_exact_in_bfp8_but_not_candidate"] = int(
                    ((~nonexact_baseline) & (group_noise != 0)).sum()
                )
                index = metrics["worst_flat_index"]
                group_index = index // 16
                metrics["worst_group_codes"] = codes.flatten()[group_index * 16 : (group_index + 1) * 16].tolist()
                metrics["worst_scale_code"] = int(scales.flatten()[group_index])
                metrics["worst_reference"] = float(selected[index])
                metrics["worst_actual"] = float(output[index])
                entries[name] = metrics
            if tensor_scale == 1.0:
                assert entries["bf16_storage"]["max_bf16_ulp"] == 0, "E2M1 x E4M3 must fit BF16 exactly"
                assert (
                    entries["precise"]["max_bf16_ulp"] == 0
                ), "Precise decoder must reconstruct scale-one samples exactly"
            failures = _accuracy_gate_failures(entries["fast_sfpu_scale"])
            gate_failures.extend(f"tensor_scale={tensor_scale}: {failure}" for failure in failures)
            report["cases"].append(
                {
                    "tensor_scale_fp32": float(torch.tensor(tensor_scale, dtype=torch.float32)),
                    "metrics": entries,
                    "sfpu_accuracy_gate": {"passed": not failures, "failures": failures},
                }
            )
    finally:
        for tensor in uploaded:
            ttnn.deallocate(tensor)
    output_path = Path(os.environ.get("NVFP4_ACCURACY_REPORT", tmp_path / "nvfp4_accuracy.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_metrics = [case["metrics"]["fast_sfpu_scale"] for case in report["cases"]]
    finite_snrs = [metrics["snr_db"] for metrics in gate_metrics if isinstance(metrics["snr_db"], (int, float))]
    report["accuracy_gate"].update(
        {
            "passed": not gate_failures,
            "observed": {
                **{
                    key: max(metrics[key] for metrics in gate_metrics)
                    for key in ("max_bf16_ulp", "mean_bf16_ulp", "median_bf16_ulp")
                },
                "minimum_snr_db": min(finite_snrs) if finite_snrs else None,
            },
        }
    )
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"NVFP4_ACCURACY_REPORT={output_path}")
    assert not gate_failures, "SFPU accuracy gate failed:\n" + "\n".join(gate_failures)


def test_nvfp4_accuracy_boundaries(device, tmp_path):
    """Check the accepted FTZ contract and retain IEEE differences as diagnostics."""
    codes = torch.arange(16, dtype=torch.int32).repeat(256).reshape(32, 128)
    scales = torch.arange(128, dtype=torch.int32).clamp_max(126).repeat_interleave(2).reshape(32, 8)
    valid_count = 127 * 2 * 16
    uploaded = []
    for fast in [False, True]:
        packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=fast)
        uploaded.extend(
            [
                ttnn.from_torch(packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(
                    blocks,
                    dtype=ttnn.bfloat16 if fast else ttnn.uint8,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                ),
            ]
        )
    cases = []
    try:
        for tensor_scale in [2**-149, 2**-134, 2**-133, 2**-127, 2**-126, 2**120, 2**127]:
            reference = _reference(codes, scales, tensor_scale).flatten()[:valid_count]
            oracle = _round_bf16_exact(reference)
            ftz_reference = _reference(codes, scales, tensor_scale, flush_subnormals=True).flatten()[:valid_count]
            ftz_oracle = _round_bf16_exact(ftz_reference)
            metrics = {}
            for name, fast, fpu in [
                ("precise", False, True),
                ("fast_sfpu_scale", True, False),
                ("fast_fpu_scale", True, True),
            ]:
                data, blocks = uploaded[2:4] if fast else uploaded[:2]
                result = ttnn.experimental.dequantize_nvfp4(
                    data, blocks, global_scale=tensor_scale, experimental_fast=fast, fpu_tensor_scale=fpu
                )
                actual = _read_and_release(result).double().flatten()[:valid_count]
                finite = torch.isfinite(oracle) & torch.isfinite(actual)
                error = (actual[finite] - reference[finite]).abs()
                quantum = _bf16_quantum(reference[finite])
                metrics[name] = {
                    "nan_count": int(torch.isnan(actual).sum()),
                    "expected_infinity_count": int(torch.isinf(oracle).sum()),
                    "actual_infinity_count": int(torch.isinf(actual).sum()),
                    "unexpected_infinity_count": int((torch.isinf(actual) & torch.isfinite(oracle)).sum()),
                    "oracle_nonzero_finite_flushed_to_zero": int(
                        ((oracle != 0) & torch.isfinite(oracle) & (actual == 0)).sum()
                    ),
                    "max_bf16_ulp_where_both_finite": float((error / quantum).max()) if error.numel() else None,
                    "nonfinite_classification_mismatches": int(
                        (torch.isfinite(actual) != torch.isfinite(oracle)).sum()
                    ),
                    "ftz_reference_mismatches": int((actual != ftz_oracle).sum()),
                }
                assert metrics[name]["nan_count"] == 0, f"Unexpected NaN in {name} at scale {tensor_scale}"
                # Only the SFPU paths share this exact arithmetic oracle.
                # The HiFi4 FPU path remains diagnostic near underflow.
                if name != "fast_fpu_scale":
                    assert (
                        metrics[name]["ftz_reference_mismatches"] == 0
                    ), f"FTZ mismatch in {name} at scale {tensor_scale}"

            cases.append(
                {"tensor_scale_fp32": float(torch.tensor(tensor_scale, dtype=torch.float32)), "metrics": metrics}
            )
    finally:
        for tensor in uploaded:
            ttnn.deallocate(tensor)
    base_path = Path(os.environ.get("NVFP4_ACCURACY_REPORT", tmp_path / "nvfp4_accuracy.json"))
    output_path = base_path.with_name(base_path.stem + "_boundaries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"valid_values_per_scale_case": valid_count, "cases": cases}, indent=2) + "\n")
    print(f"NVFP4_ACCURACY_BOUNDARIES_REPORT={output_path}")
