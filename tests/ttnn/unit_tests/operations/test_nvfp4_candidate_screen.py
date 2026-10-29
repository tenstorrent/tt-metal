# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate screen: unchanged reconstruction corpus/gate, SFPU path only.

Promising candidates still require the complete regression suite and independent
hardware timing confirmation. This avoids recomputing unrelated storage/FPU
baselines for each register-scheduling experiment.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.test_nvfp4_accuracy import (
    SFPU_ACCURACY_GATE,
    TENSOR_SCALES,
    _accuracy_gate_failures,
    _corpus,
    _metrics,
    _reference,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="NVFP4 screen requires Blackhole")


def test_nvfp4_candidate_screen(device, tmp_path):
    torch.set_num_threads(4)
    codes, scales = _corpus()
    packed, blocks = ttnn.experimental.nvfp4.pack(codes, scales, experimental_fast=True)
    data = ttnn.from_torch(packed, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    scale_data = ttnn.from_torch(blocks, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    valid_count = 127 * 256 * 16
    tensor_scales = TENSOR_SCALES
    cases, failures = [], []
    for tensor_scale in tensor_scales:
        output = ttnn.experimental.dequantize_nvfp4(
            data, scale_data, global_scale=tensor_scale, experimental_fast=True, fpu_tensor_scale=False
        )
        actual = ttnn.to_torch(output).float().flatten()[:valid_count]
        ttnn.deallocate(output)
        reference = _reference(codes, scales, tensor_scale, flush_subnormals=True).flatten()[:valid_count]
        metrics = _metrics(actual, reference)
        cases.append(dict(tensor_scale=tensor_scale, metrics=metrics))
        failures.extend(f"scale={tensor_scale}: {message}" for message in _accuracy_gate_failures(metrics))
    ttnn.deallocate(data)
    ttnn.deallocate(scale_data)
    observed = dict(
        max_bf16_ulp=max(case["metrics"]["max_bf16_ulp"] for case in cases),
        mean_bf16_ulp=max(case["metrics"]["mean_bf16_ulp"] for case in cases),
        median_bf16_ulp=max(case["metrics"]["median_bf16_ulp"] for case in cases),
        minimum_snr_db=min(
            case["metrics"]["snr_db"] for case in cases if isinstance(case["metrics"]["snr_db"], (int, float))
        ),
    )
    kernel = Path(os.environ.get("NVFP4_KERNEL_ROOT", Path(__file__).resolve().parents[4])) / (
        "ttnn/cpp/ttnn/operations/experimental/dequantize_nvfp4/device/kernels/compute/nvfp4_sfpu.hpp"
    )
    report = dict(
        kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
        valid_values_per_scale_case=valid_count,
        cases=cases,
        accuracy_gate=dict(
            path="fast_sfpu_scale",
            limits=SFPU_ACCURACY_GATE,
            subnormals="flush to zero",
            passed=not failures,
            observed=observed,
        ),
    )
    destination = Path(os.environ.get("NVFP4_ACCURACY_REPORT", tmp_path / "screen.json"))
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert not failures, "\n".join(failures)
