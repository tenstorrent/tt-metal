# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness (measurement only): DEVICE KERNEL DURATION of nomic_qkv_rotary_heads at the focus shapes, read
in-process from the device profiler. Run with
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
        scripts/run_safe_pytest.sh -s tests/ttnn/unit_tests/operations/nomic_qkv_rotary_heads/test_nomic_qkv_rotary_heads_perf.py
Correctness is the acceptance test's job; this file only prints "PERF <shape> <ns> ns"."""

import math
import os

import pytest
import torch
import ttnn

from ttnn.operations.nomic_qkv_rotary_heads import (
    default_compute_kernel_config,
    nomic_qkv_rotary_heads,
    prepare_qkv_weights,
    prepare_rotary_tables,
)
from ttnn.operations.nomic_qkv_rotary_heads.nomic_qkv_rotary_heads import model_layout_memory_config
from ttnn.operations.nomic_residual_layer_norm import (
    input_memory_config as norm_input_memory_config,
    output_memory_config as norm_output_memory_config,
)

SHAPES = [(2, 1, 512, 768), (4, 1, 512, 768), (8, 1, 512, 768)]


def _durations():
    ttnn.ReadDeviceProfiler(_durations.device)
    data = ttnn.get_latest_programs_perf_data()
    out = []
    for _chip, programs in (data or {}).items():
        for program in programs:
            results = program.program_analyses_results
            if os.environ.get("QKV_PERF_ALL"):
                print({k: int(v.duration) for k, v in results.items()})
            if "DEVICE KERNEL DURATION [ns]" in results:
                out.append((int(results["DEVICE KERNEL DURATION [ns]"].duration), int(program.core_count)))
    return out


@pytest.mark.skipif(os.environ.get("TT_METAL_DEVICE_PROFILER") != "1", reason="needs the device profiler")
@pytest.mark.parametrize("placement", ["ln_layout", "model_layout"])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{s[0]}x{s[2]}" for s in SHAPES])
def test_perf(shape, placement, device):
    _durations.device = device
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    weight = (torch.randn(2304, 768) / math.sqrt(768)).to(torch.bfloat16)
    bias = (0.02 * torch.randn(2304)).to(torch.bfloat16)
    if placement == "ln_layout":
        layout = norm_output_memory_config(shape, device, norm_input_memory_config(shape, device))
    else:
        layout = model_layout_memory_config(shape, device)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=layout)
    weights = prepare_qkv_weights(weight, bias, device)
    tables = prepare_rotary_tables(shape[2], device)
    heads_shape = ttnn.Shape((shape[0], 12, shape[2], 64))
    v, k, q = (
        ttnn.allocate_tensor_on_device(heads_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
        for _ in range(3)
    )
    _durations()  # drain whatever setup put in the profiler
    kwargs = dict(
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tensors=[q, k, v],
        compute_kernel_config=default_compute_kernel_config(),
    )
    nomic_qkv_rotary_heads(tt_x, weights, tables, **kwargs)
    ttnn.synchronize_device(device)
    results = _durations()
    assert results, "profiler returned no programs"
    ns, cores = max(results)
    print(f"\nPERF {shape[0]}x{shape[2]} {placement} {ns} ns cores={cores} all={results}")
