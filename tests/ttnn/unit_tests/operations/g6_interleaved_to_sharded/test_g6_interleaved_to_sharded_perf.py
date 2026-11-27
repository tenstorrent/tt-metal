# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench for g6_interleaved_to_sharded on the PERF FOCUS case (FLUX 58, p1).

Measurement only: run under `scripts/run_safe_pytest.sh --profile --no-py-zones`. Every test makes
exactly one op call and no readback, so the profiler CSV holds one GenericOp row per test, in
parametrize order. Correctness of each knob value is graded by test_g6_interleaved_to_sharded_knobs.py.

The "ablate_reads" variant compiles the reader with I2S_ABLATE_READS: the NoC reads are dropped and
the launch, argument loading and the barrier stay, which gives the sync-only floor of the kernel.
"""

from __future__ import annotations

import sys

import pytest
import torch
import ttnn

from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as descriptor

op_module = sys.modules["ttnn.operations.g6_interleaved_to_sharded.g6_interleaved_to_sharded"]

VARIANTS = {
    "default": {},
    "valid_rows": {"PARTIAL_TILE_TRANSFER": "valid_rows"},
    "valid_rows_span": {"PARTIAL_TILE_TRANSFER": "valid_rows_span"},
    "whole_page": {"PARTIAL_TILE_TRANSFER": "whole_page"},
    "ablate_reads": {"_ablate_reads": True},
    "brisc_noc0": {"READER_PROCESSOR": "brisc_noc0"},
    "brisc_noc1": {"READER_PROCESSOR": "brisc_noc1"},
    "split2": {"READER_SPLIT": 2},
    "split2-ablate_reads": {"READER_SPLIT": 2, "_ablate_reads": True},
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_perf_flux58(variant, device, monkeypatch):
    knobs = dict(VARIANTS[variant])
    ablate = knobs.pop("_ablate_reads", False)
    for name, value in knobs.items():
        monkeypatch.setattr(descriptor, name, value)
    # A fresh program per variant: the cache key does not carry the ablation define.
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    if ablate:
        build = op_module.create_program_descriptor

        def build_ablated(*args, **kwargs):
            program = build(*args, **kwargs)
            for kernel in program.kernels:
                kernel.defines = [("I2S_ABLATE_READS", "1")]
            return program

        monkeypatch.setattr(op_module, "create_program_descriptor", build_ablated)

    torch.manual_seed(0)
    torch_input = make_torch_input(FLUX_58["shape"], FLUX_58["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=FLUX_58["dtype"],
        layout=FLUX_58["layout"],
        memory_config=build_memory_config(FLUX_58["in_memory"], device),
    )
    output = g6_interleaved_to_sharded(ttnn_input, build_memory_config(FLUX_58["out_memory"], device))
    ttnn.synchronize_device(device)
    assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
