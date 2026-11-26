# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 perf harness (measurement only — DO NOT DELETE).

Run with:  scripts/run_safe_pytest.sh --profile --no-py-zones --run-all "<this file>::test_variant[<id>]" ...
One GenericOpDeviceOperation row per test, in the order given; device kernel ns is read from
generated/profiler/.logs/cpp_device_perf_report.csv (per-core detail: profile_log_device.csv).

A variant is (plan override, kernel defines). Defines enter the kernel hash, so define-gated
ablations never leave stale binaries in the JIT cache (verification_report.md). Ablation variants
produce wrong results by construction and skip the correctness check; every other variant is
checked against the float64 reference.
"""

import importlib

import pytest

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
)

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
desc_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)

CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES}

READER, WRITER, COMPUTE = desc_module.READER_KERNEL_INDEX, desc_module.WRITER_KERNEL_INDEX, 2

# name -> (block_ch_tiles override or None, split_rows pin or None (plan default),
#          {kernel index: [(define, value)]}, check correctness)
VARIANTS = {
    "base": (None, None, {}, True),
    "s0": (None, 0, {}, True),
    "s8": (None, 8, {}, True),
    "s16": (None, 16, {}, True),
    "s24": (None, 24, {}, True),
    "s32": (None, 32, {}, True),
    "w1s24": (1, 24, {}, True),
    "w2": (2, None, {}, True),
    "w2s0": (2, 0, {}, True),
    "w2s16": (2, 16, {}, True),
    "w2s24": (2, 24, {}, True),
    "w4s24": (4, 24, {}, True),
}


def _forced_plan(w):
    """select_plan with block_ch_tiles forced to w (same core / block / split logic otherwise)."""
    real = desc_module.select_plan

    def select_plan(tensor_seq_tiles, tensor_ch_tiles, grid_x, grid_y, regime, l1_budget=None, **kw):
        n = grid_x * grid_y
        groups = tensor_ch_tiles // w
        ranges = min(n // groups, tensor_seq_tiles)
        core_seq_tiles_max = -(-tensor_seq_tiles // ranges)
        ranges = -(-tensor_seq_tiles // core_seq_tiles_max)
        base = real(tensor_seq_tiles, tensor_ch_tiles, grid_x, grid_y, regime, **kw)
        split = kw.get("split_rows")
        if split is None:
            split = desc_module.default_split_rows() if regime == desc_module.REGIME_RM else 0
        return desc_module.Plan(
            regime=regime,
            block_ch_tiles=w,
            block_rows=1,
            num_ch_groups=groups,
            cores_per_group=ranges,
            num_cores=groups * ranges,
            core_seq_tiles_max=core_seq_tiles_max,
            split_rows=split,
        )

    return select_plan


@pytest.mark.parametrize(
    "case_id",
    [
        "conv1d-001",
        "conv1d-002",
        "conv1d-003",
        "conv1d-004",
        "conv1d-005",
        "conv1d-006",
        "conv1d-010",
        "conv1d-011",
        "conv1d-020",
    ],
)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variant(device, monkeypatch, case_id, variant):
    w, split, defines, check = VARIANTS[variant]
    op_module._PROGRAM_CACHE.clear()
    if w is not None:
        monkeypatch.setattr(op_module, "select_plan", _forced_plan(w))
    if defines:
        real_create = op_module.create_program_descriptor

        def create(*args, **kwargs):
            descriptor, work = real_create(*args, **kwargs)
            kernels = list(descriptor.kernels)
            for index, defs in defines.items():
                kernels[index].defines = list(defs)
            descriptor.kernels = kernels
            return descriptor, work

        monkeypatch.setattr(op_module, "create_program_descriptor", create)
    case = CASES[case_id]
    with op_module.plan_limits(split_rows=split):
        outputs, host, device_inputs = _run(device, case)
    print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
    op_module._PROGRAM_CACHE.clear()
    if check:
        _check_case(case, outputs, host, device_inputs)
