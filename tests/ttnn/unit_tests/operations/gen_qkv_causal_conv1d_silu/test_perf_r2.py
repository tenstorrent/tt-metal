# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 perf harness (measurement only — DO NOT DELETE).

Run with:  GQKV_PERF_R2=1 scripts/run_safe_pytest.sh --profile --no-py-zones --run-all \
               "<this file>::test_variant[variant=<v>-case_id=<c>]" ...
(skipped unless GQKV_PERF_R2=1: a measurement harness, not a regression test — the correctness of
the knobs it sweeps is pinned by test_split_reader.py.)
One GenericOpDeviceOperation row per test, in the order given; device kernel ns is read from
generated/profiler/.logs/cpp_device_perf_report.csv (per-core detail: profile_log_device.csv).

A variant is (plan override, kernel defines). Defines enter the kernel hash, so define-gated
ablations never leave stale binaries in the JIT cache (verification_report.md). Ablation variants
produce wrong results by construction and skip the correctness check; every other variant is
checked against the float64 reference.
"""

import importlib
import os

import pytest

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
)

from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_reader_addressing import (
    GOLDEN_CASES,
    _run_and_check,
)

op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
desc_module = importlib.import_module(
    "ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu_program_descriptor"
)

CASES = {pc.id.split("_")[0]: pc.values[0] for pc in PHASE0_CASES + GOLDEN_CASES}

READER, WRITER, COMPUTE = desc_module.READER_KERNEL_INDEX, desc_module.WRITER_KERNEL_INDEX, 2

# name -> (block_ch_tiles override or None, plan_limits pins, {kernel index: [(define, value)]}, check)
VARIANTS = {
    "base": (None, {}, {}, True),
    "s0": (None, dict(split_rows=0), {}, True),
    "s24": (None, dict(split_rows=24), {}, True),
    "g1": (None, dict(out_group_tiles=1), {}, True),
    "g2": (None, dict(out_group_tiles=2), {}, True),
    "g4": (None, dict(out_group_tiles=4), {}, True),
    "g8": (None, dict(out_group_tiles=8), {}, True),
    "w1": (1, {}, {}, True),
    "w2": (2, {}, {}, True),
    "w4": (4, {}, {}, True),
    "w8": (8, {}, {}, True),
    "g1s0": (None, dict(out_group_tiles=1, split_rows=0), {}, True),
    "w2g1": (2, dict(out_group_tiles=1), {}, True),
    "w8g1": (8, dict(out_group_tiles=1), {}, True),
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
        return desc_module.Plan(
            regime=regime,
            block_ch_tiles=w,
            block_rows=1,
            num_ch_groups=groups,
            cores_per_group=ranges,
            num_cores=groups * ranges,
            core_seq_tiles_max=core_seq_tiles_max,
            split_rows=base.split_rows,
            out_group_tiles=max(
                1,
                min(
                    kw.get("out_group_tiles")
                    or desc_module.default_out_group_tiles(w, kw.get("dest_tiles", 8), core_seq_tiles_max),
                    w,
                    kw.get("dest_tiles", 8),
                ),
            ),
        )

    return select_plan


@pytest.mark.skipif(os.environ.get("GQKV_PERF_R2") != "1", reason="perf harness: set GQKV_PERF_R2=1")
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
        "conv1d-014",
        "conv1d-016",
        "conv1d-020",
        "conv1d-012",
        "conv1d-013",
        "conv1d-018",
        "conv1d-019",
    ],
)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variant(device, monkeypatch, case_id, variant):
    w, limits, defines, check = VARIANTS[variant]
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
    if w is not None and (sum(case["inputs"][1]) // 32) % w:
        pytest.skip(f"block_ch_tiles {w} does not divide the channel tiles")
    if case["inputs"][3] is not None:  # channel-paged history: [1, C/P, 3P] operands (run + check)
        assert check, "channel-paged cells always check correctness"
        with op_module.plan_limits(**limits):
            _run_and_check(device, case)
        print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
        op_module._PROGRAM_CACHE.clear()
        return
    with op_module.plan_limits(**limits):
        outputs, host, device_inputs = _run(device, case)
    print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
    op_module._PROGRAM_CACHE.clear()
    if check:
        _check_case(case, outputs, host, device_inputs)
