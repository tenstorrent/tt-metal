# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 perf harness (measurement only — DO NOT DELETE).

Run with:  GQKV_PERF_R3=1 scripts/run_safe_pytest.sh --profile --no-py-zones --run-all \
               "<this file>::test_variant[variant=<v>-case_id=<c>]" ...
(skipped unless GQKV_PERF_R3=1: a measurement harness, not a regression test.)
One GenericOpDeviceOperation row per test, in the order given (device kernel ns:
generated/profiler/reports/<ts>/ops_perf_results_*.csv, or generated/profiler/.logs/cpp_device_perf_report.csv).

A variant is (plan_limits pins incl. the SiLU body, descriptor-constant overrides, kernel defines).
GQKV_ZONES=1 adds the reader / writer DM-timeline zones (R_* / W_*) to any variant; per-core timelines
come from generated/profiler/.logs/profile_log_device.csv. Defines enter the
kernel hash, so define-gated ablations never leave stale binaries in the JIT cache. Ablation variants
produce wrong results by construction and skip the correctness check; every other variant is checked
against the float64 reference.
"""

import importlib
import os

import pytest
import ttnn

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
LIB, QUARTIC = desc_module.SILU_LIBRARY, desc_module.SILU_BF16_QUARTIC
NOC0, NOC1 = ttnn.NOC.NOC_0, ttnn.NOC.NOC_1


def _rep(n):
    return {COMPUTE: [("GQKV_SILU_REPEAT", str(n))]}


# name -> (plan_limits pins, {descriptor module constant: value}, {kernel index: [(define, value)]},
#          check correctness)
VARIANTS = {
    "base": ({}, {}, {}, True),
    "lib": (dict(silu_body=LIB), {}, {}, True),
    "quartic": (dict(silu_body=QUARTIC), {}, {}, True),
    # SFPU cost of one SiLU pass = (T(rep3) - T(rep1)) / 2 (wrong results by construction)
    "lib_rep1": (dict(silu_body=LIB), {}, _rep(1), False),
    "lib_rep2": (dict(silu_body=LIB), {}, _rep(2), False),
    "lib_rep3": (dict(silu_body=LIB), {}, _rep(3), False),
    "q_rep1": (dict(silu_body=QUARTIC), {}, _rep(1), False),
    "q_rep2": (dict(silu_body=QUARTIC), {}, _rep(2), False),
    "q_rep3": (dict(silu_body=QUARTIC), {}, _rep(3), False),
    # the handshake without any SFPU work (wrong results)
    "nosilu": ({}, {}, {COMPUTE: [("GQKV_ABLATE_SILU", "1")]}, False),
    # per-tile SFPU-after-FPU stall off (parked knob)
    "lib_ns": (dict(silu_body=LIB), {"SILU_TILE_STALL": False}, {}, True),
    "q_ns": (dict(silu_body=QUARTIC), {"SILU_TILE_STALL": False}, {}, True),
    # region (b): buffer depths
    "lib_r3": (dict(silu_body=LIB), {"ROWS_DEPTH": 3}, {}, True),
    "lib_o3": (dict(silu_body=LIB), {"OUT_DEPTH": 3}, {}, True),
    "lib_r3o3": (dict(silu_body=LIB), {"ROWS_DEPTH": 3, "OUT_DEPTH": 3}, {}, True),
    "q_r3o3": (dict(silu_body=QUARTIC), {"ROWS_DEPTH": 3, "OUT_DEPTH": 3}, {}, True),
    # split reader ratio / NoC assignment / bank stagger with each body
    "lib_s0": (dict(silu_body=LIB, split_rows=0), {}, {}, True),
    "lib_s8": (dict(silu_body=LIB, split_rows=8), {}, {}, True),
    "q_s0": (dict(silu_body=QUARTIC, split_rows=0), {}, {}, True),
    "q_s8": (dict(silu_body=QUARTIC, split_rows=8), {}, {}, True),
    "q_s16": (dict(silu_body=QUARTIC, split_rows=16), {}, {}, True),
    "q_swap": (dict(silu_body=QUARTIC), {"READER_NOC": NOC1, "WRITER_NOC": NOC0}, {}, True),
    "q_stag": (dict(silu_body=QUARTIC), {"STAGGER_ROW_WALK": True}, {}, True),
}


@pytest.mark.skipif(os.environ.get("GQKV_PERF_R3") != "1", reason="perf harness: set GQKV_PERF_R3=1")
@pytest.mark.parametrize(
    "case_id",
    [
        "conv1d-004",
        "conv1d-020",
        "conv1d-001",
        "conv1d-002",
        "conv1d-003",
        "conv1d-005",
        "conv1d-006",
        "conv1d-010",
        "conv1d-011",
        "conv1d-012",
        "conv1d-013",
        "conv1d-018",
        "conv1d-019",
        "conv1d-014",
        "conv1d-016",
    ],
)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_variant(device, monkeypatch, case_id, variant):
    limits, constants, defines, check = VARIANTS[variant]
    if os.environ.get("GQKV_ZONES") == "1":  # DM timeline zones (reader + writer), any variant
        defines = dict(defines)
        for index in (READER, WRITER):
            defines[index] = list(defines.get(index, [])) + [("GQKV_ZONES", "1")]
    op_module._PROGRAM_CACHE.clear()
    for name, value in constants.items():
        monkeypatch.setattr(desc_module, name, value)
        if hasattr(op_module, name):
            monkeypatch.setattr(op_module, name, value)
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
