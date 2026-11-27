# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf tournament round 1 harness: measured breakdown by cumulative ablation (measurement only — DO NOT DELETE).

Run with:  GQKV_PERF_T1=1 scripts/run_safe_pytest.sh --profile --no-py-zones --run-all \
               "<this file>::test_ablation[variant=<v>-case_id=<c>]" ...
One GenericOpDeviceOperation row per test, in the order given. The permanent stage zones
(MaybeDeviceZoneScope) land in generated/profiler/.logs/profile_log_device.csv; summarize them with
zone_stages.py in this directory.

Ablation defines (compiled out by default; wrong results by construction, so no correctness check):
  GQKV_ABLATE_SILU    compute: the MATH_PACK handshake without any SFPU work
  GQKV_ABLATE_MAC     compute: no tap products (DEST stays zero)
  GQKV_ABLATE_TILIZE  compute: windowed tilize reduced to its CB sequence
  GQKV_ABLATE_READS   reader + writer: no input / history / halo DRAM reads (taps, CB sync, barriers stay)
  GQKV_ABLATE_WRITES  writer: no q / k / v writes (flush + CB sync stay)
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

SILU, MAC, TILIZE = ("GQKV_ABLATE_SILU", "1"), ("GQKV_ABLATE_MAC", "1"), ("GQKV_ABLATE_TILIZE", "1")
READS, WRITES = ("GQKV_ABLATE_READS", "1"), ("GQKV_ABLATE_WRITES", "1")


def _defs(compute=(), reads=False, writes=False):
    d = {}
    if compute:
        d[COMPUTE] = list(compute)
    if reads:
        d[READER] = [READS]
        d[WRITER] = [READS]
    if writes:
        d[WRITER] = d.get(WRITER, []) + [WRITES]
    return d


# name -> (plan_limits pins, {kernel index: [(define, value)]}, check correctness[, {descriptor constant: value}])
VARIANTS = {
    "full": ({}, {}, True),
    # compute peel (cumulative)
    "c_nosilu": ({}, _defs([SILU]), False),
    "c_nosilu_nomac": ({}, _defs([SILU, MAC]), False),
    "c_none": ({}, _defs([SILU, MAC, TILIZE]), False),
    # then data movement on top of no compute
    "c_none_noreads": ({}, _defs([SILU, MAC, TILIZE], reads=True), False),
    "sync_only": ({}, _defs([SILU, MAC, TILIZE], reads=True, writes=True), False),
    # the other order: data movement first
    "noreads": ({}, _defs(reads=True), False),
    "nowrites": ({}, _defs(writes=True), False),
    "nodm": ({}, _defs(reads=True, writes=True), False),
    "nodm_nosilu": ({}, _defs([SILU], reads=True, writes=True), False),
    "nodm_nosilu_nomac": ({}, _defs([SILU, MAC], reads=True, writes=True), False),
    # compute-side singles
    "notilize": ({}, _defs([TILIZE]), False),
    "nomac": ({}, _defs([MAC]), False),
    # body pins (for the multi-row plans, whose default is the library body)
    "quartic": (dict(silu_body=desc_module.SILU_BF16_QUARTIC), {}, True),
    "quartic_nodm": (dict(silu_body=desc_module.SILU_BF16_QUARTIC), _defs(reads=True, writes=True), False),
    # the same kernels with the permanent stage zones compiled out (what the zones cost under the profiler)
    "nozones": ({}, {k: [("MAYBE_DEVICE_ZONE_SCOPE_OFF", "1")] for k in (READER, WRITER, COMPUTE)}, True),
    "nozones_compute": ({}, {COMPUTE: [("MAYBE_DEVICE_ZONE_SCOPE_OFF", "1")]}, True),
    "nozones_reader": ({}, {READER: [("MAYBE_DEVICE_ZONE_SCOPE_OFF", "1")]}, True),
    "nozones_writer": ({}, {WRITER: [("MAYBE_DEVICE_ZONE_SCOPE_OFF", "1")]}, True),
    # (a 4th element {descriptor constant: value} pins a program-descriptor module constant, e.g.
    #  "lib_s8": ({}, {}, True, {"SPLIT_READER_ROWS": 8}); the split_writes A/B of Perf 1 used it)
}


@pytest.mark.skipif(os.environ.get("GQKV_PERF_T1") != "1", reason="perf harness: set GQKV_PERF_T1=1")
@pytest.mark.parametrize(
    "case_id",
    [
        "conv1d-004",
        "conv1d-001",
        "conv1d-002",
        "conv1d-003",
        "conv1d-020",
        "conv1d-005",
        "conv1d-006",
        "conv1d-010",
        "conv1d-011",
        "conv1d-012",
        "conv1d-013",
        "conv1d-014",
        "conv1d-016",
        "conv1d-019",
    ],
)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_ablation(device, monkeypatch, case_id, variant):
    limits, defines, check, *rest = VARIANTS[variant]
    constants = rest[0] if rest else {}
    op_module._PROGRAM_CACHE.clear()
    for name, value in constants.items():
        monkeypatch.setattr(desc_module, name, value)
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
    if case["inputs"][3] is not None:
        assert check, "channel-paged cells always check correctness"
        with op_module.plan_limits(**limits):
            _run_and_check(device, case)
        op_module._PROGRAM_CACHE.clear()
        return
    with op_module.plan_limits(**limits):
        outputs, host, device_inputs = _run(device, case)
    print(f"PLAN {case_id} {variant}: {op_module.get_last_plan()}")
    op_module._PROGRAM_CACHE.clear()
    if check:
        _check_case(case, outputs, host, device_inputs)
