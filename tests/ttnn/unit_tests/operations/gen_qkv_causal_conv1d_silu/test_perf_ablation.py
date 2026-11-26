# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Ablation profiling for gen_qkv_causal_conv1d_silu (perf measurement only — DO NOT DELETE).

Run with:  scripts/run_safe_pytest.sh --profile --no-py-zones --run-all <this file>
Each variant is one GenericOpDeviceOperation row in the profiler CSV, in parametrize order.
Ablated variants stub a stage's payload but keep its CB handshakes (kernels' ABLATE_* hooks);
their outputs are wrong by construction and are not checked. The baseline variant is checked.
Cell: conv1d-004 (T 4096, C 2560, ROW_MAJOR) — the DRAM-bound PERF FOCUS target — and 003.
"""

import importlib

import pytest

# The package re-exports the entry function under the submodule's name; fetch the module itself.
op_module = importlib.import_module("ttnn.operations.gen_qkv_causal_conv1d_silu.gen_qkv_causal_conv1d_silu")
from tests.ttnn.unit_tests.operations.gen_qkv_causal_conv1d_silu.test_gen_qkv_causal_conv1d_silu import (
    PHASE0_CASES,
    _check_case,
    _run,
)

_READER, _WRITER, _COMPUTE = 0, 1, 2

VARIANTS = {
    "baseline": {},
    "no_silu": {_COMPUTE: ["ABLATE_SILU"]},
    "no_mac": {_COMPUTE: ["ABLATE_MAC"]},
    "no_tilize": {_COMPUTE: ["ABLATE_TILIZE"]},
    "no_compute": {_COMPUTE: ["ABLATE_SILU", "ABLATE_MAC", "ABLATE_TILIZE"]},
    "no_dram": {_READER: ["ABLATE_READS"], _WRITER: ["ABLATE_WRITES"]},
    "no_reads": {_READER: ["ABLATE_READS"]},
    "no_writes": {_WRITER: ["ABLATE_WRITES"]},
    "read_nosync": {_READER: ["ABLATE_READ_SYNC"]},
    "read_nosync_no_compute": {_READER: ["ABLATE_READ_SYNC"], _COMPUTE: ["ABLATE_SILU", "ABLATE_MAC", "ABLATE_TILIZE"]},
    "read_one_dst": {_READER: ["ABLATE_READ_DST"]},
    "read_one_dst_no_silu": {_READER: ["ABLATE_READ_DST"], _COMPUTE: ["ABLATE_SILU"]},
    "tap_stagger": {_READER: ["TAP_STAGGER"]},
    "no_taps": {_READER: ["ABLATE_TAPS"]},
    "silu_on_math": {_COMPUTE: ["SILU_ON_MATH"]},
    "skeleton": {
        _COMPUTE: ["ABLATE_SILU", "ABLATE_MAC", "ABLATE_TILIZE"],
        _READER: ["ABLATE_READS"],
        _WRITER: ["ABLATE_WRITES"],
    },
}


@pytest.mark.parametrize("case_index", [3, 2], ids=["conv1d-004", "conv1d-003"])
@pytest.mark.parametrize(
    "variant",
    [v for v in VARIANTS if v in __import__("os").environ.get("ABLATION_VARIANTS", ",".join(VARIANTS)).split(",")],
)
def test_ablation(device, monkeypatch, variant, case_index):
    original = op_module.create_program_descriptor

    def with_defines(*args, **kwargs):
        descriptor, work = original(*args, **kwargs)
        for kernel_index, names in VARIANTS[variant].items():
            descriptor.kernels[kernel_index].defines = [(n, "1") for n in names]
        return descriptor, work

    monkeypatch.setattr(op_module, "create_program_descriptor", with_defines)
    op_module._PROGRAM_CACHE.clear()
    case = PHASE0_CASES[case_index].values[0]
    max_block_rows = __import__("os").environ.get("ABLATION_MAX_BLOCK_ROWS")
    with op_module.plan_limits(max_block_rows=int(max_block_rows) if max_block_rows else None):
        outputs, host, device_inputs = _run(device, case)
    if variant == "baseline":
        _check_case(case, outputs, host, device_inputs)
    op_module._PROGRAM_CACHE.clear()
