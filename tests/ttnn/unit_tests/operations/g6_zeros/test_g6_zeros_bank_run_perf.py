# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-time harness for bank_run_write knobs: one program per test, nothing else dispatched.

Run under `scripts/run_safe_pytest.sh --profile --no-py-zones`; the ops CSV holds one row per test,
in collection order. Correctness is the acceptance suite's job; this file only measures.
"""

import pytest
import ttnn

from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

CASES = [
    pytest.param((32831, 1, 32, 256), ttnn.bfloat8_b, id="p1"),
    pytest.param((16, 12, 128, 128), ttnn.float32, id="p2"),
    pytest.param((8, 12, 128, 128), ttnn.float32, id="p3"),
    pytest.param((1, 4, 256, 128), ttnn.bfloat16, id="p4"),
]

# Module knobs per variant; anything not named keeps its default. The defaults are the op's own plan.
KNOBS = [
    {},
    {"CORE_ORDER": "row_major", "WRITER_CORES_PER_BANK": None},
    {"CORE_ORDER": "bank_affine", "WRITER_CORES_PER_BANK": None},
    {"CORE_ORDER": "bank_affine", "WRITER_CORES_PER_BANK": 1, "ZERO_SRC_BYTES_MAX": 4096},
    {"CORE_ORDER": "bank_affine", "WRITER_CORES_PER_BANK": 1, "ZERO_SRC_BYTES_MAX": 8192},
    {"CORE_ORDER": "bank_affine", "WRITER_CORES_PER_BANK": 1, "ZERO_SRC_BYTES_MAX": 16384},
    {"CORE_ORDER": "bank_affine", "WRITER_CORES_PER_BANK": None, "ZERO_SRC_BYTES_MAX": 4096},
]


def _knob_id(knobs):
    return ",".join(f"{k}={v}" for k, v in knobs.items()) or "default"


@pytest.mark.parametrize("knobs", KNOBS, ids=_knob_id)
@pytest.mark.parametrize("shape, dtype", CASES)
def test_bank_run_device_time(shape, dtype, knobs, device, monkeypatch):
    for name, value in knobs.items():
        monkeypatch.setattr(pdm, name, value)
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = pdm.make_plan(shape, dtype, pdm.device_geometry(device))
    ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
    ttnn.synchronize_device(device)
