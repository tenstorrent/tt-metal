# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time harness for g6_zeros: the four production cases × `cores_per_bank` × `placement`.

Measurement only; correctness is the acceptance suite's job. "default" is the knob value
`make_plan` picks; the others turn `cores_per_bank` (the grid-sync lamp of the Blocking Model)
for an A/B in the same session. The descriptor is dispatched directly so the harness can also
time dtypes not yet in SUPPORTED (no validate()). Run under
`scripts/run_safe_pytest.sh --profile --no-py-zones`; one ops-CSV row per test, in collection
order (cases outer, cores_per_bank, placement inner).
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


@pytest.mark.parametrize("placement", [None, "row_major", "bank_near"], ids=lambda v: f"pl_{v or 'default'}")
@pytest.mark.parametrize("cores_per_bank", [None, 1, 2, 3, 4, 8, 13], ids=lambda v: f"cpb_{v or 'default'}")
@pytest.mark.parametrize("shape, dtype", CASES)
def test_g6_zeros_device_time(shape, dtype, cores_per_bank, placement, device):
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    plan = pdm.make_plan(
        output.buffer_num_pages(),
        output.buffer_aligned_page_size(),
        device,
        cores_per_bank=cores_per_bank,
        placement=placement,
    )
    ttnn.generic_op([output], pdm.create_program_descriptor(output, plan))
    ttnn.synchronize_device(device)
    ttnn.deallocate(output)
