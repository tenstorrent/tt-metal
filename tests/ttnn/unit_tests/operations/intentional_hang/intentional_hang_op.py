# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
intentional_hang — a test-only op whose writer blocks forever waiting
for a tile that is never produced. Used to validate that the eval
hang_plugin detects real device hangs and skips subsequent
parametrizations of the same test function.

Not exported through ttnn.operations. Import directly:
    from tests.ttnn.unit_tests.operations.intentional_hang.intentional_hang_op import intentional_hang
"""

import ttnn

from .program_descriptor import create_program_descriptor


def intentional_hang(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    device = input_tensor.device()
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)),
        input_tensor.dtype,
        input_tensor.layout,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program_descriptor = create_program_descriptor(input_tensor, output_tensor)
    return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
