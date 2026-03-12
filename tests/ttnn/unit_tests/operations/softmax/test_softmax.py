# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.unit_tests.operations.softmax.utils import DeviceGetter


def test_softmax():
    device = DeviceGetter.get_device((1, 1))

    tt_input = ttnn.ones(
        shape=ttnn.Shape([1, 100, 6800]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_output = ttnn.softmax(
        tt_input,
        dim=2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        numeric_stable=True,
    )
