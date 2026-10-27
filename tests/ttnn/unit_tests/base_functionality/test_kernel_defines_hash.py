# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

KERNEL_SOURCE = """
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
#if defined(PREFIX_APREFIX_B)
    out[0] = 2;
#elif defined(PREFIX_A)
    out[0] = 1;
#else
    out[0] = 3;
#endif
}
"""


def run_with_defines(device, prefix, defines):
    core = ttnn.CoreCoord(0, 0)
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    sharded_l1 = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(core_ranges, (1, 8), ttnn.ShardOrientation.ROW_MAJOR),
    )
    inp, out = [
        ttnn.from_torch(
            torch.zeros(1, 8, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=sharded_l1,
        )
        for _ in range(2)
    ]
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[0][0] = [out.buffer_address()]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL_SOURCE.replace("PREFIX", prefix),
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=core_ranges,
                compile_time_args=[],
                defines=[(key.replace("PREFIX", prefix), value.replace("PREFIX", prefix)) for key, value in defines],
                runtime_args=runtime_args,
                config=ttnn.WriterConfigDescriptor(),
            )
        ]
    )
    ttnn.generic_op([inp, out], program)
    return ttnn.to_torch(out)[0, 0].item()


@pytest.mark.parametrize(
    "prefix, first, first_value, second, second_value",
    [
        ("KHS", [("PREFIX_A", "PREFIX_B")], 1, [("PREFIX_APREFIX_B", "")], 2),
        ("KHC", [("PREFIX_APREFIX_B", "")], 2, [("PREFIX_A", ""), ("PREFIX_B", "")], 1),
    ],
    ids=["key_value_split", "define_count"],
)
def test_kernels_differing_only_in_define_split_compile_separately(
    device, prefix, first, first_value, second, second_value
):
    assert run_with_defines(device, prefix, first) == first_value
    assert run_with_defines(device, prefix, second) == second_value
