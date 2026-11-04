# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole SFPU format cache")
@pytest.mark.parametrize("srcb_dtype", [ttnn.uint16, ttnn.bfloat8_b])
@pytest.mark.parametrize("both_operands", [False, True])
def test_sfpu_same_family_public_reconfig(device, srcb_dtype, both_operands):
    source = r"""
#include "api/compute/common.h"
#ifdef TRISC_MATH
#include "llk_math_common_api.h"
#include "ckernel_debug.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif
void kernel_main() {
#ifdef TRISC_MATH
    llk_math_hw_configure<false>(0, 0);
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>();
    sfpi::dst_reg[0].template mode<sfpi::DataLayout::F16b>() = 1.5f;
    if constexpr (get_compile_time_arg_val(0)) {
        llk_math_reconfig_data_format<false, true>(0, 0, 0, 1);
    } else {
        llk_math_reconfig_data_format_srcb<false, true>(0, 1);
    }
    sfpi::vFloat value = sfpi::dst_reg[0];
    sfpi::dst_reg[0].template mode<sfpi::DataLayout::F32>() = value + 1.0f;
    ckernel::tensix_sync();
    auto* out = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(0));
    ckernel::dbg_copy_dest_tile<ckernel::DbgDestTileOp::Read>(DataFormat::Float32, 0, const_cast<uint32_t*>(out));
    out[1024] = (ckernel::cfg_read(ALU_FORMAT_SPEC_REG1_SrcB_ADDR32) & ALU_FORMAT_SPEC_REG1_SrcB_MASK) >> ALU_FORMAT_SPEC_REG1_SrcB_SHAMT;
    out[1025] = unpack_dst_format[0];
    out[1026] = unpack_dst_format[1];
#endif
}
"""
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, (1, 2048), ttnn.ShardOrientation.ROW_MAJOR),
    )
    scratch = ttnn.from_torch(
        torch.zeros(1, 2048, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory,
    )
    args = ttnn.RuntimeArgs()
    args[0][0] = [scratch.buffer_address()]
    cbs = [
        ttnn.CBDescriptor(
            total_size=1088 if dtype == ttnn.bfloat8_b else 2048,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i, data_format=dtype, page_size=1088 if dtype == ttnn.bfloat8_b else 2048
                )
            ],
        )
        for i, dtype in enumerate([ttnn.bfloat16, srcb_dtype])
    ]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                runtime_args=args,
                compile_time_args=[int(both_operands)],
                config=ttnn.ComputeConfigDescriptor(),
            )
        ],
        cbs=cbs,
    )
    ttnn.generic_op([scratch, scratch], program)
    result = ttnn.to_torch(scratch).reshape(-1).tolist()
    assert result[0] == 0x40200000
    assert result[1024] == 5
    assert result[1025] != result[1026]
