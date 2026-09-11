# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compile production SFPU callers independently of their multi-device pipelines."""

from pathlib import Path

import pytest

import ttnn
from models.common.utility_functions import is_blackhole


ROOT = Path(__file__).resolve().parents[5]
CALLERS = [
    "models/demos/wormhole/bge_m3/tt/custom_ops/encoder_sdpa/kernels/compute_common.hpp",
    "ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa/device/kernels/compute/compute_common.hpp",
    "ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp",
    "ttnn/cpp/ttnn/operations/experimental/ccl/moe_gpt/device/kernels/swiglu_sfpu.h",
]


@pytest.mark.skipif(not is_blackhole(), reason="Compile shared callers against the Blackhole SFPU implementation")
@pytest.mark.parametrize("caller", CALLERS, ids=["bge_sdpa", "quasar_sdpa_copy", "quasar_decode_copy", "moe_swiglu"])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_sfpu_caller_compilation(device, caller, fp32_dest_acc_en):
    if caller.endswith("swiglu_sfpu.h"):
        thread_guard = "defined(TRISC_MATH) || defined(TRISC_PACK)"
        calls = """
            ckernel::sfpu::swiglu_init();
            ckernel::llk_math_eltwise_binary_sfpu_swiglu<DST_ACCUM_MODE>(0, 1, 2);
        """
    else:
        thread_guard = "defined(TRISC_MATH)"
        calls = """
            recip_tile_first_column(0);
            exp_tile_first_column<true, 0x3f80>(0);
            exp_tile_first_column<false, 0x3f80>(0);
            fused_max_sub_exp_add_tile(0, 0x3f80);
            softplus_tile_first_column(0, 0x3f800000, 0x3f800000, 0x41a00000);
            sigmoid_sub(0, 1, 2, 1);
            logsigmoid_sub(0, 1, 2, 1);
        """
    source = f"""
        #include "api/compute/compute_kernel_api.h"
        #include "{ROOT / caller}"
        #if {thread_guard}
        void instantiate_callers() {{ {calls} }}
        #endif
        // Only compile the caller expressions; the no-op kernel needs no tile data.
        void kernel_main() {{}}
    """
    core = ttnn.CoreCoord(0, 0)
    kernel = ttnn.KernelDescriptor(
        kernel_source=source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        defines=[("EXP_APPROX_MODE", "0")],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en),
    )
    tensors = [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        for _ in range(2)
    ]
    ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[]))
    ttnn.synchronize_device(device)
