// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// One core: the device position P (uint32 ROW_MAJOR [1,1,1,1], one 4-byte page) += COUNT, in place.  The chain's
// Qwen38TTNNDevicePosition.advance is ttnn.add(scalar, count) into a fresh tensor and ttnn.copy back into the resident
// scalar (two programs); the uint32 add is the same integer add.  The read takes the page's 64-byte DRAM grain
// (derive.cpp's read of the same scalar); the write puts the 4 bytes back.
// Named compile-time args: cb_stage, count.  Compile-time args: TensorAccessorArgs(position).  Runtime args: 0 the
// position buffer address.
#include <cstdint>
#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t CB_STAGE = get_named_compile_time_arg_val("cb_stage");
constexpr uint32_t ADVANCE_COUNT = get_named_compile_time_arg_val("count");  // COUNT is core_config.h's enumerator
constexpr uint32_t DRAM_READ_GRAIN = 64;
constexpr uint32_t SCALAR_BYTES = 4;

void kernel_main() {
    FUSED_ZONE("fz_pd_adv_main");
    constexpr auto a_p = TensorAccessorArgs<0>();
    const auto position = TensorAccessor(a_p, get_arg_val<uint32_t>(0));
    Noc noc;
    DataflowBuffer stage(CB_STAGE);
    stage.reserve_back(1);
    noc.async_read(position, stage, DRAM_READ_GRAIN, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage.get_write_ptr());
    words[0] = words[0] + ADVANCE_COUNT;
    noc.async_write(stage, position, SCALAR_BYTES, {.offset_bytes = 0}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
    stage.push_back(1);
}
