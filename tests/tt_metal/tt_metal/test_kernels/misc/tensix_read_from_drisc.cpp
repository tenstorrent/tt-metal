// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tensix Kernel. Used in two test modes:
// - Default: Reads from DRISC L1 into Tensix L1
// - with MODE_TENSIX_STREAM_REG_TO_DRISC defined: Performs a write to a stream reg on DRISC and then reads it back to
// Tensix L1

#include "api/dataflow/dataflow_api.h"
#include "noc_nonblocking_api.h"
#include "experimental/noc.h"
#include "experimental/endpoints.h"
#include "experimental/core_local_mem.h"

void kernel_main() {
    constexpr uint32_t tensix_dst_addr = get_compile_time_arg_val(0);
#ifdef MODE_TENSIX_STREAM_REG_TO_DRISC
    constexpr uint32_t stream_id = get_compile_time_arg_val(1);
    constexpr uint32_t drisc_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t drisc_noc_y = get_compile_time_arg_val(3);
    constexpr uint32_t stream_reg = get_compile_time_arg_val(4);
    constexpr uint32_t value_to_write = get_compile_time_arg_val(5);
#else
    constexpr uint32_t drisc_l1_src_addr_low = get_compile_time_arg_val(1);
    constexpr uint32_t drisc_l1_src_addr_high = get_compile_time_arg_val(2);
    constexpr uint32_t drisc_noc_x = get_compile_time_arg_val(3);
    constexpr uint32_t drisc_noc_y = get_compile_time_arg_val(4);
#endif

#ifdef MODE_TENSIX_STREAM_REG_TO_DRISC
    // Stream register round trip test
    // Write to the DRISC stream register from Tensix inline reg
    experimental::UnicastEndpoint src;
    experimental::Noc noc;
    uint32_t reg_addr = STREAM_REG_ADDR(stream_id, stream_reg);
    noc.inline_dw_write<experimental::Noc::TxnIdMode::DISABLED, InlineWriteDst::REG>(
        src, value_to_write, {.noc_x = drisc_noc_x, .noc_y = drisc_noc_y, .addr = reg_addr});
    noc.async_write_barrier();

    // Read back from the DRISC stream register into Tensix L1.
    experimental::CoreLocalMem<uint32_t> dst(tensix_dst_addr);
    noc.async_read(src, dst, sizeof(uint32_t), {.noc_x = drisc_noc_x, .noc_y = drisc_noc_y, .addr = reg_addr}, {});
    noc.async_read_barrier();
#else
    // In NOC2AXI mode, DRISC L1 is accessed via DRAM_L1_NOC_OFFSET (bit 37),
    // making the address 64-bit. The standard noc_async_read / get_noc_addr APIs
    // truncate addr to 32 bits and mask NOC_TARG_ADDR_MID, dropping bit 37.
    // The _with_state 5-arg overload takes coordinates and address separately,
    // writing TARG_ADDR_MID unmasked so bit 37 is preserved
    uint64_t drisc_l1_src_addr = (static_cast<uint64_t>(drisc_l1_src_addr_high) << 32) | drisc_l1_src_addr_low;
    uint32_t drisc_src_coord = NOC_XY_COORD(drisc_noc_x, drisc_noc_y);
    noc_read_init_state<BRISC_RD_CMD_BUF>(NOC_INDEX);
    noc_read_with_state<DM_DEDICATED_NOC, BRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
        NOC_INDEX, drisc_src_coord, drisc_l1_src_addr, tensix_dst_addr, sizeof(uint32_t));
    noc_async_read_barrier();
#endif
}
