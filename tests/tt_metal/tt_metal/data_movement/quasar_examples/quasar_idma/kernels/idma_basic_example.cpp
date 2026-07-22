// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Basic IDMA copy: transfers num_elements * elem_size bytes from src to dst
// in a single IDMA transaction using direct NOC address registers (no addrgen).
//
// Sequence:
//   1. Configure complex command buffer 0 for an iDMA copy
//   2. Program the source, destination, and length registers
//   3. Issue one transaction
//   4. Wait for the iDMA acknowledgement

#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include <cstdint>

using IdmaCommandBuffer = overlay::ComplexCommandBuffer0;

constexpr std::uint32_t num_elements = 16;
constexpr std::uint32_t elem_size = 8;
constexpr std::uint32_t total_bytes = num_elements * elem_size;  // 128

constexpr std::uint32_t request_vc = 1;
constexpr std::uint32_t response_vc = 12;

void kernel_main() {
    constexpr std::uint32_t src_addr = get_arg(args::src_addr);
    constexpr std::uint32_t dst_addr = get_arg(args::dst_addr);

    IdmaCommandBuffer::reset();

    TT_ROCC_CMD_BUF_MISC_reg_u misc{};
    misc.val = TT_ROCC_CMD_BUF_MISC_REG_DEFAULT;
    misc.f.write_trans = 1;
    misc.f.idma_en = 1;
    misc.f.wrapping_en = 0;

    IdmaCommandBuffer::write_register<TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_MISC_REG_OFFSET>(misc.val);
    IdmaCommandBuffer::write_register<TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET>(request_vc);
    IdmaCommandBuffer::write_register<TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET>(response_vc);
    IdmaCommandBuffer::write_register<TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_SRC_ADDR_REG_OFFSET>(src_addr);
    IdmaCommandBuffer::write_register<TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_DEST_ADDR_REG_OFFSET>(dst_addr);
    IdmaCommandBuffer::write_register<TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_LEN_BYTES_REG_OFFSET>(total_bytes);

    IdmaCommandBuffer::issue();

    while (IdmaCommandBuffer::idma_acks_pending() != 0) {
    }

    DEVICE_PRINT("IDMA basic done: {} elements, total_bytes: {}\n", num_elements, total_bytes);
}
