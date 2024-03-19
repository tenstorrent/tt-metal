// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tt_metal/hostdevcommon/common_runtime_address_map.h"

namespace eth_l1_mem {


struct address_map {

  static constexpr std::int32_t MAX_SIZE = 256 * 1024;
  static constexpr std::int32_t MAX_L1_LOADING_SIZE = 1 * 256 * 1024;

  // Sizes
  static constexpr std::int32_t FIRMWARE_SIZE = 32 * 1024;
  static constexpr std::int32_t COMMAND_Q_SIZE = 4 * 1024;
  static constexpr std::int32_t DATA_BUFFER_SIZE_HOST = 4 * 1024;
  static constexpr std::int32_t DATA_BUFFER_SIZE_ETH = 4 * 1024;
  static constexpr std::int32_t DATA_BUFFER_SIZE_NOC = 16 * 1024;
  static constexpr std::int32_t DATA_BUFFER_SIZE = 24 * 1024;
  // Base addresses

  static constexpr std::int32_t FIRMWARE_BASE = 0x9040;
  static constexpr std::int32_t L1_EPOCH_Q_BASE = 0x9000;  // Epoch Q start in L1.
  static constexpr std::int32_t COMMAND_Q_BASE = L1_EPOCH_Q_BASE + FIRMWARE_SIZE;
  static constexpr std::int32_t DATA_BUFFER_BASE = COMMAND_Q_BASE + COMMAND_Q_SIZE;
  static constexpr std::int32_t TILE_HEADER_BUFFER_BASE = DATA_BUFFER_BASE + DATA_BUFFER_SIZE;
  static constexpr std::int32_t ERISC_MEM_MAILBOX_BASE = 0x10e94;  // See dev_msgs.h and dev_mem_map.h for restrictions
  static constexpr std::int32_t PRINT_BUFFER_ER = ERISC_MEM_MAILBOX_BASE + 128 + 12;
  // erisc early exit functionality re-uses mailboxes_t::ncrisc_halt_msg_t::stack_save memory
  static constexpr std::int32_t ERISC_MEM_MAILBOX_STACK_SAVE = ERISC_MEM_MAILBOX_BASE + 4;

  static_assert((PRINT_BUFFER_ER + PRINT_BUFFER_SIZE) < COMMAND_Q_BASE);
  static_assert((PRINT_BUFFER_ER % 32) == 0);

  // TT Metal Specific
  static constexpr std::int32_t ERISC_FIRMWARE_SIZE = 2 * 1024;
  // Total 160 * 1024 L1 starting from TILE_HEADER_BUFFER_BASE
  //    -   1 * 1024 misc args
  //    -  53 * 1024 eth app reserved buffer space
  //    - 106 * 1024 L1 unreserved buffer space
  static constexpr std::int32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 8;
  static constexpr std::int32_t ERISC_BARRIER_SIZE = 32;
  static constexpr std::int32_t ERISC_APP_ROUTING_INFO_SIZE = 48;
  static constexpr std::int32_t ERISC_APP_SYNC_INFO_SIZE = 160 + 16 * MAX_NUM_CONCURRENT_TRANSACTIONS;

  static constexpr std::int32_t ERISC_BARRIER_BASE = TILE_HEADER_BUFFER_BASE;
  static constexpr std::int32_t ERISC_APP_ROUTING_INFO_BASE = ERISC_BARRIER_BASE + ERISC_BARRIER_SIZE;
  static constexpr std::int32_t ERISC_APP_SYNC_INFO_BASE = ERISC_APP_ROUTING_INFO_BASE + ERISC_APP_ROUTING_INFO_SIZE;
  static constexpr std::uint32_t SEMAPHORE_BASE = ERISC_APP_SYNC_INFO_BASE + ERISC_APP_SYNC_INFO_SIZE;

  static constexpr uint32_t ISSUE_CQ_CB_BASE = SEMAPHORE_BASE + SEMAPHORE_SIZE;  // SIZE from shared common addr
  static constexpr uint32_t COMPLETION_CQ_CB_BASE = ISSUE_CQ_CB_BASE + 7 * L1_ALIGNMENT;

  static constexpr std::int32_t ERISC_L1_ARG_BASE = COMPLETION_CQ_CB_BASE + 7 * L1_ALIGNMENT;

  static constexpr std::int32_t ERISC_L1_UNRESERVED_BASE = TILE_HEADER_BUFFER_BASE + 1024;
  static constexpr std::int32_t ERISC_L1_UNRESERVED_SIZE = MAX_L1_LOADING_SIZE - ERISC_L1_UNRESERVED_BASE;

  static constexpr std::int32_t LAUNCH_ERISC_APP_FLAG = L1_EPOCH_Q_BASE + 4;

  // BIDIR Tunneling Kernel Space
  static constexpr std::int32_t ERISC_L1_TUNNEL_BUFFER_SIZE = ERISC_L1_UNRESERVED_SIZE / 2;

  // TODO: risky, is there a check for FW size we can add?
  template <std::size_t A, std::size_t B>
  struct TAssertEquality {
      static_assert(A == B, "Not equal");
      static constexpr bool _cResult = (A == B);
  };

  static constexpr std::int32_t RISC_LOCAL_MEM_BASE = 0xffb00000; // Actaul local memory address as seen from risc firmware
                                                                   // As part of the init risc firmware will copy local memory data from
                                                                   // l1 locations listed above into internal local memory that starts
                                                                   // at RISC_LOCAL_MEM_BASE address

  static constexpr std::uint32_t FW_VERSION_ADDR = 0x210;
};
}  // namespace eth_l1_mem
