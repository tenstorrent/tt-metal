// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "noc_nonblocking_api.h"
#include "hw/inc/ethernet/tunneling.h"
#include "risc_common.h"
#include "lf_dev_mem_map.hpp"

namespace lite_fabric {

// Interface to the connected RISC processor via ethernet
struct ConnectedRiscInterface {
    static constexpr uint32_t k_Txq = 0;
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;

    // Put the connected RISC into reset
    inline static void assert_connected_dm1_reset() {
        constexpr uint32_t k_ResetValue = 0x47000;
        internal_::eth_write_remote_reg(k_Txq, k_SoftResetAddr, k_ResetValue);
        while (internal_::eth_txq_is_busy(k_Txq)) {
        }
    }

    // Take the connected RISC out of reset
    inline static void deassert_connected_dm1_reset() {
        constexpr uint32_t k_ResetValue = k_Txq;
        internal_::eth_write_remote_reg(0, k_SoftResetAddr, k_ResetValue);
        while (internal_::eth_txq_is_busy(k_Txq)) {
        }
    }

    inline static void set_pc(uint32_t pc) {
        constexpr uint32_t k_ResetPcAddr = LITE_FABRIC_RESET_PC;
        internal_::eth_write_remote_reg(k_Txq, k_ResetPcAddr, pc);
        while (internal_::eth_txq_is_busy(k_Txq)) {
        }
    }
};

}  // namespace lite_fabric
