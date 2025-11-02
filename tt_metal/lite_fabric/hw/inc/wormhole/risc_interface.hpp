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
// Everything is no op for wormhole as there is only 1 ethernet core and that is running base firmware
struct ConnectedRiscInterface {
    static constexpr uint32_t k_Txq = 0;
    static constexpr uint32_t k_SoftResetAddr = 0xFFB121B0;

    // Put the connected RISC into reset
    inline static void assert_connected_dm1_reset() {}

    // Take the connected RISC out of reset
    inline static void deassert_connected_dm1_reset() {}

    inline static void set_pc(uint32_t pc) {}
};

}  // namespace lite_fabric
