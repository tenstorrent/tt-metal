// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal {

enum class DataMovementProcessor {
    RISCV_0 = 0,  // BRISC
    RISCV_1 = 1,  // NCRISC
};

enum NOC : uint8_t {
    RISCV_0_default = 0,
    RISCV_1_default = 1,
    NOC_0 = 0,
    NOC_1 = 1,
};

namespace detail {

inline NOC preferred_noc_for_dram_read(ARCH arch) {
    switch (arch) {
        case ARCH::WORMHOLE_B0:
        default: return NOC::NOC_0;
    }
}

inline NOC preferred_noc_for_dram_write(ARCH arch) {
    switch (arch) {
        case ARCH::WORMHOLE_B0:
        default: return NOC::NOC_1;
    }
}

}  // namespace detail

enum NOC_MODE : uint8_t {
    DM_DEDICATED_NOC = 0,
    DM_DYNAMIC_NOC = 1,
};

enum Eth : uint8_t {
    SENDER = 0,
    RECEIVER = 1,
    IDLE = 2,
};

}  // namespace tt::tt_metal
