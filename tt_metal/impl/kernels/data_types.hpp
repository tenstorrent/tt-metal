// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

enum NOC_MODE : uint8_t {
    DEDICATED_NOC_PER_DM = 0,
    ANY_NOC_PER_DM = 1,
};

enum Eth : uint8_t {
    SENDER = 0,
    RECEIVER = 1,
    IDLE = 2,
};

} // namespace tt::tt_metal
