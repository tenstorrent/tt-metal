// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Hardware choices a kernel is configured with: its data movement processor and NOC, and its build optimization
// level. Shared by the Metal 2.0 kernel specs and the legacy kernel configs (workload/legacy/kernel_types.hpp).

namespace tt::tt_metal {

enum class DataMovementProcessor {
    RISCV_0 = 0,  // BRISC; Core DM0 on Quasar
    RISCV_1 = 1,  // NCRISC; Core DM1 on Quasar
    RISCV_2 = 2,  // Core DM2 on Quasar
    RISCV_3 = 3,  // Core DM3 on Quasar
    RISCV_4 = 4,  // Core DM4 on Quasar
    RISCV_5 = 5,  // Core DM5 on Quasar
    RISCV_6 = 6,  // Core DM6 on Quasar
    RISCV_7 = 7,  // Core DM7 on Quasar
};

enum NOC : uint8_t {
    RISCV_0_default = 0,
    RISCV_1_default = 1,
    NOC_0 = 0,
    NOC_1 = 1,
};

enum NOC_MODE : uint8_t {
    DM_DEDICATED_NOC = 0,
    DM_DYNAMIC_NOC = 1,
};

// Option that controls build optimization level
enum class KernelBuildOptLevel : uint8_t {
    O1,     // Turns on level 1 optimization. Same as O.
    O2,     // Turns on level 2 optimization and also all flags specified by O1.
    O3,     // Turns on level 3 optimization and also all flags specified by O2.
    O0,     // Reduce compilation time and make debugging produce the expected results.
    Os,     // Optimize for size and also O2 optimizations except for those that increase binary size.
    Ofast,  // Turns on level O3 and also non standard optimizations.
    Oz,     // Aggressively optimize for size rather than speed.
};

}  // namespace tt::tt_metal
