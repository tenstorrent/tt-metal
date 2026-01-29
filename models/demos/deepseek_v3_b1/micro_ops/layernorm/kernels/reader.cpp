// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// LayerNorm Reader Kernel (NCRISC/BRISC)
// Purpose: Read row-major data from DRAM into circular buffers
// - Input tensor sticks -> CB_INPUT_RM
// - Gamma tensor sticks -> CB_GAMMA_RM (once)
// - Beta tensor sticks -> CB_BETA_RM (once)
// - Generate scalar constants -> CB_SCALARS

#include <cstdint>

void kernel_main() {
    // Placeholder - implementation in Step 2.1
}
