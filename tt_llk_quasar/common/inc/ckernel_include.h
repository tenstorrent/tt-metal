// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file lists the includes that are safe to be included for both firmware and ckernels
//
#include "ckernel_defs.h"
#include "ckernel_gpr_map.h"
#include "ckernel_instr_params.h"

namespace ckernel
{

// Firmware messages to ckernels
enum firmware_msg_e
{
    FLIP_STATE_ID        = 1,
    RUN_INSTRUCTIONS     = 2,
    RESET_DEST_OFFSET_ID = 3
};

} // namespace ckernel
