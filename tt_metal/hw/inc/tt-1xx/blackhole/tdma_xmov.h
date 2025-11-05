// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// TDMA XMOV functions for the firmware
//

#pragma once

#include "tensix_types.h"

// Write TDMA registers and initiate a move
void tdma_xmov(uint mover_number, uint source_addr, uint dest_addr, uint size, xmov_direction_t direction);
void wait_tdma_movers_done(uint mover_busy_mask);
