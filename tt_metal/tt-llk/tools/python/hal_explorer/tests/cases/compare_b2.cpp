// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hal/cfg/detail/state_bank.h"
using namespace ckernel;
void reference_c(const std::uint32_t v) { cfg_write(184, v); }
