// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hal/cfg/detail/state_bank.h"
using namespace ckernel;
// the same bank-relative write built two ways: cfg_addr() select vs get_cfg_pointer() branch vs hal state_cfg_bank()
void reference_a(const std::uint32_t v) { cfg_write(184, v); }
void a(const std::uint32_t v) { get_cfg_pointer()[184] = v; }
void reference_b(const std::uint32_t v) { cfg_write(184, v); }
void b(const std::uint32_t v) { hal::cfg::detail::state_cfg_bank()[184] = v; }
// bank write on one side only
void reference_c(const std::uint32_t v) { TTI_NOP; }
void c(const std::uint32_t v) { TTI_NOP; cfg_write(184, v); }
