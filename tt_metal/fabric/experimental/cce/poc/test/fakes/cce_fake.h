// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-side model of one CCE for the fakes: SRAM sits at the uncached alias
// (addr + MEM_L1_UNCACHED_BASE) and the CPU's cached view is a separate copy at
// the CCE address itself, so data moves between them only through the flush
// and invalidate fakes.

#pragma once

#include <cstddef>
#include <cstdint>

namespace cce_fake {

// The CCE-local window the test maps: the buffer regions and launch block.
constexpr uintptr_t kWindowBase = 0x3D0000;
constexpr size_t kWindowSize = 0x30000;

// Thread ids and runtime arguments the fakes hand to the kernel.
void set_thread_id(uint32_t id);
void set_args(const uint32_t* rt_args, size_t num_rt_args, const uint32_t* common_args, size_t num_common_args);

// The SRAM cell behind a CCE address.
uint8_t* sram(uintptr_t addr);

}  // namespace cce_fake
