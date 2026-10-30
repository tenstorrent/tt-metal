// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llk_pack_common.h"
void acquire_rt(const hal::sync::Mutex mutex) { hal::sync::mutex::acquire(mutex); }
void atgetm_rt(const std::uint32_t m) { TT_ATGETM(m); }
void stall_rt(const std::uint32_t a, const std::uint32_t b) { TT_STALLWAIT(a, b); }
