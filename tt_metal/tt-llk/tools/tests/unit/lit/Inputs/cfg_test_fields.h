// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hal/cfg.h"

using namespace hal::cfg;

// Synthetic descriptors exercise bank boundaries and identical addresses in
// different scopes without depending on which real fields occupy those words.
inline constexpr Field state_first {RegisterScope::State, 32, 0, 0, 0, 32, 1, 0};
inline constexpr Field state_last {RegisterScope::State, 32, 223, 0, 0, 32, 1, 0};
inline constexpr Field state_last_block {RegisterScope::State, 32, 220, 0, 0, 32, 1, 0};
inline constexpr Field state_crossing {RegisterScope::State, 32, 222, 0, 0, 32, 1, 0};
inline constexpr Field state_outside {RegisterScope::State, 32, 224, 0, 0, 32, 1, 0};
inline constexpr Field thread_outside {RegisterScope::Thread, 16, 68, 0, 0, 16, 1, 0};
inline constexpr Field state_word5 {RegisterScope::State, 32, 5, 0, 0, 4, 1, 0};
inline constexpr Field thcon_last_block {RegisterScope::State, 32, 176, 0, 0, 32, 1, 0};
