// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

static inline bool reload_next_stage(launch_msg_t*, uint32_t&, uint32_t&) { return false; }
