// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

uint32_t get_num_threads();
uint32_t get_my_thread_id();
void sync_threads();
