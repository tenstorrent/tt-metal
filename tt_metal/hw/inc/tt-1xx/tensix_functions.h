// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensix_functions_common.h"

extern "C" void wzerorange(uint32_t* start, uint32_t* end);
inline void wzeromem(uint32_t start, uint32_t len) { wzerorange((uint32_t*)start, (uint32_t*)(start + len)); }
