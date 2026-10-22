// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/circular_buffer_interface.h"
#include "llk_assert.h"

inline uint32_t get_cb_page_size(const uint32_t cb_id, const uint32_t page_size) {
    LLK_ASSERT(
        page_size == get_local_cb_interface(cb_id).fifo_page_size,
        "CB page size differs from the one the kernel was compiled with");
    return page_size;
}
