// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

extern "C"
{
    __attribute__((always_inline)) inline void *memset(void *ptr, int c, size_t len)
    {
        char *end  = (char *)ptr + len;
        char *iter = (char *)ptr;

        for (; iter < end; iter++)
        {
            *iter = c;
        }

        return ptr;
    }
}
