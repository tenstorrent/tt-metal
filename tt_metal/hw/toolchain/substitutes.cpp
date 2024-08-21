// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstddef>

using namespace std;

extern "C" int atexit(void (*f)(void))
{
    return 0;
}

extern "C" void exit(int ec)
{
    while (1) { asm volatile ("" ::: "memory"); }
}

extern "C" void wzerorange(uint32_t *start, uint32_t *end) __attribute__((aligned(16)));

extern "C" void wzerorange(uint32_t *start, uint32_t *end)
{
    for (; start != end; start++)
    {
        *start = 0;
    }
}
