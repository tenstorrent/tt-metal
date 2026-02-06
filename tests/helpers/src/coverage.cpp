// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "llk_assert.h"

extern "C"
{
#include <gcov.h>

    typedef std::int64_t gcov_type;

    // Symbols pointing to per-TU coverage data from -fprofile-info-section.
    extern const struct gcov_info* __gcov_info_start[];
    extern const struct gcov_info* __gcov_info_end[];

    // Start address and region length of per-RISC REGION_GCOV.
    // This region stores the actual gcda, and the host reads it
    // and dumps it into a file.
    extern std::uint8_t __coverage_start[];
    extern std::uint8_t __coverage_end[];

    // Included so there are no compilation errors
    void __gcov_merge_add(gcov_type*, unsigned)
    {
        LLK_ASSERT(false, "__gcov_merge_add: panic");
    }

    void abort()
    {
        LLK_ASSERT(false, "abort: panic");
    }

    std::uint32_t __umodsi3(std::uint32_t a, std::uint32_t b)
    {
        return a % b;
    }

    std::uint32_t __udivsi3(std::uint32_t a, std::uint32_t b)
    {
        return a / b;
    }

    static void write_data(const void* _data, unsigned int length, void*)
    {
        const std::uint8_t* data = static_cast<const std::uint8_t*>(_data);
        auto* written            = reinterpret_cast<std::uint32_t*>(__coverage_start);

        std::uint8_t* mem = __coverage_start + *written;

        for (unsigned int i = 0; i < length; i++)
        {
            mem[i] = data[i];
        }

        *written += length;
    }

    void filename_dump(const char* f, void*)
    {
        __gcov_filename_to_gcfn(f, write_data, NULL);
    }

    // The first value in the coverage segment is the number of bytes written to it.
    void gcov_dump(void)
    {
        // First word is for file length, start writing past that.
        *reinterpret_cast<std::uint32_t*>(__coverage_start) = 4;

        const struct gcov_info* const* info = __gcov_info_start;
        asm volatile("" : "+r"(info)); // Prevent optimizations.
        std::uint16_t compilation_units = (__gcov_info_end - __gcov_info_start);

        for (std::uint16_t ind = 0; ind < compilation_units; ind++)
        {
            __gcov_info_to_gcda(info[ind], filename_dump, write_data, NULL, NULL);
        }
    }
}
