// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "build.h"
#include "ckernel.h"

// Logic to convert zone name -> 16bit numeric id
#define Stringize(L)       #L
#define ExpandStringize(L) Stringize(L)

template <size_t N>
constexpr std::uint16_t hashString16(const char (&s)[N])
{
    std::uint32_t hash32 = UINT32_C(2166136261);
    for (std::size_t i = 0; i < N - 1; ++i)
    {
        std::uint8_t c = static_cast<std::uint8_t>(s[i]);
        hash32 ^= c;
        hash32 *= UINT32_C(16777619);
    }
    return static_cast<uint16_t>(hash32 ^ (hash32 >> 16));
}

// clang-format off
#define MARKER_FULL(marker) "LLK_PROFILER" ":" __FILE__ ":" ExpandStringize(__LINE__) ":" marker
// clang-format on

#define MARKER_ID(marker) hashString16(MARKER_FULL(marker))

/* Push a string containing the full marker into the .profiler_meta section.
 * This section will be processed by the host code to construct a mapping
 * MARKER_ID -> { filename, line, marker } for parsing the profiler buffer.
 */
// clang-format off
#define PROFILER_META(full_marker)                          \
    __attribute__((section(".profiler_meta"), used))        \
    static const char _profiler_meta_##__COUNTER__[] = full_marker;
// clang-format on

#if defined(LLK_PROFILER)

namespace llk_profiler
{

constexpr uint32_t ENTRY_TYPE_SHAMT = 28;
constexpr uint32_t ENTRY_ID_SHAMT   = ENTRY_TYPE_SHAMT - 16;
constexpr uint32_t ENTRY_META_SHAMT = ENTRY_ID_SHAMT;

constexpr uint32_t ENTRY_META_MASK = ~((1 << ENTRY_META_SHAMT) - 1);

constexpr uint32_t ENTRY_EXISTS_BIT = 0b1000 << ENTRY_TYPE_SHAMT;

enum class EntryType : uint32_t
{
    TIMESTAMP      = 0b1000,
    TIMESTAMP_DATA = 0b1001,
    ZONE_START     = 0b1010,
    ZONE_END       = 0b1011
};

// Initialize id of the core executing the kernel
#if defined(LLK_TRISC_UNPACK)
constexpr uint32_t TRISC_ID = 0;
#elif defined(LLK_TRISC_MATH)
constexpr uint32_t TRISC_ID = 1;
#elif defined(LLK_TRISC_PACK)
constexpr uint32_t TRISC_ID = 2;
#else
#error "Profiler can only be used on TRISC cores"
#endif

constexpr uint32_t BUFFER_LENGTH = 0x400; // 1024 entries per core
constexpr uint32_t NUM_CORES     = 3;     // TRISC cores: unpack, math, pack
constexpr uint32_t BUFFERS_END   = 0x16E000;
constexpr uint32_t BUFFERS_START = BUFFERS_END - (NUM_CORES * BUFFER_LENGTH * sizeof(uint32_t));

constexpr uint32_t BARRIER_END   = BUFFERS_START;
constexpr uint32_t BARRIER_START = BARRIER_END - (NUM_CORES * sizeof(uint32_t));

using barrier_ptr_t = volatile uint32_t (*)[NUM_CORES];
using buffer_ptr_t  = uint32_t (*)[BUFFER_LENGTH];

extern barrier_ptr_t barrier_ptr;
extern buffer_ptr_t buffer;
extern uint32_t write_idx;
extern uint32_t open_zone_cnt;

__attribute__((always_inline)) inline void sync_threads()
{
    auto& barrier = *barrier_ptr;

    // wait for all the threads to set the barrier
    barrier[TRISC_ID] = 1;
    asm volatile("fence" ::: "memory");
    for (uint32_t i = 0; i < NUM_CORES; ++i)
    {
        if (i == TRISC_ID)
        {
            continue;
        }
        while (barrier[i] != 1)
        {
            asm volatile("fence" ::: "memory");
        }
    }
}

__attribute__((always_inline)) inline void reset()
{
    barrier_ptr   = reinterpret_cast<barrier_ptr_t>(BARRIER_START);
    buffer        = reinterpret_cast<buffer_ptr_t>(BUFFERS_START);
    write_idx     = 0;
    open_zone_cnt = 0;

    memset(buffer[TRISC_ID], 0, BUFFER_LENGTH * sizeof(buffer[TRISC_ID][0]));
}

__attribute__((always_inline)) inline bool is_buffer_full()
{
    // the buffer is considered full when there is not enough space to store:
    // - timestamp with data (TIMESTAMP_DATA_ENTRY) (size = 16B)
    // - new zone (ZONE_START_ENTRY + ZONE_END_ENTRY) (size = 16B)
    // after closing all of the currently open zones
    return (BUFFER_LENGTH - (write_idx + open_zone_cnt)) < 4;
}

__attribute__((always_inline)) inline void write_entry(EntryType type, uint16_t id16)
{
    uint64_t timestamp      = ckernel::read_wall_clock();
    uint32_t timestamp_high = static_cast<uint32_t>(timestamp >> 32);

    uint32_t type_numeric = static_cast<uint32_t>(type);
    uint32_t meta         = (type_numeric << ENTRY_TYPE_SHAMT) | ((uint32_t)id16 << ENTRY_ID_SHAMT);

    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    buffer[TRISC_ID][write_idx++] = static_cast<uint32_t>(timestamp);
}

__attribute__((always_inline)) inline void write_data(uint64_t data)
{
    buffer[TRISC_ID][write_idx++] = static_cast<uint32_t>(data >> 32);
    buffer[TRISC_ID][write_idx++] = static_cast<uint32_t>(data);
}

template <uint16_t id16>
class zone_scoped
{
private:
    bool is_opened = false;

public:
    zone_scoped(const zone_scoped&)            = delete;
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped()
    {
        if (!is_buffer_full())
        {
            is_opened = true;
            write_entry(EntryType::ZONE_START, id16);
            ++open_zone_cnt;
        }
    }

    ~zone_scoped()
    {
        if (is_opened)
        {
            write_entry(EntryType::ZONE_END, id16);
            --open_zone_cnt;
        }
    }
};

__attribute__((always_inline)) inline void write_timestamp(uint16_t id16)
{
    if (!is_buffer_full())
    {
        write_entry(EntryType::TIMESTAMP, id16);
    }
}

__attribute__((always_inline)) inline void write_timestamp(uint16_t id16, uint64_t data)
{
    if (!is_buffer_full())
    {
        write_entry(EntryType::TIMESTAMP_DATA, id16);
        write_data(data);
    }
}

} // namespace llk_profiler

#define ZONE_SCOPED(marker)            \
    PROFILER_META(MARKER_FULL(marker)) \
    const auto _zone_scoped_ = llk_profiler::zone_scoped<MARKER_ID(marker)>();

#define TIMESTAMP(marker)              \
    PROFILER_META(MARKER_FULL(marker)) \
    llk_profiler::write_timestamp(MARKER_ID(marker));

#define TIMESTAMP_DATA(marker, data)   \
    PROFILER_META(MARKER_FULL(marker)) \
    llk_profiler::write_timestamp(MARKER_ID(marker), data);

#else

#define ZONE_SCOPED(marker)

#define TIMESTAMP(marker)

#define TIMESTAMP_DATA(marker, data)

#endif
