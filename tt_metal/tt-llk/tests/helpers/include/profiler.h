// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(LLK_PROFILER)

#include <cstdint>
#include <cstring>

#include "ckernel.h"
#include "metadata.h"

// Logic to convert zone name -> 16bit numeric id
#define Stringize(L)       #L
#define ExpandStringize(L) Stringize(L)

/* Push a string containing the full marker into the .profiler_meta section.
 * This section will be processed by the host code to construct a mapping
 * MARKER_ID -> { filename, line, marker } for parsing the profiler buffer.
 */
// clang-format off
#define PROFILER_META(name)                                                 \
    []() {                                                                  \
        static constexpr llk::sstring::container section(".meta.profiler"); \
        static constexpr llk::sstring::container marker(name);              \
        static constexpr llk::sstring::container file(__FILE__);            \
        static constexpr size_t line = __LINE__;                            \
                                                                            \
        auto meta = llk::MetadataBuilder<section>()                         \
            .add(llk::StringField<marker>{})                                \
            .add(llk::StringField<file>{})                                  \
            .add(llk::IntegralField<line>{})                                \
            .create();                                                      \
                                                                            \
        return reinterpret_cast<uintptr_t>(meta);                           \
    }()
// clang-format on

namespace llk_profiler
{

constexpr std::uint32_t ENTRY_TYPE_SHAMT = 28;
constexpr std::uint32_t ENTRY_ID_SHAMT   = ENTRY_TYPE_SHAMT - 16;
constexpr std::uint32_t ENTRY_META_SHAMT = ENTRY_ID_SHAMT;

constexpr std::uint32_t ENTRY_META_MASK = ~((1 << ENTRY_META_SHAMT) - 1);

constexpr std::uint32_t ENTRY_EXISTS_BIT = 0b1000 << ENTRY_TYPE_SHAMT;

enum class EntryType : std::uint32_t
{
    TIMESTAMP      = 0b1000,
    TIMESTAMP_DATA = 0b1001,
    ZONE_START     = 0b1010,
    ZONE_END       = 0b1011
};

// Initialize id of the core executing the kernel
#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t TRISC_ID = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t TRISC_ID = 1;
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t TRISC_ID = 2;
#elif defined(LLK_TRISC_ISOLATE_SFPU)
constexpr std::uint32_t TRISC_ID = 3;
#else
#error "Profiler can only be used on TRISC cores"
#endif

constexpr std::uint32_t BUFFER_LENGTH = 0x400; // 1024 entries per core
// Quasar: 4 TRISCs (UNPACK, MATH, PACK, SFPU); Wormhole/Blackhole: 3 TRISCs
#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_CORES   = 4;
constexpr std::uint32_t BUFFERS_END = 0x16F000;
#else
constexpr std::uint32_t NUM_CORES   = 3;
constexpr std::uint32_t BUFFERS_END = 0x16E000;
#endif
constexpr std::uint32_t BUFFERS_START = BUFFERS_END - (NUM_CORES * BUFFER_LENGTH * sizeof(std::uint32_t));

constexpr std::uint32_t BARRIER_END   = BUFFERS_START;
constexpr std::uint32_t BARRIER_START = BARRIER_END - (NUM_CORES * sizeof(std::uint32_t));

using barrier_ptr_t = volatile std::uint32_t (*)[NUM_CORES];
using buffer_ptr_t  = std::uint32_t (*)[BUFFER_LENGTH];

extern barrier_ptr_t barrier_ptr;
extern buffer_ptr_t buffer;
extern std::uint32_t write_idx;
extern std::uint32_t open_zone_cnt;

__attribute__((always_inline)) inline void sync_threads()
{
    auto& barrier = *barrier_ptr;

    // wait for all the threads to set the barrier
    barrier[TRISC_ID] = 1;
    ckernel::invalidate_data_cache();
    for (std::uint32_t i = 0; i < NUM_CORES; ++i)
    {
        if (i == TRISC_ID)
        {
            continue;
        }
        while (barrier[i] != 1)
        {
            ckernel::invalidate_data_cache();
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

__attribute__((always_inline)) inline void write_entry(EntryType type, std::uint16_t id16)
{
    std::uint64_t timestamp      = ckernel::read_wall_clock();
    std::uint32_t timestamp_high = static_cast<std::uint32_t>(timestamp >> 32);

    std::uint32_t type_numeric = static_cast<std::uint32_t>(type);
    std::uint32_t meta         = (type_numeric << ENTRY_TYPE_SHAMT) | (static_cast<std::uint32_t>(id16) << ENTRY_ID_SHAMT);

    buffer[TRISC_ID][write_idx++] = meta | (timestamp_high & ~ENTRY_META_MASK);
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(timestamp);
}

__attribute__((always_inline)) inline void write_data(std::uint64_t data)
{
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(data >> 32);
    buffer[TRISC_ID][write_idx++] = static_cast<std::uint32_t>(data);
}

class zone_scoped
{
private:
    const std::uint16_t id16;
    bool is_opened = false;

public:
    zone_scoped(const zone_scoped&)            = delete;
    zone_scoped(zone_scoped&&)                 = delete;
    zone_scoped& operator=(const zone_scoped&) = delete;
    zone_scoped& operator=(zone_scoped&&)      = delete;

    inline __attribute__((always_inline)) zone_scoped(std::uint16_t id16) : id16(id16)
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

__attribute__((always_inline)) inline void write_timestamp(std::uint16_t id16)
{
    if (!is_buffer_full())
    {
        write_entry(EntryType::TIMESTAMP, id16);
    }
}

__attribute__((always_inline)) inline void write_timestamp(std::uint16_t id16, std::uint64_t data)
{
    if (!is_buffer_full())
    {
        write_entry(EntryType::TIMESTAMP_DATA, id16);
        write_data(data);
    }
}

} // namespace llk_profiler

#define ZONE_SCOPED(marker) const auto _zone_scoped_ = llk_profiler::zone_scoped(PROFILER_META(marker));

#define TIMESTAMP(marker) llk_profiler::write_timestamp(PROFILER_META(marker));

#define TIMESTAMP_DATA(marker, data) llk_profiler::write_timestamp(PROFILER_META(marker), data);

#define PROFILER_SYNC() tensix_sync()

#else

#define ZONE_SCOPED(marker)

#define TIMESTAMP(marker)

#define TIMESTAMP_DATA(marker, data)

#define PROFILER_SYNC()

#endif
