// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace llk::perf::noc
{

// The 32-bit event counters of a tile's NoC interface unit (NIU), one set per NoC, shared by every core of the tile.
// They count events (requests, data words), never stalls, and nothing resets them: a capture is a delta modulo 2^32.
// Master side counters count what the tile issues, slave side counters what it serves.
// Names only: the profiler record is positional, the host decodes it through the arch table. Append, never reorder.
enum class NocCounter : std::uint16_t
{
    UNDEF = 0,
    MST_RD_REQ_SENT,
    MST_RD_DATA_WORD_RECEIVED,
    MST_NONPOSTED_WR_REQ_SENT,
    MST_NONPOSTED_WR_DATA_WORD_SENT,
    MST_POSTED_WR_REQ_SENT,
    MST_POSTED_WR_DATA_WORD_SENT,
    MST_NONPOSTED_ATOMIC_SENT,
    MST_POSTED_ATOMIC_SENT,
    SLV_RD_REQ_RECEIVED,
    SLV_RD_DATA_WORD_SENT,
    SLV_NONPOSTED_WR_REQ_RECEIVED,
    SLV_NONPOSTED_WR_DATA_WORD_RECEIVED,
    SLV_POSTED_WR_REQ_RECEIVED,
    SLV_POSTED_WR_DATA_WORD_RECEIVED,
    SLV_NONPOSTED_ATOMIC_RECEIVED,
    SLV_POSTED_ATOMIC_RECEIVED,
};

// Capture groups; the bit of a group is 1 << group in the option mask.
enum class NocGroup : std::uint8_t
{
    MASTER = 0, // requests and data words the tile issues
    SLAVE  = 1, // requests and data words the tile serves
    ATOMIC = 2, // atomic requests, both sides
};

inline constexpr std::uint32_t GROUP_MASTER = 1u << static_cast<unsigned>(NocGroup::MASTER);
inline constexpr std::uint32_t GROUP_SLAVE  = 1u << static_cast<unsigned>(NocGroup::SLAVE);
inline constexpr std::uint32_t GROUP_ATOMIC = 1u << static_cast<unsigned>(NocGroup::ATOMIC);
inline constexpr std::uint32_t GROUP_ALL    = GROUP_MASTER | GROUP_SLAVE | GROUP_ATOMIC;

// reg is the index into the NoC status register file; the register address is
// NIU_STATUS_BASE + noc * NIU_NOC_STRIDE + reg * 4 (noc_counters_<arch>.h). Table order is the order of a capture.
struct Entry
{
    NocCounter counter;
    std::uint8_t reg;
    NocGroup group;
};

} // namespace llk::perf::noc
