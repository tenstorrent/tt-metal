// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace llk::perf::router
{

// The router counters are free running, so a capture is a delta modulo 2^32.
// Append, never reorder: the profiler record is positional and the host decodes it with the arch table.
enum class RouterPort : std::uint8_t
{
    UNDEF = 0,
    OUT_PORT1, // leaves the tile; the ejection port has no counter
    OUT_PORT2,
    OUT_PORT3,
    OUT_PORT4,
    IN_PORT1,
    IN_PORT2,
    IN_PORT3,
    IN_PORT4,
};

// What one count of a counter is.
enum class CountKind : std::uint8_t
{
    NONE = 0,
    HEAD_FLIT, // one count per packet that leaves through the port
    ALL_FLIT,  // one count per flit of any type, head, body and tail; not proven on silicon
};

// One bank is the 16 virtual channel counters of one port.
// offset is in bytes from the tile's NoC register base, plus ROUTER_NOC_STRIDE per NoC. Table order is capture order.
struct Bank
{
    RouterPort port;
    std::uint16_t offset;
};

inline constexpr std::uint32_t NUM_VCS = 16;

// Option mask: bits 0-15 select virtual channels, bits 16-23 banks (bit 16 is the first table entry), bit 24 keeps the
// channels apart, else a bank's channels are summed. Bank bits past the table are ignored.
inline constexpr std::uint32_t VC_MASK_ALL  = 0x0000FFFFu;
inline constexpr std::uint32_t BANK_SHIFT   = 16;
inline constexpr std::uint32_t BANKS_ALL    = 0x00FF0000u;
inline constexpr std::uint32_t PER_VC       = 0x01000000u;
inline constexpr std::uint32_t OPTION_ALL   = VC_MASK_ALL | BANKS_ALL;
inline constexpr std::uint32_t OPTION_VALID = OPTION_ALL | PER_VC;
// A record holds two values and a capture is at most 16 records.
inline constexpr std::uint32_t MAX_VALUES = 32;

} // namespace llk::perf::router
