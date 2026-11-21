// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace llk::perf::router
{

// The NoC router of a tile has one counter per virtual channel on each of its output ports, and on Quasar on each input
// port too. They are free running and nothing here resets them: a capture is a delta modulo 2^32.
// Names only: the profiler record is positional, the host decodes it through the arch table. Append, never reorder.
enum class RouterPort : std::uint8_t
{
    UNDEF = 0,
    OUT_PORT1, // output ports leave the tile; the port that ejects into the tile's own NoC interface has no counter
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
// offset is the byte offset of the bank from the NoC register base of the tile, plus ROUTER_NOC_STRIDE per NoC.
// Table order is the order of a capture.
struct Bank
{
    RouterPort port;
    std::uint16_t offset;
};

inline constexpr std::uint32_t NUM_VCS = 16;

// The profiler option is one mask: bits 0 to 15 select virtual channels, bits 16 to 23 select banks (bit 16 is the
// first table entry), bit 24 keeps every selected virtual channel apart. Without bit 24 the selected virtual channels
// of a bank are added up into one value. Bank bits beyond the table of the architecture are ignored.
inline constexpr std::uint32_t VC_MASK_ALL  = 0x0000FFFFu;
inline constexpr std::uint32_t BANK_SHIFT   = 16;
inline constexpr std::uint32_t BANKS_ALL    = 0x00FF0000u;
inline constexpr std::uint32_t PER_VC       = 0x01000000u;
inline constexpr std::uint32_t OPTION_ALL   = VC_MASK_ALL | BANKS_ALL;
inline constexpr std::uint32_t OPTION_VALID = OPTION_ALL | PER_VC;
// One profiler record holds two values; a capture is at most 16 records.
inline constexpr std::uint32_t MAX_VALUES = 32;

} // namespace llk::perf::router
