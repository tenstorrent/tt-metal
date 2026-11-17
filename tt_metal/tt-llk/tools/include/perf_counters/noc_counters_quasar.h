// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/noc_counters_types.h"

// Quasar NIU counters (one NoC). Include through perf_counters/noc_counters_inventory.h.

namespace llk::perf::noc
{

inline constexpr std::uint32_t NIU_STATUS_BASE = 0x02000200; // NoC register space base plus the status file offset 0x200
inline constexpr std::uint32_t NIU_NOC_STRIDE  = 0x10000;
inline constexpr std::uint32_t NIU_NOCS        = 1;
// Bytes in one data word of the *_DATA_WORD_* counters. ASSUMED 256 (the 2048-bit NoC payload width); not yet checked on the emulator.
inline constexpr std::uint32_t NIU_DATA_WORD_BYTES = 256;

// clang-format off
inline constexpr std::array<Entry, 16> noc_counters = {
    {
    {NocCounter::MST_RD_REQ_SENT, 0x5, NocGroup::MASTER},
    {NocCounter::MST_RD_DATA_WORD_RECEIVED, 0x3, NocGroup::MASTER},
    {NocCounter::MST_NONPOSTED_WR_REQ_SENT, 0xA, NocGroup::MASTER},
    {NocCounter::MST_NONPOSTED_WR_DATA_WORD_SENT, 0x8, NocGroup::MASTER},
    {NocCounter::MST_POSTED_WR_REQ_SENT, 0xB, NocGroup::MASTER},
    {NocCounter::MST_POSTED_WR_DATA_WORD_SENT, 0x9, NocGroup::MASTER},
    {NocCounter::SLV_RD_REQ_RECEIVED, 0x35, NocGroup::SLAVE},
    {NocCounter::SLV_RD_DATA_WORD_SENT, 0x33, NocGroup::SLAVE},
    {NocCounter::SLV_NONPOSTED_WR_REQ_RECEIVED, 0x3A, NocGroup::SLAVE},
    {NocCounter::SLV_NONPOSTED_WR_DATA_WORD_RECEIVED, 0x38, NocGroup::SLAVE},
    {NocCounter::SLV_POSTED_WR_REQ_RECEIVED, 0x3B, NocGroup::SLAVE},
    {NocCounter::SLV_POSTED_WR_DATA_WORD_RECEIVED, 0x39, NocGroup::SLAVE},
    {NocCounter::MST_NONPOSTED_ATOMIC_SENT, 0x6, NocGroup::ATOMIC},
    {NocCounter::MST_POSTED_ATOMIC_SENT, 0x7, NocGroup::ATOMIC},
    {NocCounter::SLV_NONPOSTED_ATOMIC_RECEIVED, 0x36, NocGroup::ATOMIC},
    {NocCounter::SLV_POSTED_ATOMIC_RECEIVED, 0x37, NocGroup::ATOMIC}}};
// clang-format on

} // namespace llk::perf::noc
