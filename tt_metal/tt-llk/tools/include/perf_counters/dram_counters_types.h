// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

namespace llk::perf::dram
{

// Counters that live in a DRAM channel tile, not in a worker core. The host reads them over PCIe.
// Names only: append new counters at the end, never reorder or remove.
enum class DramCounter : std::uint16_t
{
    UNDEF = 0,
    // NoC interface of the tile, free running, counted per NoC.
    NIU_SLV_RD_REQ_RECEIVED,
    NIU_SLV_RD_DATA_WORD_SENT,
    NIU_SLV_NONPOSTED_WR_REQ_RECEIVED,
    NIU_SLV_POSTED_WR_REQ_RECEIVED,
    NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED,
    NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED,
};

enum class Arch : std::uint8_t
{
    BLACKHOLE = 0,
    WORMHOLE,
    QUASAR,
};

inline constexpr std::size_t NUM_ARCHES = 3;

// {counter name, offset from the NoC register base of the tile, plus noc_instance_stride per NoC}; every counter is a free running 32-bit register.
using Entry = std::pair<DramCounter, std::uint32_t>;

struct Table
{
    const Entry* data;
    std::size_t size;
};

// What a table of one architecture is relative to. A zero niu_base or word_bytes means the value is not known.
struct Layout
{
    std::uint32_t niu_base;            // address of NoC 0 register space on the tile
    std::uint32_t noc_instance_stride; // address distance between the register spaces of two NoCs
    std::uint32_t word_bytes;          // bytes behind one NIU data word
    std::uint32_t num_nocs;
};

} // namespace llk::perf::dram
