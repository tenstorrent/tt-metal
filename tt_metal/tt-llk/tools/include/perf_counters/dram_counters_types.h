// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "perf_counters/noc_counters_types.h"

namespace llk::perf::dram
{

// The DRAM tile counters are the slave side NIU counters of noc_counters_types.h, read by the host at
// Layout::niu_base + NIU_STATUS_OFFSET + 4 * reg (plus the NoC and tile instance terms of the Layout).
inline constexpr std::uint32_t NIU_STATUS_OFFSET = 0x200;

enum class Arch : std::uint8_t
{
    BLACKHOLE = 0,
    WORMHOLE,
    QUASAR,
};

inline constexpr std::size_t NUM_ARCHES = 3;

// What the register offsets are relative to; a zero niu_base or word_bytes means unknown.
// The instance of a tile is (noc0 y + tile_instance_bias) % tile_instance_count.
struct Layout
{
    std::uint64_t niu_base;            // address of the NoC 0 register space of instance 0
    std::uint32_t noc_instance_stride; // address distance between the register spaces of two NoCs
    std::uint32_t tile_instance_stride;
    std::uint32_t tile_instance_count;
    std::uint32_t tile_instance_bias;
    std::uint32_t word_bytes; // bytes behind one NIU data word
    std::uint32_t num_nocs;
};

// A 32-bit register of the memory controller activity monitor holding the count of the last interval.
struct ControllerEntry
{
    const char* name;
    std::uint32_t address;
};

struct ControllerTable
{
    const ControllerEntry* data;
    std::size_t size;
};

// Registers that frame a capture interval; a zero register_window means no known controller map.
struct ControllerLayout
{
    std::uint64_t register_window;       // OR-ed into every controller register address
    std::uint32_t interval;              // interval length in memory clocks; 0 stops counting and latches the counts
    std::uint32_t axi_control;           // AXI port resets and clock gating of the controller
    std::uint32_t axi_reset_bits;        // all set on a controller whose registers answer through this tile
    std::uint32_t axi_dynamic_gate_bits; // a dynamically gated port's monitor stops counting with its clock
    std::uint32_t command_bytes;         // bytes a full read or write command moves
};

// The files of the tt-metal capture (TT_METAL_PROFILE_DRAM_COUNTERS) and the bits of their flags column; the host
// decoder reads them from here.
inline constexpr const char TRAFFIC_LOG[]       = "dram_traffic.csv";
inline constexpr const char IDLE_LOG[]          = "dram_idle.csv";
inline constexpr const char CONTROLLER_LOG[]    = "dram_controller.csv";
inline constexpr const char TRAFFIC_HEADER[]    = "device_id,program_id,launch_seq,flags,elapsed_ns,dram_x,dram_y,noc,counter,delta_words,self_words";
inline constexpr const char IDLE_HEADER[]       = "device_id,dram_x,dram_y,noc,counter,words_per_snapshot_pair,words_per_second";
inline constexpr const char CONTROLLER_HEADER[] = "device_id,program_id,launch_seq,flags,elapsed_ns,channel,dram_x,dram_y,counter,value";

inline constexpr std::uint32_t FLAG_OVERLAPPED       = 1u << 0; // another program was in flight on the device
inline constexpr std::uint32_t FLAG_UNSETTLED        = 1u << 1; // write counters were still moving when the settle loop gave up
inline constexpr std::uint32_t FLAG_INTERVAL_WRAPPED = 1u << 2; // the window was long enough for the controller interval to run out
inline constexpr std::uint32_t FLAG_NOT_FINISHED     = 1u << 3; // nobody waited for the program; closed when the device closed
inline constexpr std::uint32_t FLAG_SERVICE_ACTIVE   = 1u << 4; // a service workload ran on the device during the window

} // namespace llk::perf::dram
