// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/device_wall_clock.hpp"

#include <tt_stl/assert.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/io_window/io_window.hpp>
#include <tt-umd/pcie/tlb_handle.hpp>
#include <tt-umd/pcie/tlb_window.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/io_window_config.hpp>

#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace {

// A Tensix core's RISC debug register block, and its wall clock's low and high words inside it
// (RISCV_DEBUG_REG_WALL_CLOCK_L and _H).
constexpr uint64_t kRiscDebugRegs = 0xFFB12000ull;
constexpr uint32_t kWallClockLo = 0x1F0, kWallClockHi = 0x1F8;

}  // namespace

const volatile uint32_t* mapped_word(tt::umd::IoWindow& window, uint64_t offset) {
    auto& tlb = dynamic_cast<tt::umd::TlbWindow&>(window);
    const tt::umd::TlbHandle& handle = tlb.handle_ref();
    TT_FATAL(handle.get_base() != nullptr, "I/O window has no host mapping");
    return reinterpret_cast<const volatile uint32_t*>(
        handle.get_base() + (tlb.get_base_address() - handle.get_config().local_offset) + offset);
}

DeviceWallClock::DeviceWallClock(tt::Cluster& cluster, uint32_t chip_id, const CoreCoord& virtual_core) :
    window_(cluster.get_driver()->create_io_window(
        chip_id,
        tt::umd::CoreCoord(virtual_core, CoreType::TENSIX, CoordSystem::TRANSLATED),
        kRiscDebugRegs,
        tt::umd::HostIoWindowConfig{
            .mapping = tt::umd::HostMemoryCaching::UC, .size = kWallClockHi + sizeof(uint32_t)})),
    lo_(mapped_word(*window_, kWallClockLo)),
    hi_(mapped_word(*window_, kWallClockHi)) {}

DeviceWallClock::~DeviceWallClock() = default;

}  // namespace tt::tt_metal
