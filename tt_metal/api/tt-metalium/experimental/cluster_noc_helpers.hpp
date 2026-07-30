// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace tt::tt_metal::distributed {

/**
 * Thin shims over @c Cluster::write_core / @c Cluster::read_core /
 * @c Cluster::write_core_immediate / @c Cluster::read_reg.
 *
 * Why a separate translation unit? @c Cluster lives in @c tt_metal/llrt/ and
 * @c MetalContext lives in @c tt_metal/impl/, both of which are internal
 * tt-metal headers (not part of the @c tt-metalium/ public surface). The
 * ttnn-nanobind layer (@c ttnn/cpp/ttnn-nanobind/cluster.cpp) wants to
 * expose Python bindings for raw NOC reads/writes without pulling in those
 * internal headers; this header is the public-API gateway.
 *
 * All four helpers are one-line wrappers. They take TRANSLATED NOC
 * coordinates and a logical chip id; bytes are passed through verbatim.
 * Thread safety: callers must serialise host-side writes to a given
 * (device_id, x, y) tile.
 */

/**
 * @brief NOC write via @c Cluster::write_core (WC TLB window, Relaxed
 *        ordering, may use PCIe DMA fast path above the size threshold).
 *
 * @param device_id Logical chip id (matches @c IDevice::id()).
 * @param x TRANSLATED NOC x coord of the target tile.
 * @param y TRANSLATED NOC y coord of the target tile.
 * @param addr Device-side address (64-bit).
 * @param data Bytes to write.
 */
void noc_write(std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::string_view data);

/**
 * @brief NOC read via @c Cluster::read_core (counterpart to @ref noc_write).
 *
 * @return @p size bytes read from the target tile.
 */
std::vector<std::uint8_t> noc_read(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::uint32_t size);

/**
 * @brief NOC write via @c Cluster::write_core_immediate (UC TLB window,
 *        Strict ordering).
 *
 * No host-side write-combining, no DMA fast path; every byte hits the chip
 * in program order. Use when the target is order-sensitive (control
 * registers) or when a payload must not be merged into a single bursted
 * line; @ref noc_write is the right choice for bulk transfers where
 * throughput matters and ordering does not.
 *
 * @see Cluster::write_core_immediate in tt_metal/llrt/tt_cluster.cpp
 */
void noc_write_immediate(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::string_view data);

/**
 * @brief Single-u32 UC-path register read via @c Cluster::read_reg
 *        (counterpart to @ref noc_write_immediate).
 */
std::uint32_t noc_read_reg_u32(std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr);

/**
 * @brief One entry of the DRAM bank table used by tt-blaze's Tensix
 *        migration kernel and mirrored by the X280 migration worker.
 *
 * Each entry maps a logical @c bank_id to the NOC coordinate of the DRAM
 * controller that backs it plus the per-bank base address that goes into
 * the BRISC's @c bank_to_dram_offset[] table. Coordinates are TRANSLATED
 * on virtualized-DRAM SKUs (Blackhole) and raw NOC0 on the rest -- i.e.
 * the same value @c RiscFirmwareInitializer programs into
 * @c dram_bank_to_noc_xy[NOC0] for @c device_id.
 */
struct DramBankInfo {
    std::uint32_t bank_id;
    std::uint32_t noc_x;  // TRANSLATED on virtualized-DRAM SKUs, NOC0 elsewhere (NOC=0).
    std::uint32_t noc_y;
    std::uint64_t base_addr;  // = Allocator::get_bank_offset(BufferType::DRAM, bank_id).
    std::uint64_t bank_size;  // = metal_SocDescriptor::dram_view_size.
};

/**
 * @brief Snapshot of BRISC's @c dram_bank_to_noc_xy[NOC0] +
 *        @c bank_to_dram_offset[] for an opened device.
 *
 * Reproduces the table that @c RiscFirmwareInitializer::
 * @c generate_device_bank_to_noc_tables would multicast into Tensix L1 at
 * boot, so an X280 (or any other host-driven kernel) can populate its own
 * NOC translation table from Python and route DRAM transactions to any of
 * the @c NUM_DRAM_BANKS banks the way a Tensix kernel would.
 *
 * Requires the device to be opened first (e.g. via @c
 * ttnn.open_mesh_device); otherwise this throws.
 *
 * @param device_id Logical chip id (matches @c IDevice::id()).
 * @return @c NUM_DRAM_BANKS entries indexed by @c bank_id.
 */
std::vector<DramBankInfo> get_dram_bank_table(std::uint32_t device_id);

}  // namespace tt::tt_metal::distributed
