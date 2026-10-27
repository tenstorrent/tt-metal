// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/span.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tt::tt_metal::internal {

/**
 * @warning INTERNAL. Everything declared in this header lives under
 * @c api/internal: it exists to serve tt-metal's own tooling and
 * bindings, is not part of the supported user-facing API, and may change or be
 * removed without a deprecation period.
 *
 * These are raw NOC accessors with no bounds checking beyond what the Watcher
 * host sanitizer applies; callers are responsible for targeting a valid core
 * and address range.
 */

/**
 * Wrappers over @c Cluster::write_core / @c read_core / @c write_core_immediate /
 * @c read_reg, declared here so callers can issue raw NOC access without including
 * the internal @c Cluster and @c MetalContext headers.
 *
 * All take TRANSLATED NOC coordinates and a logical chip id. Callers must serialise
 * host-side writes to a given (device_id, x, y) tile.
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
void noc_write(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, ttsl::Span<const std::byte> data);

/**
 * @brief NOC read via @c Cluster::read_core (counterpart to @ref noc_write).
 *
 * @return @p size bytes read from the target tile.
 */
std::vector<std::byte> noc_read(
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
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, ttsl::Span<const std::byte> data);

/**
 * @brief Single-u32 UC-path register read via @c Cluster::read_reg
 *        (counterpart to @ref noc_write_immediate).
 */
std::uint32_t noc_read_reg_u32(std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr);

/**
 * @brief One entry of the per-bank DRAM NOC routing table.
 *
 * Maps a logical @c bank_id to the NOC coordinate of the DRAM controller backing it
 * and the per-bank base address. Coordinates are TRANSLATED on virtualized-DRAM
 * architectures and NOC0 elsewhere.
 */
struct DramBankInfo {
    std::uint32_t bank_id;
    std::uint32_t noc_x;  // TRANSLATED on virtualized-DRAM SKUs, NOC0 elsewhere (NOC=0).
    std::uint32_t noc_y;
    std::uint64_t base_addr;  // = Allocator::get_bank_offset(BufferType::DRAM, bank_id).
    std::uint64_t bank_size;  // = metal_SocDescriptor::dram_view_size.
};

/**
 * @brief Returns the per-bank DRAM NOC routing table for an opened device.
 *
 * Mirrors the @c dram_bank_to_noc_xy[NOC0] + @c bank_to_dram_offset[] tables that
 * @c RiscFirmwareInitializer::generate_device_bank_to_noc_tables programs at boot.
 *
 * @param device_id Logical chip id (matches @c IDevice::id()). The device must
 *                  already be open, otherwise this throws.
 * @return One entry per DRAM bank, indexed by @c bank_id.
 */
std::vector<DramBankInfo> get_dram_bank_table(std::uint32_t device_id);

}  // namespace tt::tt_metal::internal
