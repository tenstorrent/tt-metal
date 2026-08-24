// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "internal/cluster_noc_helpers.hpp"

#include <tt_stl/assert.hpp>
#include <umd/device/cluster_descriptor.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "tt_metal/api/tt-metalium/allocator.hpp"
#include "tt_metal/api/tt-metalium/buffer_types.hpp"
#include "tt_metal/api/tt-metalium/device.hpp"
#include "tt_metal/llrt/hal.hpp"
#include "tt_metal/llrt/metal_soc_descriptor.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

// Kept in a separate TU so callers do not need MetalContext (impl/) or
// tt_cluster (llrt/) on their include path.

namespace tt::tt_metal::internal {

void noc_write(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, ttsl::Span<const std::byte> data) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    cluster.write_core(data.data(), static_cast<std::uint32_t>(data.size()), target, addr);
}

std::vector<std::byte> noc_read(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::uint32_t size) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    std::vector<std::byte> buf(size);
    cluster.read_core(buf.data(), size, target, addr);
    return buf;
}

void noc_write_immediate(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, ttsl::Span<const std::byte> data) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    cluster.write_core_immediate(data.data(), static_cast<std::uint32_t>(data.size()), target, addr);
}

std::uint32_t noc_read_reg_u32(std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    std::uint32_t value = 0;
    cluster.read_reg(&value, target, addr);
    return value;
}

// Mirrors RiscFirmwareInitializer::generate_device_bank_to_noc_tables for NOC=0.
// On virtualized-DRAM architectures the table holds TRANSLATED coords as-is;
// elsewhere the NOC=0 conversion is the identity.
//
// DRAM is allocated one bank per channel (bank_id == channel), so bank_id is passed
// straight to get_preferred_worker_core_for_dram_view().
std::vector<DramBankInfo> get_dram_bank_table(std::uint32_t device_id) {
    auto& mctx = MetalContext::instance();
    auto& dm = mctx.device_manager();
    TT_FATAL(dm != nullptr, "DeviceManager is not initialised; open a device before calling get_dram_bank_table");
    auto* dev = dm->get_active_device(static_cast<ChipId>(device_id));
    TT_FATAL(
        dev != nullptr,
        "Device {} is not opened; open via ttnn.open_mesh_device before calling get_dram_bank_table",
        device_id);

    const auto& allocator = *dev->allocator();
    const auto& cluster = mctx.get_cluster();
    const auto& soc_d = cluster.get_soc_desc(static_cast<ChipId>(device_id));
    const auto& hal = mctx.hal();

    const bool noc_translation_enabled =
        !cluster.is_mock_or_emulated() && cluster.get_cluster_desc()->get_noc_translation_table_en().at(device_id);
    const bool dram_is_virtualized =
        noc_translation_enabled && hal.get_virtualized_core_types().contains(dev_msgs::AddressableCoreType::DRAM);

    const std::size_t num_banks = allocator.get_num_banks(BufferType::DRAM);
    std::vector<DramBankInfo> out;
    out.reserve(num_banks);
    constexpr std::uint8_t kNoc = 0;
    for (std::uint32_t bank_id = 0; bank_id < num_banks; ++bank_id) {
        CoreCoord dram_noc_coord = soc_d.get_preferred_worker_core_for_dram_view(static_cast<int>(bank_id), kNoc);
        std::uint32_t x = static_cast<std::uint32_t>(dram_noc_coord.x);
        std::uint32_t y = static_cast<std::uint32_t>(dram_noc_coord.y);
        if (!dram_is_virtualized) {
            x = static_cast<std::uint32_t>(hal.noc_coordinate(kNoc, soc_d.grid_size.x, dram_noc_coord.x));
            y = static_cast<std::uint32_t>(hal.noc_coordinate(kNoc, soc_d.grid_size.y, dram_noc_coord.y));
        }
        const std::int32_t bank_offset = allocator.get_bank_offset(BufferType::DRAM, bank_id);
        out.push_back(DramBankInfo{
            /*bank_id=*/bank_id,
            /*noc_x=*/x,
            /*noc_y=*/y,
            /*base_addr=*/static_cast<std::uint64_t>(static_cast<std::int64_t>(bank_offset)),
            /*bank_size=*/soc_d.dram_view_size,
        });
    }
    return out;
}

}  // namespace tt::tt_metal::internal
