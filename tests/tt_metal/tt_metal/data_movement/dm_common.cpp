// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dm_common.hpp"
#include "device_fixture.hpp"
#include <tuple>

namespace tt::tt_metal::unit_tests::dm {

// Hardcoded constants for L1 memory base address and size
// constexpr uint64_t HARDCODED_L1_MEMORY_BASE_ADDRESS = 1024 * 256;  // 256 KB
// constexpr uint64_t HARDCODED_L1_MEMORY_SIZE_BYTES = 1024 * 1024;   // 1 MB

uint32_t runtime_host_id = 0;

// Static function for internal use only
static uint32_t obtain_page_size_bytes(tt::ARCH arch) { return (arch == tt::ARCH::BLACKHOLE) ? 64 : 32; }

L1AddressInfo get_l1_address_and_size(const IDevice* device, const CoreCoord& core_coord) {
    // Obtaining L1 address and size for a specific core //

    CoreCoord physical_core = device->worker_core_from_logical_core(core_coord);

    uint64_t core_l1_base_address = device->get_dev_addr(physical_core, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint64_t core_l1_size = device->get_dev_size(physical_core, HalL1MemAddrType::DEFAULT_UNRESERVED);

    return {core_l1_base_address, core_l1_size};

    // Obtaining hardcoded values for L1 address and size //

    // return {HARDCODED_L1_MEMORY_BASE_ADDRESS, HARDCODED_L1_MEMORY_SIZE_BYTES};
}

DramAddressInfo get_dram_address_and_size(const IDevice* device) {
    // Obtaining DRAM address and size //

    auto dram_base_address = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    uint32_t dram_size = tt::tt_metal::MetalContext::instance().hal().get_dev_size(HalDramMemAddrType::UNRESERVED);

    return {dram_base_address, dram_size};
}

std::tuple<uint32_t, uint32_t, uint32_t> compute_physical_constraints(const tt::ARCH arch, const IDevice* device) {
    auto [_, max_transmittable_bytes] = get_l1_address_and_size(device);
    uint32_t bytes_per_page = obtain_page_size_bytes(arch);
    uint32_t max_transmittable_pages = max_transmittable_bytes / bytes_per_page;

    return {bytes_per_page, static_cast<uint32_t>(max_transmittable_bytes), max_transmittable_pages};
}

}  // namespace tt::tt_metal::unit_tests::dm
