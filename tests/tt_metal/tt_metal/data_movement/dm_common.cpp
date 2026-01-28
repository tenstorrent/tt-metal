// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dm_common.hpp"
#include "device_fixture.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tuple>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal::unit_tests::dm {

// Hardcoded constants for L1 memory base address and size
// constexpr uint64_t HARDCODED_L1_MEMORY_BASE_ADDRESS = 1024 * 256;  // 256 KB
// constexpr uint64_t HARDCODED_L1_MEMORY_SIZE_BYTES = 1024 * 1024;   // 1 MB

uint32_t runtime_host_id = 0;

// Static function for internal use only
static uint32_t obtain_page_size_bytes(ARCH arch) { return (arch == ARCH::BLACKHOLE) ? 64 : 32; }

L1AddressInfo get_l1_address_and_size(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CoreCoord& core_coord) {
    // Obtaining L1 address and size for a specific core //
    const IDevice* device = mesh_device->impl().get_device(0);

    CoreCoord physical_core = device->worker_core_from_logical_core(core_coord);

    uint64_t core_l1_base_address = device->get_dev_addr(physical_core, HalL1MemAddrType::DEFAULT_UNRESERVED);
    uint64_t core_l1_size = device->get_dev_size(physical_core, HalL1MemAddrType::DEFAULT_UNRESERVED);

    return {core_l1_base_address, core_l1_size};

    // Obtaining hardcoded values for L1 address and size //

    // return {HARDCODED_L1_MEMORY_BASE_ADDRESS, HARDCODED_L1_MEMORY_SIZE_BYTES};
}

DramAddressInfo get_dram_address_and_size() {
    // Obtaining DRAM address and size //

    auto dram_base_address = MetalContext::instance().hal().get_dev_addr(HalDramMemAddrType::UNRESERVED);
    uint32_t dram_size = MetalContext::instance().hal().get_dev_size(HalDramMemAddrType::UNRESERVED);

    return {dram_base_address, dram_size};
}

std::tuple<uint32_t, uint32_t, uint32_t> compute_physical_constraints(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const IDevice* device = mesh_device->impl().get_device(0);
    ARCH arch = device->arch();

    // Use core {0,0} as representative core for computing physical constraints
    CoreCoord representative_core = {0, 0};
    auto [_, max_transmittable_bytes] = get_l1_address_and_size(mesh_device, representative_core);
    uint32_t bytes_per_page = obtain_page_size_bytes(arch);
    uint32_t max_transmittable_pages = max_transmittable_bytes / bytes_per_page;

    return {bytes_per_page, static_cast<uint32_t>(max_transmittable_bytes), max_transmittable_pages};
}

}  // namespace tt::tt_metal::unit_tests::dm
