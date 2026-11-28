// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/dram_geometry.hpp>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/impl/device/device_impl.hpp"

namespace tt::tt_metal::experimental {

namespace {

// (the concrete device: experimental::Device is a namespace here)
using ConcreteDevice = ::tt::tt_metal::Device;

const ConcreteDevice& local_device(
    const distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord) {
    TT_FATAL(
        mesh_device.impl().is_local(mesh_coord),
        "DRAM geometry: MeshCoordinate {} maps to a device this rank does not drive",
        mesh_coord);
    const ConcreteDevice* device = mesh_device.impl().get_device(mesh_coord);
    TT_FATAL(device != nullptr, "DRAM geometry: no device at MeshCoordinate {}", mesh_coord);
    return *device;
}

const metal_SocDescriptor& soc_of(const distributed::MeshDevice& mesh_device, const ConcreteDevice& device) {
    return MetalContext::instance(mesh_device.impl().get_context_id()).get_cluster().get_soc_desc(device.id());
}

}  // namespace

std::vector<DramBankTile> GetDramBankTiles(
    distributed::MeshDevice& mesh_device, const distributed::MeshCoordinate& mesh_coord, uint32_t bank) {
    const ConcreteDevice& device = local_device(mesh_device, mesh_coord);
    const auto& soc = soc_of(mesh_device, device);
    TT_FATAL(bank < soc.dram_bank_endpoint_coords.size(), "GetDramBankTiles: bank {} out of range", bank);
    std::vector<DramBankTile> tiles;
    for (uint32_t i = 0; i < soc.dram_bank_endpoint_coords[bank].size(); ++i) {
        const CoreCoord logical(bank, i);
        const CoreCoord translated = device.virtual_core_from_logical_core(logical, tt::CoreType::DRAM);
        const auto noc0 = soc.translate_coord_to(
            tt::umd::CoreCoord(translated.x, translated.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED),
            tt::CoordSystem::NOC0);
        tiles.push_back(DramBankTile{
            .logical_core = logical,
            .virtual_core = translated,
            .noc0_core = CoreCoord(noc0.x, noc0.y),
            .endpoint_noc_mask = soc.get_dram_endpoint_noc_mask(translated)});
    }
    return tiles;
}

CoreCoord GetWorkerNoc0Core(
    distributed::MeshDevice& mesh_device,
    const distributed::MeshCoordinate& mesh_coord,
    const CoreCoord& logical_core) {
    const ConcreteDevice& device = local_device(mesh_device, mesh_coord);
    return soc_of(mesh_device, device).get_physical_tensix_core_from_logical(logical_core);
}

}  // namespace tt::tt_metal::experimental
