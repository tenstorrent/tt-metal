// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <memory>

#include "host_api.hpp"
#include "tt_metal.hpp"
#include "lite_fabric.hpp"

#include "kernel_types.hpp"
#include "program.hpp"
#include "rtoptions.hpp"
#include "tt_cluster.hpp"
#include "assert.hpp"
#include "context/metal_context.hpp"
#include "hal_types.hpp"
#include "jit_build/build_env_manager.hpp"
#include "impl/kernels/kernel_impl.hpp"
#include "core_coord.hpp"
#include "data_types.hpp"
#include <umd/device/types/xy_pair.h>
#include <tt-metalium/control_plane.hpp>

namespace lite_fabric {

// Depth of 1
struct TunnelDescriptor {
    chip_id_t mmio_id{0xffffffff};
    CoreCoord mmio_core_virtual{-1, -1};
    CoreCoord mmio_core_logical{-1, -1};
    tt::tt_metal::KernelHandle mmio_kernel = 0;

    chip_id_t connected_id{0xffffffff};
    CoreCoord connected_core_virtual{-1, -1};
    CoreCoord connected_core_logical{-1, -1};

    tt_cxy_pair mmio_cxy_virtual() const { return {mmio_id, mmio_core_virtual}; }

    tt_cxy_pair connected_cxy_virtual() const { return {connected_id, connected_core_virtual}; }
};

struct SystemDescriptor {
    std::map<chip_id_t, uint32_t> enabled_eth_channels;
    std::vector<TunnelDescriptor> tunnels_from_mmio;
};

uint32_t GetEthChannelMask(chip_id_t device_id);

SystemDescriptor GetSystemDescriptor2Devices(chip_id_t mmio_device_id, chip_id_t connected_device_id);

uint32_t GetLocalInitAddr(std::shared_ptr<tt::tt_metal::Kernel> kernel);

std::pair<const ll_api::memory&, uint32_t> GetBinaryMetadata(
    uint32_t build_id, tt::tt_metal::Program& pgm, tt::tt_metal::KernelHandle kernel_handle);

std::pair<const ll_api::memory&, uint32_t> GetBinaryMetadata(std::string path, const tt::tt_metal::Hal& hal);

void SetResetState(tt::Cluster& cluster, tt_cxy_pair virtual_core, bool assert_reset);

void SetResetState(tt::Cluster& cluster, const SystemDescriptor& desc, bool assert_reset);

void SetPC(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t pc_addr, uint32_t pc_val);

void SetPC(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t pc_addr, uint32_t pc_val);

void WaitForState(tt::Cluster& cluster, tt_cxy_pair virtual_core, uint32_t addr, lite_fabric::InitState state);

void WaitForState(tt::Cluster& cluster, const SystemDescriptor& desc, uint32_t addr, lite_fabric::InitState state);

std::unique_ptr<tt::tt_metal::Program> LaunchLiteFabricWithMetal(
    std::map<chip_id_t, tt::tt_metal::IDevice*>& devices, const SystemDescriptor& desc);

void LaunchLiteFabric(
    tt::Cluster& cluster,
    const tt::tt_metal::Hal& hal,
    const SystemDescriptor& desc,
    const std::filesystem::path& elf_path);

void LaunchLiteFabric(tt::Cluster& cluster, const tt::tt_metal::Hal& hal, const SystemDescriptor& desc);

void TerminateLiteFabricWithMetal(tt::Cluster& cluster, const SystemDescriptor& desc);
void TerminateLiteFabric(tt::Cluster& cluster, const SystemDescriptor& desc);

}  // namespace lite_fabric
