// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <gtest/gtest.h>
#include <umd/device/types/core_coordinates.hpp>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "mesh_device.hpp"

namespace tt::tt_metal {

struct TestBufferConfig {
    uint32_t num_pages{};
    uint32_t page_size{};
    BufferType buftype{0};
    std::optional<BufferShardingArgs> sharding_args = std::nullopt;  // only used for sharded buffers
};

struct CoreCoordsL1 {
    uint32_t my_x;
    uint32_t my_y;
    uint32_t my_logical_x;
    uint32_t my_logical_y;
    uint32_t my_sub_device_x;
    uint32_t my_sub_device_y;
};

static_assert(sizeof(CoreCoordsL1) == 24);  // Must match kernel

inline std::vector<uint32_t> generate_arange_vector(uint32_t size_bytes, uint32_t start = 0) {
    TT_FATAL(size_bytes % sizeof(uint32_t) == 0, "Error");
    std::vector<uint32_t> src(size_bytes / sizeof(uint32_t), 0);

    for (uint32_t i = 0; i < src.size(); i++) {
        src.at(i) = start + i;
    }
    return src;
}

inline bool does_device_have_active_eth_cores(const IDevice* device) {
    return !(device->get_active_ethernet_cores(true).empty());
}

inline bool does_device_have_idle_eth_cores(const IDevice* device) {
    return !(device->get_inactive_ethernet_cores().empty());
}

inline std::pair<std::vector<uint32_t>, std::vector<uint32_t>> create_runtime_args(
    const uint32_t num_unique_rt_args,
    const uint32_t num_common_rt_args,
    const uint32_t unique_base,
    const uint32_t common_base) {
    TT_FATAL(
        num_unique_rt_args + num_common_rt_args <= tt::tt_metal::max_runtime_args,
        "Number of unique runtime args and common runtime args exceeds the maximum limit of {} runtime args",
        tt::tt_metal::max_runtime_args);

    std::vector<uint32_t> common_rt_args;
    common_rt_args.reserve(num_common_rt_args);
    for (uint32_t i = 0; i < num_common_rt_args; i++) {
        common_rt_args.push_back(common_base + i);
    }

    std::vector<uint32_t> unique_rt_args;
    unique_rt_args.reserve(num_unique_rt_args);
    for (uint32_t i = 0; i < num_unique_rt_args; i++) {
        unique_rt_args.push_back(unique_base + i);
    }

    return std::make_pair(unique_rt_args, common_rt_args);
}

// Create randomly sized pair of unique and common runtime args vectors, with careful not to exceed max between the two.
// Optionally force the max size for one of the vectors.
inline std::pair<std::vector<uint32_t>, std::vector<uint32_t>> create_runtime_args(
    const bool force_max_size = false, const uint32_t unique_base = 0, const uint32_t common_base = 100) {
    uint32_t num_rt_args_unique = rand() % (tt::tt_metal::max_runtime_args + 1);
    uint32_t num_rt_args_common = num_rt_args_unique < tt::tt_metal::max_runtime_args
                                      ? rand() % (tt::tt_metal::max_runtime_args - num_rt_args_unique + 1)
                                      : 0;

    if (force_max_size) {
        if (rand() % 2) {
            num_rt_args_unique = tt::tt_metal::max_runtime_args;
            num_rt_args_common = 0;
        } else {
            num_rt_args_common = tt::tt_metal::max_runtime_args;
            num_rt_args_unique = 0;
        }
    }

    log_trace(
        tt::LogTest,
        "{} - num_rt_args_unique: {} num_rt_args_common: {} force_max_size: {}",
        __FUNCTION__,
        num_rt_args_unique,
        num_rt_args_common,
        force_max_size);

    return create_runtime_args(num_rt_args_unique, num_rt_args_common, unique_base, common_base);
}

inline void verify_kernel_coordinates(
    HalProgrammableCoreType hal_core_type,
    const CoreRangeSet& cr_set,
    const tt::tt_metal::distributed::MeshDevice* mesh_device,
    tt::tt_metal::SubDeviceId sub_device_id,
    uint32_t cb_addr) {
    for (const auto& device : mesh_device->get_devices()) {
        tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(device->id());
    }

    CoreType core_type = hal_core_type == HalProgrammableCoreType::TENSIX     ? CoreType::WORKER
                         : hal_core_type == HalProgrammableCoreType::IDLE_ETH ? CoreType::IDLE_ETH
                                                                              : CoreType::ETH;

    const auto& sub_device_origin = mesh_device->worker_cores(hal_core_type, sub_device_id).bounding_box().start_coord;
    for (const auto& cr : cr_set.ranges()) {
        for (auto core = cr.begin(); core != cr.end(); ++core) {
            const auto& logical_coord = *core;
            const auto& virtual_coord = mesh_device->virtual_core_from_logical_core(logical_coord, core_type);
            CoreCoord relative_coord{logical_coord.x - sub_device_origin.x, logical_coord.y - sub_device_origin.y};
            for (const auto& device : mesh_device->get_devices()) {
                auto read_coords_raw = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                    device->id(), virtual_coord, cb_addr, sizeof(tt::tt_metal::CoreCoordsL1));
                auto read_coords = reinterpret_cast<volatile tt::tt_metal::CoreCoordsL1*>(read_coords_raw.data());

                EXPECT_EQ(read_coords->my_logical_x, logical_coord.x) << "Logical X";
                EXPECT_EQ(read_coords->my_logical_y, logical_coord.y) << "Logical Y";

                EXPECT_EQ(read_coords->my_sub_device_x, (relative_coord).x) << "SubDevice Logical X";
                EXPECT_EQ(read_coords->my_sub_device_y, (relative_coord).y) << "SubDevice Logical Y";
            }
        }
    }
}

}  // namespace tt::tt_metal
