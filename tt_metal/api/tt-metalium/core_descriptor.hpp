// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <umd/device/types/arch.h>                      // tt::ARCH
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>

namespace tt {

struct core_descriptor_t {
    CoreCoord compute_grid_size;
    std::vector<RelativeCoreCoord> relative_compute_cores;
    std::vector<RelativeCoreCoord> relative_storage_cores;
    std::optional<uint32_t> storage_core_bank_size = std::nullopt;
    std::vector<RelativeCoreCoord> relative_dispatch_cores;
    std::vector<CoreCoord> logical_compute_cores;
    std::vector<CoreCoord> logical_storage_cores;
    std::vector<CoreCoord> logical_dispatch_cores;
};

inline const std::string& get_product_name(tt::ARCH arch, uint32_t num_harvested_on_axis) {
    const static std::map<tt::ARCH, std::map<uint32_t, std::string>> product_name = {
        {tt::ARCH::GRAYSKULL, {{0, "E150"}, {2, "E75"}}},
        {tt::ARCH::WORMHOLE_B0, {{0, "galaxy"}, {1, "nebula_x1"}, {2, "nebula_x2"}}},
        {tt::ARCH::BLACKHOLE, {{0, "unharvested"}, {1, "1xharvested"}, {2, "2xharvested"}}}};

    return product_name.at(arch).at(num_harvested_on_axis);
}

const core_descriptor_t& get_core_descriptor_config(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config);

const std::tuple<uint32_t, CoreRange>& get_physical_worker_grid_config(
    chip_id_t chip, uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config);

std::optional<uint32_t> get_storage_core_bank_size(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config);

inline const std::vector<CoreCoord>& get_logical_storage_cores(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    const core_descriptor_t& core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_config);
    return core_desc.logical_storage_cores;
}

inline const CoreCoord& get_compute_grid_size(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    const core_descriptor_t& core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_config);
    return core_desc.compute_grid_size;
}

inline const std::vector<CoreCoord>& get_logical_compute_cores(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    const core_descriptor_t& core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_config);
    return core_desc.logical_compute_cores;
}

inline const std::vector<CoreCoord>& get_logical_dispatch_cores(
    chip_id_t device_id, const uint8_t num_hw_cqs, const tt_metal::DispatchCoreConfig& dispatch_core_config) {
    const core_descriptor_t& core_desc = get_core_descriptor_config(device_id, num_hw_cqs, dispatch_core_config);
    return core_desc.logical_dispatch_cores;
}

}  // namespace tt
