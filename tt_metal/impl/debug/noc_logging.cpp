// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_logging.hpp"

#include <cstdint>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include <tt_stl/assert.hpp>
#include <tt_stl/fmt.hpp>
#include "context/context_types.hpp"
#include "core_coord.hpp"
#include "debug_helpers.hpp"
#include "hostdevcommon/dprint_common.h"
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <impl/dispatch/dispatch_core_manager.hpp>
#include <llrt/tt_cluster.hpp>
#include "common/filesystem_utils.hpp"

using namespace tt::tt_metal;

// 32 buckets to match the number of bits in uint32_t lengths on device
#define NOC_DATA_SIZE (sizeof(uint32_t) * 8)
using noc_data_t = std::array<uint64_t, NOC_DATA_SIZE>;

namespace tt {

constexpr auto logfile_path = "generated/noc_data/";
void PrintNocData(MetalEnvImpl& env, noc_data_t noc_data, const std::string& file_name) {
    const auto& rtoptions = env.get_rtoptions();
    std::filesystem::path output_dir(rtoptions.get_logs_dir() + logfile_path);
    if (!tt::filesystem::safe_create_directories(output_dir)) {
        log_warning(tt::LogMetal, "NocLogging: failed to create output directory {}", output_dir.string());
    }
    std::string filename = rtoptions.get_logs_dir() + logfile_path + file_name;
    std::ofstream outfile(filename);

    for (uint32_t idx = 0; idx < NOC_DATA_SIZE; idx++) {
        uint64_t lower = 1UL << idx;
        uint64_t upper = 1UL << (idx + 1);
        outfile << fmt::format("[{},{}): {}\n", lower, upper, noc_data[idx]);
    }

    outfile.close();
}

void DumpCoreNocData(
    MetalEnvImpl& env, ChipId device_id, const umd::CoreDescriptor& logical_core, noc_data_t& noc_data) {
    CoreCoord virtual_core = env.get_cluster().get_virtual_coordinate_from_logical_coordinates(
        device_id, logical_core.coord, logical_core.type);
    uint32_t num_processors = env.get_hal().get_num_risc_processors(llrt::get_core_type(device_id, virtual_core));
    for (int risc_id = 0; risc_id < num_processors; risc_id++) {
        // Read out the DPRINT buffer, we stored our data in the "data field"
        uint64_t addr = GetDprintBufAddr(device_id, virtual_core, risc_id);
        auto from_dev = env.get_cluster().read_core(device_id, virtual_core, addr, DPRINT_BUFFER_SIZE);
        DebugPrintMemLayout* l = reinterpret_cast<DebugPrintMemLayout*>(from_dev.data());
        uint32_t* data = reinterpret_cast<uint32_t*>(l->data);

        // Append the data for this core to existing data
        for (int idx = 0; idx < NOC_DATA_SIZE; idx++) {
            noc_data[idx] += data[idx];
        }
    }
}

void DumpDeviceNocData(
    tt_metal::MetalEnvImpl& env,
    ChipId device_id,
    noc_data_t& noc_data,
    noc_data_t& dispatch_noc_data,
    uint8_t num_hw_cqs,
    const DispatchCoreConfig& dispatch_core_config) {
    // Need to treat dispatch cores and normal cores separately, so keep track of which cores are dispatch.
    auto& cluster = env.get_cluster();
    auto& control_plane = env.get_control_plane();
    CoreDescriptorSet dispatch_cores = GetDispatchCores(env, device_id, num_hw_cqs, dispatch_core_config);

    // Now go through all cores on the device, and dump noc data for them.
    CoreDescriptorSet all_cores = GetAllCores(cluster, control_plane, device_id);
    for (const umd::CoreDescriptor& logical_core : all_cores) {
        if (dispatch_cores.contains(logical_core)) {
            DumpCoreNocData(env, device_id, logical_core, dispatch_noc_data);
        } else {
            DumpCoreNocData(env, device_id, logical_core, noc_data);
        }
    }
}

void DumpNocData(
    tt_metal::MetalEnvImpl& env,
    const std::vector<ChipId>& devices,
    uint8_t num_hw_cqs,
    const DispatchCoreConfig& dispatch_core_config) {
    // Skip if feature is not enabled
    if (!env.get_rtoptions().get_record_noc_transfers()) {
        return;
    }

    noc_data_t noc_data = {}, dispatch_noc_data = {};
    for (ChipId device_id : devices) {
        log_info(tt::LogMetal, "Dumping noc data for Device {}...", device_id);
        DumpDeviceNocData(env, device_id, noc_data, dispatch_noc_data, num_hw_cqs, dispatch_core_config);
    }

    PrintNocData(env, noc_data, "noc_data.txt");
    PrintNocData(env, dispatch_noc_data, "dispatch_noc_data.txt");
}

void ClearNocData(MetalEnvImpl& env, ChipId device_id) {
    // Skip if feature is not enabled
    if (!env.get_rtoptions().get_record_noc_transfers()) {
        return;
    }

    // This feature is incomatible with dprint since they share memory space
    TT_FATAL(
        env.get_rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint) == false,
        "NOC transfer recording is incompatible with DPRINT");

    auto& cluster = env.get_cluster();
    auto& control_plane = env.get_control_plane();
    CoreDescriptorSet all_cores = GetAllCores(cluster, control_plane, device_id);
    for (const umd::CoreDescriptor& logical_core : all_cores) {
        CoreCoord virtual_core =
            cluster.get_virtual_coordinate_from_logical_coordinates(device_id, logical_core.coord, logical_core.type);
        uint32_t num_processors = env.get_hal().get_num_risc_processors(llrt::get_core_type(device_id, virtual_core));
        for (int risc_id = 0; risc_id < num_processors; risc_id++) {
            uint64_t addr = GetDprintBufAddr(device_id, virtual_core, risc_id);
            std::vector<uint32_t> initbuf = std::vector<uint32_t>(DPRINT_BUFFER_SIZE / sizeof(uint32_t), 0);
            cluster.write_core(device_id, virtual_core, initbuf, addr);
        }
    }
}

}  // namespace tt
