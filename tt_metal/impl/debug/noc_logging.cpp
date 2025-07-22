// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_logging.hpp"

#include <stdint.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <string>

#include "assert.hpp"
#include "core_coord.hpp"
#include "debug_helpers.hpp"
#include "hostdevcommon/dprint_common.h"
#include "llrt.hpp"
#include "impl/context/metal_context.hpp"
#include <tt-logger/tt-logger.hpp>
#include <umd/device/tt_soc_descriptor.h>

using namespace tt::tt_metal;

// 32 buckets to match the number of bits in uint32_t lengths on device
#define NOC_DATA_SIZE sizeof(uint32_t) * 8
using noc_data_t = std::array<uint64_t, NOC_DATA_SIZE>;

namespace tt {

static std::string logfile_path = "generated/noc_data/";
void PrintNocData(noc_data_t noc_data, const std::string& file_name) {
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    std::filesystem::path output_dir(rtoptions.get_root_dir() + logfile_path);
    std::filesystem::create_directories(output_dir);
    std::string filename = rtoptions.get_root_dir() + logfile_path + file_name;
    std::ofstream outfile(filename);

    for (uint32_t idx = 0; idx < NOC_DATA_SIZE; idx++) {
        uint64_t lower = 1UL << idx;
        uint64_t upper = 1UL << (idx + 1);
        outfile << fmt::format("[{},{}): {}\n", lower, upper, noc_data[idx]);
    }

    outfile.close();
}

void DumpCoreNocData(chip_id_t device_id, const CoreDescriptor& logical_core, noc_data_t& noc_data) {
    CoreCoord virtual_core =
        tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
            device_id, logical_core.coord, logical_core.type);
    for (int risc_id = 0; risc_id < GetNumRiscs(device_id, logical_core); risc_id++) {
        // Read out the DPRINT buffer, we stored our data in the "data field"
        uint64_t addr = GetDprintBufAddr(device_id, virtual_core, risc_id);
        auto from_dev = tt::llrt::read_hex_vec_from_core(device_id, virtual_core, addr, DPRINT_BUFFER_SIZE);
        DebugPrintMemLayout* l = reinterpret_cast<DebugPrintMemLayout*>(from_dev.data());
        uint32_t* data = reinterpret_cast<uint32_t*>(l->data);

        // Append the data for this core to existing data
        for (int idx = 0; idx < NOC_DATA_SIZE; idx++) {
            noc_data[idx] += data[idx];
        }
    }
}

void DumpDeviceNocData(chip_id_t device_id, noc_data_t& noc_data, noc_data_t& dispatch_noc_data) {
    // Need to treat dispatch cores and normal cores separately, so keep track of which cores are dispatch.
    CoreDescriptorSet dispatch_cores = GetDispatchCores(device_id);

    // Now go through all cores on the device, and dump noc data for them.
    CoreDescriptorSet all_cores = GetAllCores(device_id);
    for (const CoreDescriptor& logical_core : all_cores) {
        if (dispatch_cores.count(logical_core)) {
            DumpCoreNocData(device_id, logical_core, dispatch_noc_data);
        } else {
            DumpCoreNocData(device_id, logical_core, noc_data);
        }
    }
}

void DumpNocData(const std::vector<chip_id_t>& devices) {
    // Skip if feature is not enabled
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_record_noc_transfers()) {
        return;
    }

    noc_data_t noc_data = {}, dispatch_noc_data = {};
    for (chip_id_t device_id : devices) {
        log_info(tt::LogMetal, "Dumping noc data for Device {}...", device_id);
        DumpDeviceNocData(device_id, noc_data, dispatch_noc_data);
    }

    PrintNocData(noc_data, "noc_data.txt");
    PrintNocData(dispatch_noc_data, "dispatch_noc_data.txt");
}

void ClearNocData(chip_id_t device_id) {
    // Skip if feature is not enabled
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_record_noc_transfers()) {
        return;
    }

    // This feature is incomatible with dprint since they share memory space
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().rtoptions().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint) ==
            false,
        "NOC transfer recording is incompatible with DPRINT");

    CoreDescriptorSet all_cores = GetAllCores(device_id);
    for (const CoreDescriptor& logical_core : all_cores) {
        CoreCoord virtual_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_virtual_coordinate_from_logical_coordinates(
                device_id, logical_core.coord, logical_core.type);
        for (int risc_id = 0; risc_id < GetNumRiscs(device_id, logical_core); risc_id++) {
            uint64_t addr = GetDprintBufAddr(device_id, virtual_core, risc_id);
            std::vector<uint32_t> initbuf = std::vector<uint32_t>(DPRINT_BUFFER_SIZE / sizeof(uint32_t), 0);
            tt::llrt::write_hex_vec_to_core(device_id, virtual_core, initbuf, addr);
        }
    }
}

}  // namespace tt
