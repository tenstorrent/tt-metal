// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "noc_logging.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
// #include <iomanip>
#include <set>

#include "debug_helpers.hpp"
#include "hostdevcommon/dprint_common.h"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/llrt.hpp"

using namespace tt::tt_metal;

// 32 buckets to match the number of bits in uint32_t lengths on device
#define NOC_DATA_SIZE sizeof(uint32_t) * 8
using noc_data_t = std::array<uint64_t, NOC_DATA_SIZE>;

namespace tt {

static string logfile_path = "generated/noc_data/";
void PrintNocData(noc_data_t noc_data, const string& file_name) {
    std::filesystem::path output_dir(tt::llrt::RunTimeOptions::get_instance().get_root_dir() + logfile_path);
    std::filesystem::create_directories(output_dir);
    std::string filename = tt::llrt::RunTimeOptions::get_instance().get_root_dir() + logfile_path + file_name;
    std::ofstream outfile(filename);

    for (uint32_t idx = 0; idx < NOC_DATA_SIZE; idx++) {
        uint64_t lower = 1UL << idx;
        uint64_t upper = 1UL << (idx + 1);
        outfile << fmt::format("[{},{}): {}\n", lower, upper, noc_data[idx]);
    }

    outfile.close();
}

void DumpCoreNocData(Device* device, const CoreDescriptor& logical_core, noc_data_t& noc_data) {
    CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
    for (int risc_id = 0; risc_id < GetNumRiscs(logical_core); risc_id++) {
        // Read out the DPRINT buffer, we stored our data in the "data field"
        uint64_t addr = GetDprintBufAddr(device, phys_core, risc_id);
        auto from_dev = tt::llrt::read_hex_vec_from_core(device->id(), phys_core, addr, DPRINT_BUFFER_SIZE);
        DebugPrintMemLayout* l = reinterpret_cast<DebugPrintMemLayout*>(from_dev.data());
        uint32_t* data = reinterpret_cast<uint32_t*>(l->data);

        // Append the data for this core to existing data
        for (int idx = 0; idx < NOC_DATA_SIZE; idx++) {
            noc_data[idx] += data[idx];
        }
    }
}

void DumpDeviceNocData(Device* device, noc_data_t& noc_data, noc_data_t& dispatch_noc_data) {
    // Need to treat dispatch cores and normal cores separately, so keep track of which cores are dispatch.
    CoreDescriptorSet dispatch_cores = GetDispatchCores(device);

    // Now go through all cores on the device, and dump noc data for them.
    CoreDescriptorSet all_cores = GetAllCores(device);
    for (const CoreDescriptor& logical_core : all_cores) {
        if (dispatch_cores.count(logical_core)) {
            DumpCoreNocData(device, logical_core, dispatch_noc_data);
        } else {
            DumpCoreNocData(device, logical_core, noc_data);
        }
    }
}

void DumpNocData(const std::vector<Device*>& devices) {
    // Skip if feature is not enabled
    if (!tt::llrt::RunTimeOptions::get_instance().get_record_noc_transfers()) {
        return;
    }

    noc_data_t noc_data = {}, dispatch_noc_data = {};
    for (Device* device : devices) {
        log_info("Dumping noc data for Device {}...", device->id());
        DumpDeviceNocData(device, noc_data, dispatch_noc_data);
    }

    PrintNocData(noc_data, "noc_data.txt");
    PrintNocData(dispatch_noc_data, "dispatch_noc_data.txt");
}

void ClearNocData(Device* device) {
    // Skip if feature is not enabled
    if (!tt::llrt::RunTimeOptions::get_instance().get_record_noc_transfers()) {
        return;
    }

    // This feature is incomatible with dprint since they share memory space
    TT_FATAL(
        tt::llrt::RunTimeOptions::get_instance().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint) == false,
        "NOC transfer recording is incompatible with DPRINT");

    CoreDescriptorSet all_cores = GetAllCores(device);
    for (const CoreDescriptor& logical_core : all_cores) {
        CoreCoord phys_core = device->virtual_core_from_logical_core(logical_core.coord, logical_core.type);
        for (int risc_id = 0; risc_id < GetNumRiscs(logical_core); risc_id++) {
            uint64_t addr = GetDprintBufAddr(device, phys_core, risc_id);
            std::vector<uint32_t> initbuf = std::vector<uint32_t>(DPRINT_BUFFER_SIZE / sizeof(uint32_t), 0);
            tt::llrt::write_hex_vec_to_core(device->id(), phys_core, initbuf, addr);
        }
    }
}

}  // namespace tt
