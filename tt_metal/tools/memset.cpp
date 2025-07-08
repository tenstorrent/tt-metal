// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "impl/context/metal_context.hpp"
#include <tt_stl/span.hpp>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "assert.hpp"
#include "fmt/base.h"
#include "llrt.hpp"
#include <tt-logger/tt-logger.hpp>
#include "metal_soc_descriptor.h"
#include <umd/device/tt_core_coordinates.h>

void memset_l1(tt::stl::Span<const uint32_t> mem_vec, uint32_t chip_id, uint32_t start_addr) {
    // Utility function that writes a memory vector to L1 for all cores at a specific start address.
    const metal_SocDescriptor& sdesc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip_id);
    for (auto& worker_core : sdesc.get_cores(CoreType::TENSIX, CoordSystem::PHYSICAL)) {
        tt::llrt::write_hex_vec_to_core(chip_id, worker_core, mem_vec, start_addr);
    }
}

void memset_dram(std::vector<uint32_t> mem_vec, uint32_t chip_id, uint32_t start_addr) {
    // Utility function that writes a memory to all channels and subchannels at a specific start address.
    const metal_SocDescriptor& sdesc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(chip_id);
    for (uint32_t dram_view = 0; dram_view < sdesc.get_num_dram_views(); dram_view++) {
        tt::tt_metal::MetalContext::instance().get_cluster().write_dram_vec(
            mem_vec.data(), mem_vec.size() * sizeof(uint32_t), chip_id, dram_view, start_addr);
    }
}

int main(int argc, char* argv[]) {
    int num_user_provided_arguments = argc - 1;

    if (std::getenv("RUNNING_FROM_PYTHON") == nullptr) {
        log_warning(tt::LogDevice, "It is recommended to run this script from 'memset.py'");
        sleep(2);
    }

    // Since memset.py would always correctly launch
    TT_FATAL(
        num_user_provided_arguments == 5,
        "Invalid number of supplied arguments. For an example of usage on how to launch the program, read "
        "tools/README.md. "
        "If you don't want to launch from memset.py, the order of arguments supplied is specified by the command list "
        "in memset.py.");

    std::string mem_type = argv[1];
    uint32_t chip_id = std::stoi(argv[2]);
    uint32_t start_addr = std::stoi(argv[3]);
    uint32_t size = std::stoi(argv[4]);
    uint32_t val = std::stoi(argv[5]);

    std::vector<uint32_t> mem_vec(size, val);  // Create a vector all filled with user-chosen values

    if (mem_type == "dram") {
        memset_dram(mem_vec, chip_id, start_addr);
    } else {  // Write to L1
        memset_l1(mem_vec, chip_id, start_addr);
    }

    return 0;
}
