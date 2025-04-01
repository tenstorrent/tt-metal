// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt_cluster.hpp>
#include <tt_stl/span.hpp>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <vector>

#include "assert.hpp"
#include "fmt/base.h"
#include "llrt.hpp"
#include "logger.hpp"
#include "metal_soc_descriptor.h"
#include <umd/device/tt_core_coordinates.h>
#include "utils.hpp"

void memset_l1(tt::stl::Span<const uint32_t> mem_vec, uint32_t chip_id, uint32_t start_addr) {
    // Utility function that writes a memory vector to L1 for all cores at a specific start address.
    const metal_SocDescriptor& sdesc = tt::Cluster::instance().get_soc_desc(chip_id);
    for (auto& worker_core : sdesc.get_cores(CoreType::TENSIX, CoordSystem::PHYSICAL)) {
        tt::llrt::write_hex_vec_to_core(chip_id, worker_core, mem_vec, start_addr);
    }
}

void memset_dram(std::vector<uint32_t> mem_vec, uint32_t chip_id, uint32_t start_addr) {
    // Utility function that writes a memory to all channels and subchannels at a specific start address.
    const metal_SocDescriptor& sdesc = tt::Cluster::instance().get_soc_desc(chip_id);
    for (uint32_t dram_view = 0; dram_view < sdesc.get_num_dram_views(); dram_view++) {
        for (uint32_t dram_src_subchannel_id = 0;
             dram_src_subchannel_id < sdesc.get_dram_cores().at(sdesc.get_channel_for_dram_view(dram_view)).size();
             dram_src_subchannel_id++) {
            tt::Cluster::instance().write_dram_vec(
                mem_vec, tt_target_dram{chip_id, dram_view, dram_src_subchannel_id}, start_addr);
        }
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

    string mem_type = argv[1];
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
