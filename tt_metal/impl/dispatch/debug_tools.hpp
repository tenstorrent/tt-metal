// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <fstream>

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "command_queue_interface.hpp"

namespace internal {


void wait_for_program_vector_to_arrive_and_compare_to_host_program_vector(const char *DISPATCH_MAP_DUMP, tt::tt_metal::Device * device);
void match_device_program_data_with_host_program_data(const char* host_file, const char* device_file);

// Dumps host-side CQ data into files.
void dump_cqs(
    std::ofstream &cq_file, std::ofstream &iq_file, SystemMemoryManager &sysmem_manager, bool dump_raw_data = false);

} // end namespace internal
