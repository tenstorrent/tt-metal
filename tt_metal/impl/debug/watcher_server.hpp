// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core_coord.hpp>
#include <stdint.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <string>

struct metal_SocDescriptor;

namespace tt {

void watcher_init(chip_id_t device_id);
void watcher_attach(chip_id_t device_id);
void watcher_detach(chip_id_t device_id);

void watcher_sanitize_host_noc_read(const metal_SocDescriptor& soc_d, CoreCoord core, uint64_t addr, uint32_t len);
void watcher_sanitize_host_noc_write(const metal_SocDescriptor& soc_d, CoreCoord core, uint64_t addr, uint32_t len);

int watcher_register_kernel(const std::string& name);
void watcher_register_kernel_elf_paths(int id, std::vector<std::string> paths);

// Helper functions for manually dumping watcher contents.
void watcher_dump();
void watcher_read_kernel_ids_from_file();

// Check whether the watcher server has been killed due to an error detected, and a function to set
// that flag. Used in test mode only.
bool watcher_server_killed_due_to_error();
void watcher_server_set_error_flag(bool val);

// Description of thrown exception from watcher server, used for testing purposes.
std::string get_watcher_exception_message();

// Helper function to clear the watcher log file
void watcher_clear_log();

// Helper function to get the current watcher log file name/path
std::string watcher_get_log_file_name();

// Helper function to get the current watcher dump count
int watcher_get_dump_count();

}  // namespace tt
