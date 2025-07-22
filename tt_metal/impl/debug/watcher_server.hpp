// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <core_coord.hpp>
#include <stdint.h>
#include <umd/device/types/cluster_descriptor_types.h>
#include <string>

struct metal_SocDescriptor;

namespace tt::tt_metal {
class WatcherServer {
public:
    WatcherServer();
    ~WatcherServer();

    void init_devices();    // Always runs, puts watcher mailboxes in a default state
    void attach_devices();  // Start watcher server and attach all devices
    void detach_devices();  // Detach all devices and stop watcher server

    // Helper function to clear the log file, get the log file name
    void clear_log();
    std::string log_file_name();

    // Functions to register kernel ids & elf paths with the watcher server, which is responsible for dumping them to
    // files for use after the metal run is completed.
    int register_kernel(const std::string& name);
    void register_kernel_elf_paths(int id, std::vector<std::string>& paths);

    // Test-only helper functions
    bool killed_due_to_error();
    void set_killed_due_to_error_flag(bool val);
    std::string exception_message();
    void set_exception_message(const std::string& msg);
    int dump_count();

    // Helper to return the watcher mutex lock. Use this when writing to mailboxes if watcher server has started.
    std::unique_lock<std::mutex> get_lock();

    // Helper function for manually dumping watcher contents. TODO: remove when watcher_dump tool is removed.
    void isolated_dump(std::vector<chip_id_t>& device_ids);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;  // Pointer to implementation
};
}  // namespace tt::tt_metal
