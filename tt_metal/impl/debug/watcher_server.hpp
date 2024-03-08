// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/device/device.hpp"

namespace tt {

void watcher_init(Device *device);
void watcher_attach(Device *device, const string& log_path);
void watcher_detach(Device *dev);

void watcher_sanitize_host_noc_read(const metal_SocDescriptor &soc_d, CoreCoord core, uint64_t addr, uint32_t len);
void watcher_sanitize_host_noc_write(const metal_SocDescriptor &soc_d, CoreCoord core, uint64_t addr, uint32_t len);

int watcher_register_kernel(const string& name);

// Check whether the watcher server has been killed due to an error detected.
bool watcher_server_killed_due_to_error();
// Function to set this flag to true/false, so that non-watcher runs can continue as normal when set to false.
// TODO(dma): this doesn't currently clear the actual error codes on the device. Once watcher is
// moved out of llrt we can change this to watcher_clear_errors().
void watcher_server_set_error_flag(bool val);

// Helper function to clear the watcher log file
void watcher_clear_log();

// Helper function to get the current watcher log file name/path
string watcher_get_log_file_name();

} // namespace tt
