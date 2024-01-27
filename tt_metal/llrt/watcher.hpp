// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_cluster.hpp"

namespace tt {
namespace llrt {

void watcher_init(int device_id,
                  std::function<CoreCoord ()>get_grid_size,
                  std::function<CoreCoord (CoreCoord)>worker_from_logical,
                  const std::function<const std::set<CoreCoord>& ()> &ethernet_cores,
                  const std::function<CoreCoord (CoreCoord)> &ethernet_from_logical
                  );
void watcher_attach(
    void *dev,
    int device_id,
    const std::function<CoreCoord ()>& get_grid_size,
    const std::function<CoreCoord (CoreCoord)>& worker_from_logical,
    const std::function<const std::set<CoreCoord> &()>& storage_only_cores,
    const std::function<const std::set<CoreCoord>& ()> &ethernet_cores,
    const std::function<CoreCoord (CoreCoord)> &ethernet_from_logical,
    const string& log_path
);
void watcher_detach(void *dev);

void watcher_sanitize_host_noc_read(const metal_SocDescriptor &soc_d, CoreCoord core, uint64_t addr, uint32_t len);
void watcher_sanitize_host_noc_write(const metal_SocDescriptor &soc_d, CoreCoord core, uint64_t addr, uint32_t len);

int watcher_register_kernel(const string& name);

// Check whether the watcher server has been killed due to an error detected.
bool watcher_server_killed_due_to_error();
// Function to clear this flag, so that non-watcher runs can continue as normal.
// TODO(dma): this doesn't currently clear the actual error codes on the device. Once watcher is
// moved out of llrt we can change this to watcher_clear_errors().
void watcher_server_clear_error_flag();

// Helper function to clear the watcher log file
void watcher_clear_log();

} // namespace llrt
} // namespace tt
