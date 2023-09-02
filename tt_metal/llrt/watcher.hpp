/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_cluster.hpp"

namespace tt {
namespace llrt {

void watcher_attach(void *dev, tt_cluster *cluster, int device_id, const std::function<CoreCoord ()>& get_grid_size, const std::function<CoreCoord (CoreCoord)>& worker_from_logical, const string& log_path);
void watcher_detach(void *dev);

void watcher_sanitize_host_noc_read(tt_SocDescriptor soc_d, CoreCoord core, uint64_t addr, uint32_t len);
void watcher_sanitize_host_noc_write(tt_SocDescriptor soc_d, CoreCoord core, uint64_t addr, uint32_t len);

} // namespace llrt
} // namespace tt
