/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_cluster.hpp"

namespace tt {
namespace llrt {

void watcher_attach(void *dev, tt_cluster *cluster, int pcie_slot, const std::function<CoreCoord ()>& get_grid_size, const std::function<CoreCoord (CoreCoord)>& worker_from_logical);
void watcher_detach(void *dev);

} // namespace llrt
} // namespace tt
