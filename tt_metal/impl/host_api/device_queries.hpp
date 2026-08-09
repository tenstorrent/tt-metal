// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {

/**
 * Returns whether Tenstorrent devices are in a Galaxy cluster.
 *
 * Return value: bool
 */
bool IsGalaxyCluster();

}  // namespace tt::tt_metal
