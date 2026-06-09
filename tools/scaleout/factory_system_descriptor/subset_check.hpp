// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>
#include <string>

#include <cabling_generator/cabling_generator.hpp>

namespace tt::scaleout_tools {

// Return the connections required by the skinny FSD that are absent from the good FSD
// (skinny \ good). An empty result means skinny is a subset of good: the good (currently
// working) topology still provides everything the skinny (minimum validated) topology needs,
// so a workload validated on skinny can run on the degraded cluster.
//
// Connections are matched by hostname (resolved from each FSD's own host_id index) and by
// canonicalized endpoint order, so the result is independent of host ordering within either FSD.
// Throws std::runtime_error on missing file, parse failure, or out-of-range host_id.
std::set<PhysicalChannelConnection> missing_skinny_connections(
    const std::string& skinny_fsd_path, const std::string& good_fsd_path);

}  // namespace tt::scaleout_tools
