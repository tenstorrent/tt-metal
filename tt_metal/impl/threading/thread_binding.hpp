// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tt::tt_metal::thread_binding {

// Picks the CPU for a pool's next worker from its NUMA node's CPUs: one the pool does not hold yet, then the
// least-used, round robin from `next` among ties. Counts the pick in workers_per_cpu and moves `next` past it.
uint32_t pick_cpu(
    const std::vector<uint32_t>& node_cpus,
    const std::vector<uint32_t>& pool_cpus,
    std::unordered_map<uint32_t, uint32_t>& workers_per_cpu,
    size_t& next);

// Orders CPUs core by core, a core's hardware threads together (core_of: package, then core id), so that picks in
// that order fill whole cores. CPUs of one core keep their order; if any CPU's core is unknown (-1), all do.
void order_by_core(std::vector<uint32_t>& cpus, const std::function<std::pair<long, long>(uint32_t)>& core_of);

}  // namespace tt::tt_metal::thread_binding
