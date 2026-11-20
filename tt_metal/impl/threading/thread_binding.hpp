// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal::thread_binding {

// Picks the CPU for a pool's next worker from its NUMA node's CPUs: `preferred` if it is one of them, outside the pool
// and unused; else one the pool does not hold yet, then the least-used, round robin from `next` among ties. Counts the
// pick in workers_per_cpu and moves `next` past a round-robin pick.
uint32_t pick_cpu(
    const std::vector<uint32_t>& node_cpus,
    const std::vector<uint32_t>& pool_cpus,
    std::unordered_map<uint32_t, uint32_t>& workers_per_cpu,
    size_t& next,
    std::optional<uint32_t> preferred = std::nullopt);

// The CPUs of a sysfs CPU list such as "0,32" or "8-9"; empty if it does not parse.
std::vector<uint32_t> parse_cpu_list(std::string_view list);

// The other hardware thread of `cpu`'s core, if it has one.
std::optional<uint32_t> smt_sibling(uint32_t cpu);

}  // namespace tt::tt_metal::thread_binding
