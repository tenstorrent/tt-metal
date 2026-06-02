// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt-metalium/experimental/cluster_noc_helpers.hpp"

#include "impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

// One-line shims over Cluster::write_core / read_core / write_core_immediate /
// read_reg. The only reason these live in their own TU is to keep
// MetalContext (impl/) and tt_cluster (llrt/) out of the ttnn-nanobind
// include path. See cluster_noc_helpers.hpp for the rationale.

namespace tt::tt_metal::distributed {

void noc_write(std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::string_view data) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    cluster.write_core(data.data(), static_cast<std::uint32_t>(data.size()), target, addr);
}

std::vector<std::uint8_t> noc_read(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::uint32_t size) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    std::vector<std::uint8_t> buf(size);
    cluster.read_core(buf.data(), size, target, addr);
    return buf;
}

void noc_write_immediate(
    std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr, std::string_view data) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    cluster.write_core_immediate(data.data(), static_cast<std::uint32_t>(data.size()), target, addr);
}

std::uint32_t noc_read_reg_u32(std::uint32_t device_id, std::uint32_t x, std::uint32_t y, std::uint64_t addr) {
    const auto& cluster = MetalContext::instance().get_cluster();
    tt_cxy_pair target(device_id, x, y);
    std::uint32_t value = 0;
    cluster.read_reg(&value, target, addr);
    return value;
}

}  // namespace tt::tt_metal::distributed
