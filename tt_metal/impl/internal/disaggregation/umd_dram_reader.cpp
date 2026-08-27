// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Reads a KV chunk's DRAM bytes over a bare tt::umd::Cluster and reads through on-demand TLB windows.
// This lets a process that has NOT opened the mesh (an external scheduler / the prefill_producer)
// read a live server's KV cache CONCURRENTLY with the runner that owns the chips via CreateDevice.
// It is a direct port of the migration worker's device_io.cpp read path
// (tt-llm-engine disaggregation/migration/src/worker/device_io.cpp). The chip is selected by
// ASIC unique_id (the caller resolves fabric_node -> unique_id from the runner's device-map
// sidecar), so no MetalContext / ControlPlane is touched (either would start_device the chips).
// ---------------------------------------------------------------------------

#include "tt_metal/impl/internal/disaggregation/noc_addr.hpp"

#include <tt_stl/assert.hpp>

#include "llrt/metal_soc_descriptor.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <umd/device/cluster.hpp>
#include <umd/device/cluster_descriptor.hpp>
#include <umd/device/soc_descriptor.hpp>
#include <umd/device/types/arch.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include <umd/device/types/core_coordinates.hpp>
#pragma GCC diagnostic pop

namespace tt::tt_metal::internal::disaggregation {

namespace {

// Per-DRAM-view TRANSLATED core + intra-chip byte offset (mirrors device_io.cpp DramViewInfo).
struct UmdKvDramView {
    tt::umd::CoreCoord core;
    uint64_t address_offset;
};

// Metal SOC-descriptor YAML for an arch (mirrors device_io.cpp / Cluster::get_metal_desc_from_tt_desc).
std::string umd_metal_soc_yaml_path(tt::ARCH arch) {
    std::string root;
    for (const char* var : {"TT_METAL_RUNTIME_ROOT", "TT_METAL_HOME"}) {
        if (const char* v = std::getenv(var); v != nullptr && *v != '\0') {
            root = v;
            break;
        }
    }
    if (root.empty()) {
        root = std::filesystem::current_path().string();
    }
    if (root.back() != '/') {
        root.push_back('/');
    }
    root += "tt_metal/soc_descriptors/";
    switch (arch) {
        case tt::ARCH::WORMHOLE_B0: return root + "wormhole_b0_80_arch.yaml";
        case tt::ARCH::BLACKHOLE: return root + "blackhole_140_arch.yaml";
        case tt::ARCH::QUASAR: return root + "quasar_32_arch.yaml";
        default: TT_THROW("read_dram_umd: unsupported arch for dram_view mapping");
    }
}

// One bare UMD cluster per process — constructed WITHOUT start_device() (no CHIP_IN_USE lock),
// so it coexists with the server that owns the chips. Built lazily, shared for the process lifetime.
std::shared_ptr<tt::umd::Cluster> umd_read_cluster() {
    static std::shared_ptr<tt::umd::Cluster> cluster = std::make_shared<tt::umd::Cluster>();
    return cluster;
}

// ASIC unique_id -> UMD ChipId (from the cluster descriptor); built once.
tt::ChipId umd_chip_for_unique_id(uint64_t unique_id) {
    static const std::unordered_map<uint64_t, tt::ChipId> uid_to_chip = [] {
        std::unordered_map<uint64_t, tt::ChipId> m;
        for (const auto& [chip, uid] : umd_read_cluster()->get_cluster_description()->get_chip_unique_ids()) {
            m.emplace(uid, chip);
        }
        return m;
    }();
    auto it = uid_to_chip.find(unique_id);
    TT_FATAL(
        it != uid_to_chip.end(),
        "read_dram_umd: no visible chip with ASIC unique_id {} (0x{:x}); check TT_VISIBLE_DEVICES / device map",
        unique_id,
        unique_id);
    return it->second;
}

// DRAM views (TRANSLATED core + address offset per view) for a chip, from the Metal SOC descriptor.
// Cached per chip. Mirrors device_io.cpp::load_dram_views.
const std::vector<UmdKvDramView>& umd_dram_views_for(tt::ChipId chip_id) {
    static std::mutex mu;
    static std::unordered_map<tt::ChipId, std::vector<UmdKvDramView>> cache;
    std::lock_guard<std::mutex> lock(mu);
    if (auto it = cache.find(chip_id); it != cache.end()) {
        return it->second;
    }
    tt::umd::SocDescriptor soc = umd_read_cluster()->get_soc_descriptor(chip_id);
    soc.device_descriptor_file_path = umd_metal_soc_yaml_path(soc.arch);
    metal_SocDescriptor metal_soc(soc, tt::BoardType::UNKNOWN);

    std::vector<UmdKvDramView> views;
    const size_t num_views = metal_soc.get_num_dram_views();
    views.reserve(num_views);
    for (int v = 0; v < static_cast<int>(num_views); ++v) {
        tt::umd::CoreCoord worker = metal_soc.get_preferred_worker_core_for_dram_view(v, /*noc=*/0);
        views.push_back(UmdKvDramView{
            tt::umd::CoreCoord{worker.x, worker.y, tt::CoreType::DRAM, tt::CoordSystem::TRANSLATED},
            static_cast<uint64_t>(metal_soc.get_address_offset(v))});
    }
    return cache.emplace(chip_id, std::move(views)).first->second;
}

}  // namespace

std::vector<uint8_t> read_dram_umd(uint64_t unique_id, uint64_t noc_addr, uint32_t size_bytes) {
    const tt::ChipId chip_id = umd_chip_for_unique_id(unique_id);
    const auto& views = umd_dram_views_for(chip_id);
    const uint32_t view = addr_channel(noc_addr);
    const uint32_t local = addr_local(noc_addr);
    TT_FATAL(view < views.size(), "read_dram_umd: dram view {} out of range ({} views)", view, views.size());
    const auto& info = views[view];
    std::vector<uint8_t> buf(size_bytes);
    umd_read_cluster()->read_from_device(
        buf.data(), chip_id, info.core, static_cast<uint64_t>(local) + info.address_offset, size_bytes);
    return buf;
}

}  // namespace tt::tt_metal::internal::disaggregation
