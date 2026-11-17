// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280_metal_transport.hpp"

#include "x280/device/x280_state.hpp"
#include "x280_format.hpp"

#include "impl/context/metal_context.hpp"

#include <llrt/tt_cluster.hpp>
#include <mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-umd/arc/firmware_telemetry_reader.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/types/noc_id.hpp>
#include <tt-umd/tt_device/tt_device.hpp>

#include <algorithm>
#include <limits>

namespace x280 {

namespace {

using tt::tt_metal::distributed::MeshDevice;

// is_initialized() is the only MeshDevice query that is safe after close: it
// short-circuits before touching device pointers destroyed by close().
void require_open(const MeshDevice& mesh, const char* what) {
    if (!mesh.is_initialized()) {
        fail(
            std::string(what) +
            ": the MeshDevice is closed. X280 borrows the cluster the mesh runs on, so it "
            "cannot outlive the mesh.");
    }
}

void require_aligned(uint64_t value, uint64_t alignment, const char* what) {
    if (value % alignment != 0) {
        fail(std::string(what) + " " + hex(value) + " is not " + std::to_string(alignment) + "-byte aligned");
    }
}

uint32_t checked_size(size_t size) {
    if (size > std::numeric_limits<uint32_t>::max()) {
        fail("Metal wrapper access size " + std::to_string(size) + " exceeds the 32-bit wrapper limit");
    }
    return static_cast<uint32_t>(size);
}

tt_cxy_pair translated_target(const tt::Cluster& cluster, int chip_id, const tt::umd::CoreCoord& noc0) {
    // X280 identity is defined in NOC0, while tt::Cluster resolves host-I/O
    // coordinates as TRANSLATED.
    const auto translated = cluster.get_soc_desc(chip_id).translate_coord_to(noc0, tt::CoordSystem::TRANSLATED);
    return tt_cxy_pair(chip_id, translated.x, translated.y);
}

}  // namespace

X280MetalTransport::X280MetalTransport(const tt::Cluster& cluster, tt::umd::Cluster& telemetry_driver) :
    cluster_(&cluster), telemetry_driver_(&telemetry_driver) {}

std::shared_ptr<X280MetalTransport> X280MetalTransport::bind(const MeshDevice& mesh) {
    require_open(mesh, "cannot bind the Metal transport");

    // Silicon has one default context. MeshDeviceImpl does not expose its
    // context id in an installed header, so the deprecated borrow is
    // confined here.
    auto& context = tt::tt_metal::MetalContext::instance();
    const auto& cluster = context.get_cluster();
    if (context.rtoptions().get_watcher_enabled()) {
        log_warning(
            tt::LogMetal,
            "Metal watcher is enabled, but tt::Cluster's host-NOC sanitizer does not admit Blackhole ARC or L2CPU "
            "coordinates. X280 will keep using the Metal wrapper, so the first ARC/L2CPU access will fail unless "
            "the sanitizer gains a supported route.");
    }

    auto* driver = cluster.get_driver().get();
    if (driver == nullptr) {
        fail("Metal's cluster wrapper has no UMD telemetry driver");
    }
    return std::shared_ptr<X280MetalTransport>(new X280MetalTransport(cluster, *driver));
}

void X280MetalTransport::require_chip(int chip_id) const {
    if (!has_chip(chip_id)) {
        fail("chip " + std::to_string(chip_id) + " is not present in the cluster Metal opened");
    }
}

void X280MetalTransport::require_mmio_chip(int chip_id) const {
    require_chip(chip_id);
    if (!is_mmio_local(chip_id)) {
        fail(
            "chip " + std::to_string(chip_id) +
            " is not MMIO-local. A remote chip has no TLB manager and its register accesses degrade to plain "
            "reads, so X280 refuses to drive it.");
    }
}

bool X280MetalTransport::has_chip(int chip_id) const {
    const auto chips = cluster_->all_chip_ids();
    return chips.contains(chip_id);
}

bool X280MetalTransport::is_mmio_local(int chip_id) const {
    const auto chips = cluster_->mmio_chip_ids();
    return chips.contains(chip_id);
}

std::vector<tt::umd::CoreCoord> X280MetalTransport::l2cpu_cores(int chip_id) const {
    require_chip(chip_id);
    return cluster_->get_soc_desc(chip_id).get_cores(tt::CoreType::L2CPU, tt::CoordSystem::NOC0);
}

tt::tt_metal::CoreCoord X280MetalTransport::virtual_core_from_noc0(
    int chip_id, const tt::tt_metal::CoreCoord& noc0) const {
    require_mmio_chip(chip_id);
    const auto cores = l2cpu_cores(chip_id);
    const auto physical = std::find_if(
        cores.begin(), cores.end(), [&noc0](const auto& core) { return core.x == noc0.x && core.y == noc0.y; });
    if (physical == cores.end()) {
        fail(
            "NOC0 coordinate (" + std::to_string(noc0.x) + ", " + std::to_string(noc0.y) + ") on chip " +
            std::to_string(chip_id) + " is not an enabled L2CPU core");
    }
    const auto translated = translated_target(*cluster_, chip_id, *physical);
    return {translated.x, translated.y};
}

tt::umd::CoreCoord X280MetalTransport::arc_core(int chip_id) const {
    require_chip(chip_id);
    const auto cores = cluster_->get_soc_desc(chip_id).get_cores(tt::CoreType::ARC, tt::CoordSystem::NOC0);
    if (cores.empty()) {
        fail("chip " + std::to_string(chip_id) + " reports no ARC core");
    }
    return cores.front();
}

uint32_t X280MetalTransport::read_reg32(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr) const {
    require_aligned(addr, sizeof(uint32_t), "register address");
    require_mmio_chip(chip_id);
    const tt::umd::NocIdSwitcher noc0(tt::umd::NocId::NOC0);
    uint32_t value = 0;
    cluster_->read_reg(&value, translated_target(*cluster_, chip_id, core), addr);
    return value;
}

uint32_t X280MetalTransport::read_reg32_checked(
    int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, const char* what) const {
    if (what == nullptr) {
        fail("checked register read requires a description");
    }
    const uint32_t value = read_reg32(chip_id, core, addr);
    if (is_all_ones(value)) {
        fail(
            std::string(what) + " read all ones at " + hex(addr) +
            ": the tile is unclocked or the transport is broken. This is not a valid state value and is not "
            "masked into one.");
    }
    return value;
}

void X280MetalTransport::write_reg32(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint32_t value) const {
    require_aligned(addr, sizeof(uint32_t), "register address");
    require_mmio_chip(chip_id);
    const tt::umd::NocIdSwitcher noc0(tt::umd::NocId::NOC0);
    cluster_->write_reg(&value, translated_target(*cluster_, chip_id, core), addr);
}

void X280MetalTransport::write_reg32_verified(
    int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint32_t value) const {
    write_reg32(chip_id, core, addr, value);
    const uint32_t seen = read_reg32(chip_id, core, addr);
    if (seen != value) {
        fail("register readback mismatch at " + hex(addr) + ": " + format_readback_values(value, seen));
    }
}

void X280MetalTransport::read_block(
    int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, void* dst, size_t size) const {
    require_aligned(addr, sizeof(uint32_t), "block address");
    if (size % sizeof(uint32_t) != 0) {
        fail("block read size " + std::to_string(size) + " is not a multiple of 4 bytes");
    }
    if (size != 0 && dst == nullptr) {
        fail("null destination for a " + std::to_string(size) + "-byte Metal wrapper read");
    }
    require_mmio_chip(chip_id);
    if (size == 0) {
        return;
    }
    const tt::umd::NocIdSwitcher noc0(tt::umd::NocId::NOC0);
    cluster_->read_core(dst, checked_size(size), translated_target(*cluster_, chip_id, core), addr);
}

void X280MetalTransport::write_block(
    int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, const void* src, size_t size) const {
    require_aligned(addr, sizeof(uint32_t), "block address");
    if (size % sizeof(uint32_t) != 0) {
        fail("block write size " + std::to_string(size) + " is not a multiple of 4 bytes");
    }
    if (size != 0 && src == nullptr) {
        fail("null source for a " + std::to_string(size) + "-byte Metal wrapper write");
    }
    require_mmio_chip(chip_id);
    if (size == 0) {
        return;
    }
    const tt::umd::NocIdSwitcher noc0(tt::umd::NocId::NOC0);
    cluster_->write_core(src, checked_size(size), translated_target(*cluster_, chip_id, core), addr);
}

uint64_t X280MetalTransport::read_u64(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr) const {
    uint64_t value = 0;
    read_block(chip_id, core, addr, &value, sizeof(value));
    return value;
}

void X280MetalTransport::write_u64(int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t value) const {
    write_block(chip_id, core, addr, &value, sizeof(value));
}

std::optional<uint32_t> X280MetalTransport::telemetry(int chip_id, uint8_t tag) const {
    require_chip(chip_id);
    auto* device = telemetry_driver_->get_tt_device(chip_id);
    if (device == nullptr) {
        return std::nullopt;
    }
    auto* reader = device->get_firmware_telemetry_reader();
    if (reader == nullptr || !reader->is_entry_available(tag)) {
        return std::nullopt;
    }
    return reader->read_entry(tag);
}

std::vector<int> sorted_usable_chip_ids(const MeshDevice& mesh, const X280MetalTransport& transport) {
    require_open(mesh, "cannot bind to the mesh");

    if (mesh.arch() != tt::ARCH::BLACKHOLE) {
        fail("the MeshDevice is not Blackhole. Every L2CPU address X280 uses is Blackhole-specific.");
    }

    auto ids = mesh.get_device_ids();
    if (ids.empty()) {
        fail("the MeshDevice holds no devices");
    }
    std::sort(ids.begin(), ids.end());
    if (const auto duplicate = std::adjacent_find(ids.begin(), ids.end()); duplicate != ids.end()) {
        fail("the MeshDevice reports physical chip " + std::to_string(*duplicate) + " more than once");
    }
    for (int chip_id : ids) {
        if (!transport.has_chip(chip_id)) {
            fail(
                "the MeshDevice reports physical chip " + std::to_string(chip_id) +
                ", but Metal's cluster wrapper does not contain it");
        }
    }
    return ids;
}

}  // namespace x280
