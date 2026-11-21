// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/pcie_core_writer.hpp"

#include <optional>
#include <stdexcept>

#include <tt-logger/tt-logger.hpp>
#include <tt-umd-workload/cluster.hpp>
#include <tt-umd/io_window/io_window.hpp>
#include <tt-umd/pcie/pci_device.hpp>
#include <tt-umd/types/core_coordinates.hpp>
#include <tt-umd/types/io_window_config.hpp>
#include <tt_stl/assert.hpp>

namespace tt::tt_metal::distributed {

std::mutex PCIeCoreWriter::cluster_mutex_;
std::shared_ptr<tt::umd::Cluster> PCIeCoreWriter::shared_cluster_;
// The descriptor carries the *physical* PCIe device number (/dev/tenstorrent/N), which is
// stable across processes regardless of TT_VISIBLE_DEVICES. UMD's Cluster, however, addresses
// chips by *logical* id (the index into this process's device enumeration). Translate the
// physical device number into this process's local logical chip id.
static uint32_t physical_to_local_logical_id(uint32_t physical_device_num) {
    const auto enumerated = tt::umd::PCIDevice::enumerate_devices();
    for (size_t i = 0; i < enumerated.size(); ++i) {
        if (enumerated[i] == static_cast<int>(physical_device_num)) {
            return static_cast<uint32_t>(i);
        }
    }
    TT_THROW(
        "H2D socket connector cannot find physical PCIe device /dev/tenstorrent/{} in this process. "
        "Ensure the device is visible to the connector (check TT_VISIBLE_DEVICES).",
        physical_device_num);
}

std::unique_ptr<tt::umd::IoWindow> create_posted_window(
    tt::umd::Cluster& cluster, int chip_id, const tt::umd::CoreCoord& core, uint64_t addr, uint64_t size) {
    std::unique_ptr<tt::umd::IoWindow> window;
    try {
        window = cluster.create_io_window(
            chip_id,
            core,
            addr,
            /*host=*/{},
            tt::umd::IoOrdering::Posted,
            /*core_end=*/std::nullopt,
            tt::umd::WindowFlags::UnicastWrite);
    } catch (const std::runtime_error& e) {
        log_warning(
            tt::LogMetal,
            "H2DSocket: no TLB window for core ({}, {}) on chip {}; its writes go through the driver, which is "
            "slower: {}",
            core.x,
            core.y,
            chip_id,
            e.what());
        return nullptr;
    }
    if (size > window->get_size()) {
        return nullptr;
    }
    return window;
}

std::shared_ptr<tt::umd::Cluster> PCIeCoreWriter::get_or_create_cluster() {
    std::lock_guard lock(cluster_mutex_);
    if (!shared_cluster_) {
        // No target_devices: it's a no-op for SILICON, so the ctor discovers all visible PCIe chips.
        shared_cluster_ = std::make_shared<tt::umd::Cluster>(tt::umd::ClusterOptions{});
    }
    return shared_cluster_;
}

PCIeCoreWriter::PCIeCoreWriter(uint32_t device_id, uint32_t virtual_core_x, uint32_t virtual_core_y) :
    // Init eagerly so topology-discovery cost is paid here rather than lazily during the first send.
    cluster_(get_or_create_cluster()),
    device_id_(physical_to_local_logical_id(device_id)),
    virtual_core_x_(virtual_core_x),
    virtual_core_y_(virtual_core_y) {}

PCIeCoreWriter::~PCIeCoreWriter() = default;
PCIeCoreWriter::PCIeCoreWriter(PCIeCoreWriter&&) noexcept = default;
PCIeCoreWriter& PCIeCoreWriter::operator=(PCIeCoreWriter&&) noexcept = default;

std::function<void(void*, uint32_t, uint64_t)> PCIeCoreWriter::get_pcie_writer() const {
    auto* cluster = cluster_.get();
    auto chip_id = static_cast<int>(device_id_);
    auto vx = virtual_core_x_;
    auto vy = virtual_core_y_;
    tt::umd::CoreCoord core(vx, vy, CoreType::TENSIX, CoordSystem::TRANSLATED);
    return [cluster, chip_id, core](void* data, uint32_t num_bytes, uint64_t device_addr) {
        cluster->write_to_device(data, num_bytes, chip_id, core, device_addr);
    };
}

std::function<void(void*, uint32_t, uint64_t)> PCIeCoreWriter::get_posted_pcie_writer(uint64_t window_end) {
    const auto chip_id = static_cast<int>(device_id_);
    // The owner's receiver window (H2DSocketImpl::init_receiver_tlb), anchored at the start of the core's L1.
    if (window_ == nullptr && cluster_->get_soc_descriptor(chip_id).arch == tt::ARCH::BLACKHOLE) {
        tt::umd::CoreCoord core(virtual_core_x_, virtual_core_y_, CoreType::TENSIX, CoordSystem::TRANSLATED);
        window_ = create_posted_window(*cluster_, chip_id, core, /*addr=*/0, window_end);
    }
    if (window_ == nullptr) {
        return get_pcie_writer();
    }
    auto* window = window_.get();
    return [window](void* data, uint32_t num_bytes, uint64_t device_addr) {
        window->write_block(device_addr, data, num_bytes);
    };
}

}  // namespace tt::tt_metal::distributed
