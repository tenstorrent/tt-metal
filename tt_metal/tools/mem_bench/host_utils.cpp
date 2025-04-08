// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_utils.hpp"

#include "impl/context/metal_context.hpp"
#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <unordered_set>

namespace tt::tt_metal::tools::mem_bench {

void* get_hugepage(int device_id, uint32_t base_offset) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto mmio_device_id = cluster.get_associated_mmio_device(device_id);
    auto channel = cluster.get_assigned_channel_for_device(device_id);
    return (void*)(cluster.host_dma_address(base_offset, mmio_device_id, channel));
}

uint32_t get_hugepage_size(int device_id) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto mmio_device_id = cluster.get_associated_mmio_device(device_id);
    auto channel = cluster.get_assigned_channel_for_device(device_id);
    return cluster.get_host_channel_size(mmio_device_id, channel);
}

tt::tt_metal::vector_aligned<uint32_t> generate_random_src_data(uint32_t num_bytes) {
    std::uniform_int_distribution<uint32_t> distribution(
        std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
    std::default_random_engine generator;

    tt::tt_metal::vector_aligned<uint32_t> vec(num_bytes / sizeof(uint32_t));
    std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });

    return vec;
}

double get_current_time_seconds() {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

std::vector<int> get_mmio_device_ids(int number_of_devices, int numa_node) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto pcie_devices = cluster.number_of_pci_devices();
    std::vector<int> device_ids;

    // Assumes PCIe device IDs are iterated first
    for (int device_id = 0; device_id < pcie_devices && device_ids.size() < number_of_devices; ++device_id) {
        // Not an MMIO device
        if (cluster.get_associated_mmio_device(device_id) != device_id) {
            continue;
        }

        auto associated_node = cluster.get_numa_node_for_device(device_id);
        if (numa_node == -1 || associated_node == numa_node) {
            device_ids.push_back(device_id);
        }
    }

    return device_ids;
}

std::vector<int> get_mmio_device_ids_unique_nodes(int number_of_devices) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto pcie_devices = cluster.number_of_pci_devices();
    std::vector<int> device_ids;
    std::unordered_set<uint32_t> numa_nodes;

    for (int device_id = 0; device_id < pcie_devices && device_ids.size() < number_of_devices; ++device_id) {
        auto associated_node = cluster.get_numa_node_for_device(device_id);
        if (!numa_nodes.contains(associated_node)) {
            device_ids.push_back(device_id);
            numa_nodes.insert(associated_node);
        }
    }

    return device_ids;
}

int get_number_of_mmio_devices() {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    return cluster.number_of_pci_devices();
}

}  // namespace tt::tt_metal::tools::mem_bench
