// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_pool.hpp"

#include <numa.h>

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

namespace device_cpu_allocator {
std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node(std::unordered_set<uint32_t>& free_cores) {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    if (numa_available() != -1) {
        // Host has NUMA enabled. Group CPU IDs by the NUMA nodes they belong to.
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            if (cpu_cores_per_numa_node.find(node) == cpu_cores_per_numa_node.end()) {
                cpu_cores_per_numa_node.insert({node, {}});
            }
            free_cores.insert(cpu);
            cpu_cores_per_numa_node.at(node).push_back(cpu);
        }
    } else {
        // Host does not have NUMA. Place all CPU Ids under a single node (0).
        log_warning(tt::LogMetal, "Host does not use NUMA. May see reduced performance.");
        for (int cpu = 0; cpu < sysconf(_SC_NPROCESSORS_ONLN); ++cpu) {
            free_cores.insert(cpu);
        }
    }
    return cpu_cores_per_numa_node;
}

int get_cpu_core_for_device_worker_thread(
    int mmio_controlled_device_id,
    const std::unordered_map<int, std::vector<uint32_t>>& cpu_cores_per_numa_node,
    std::unordered_set<uint32_t>& free_cores) {
    int core_assigned_to_device = 0;
    if (numa_available() != -1) {
        // Get NUMA node that the current device is mapped to through UMD
        int numa_node_for_device = tt::Cluster::instance().get_numa_node_for_device(mmio_controlled_device_id);
        if (cpu_cores_per_numa_node.find(numa_node_for_device) != cpu_cores_per_numa_node.end()) {
            // NUMA node reported by UMD exists on host. Choose a core on this numa-node using round robin policy
            int num_cores_in_numa_node = cpu_cores_per_numa_node.at(numa_node_for_device).size();
            core_assigned_to_device =
                cpu_cores_per_numa_node.at(numa_node_for_device).at(mmio_controlled_device_id % num_cores_in_numa_node);
        } else {
            // NUMA node reported by UMD does not exist on host. Use round-robin binding policy for this worker thread.
            log_warning(
                tt::LogMetal,
                "NUMA node {} for device {} does not exist on host.",
                numa_node_for_device,
                mmio_controlled_device_id);
            core_assigned_to_device = mmio_controlled_device_id % sysconf(_SC_NPROCESSORS_ONLN);
        }
    } else {
        // System does not use NUMA. Use-round robin binding strategy.
        core_assigned_to_device = mmio_controlled_device_id % sysconf(_SC_NPROCESSORS_ONLN);
    }
    free_cores.erase(core_assigned_to_device);
    return core_assigned_to_device;
}

std::unordered_map<uint32_t, uint32_t> get_device_id_to_core_map(
    const std::vector<chip_id_t>& device_ids,
    std::unordered_set<uint32_t>& free_cores,
    bool use_numa_node_based_thread_binding) {
    std::unordered_map<uint32_t, uint32_t> device_to_core_map = {};
    if (use_numa_node_based_thread_binding) {
        auto cpu_cores_per_numa_node = device_cpu_allocator::get_cpu_cores_per_numa_node(free_cores);
        for (const auto& device_id : device_ids) {
            device_to_core_map.insert(
                {device_id,
                 device_cpu_allocator::get_cpu_core_for_device_worker_thread(
                     device_id, cpu_cores_per_numa_node, free_cores)});
        }
    } else {
        for (const auto& device_id : device_ids) {
            device_to_core_map.insert({device_id, device_id % sysconf(_SC_NPROCESSORS_ONLN)});
        }
    }
    return device_to_core_map;
}

void bind_current_thread_to_free_cores(const std::unordered_set<uint32_t>& free_cores) {
    cpu_set_t cpuset;
    pthread_t current_thread = pthread_self();
    CPU_ZERO(&cpuset);

    for (const auto& free_core : free_cores) {
        CPU_SET(free_core, &cpuset);
    }
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind main thread to free CPU cores. May see performance degradation. Error Code: {}",
            rc);
    }
}

}  // namespace device_cpu_allocator

DevicePool* DevicePool::_inst = nullptr;
// Should probably add a dispatch_core_manager.cpp and move this there
tt_metal::dispatch_core_manager* tt_metal::dispatch_core_manager::_inst = nullptr;

void DevicePool::initialize(
    std::vector<chip_id_t> device_ids,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    DispatchCoreType dispatch_core_type,
    const std::vector<uint32_t> &l1_bank_remap) noexcept {
    log_debug(tt::LogMetal, "DevicePool initialize");
    tt::tt_metal::dispatch_core_manager::initialize(dispatch_core_type);

    if (_inst == nullptr) {
        static DevicePool device_pool(device_ids, num_hw_cqs, l1_small_size, trace_region_size, l1_bank_remap);
        _inst = &device_pool;
    }
    _inst->l1_small_size = l1_small_size;
    _inst->trace_region_size = trace_region_size;
    _inst->num_hw_cqs = num_hw_cqs;
    _inst->l1_bank_remap = l1_bank_remap;

     // Never skip for TG Cluster
    bool skip = not tt::Cluster::instance().is_galaxy_cluster();
    for (const auto& device_id : device_ids) {
        TT_FATAL(
            device_id < tt::Cluster::instance().number_of_devices(),
            "Device index {} out of range. There are {} devices available.",
            device_id,
            tt::Cluster::instance().number_of_devices());
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        skip &= (device_id == mmio_device_id);
    }
     _inst->skip_remote_devices = skip;

    _inst->add_devices_to_pool(device_ids);
    _inst->init_firmware_on_active_devices();
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
}

void DevicePool::initialize_device(Device* dev) const {
    detail::ClearProfilerControlBuffer(dev);

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e.
    // hugepage). Need to do this before FW init so we know what dispatch cores to reset.
    bool using_fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);
    if (using_fast_dispatch) {
        detail::DispatchStateCheck(true);
        dev->init_command_queue_host();
    } else {
        detail::DispatchStateCheck(false);
        dev->initialize_synchronous_sw_cmd_queue();
        TT_ASSERT(dev->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }

    DprintServerAttach(dev);
    watcher_init(dev);

    // TODO: as optimization, investigate removing all this call for already initialized devivces
    dev->reset_cores();
    dev->initialize_and_launch_firmware();

    watcher_attach(dev);

    // Set up HW command queues on device for FD
    if (using_fast_dispatch)
        dev->init_command_queue_device();
    detail::InitDeviceProfiler(dev);
}

void DevicePool::activate_device(chip_id_t id) {
    TT_FATAL(
        id < tt::Cluster::instance().number_of_devices(),
        "Device index {} out of range. There are {} devices available.",
        id,
        tt::Cluster::instance().number_of_devices());
    const std::lock_guard<std::mutex> lock(this->lock);
    if (this->devices.size() < id + 1) {
        this->devices.resize(id + 1);
    }
    if (this->devices[id] == nullptr) {
        log_debug(tt::LogMetal, "DevicePool new device {}", id);
        int core_assigned_to_device = this->device_to_core_map.at(id);
        auto dev =
            new Device(id, this->num_hw_cqs, this->l1_small_size, this->trace_region_size, this->l1_bank_remap, false, core_assigned_to_device);
        dev->update_dispatch_cores_for_multi_cq_eth_dispatch();
        if (!this->firmware_built_keys.contains(dev->build_key())) {
            dev->build_firmware();
            this->firmware_built_keys.insert(dev->build_key());
        }
        this->devices[id] = std::unique_ptr<Device>(dev);
    } else {
        const auto& dev = this->devices[id];
        log_debug(tt::LogMetal, "DevicePool re-initialize device {}", id);
        if (not dev->is_initialized()) {
            dev->initialize(num_hw_cqs, this->l1_small_size, this->trace_region_size, this->l1_bank_remap);
            if (!this->firmware_built_keys.contains(dev->build_key())) {
                dev->build_firmware();
                this->firmware_built_keys.insert(dev->build_key());
            }
        } else {
            TT_THROW("Cannot re-initialize device {}, must first call close()", id);
        }
    }
}

bool DevicePool::is_device_active(chip_id_t id) const {
    if (this->devices.size() < id + 1 || this->devices[id] == nullptr) {
        return false;
    } else {
        return this->devices[id]->is_initialized();
    }
}

void DevicePool::add_devices_to_pool(std::vector<chip_id_t> device_ids) {
    if (this->skip_remote_devices) {
        for (const auto& device_id : device_ids) {
            const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
            TT_ASSERT(device_id == mmio_device_id, "Skipping remote devices is only available for mmio devices");
            if (not this->is_device_active(device_id)) {
                this->activate_device(device_id);
            }
        }
    } else {
        std::vector<chip_id_t> all_device_ids = {};
        for (const auto& device_id : device_ids) {
            // Get list of all devices in the cluster connected to the passed in device_ids
            const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
            for (const auto& mmio_controlled_device_id :
                 tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
                if (not this->is_device_active(mmio_controlled_device_id)) {
                    this->activate_device(mmio_controlled_device_id);
                }
            }
        }
    }
}

void DevicePool::init_firmware_on_active_devices() const {
    for (const auto& dev : this->get_all_active_devices()) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        log_debug(
            tt::LogMetal,
            "MMIO Device {} Tunnel Count: {}",
            mmio_device_id,
            tt::Cluster::instance().get_mmio_device_tunnel_count(mmio_device_id));
        log_debug(
            tt::LogMetal,
            "MMIO Device {} Tunnel Depth: {}",
            mmio_device_id,
            tt::Cluster::instance().get_mmio_device_max_tunnel_depth(mmio_device_id));
        log_debug(
            tt::LogMetal,
            "MMIO Device {} Tunnel Stop: {}",
            mmio_device_id,
            tt::Cluster::instance().get_device_tunnel_depth(mmio_device_id));

        auto tunnels_from_mmio = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id);
        this->initialize_device(dev);
        if (not this->skip_remote_devices) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_debug(tt::LogMetal, "Tunnel {} Device {} Tunnel Stop: {}", t, mmio_controlled_device_id, ts);
                    this->initialize_device(this->devices[mmio_controlled_device_id].get());
                }
            }
        }
    }
}

DevicePool::DevicePool(
    std::vector<chip_id_t> device_ids,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const std::vector<uint32_t>& l1_bank_remap) {
    ZoneScoped;
    log_debug(tt::LogMetal, "DevicePool constructor");
    bool use_numa_node_based_thread_binding = parse_env("TT_METAL_NUMA_BASED_AFFINITY", false);
    std::vector<chip_id_t> all_device_ids;
    for (int i = 0; i < tt::Cluster::instance().number_of_devices(); i++) {
        all_device_ids.emplace_back((chip_id_t)i);
    }
    std::unordered_set<uint32_t> free_cores = {};
    this->device_to_core_map =
        device_cpu_allocator::get_device_id_to_core_map(all_device_ids, free_cores, use_numa_node_based_thread_binding);
    if (use_numa_node_based_thread_binding) {
        // Bind main thread to cores not being used by workers
        device_cpu_allocator::bind_current_thread_to_free_cores(free_cores);
    }
}

Device* DevicePool::get_active_device(chip_id_t device_id) const {
    TT_ASSERT(this->is_device_active(device_id), "DevicePool does not contain active device {}", device_id);
    return this->devices[device_id].get();
}

std::vector<Device*> DevicePool::get_all_active_devices() const {
    std::vector<Device*> user_devices;
    for (int id = 0; id < this->devices.size(); id++) {
        if (this->is_device_active(id)) {
            user_devices.emplace_back(this->devices[id].get());
        }
    }
    return user_devices;
}

bool DevicePool::close_device(chip_id_t device_id) const {
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    bool pass = true;
    const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (const auto& mmio_controlled_device_id :
         tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
        if (this->is_device_active(mmio_controlled_device_id)) {
            pass &= this->devices[mmio_controlled_device_id]->close();
        }
    }
    return pass;
}

DevicePool::~DevicePool() {
    log_debug(tt::LogMetal, "DevicePool destructor");
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);

    for (const auto& dev : this->devices) {
        if (dev != nullptr and dev->is_initialized()) {
            dev->close();
        }
    }
    this->devices.clear();
}

}  // namespace tt
