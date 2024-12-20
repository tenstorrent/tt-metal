// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_pool.hpp"

#include <numa.h>

#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/device/device_handle.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"

using namespace tt::tt_metal;

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

std::pair<int, int> get_cpu_cores_for_dispatch_threads(
    int mmio_controlled_device_id,
    const std::unordered_map<int, std::vector<uint32_t>>& cpu_cores_per_numa_node,
    std::unordered_set<uint32_t>& free_cores,
    uint32_t num_devices,
    bool use_separate_procs) {
    int core_assigned_to_device_worker_thread = 0;
    int core_assigned_to_device_completion_queue_reader = 0;
    uint32_t num_online_processors = sysconf(_SC_NPROCESSORS_ONLN);
    // Get NUMA node that the current device is mapped to through UMD
    int numa_node_for_device = tt::Cluster::instance().get_numa_node_for_device(mmio_controlled_device_id);

    if (numa_available() != -1 and
        cpu_cores_per_numa_node.find(numa_node_for_device) != cpu_cores_per_numa_node.end()) {
        // NUMA node reported by UMD exists on host. Choose a core on this numa-node using round robin policy
        const auto& cpu_core_for_numa_node = cpu_cores_per_numa_node.at(numa_node_for_device);
        int num_cores_in_numa_node = cpu_core_for_numa_node.size();
        core_assigned_to_device_worker_thread =
            cpu_core_for_numa_node.at(mmio_controlled_device_id % num_cores_in_numa_node);
        if (use_separate_procs) {
            core_assigned_to_device_completion_queue_reader =
                cpu_core_for_numa_node.at((mmio_controlled_device_id + num_devices) % num_cores_in_numa_node);
        } else {
            core_assigned_to_device_completion_queue_reader = core_assigned_to_device_worker_thread;
        }
    } else {
        // NUMA node reported by UMD does not exist on host. Use round-robin binding policy for this worker thread.
        log_warning(
            tt::LogMetal,
            "NUMA node {} for device {} does not exist on host or NUMA is not available.",
            numa_node_for_device,
            mmio_controlled_device_id);
        core_assigned_to_device_worker_thread = mmio_controlled_device_id % num_online_processors;
        if (use_separate_procs) {
            core_assigned_to_device_completion_queue_reader =
                (mmio_controlled_device_id + num_devices) % num_online_processors;
        } else {
            core_assigned_to_device_completion_queue_reader = core_assigned_to_device_worker_thread;
        }
    }

    free_cores.erase(core_assigned_to_device_worker_thread);
    if (use_separate_procs) {
        free_cores.erase(core_assigned_to_device_completion_queue_reader);
    }
    return std::make_pair(core_assigned_to_device_worker_thread, core_assigned_to_device_completion_queue_reader);
}

std::unordered_map<uint32_t, uint32_t> get_device_id_to_core_map(
    const std::vector<chip_id_t>& device_ids,
    std::unordered_set<uint32_t>& free_cores,
    bool use_numa_node_based_thread_binding,
    const uint8_t num_hw_cqs,
    std::unordered_map<uint32_t, uint32_t>& completion_queue_reader_to_cpu_core_map) {
    uint32_t num_online_processors = sysconf(_SC_NPROCESSORS_ONLN);
    constexpr uint32_t max_num_procs_per_device = 2;
    // When using multiple command queues, assign separate CPU cores to worker and completion queue reader threads,
    // if enough processors exist on host. Atleast one core is given to the main thread.
    bool separate_procs_for_worker_and_reader =
        (num_hw_cqs > 1) && (max_num_procs_per_device * device_ids.size() <= num_online_processors - 1);
    std::unordered_map<uint32_t, uint32_t> worker_thread_to_cpu_core_map = {};
    if (use_numa_node_based_thread_binding) {
        auto cpu_cores_per_numa_node = device_cpu_allocator::get_cpu_cores_per_numa_node(free_cores);
        for (const auto& device_id : device_ids) {
            auto [worker_thread_core, completion_queue_reader_core] =
                device_cpu_allocator::get_cpu_cores_for_dispatch_threads(
                    device_id,
                    cpu_cores_per_numa_node,
                    free_cores,
                    device_ids.size(),
                    separate_procs_for_worker_and_reader);
            worker_thread_to_cpu_core_map.insert({device_id, worker_thread_core});
            completion_queue_reader_to_cpu_core_map.insert({device_id, completion_queue_reader_core});
        }
    } else {
        // Round Robin CPU assignment for worker and completion queue reader threads
        for (const auto& device_id : device_ids) {
            uint32_t worker_thread_proc = device_id % num_online_processors;
            worker_thread_to_cpu_core_map.insert({device_id, worker_thread_proc});
            if (separate_procs_for_worker_and_reader) {
                uint32_t completion_queue_reader_proc = (device_id + device_ids.size()) % num_online_processors;
                completion_queue_reader_to_cpu_core_map.insert({device_id, completion_queue_reader_proc});
            } else {
                completion_queue_reader_to_cpu_core_map.insert({device_id, worker_thread_proc});
            }
        }
    }
    return worker_thread_to_cpu_core_map;
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

void DevicePool::init_profiler_devices() const {
#if defined(TRACY_ENABLE)
    for (const auto& dev : this->get_all_active_devices()) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        auto tunnels_from_mmio = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id);
        detail::InitDeviceProfiler(dev);
        log_info(tt::LogMetal, "Profiler started on device {}", mmio_device_id);
        if (not this->skip_remote_devices) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_info(
                        tt::LogMetal,
                        "Starting profiler on device {}",
                        this->devices[mmio_controlled_device_id].get()->id());
                    detail::InitDeviceProfiler(this->devices[mmio_controlled_device_id].get());
                }
            }
        }
    }
#endif
}

void DevicePool::initialize(
    const std::vector<chip_id_t>& device_ids,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap) noexcept {
    ZoneScoped;
    log_debug(tt::LogMetal, "DevicePool initialize");
    tt::tt_metal::dispatch_core_manager::initialize(dispatch_core_config, num_hw_cqs);

    if (_inst == nullptr) {
        static DevicePool device_pool{};
        _inst = &device_pool;
    }
    _inst->l1_small_size = l1_small_size;
    _inst->trace_region_size = trace_region_size;
    _inst->num_hw_cqs = num_hw_cqs;
    _inst->l1_bank_remap.assign(l1_bank_remap.begin(), l1_bank_remap.end());
    // Track the thread where the Device Pool was created. Certain functions
    // modifying the state of this instance, for example those responsible for
    // (un)registering worker threads, can only be called in the creation thread
    _inst->device_pool_creation_thread_id = std::this_thread::get_id();

    // Never skip for TG Cluster
    bool skip = not tt::Cluster::instance().is_galaxy_cluster();
    std::vector<chip_id_t> target_mmio_ids;
    for (const auto& device_id : device_ids) {
        TT_FATAL(
            device_id < tt::Cluster::instance().number_of_devices(),
            "Device index {} out of range. There are {} devices available.",
            device_id,
            tt::Cluster::instance().number_of_devices());
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
        if (std::find(target_mmio_ids.begin(), target_mmio_ids.end(), mmio_device_id) == target_mmio_ids.end()) {
            target_mmio_ids.push_back(mmio_device_id);
        }
        skip &= (device_id == mmio_device_id);
    }
    if (target_mmio_ids.size() != tt::Cluster::instance().number_of_pci_devices()) {
        log_warning(
            tt::LogMetal,
            "Opening subset of mmio devices slows down UMD read/write to remote chips. If opening more devices, "
            "consider using CreateDevices API.");
    }

    _inst->skip_remote_devices = skip;

    _inst->add_devices_to_pool(device_ids);
    _inst->init_firmware_on_active_devices();
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true, target_mmio_ids);
    _inst->init_profiler_devices();
}

void DevicePool::initialize_device(v1::DeviceHandle handle) const {
    const auto dev = devices[handle.key.index()].get();
    detail::ClearProfilerControlBuffer(dev);

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e.
    // hugepage). Need to do this before FW init so we know what dispatch cores to reset.
    if (this->using_fast_dispatch) {
        detail::DispatchStateCheck(true);
        dev->init_command_queue_host();
    } else {
        detail::DispatchStateCheck(false);
        dev->initialize_synchronous_sw_cmd_queue();
        TT_ASSERT(dev->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }

    ClearNocData(dev);
    DprintServerAttach(dev);
    watcher_init(dev);

    // TODO: as optimization, investigate removing all this call for already initialized devivces
    if (!llrt::RunTimeOptions::get_instance().get_skip_reset_cores_on_init()) {
        dev->reset_cores();
    }
    dev->initialize_and_launch_firmware();

    watcher_attach(dev);

    // Set up HW command queues on device for FD
    if (this->using_fast_dispatch) {
        dev->init_command_queue_device();
    }
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
        int worker_core_thread_core = this->worker_thread_to_cpu_core_map.at(id);
        int completion_queue_reader_core = this->completion_queue_reader_to_cpu_core_map.at(id);
        auto dev = new Device(
            id,
            this->num_hw_cqs,
            this->l1_small_size,
            this->trace_region_size,
            this->l1_bank_remap,
            false,
            worker_core_thread_core,
            completion_queue_reader_core);
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
            if (dev->num_hw_cqs() != num_hw_cqs) {
                // The dispatch core manager was reset, since the number of CQs was toggled.
                // Account for chip specific idle eth dispatch cores.
                dev->update_dispatch_cores_for_multi_cq_eth_dispatch();
            }
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

void DevicePool::add_devices_to_pool(const std::vector<chip_id_t>& device_ids) {
    std::set<chip_id_t> devices_to_activate;
    if (this->skip_remote_devices) {
        for (const auto& device_id : device_ids) {
            const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
            TT_ASSERT(device_id == mmio_device_id, "Skipping remote devices is only available for mmio devices");
            devices_to_activate.insert(device_id);
        }
    } else {
        std::vector<chip_id_t> all_device_ids = {};
        for (const auto& device_id : device_ids) {
            // Get list of all devices in the cluster connected to the passed in device_ids
            const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
            for (const auto& mmio_controlled_device_id :
                 tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
                devices_to_activate.insert(mmio_controlled_device_id);
            }
        }
    }

    for (const auto& device_id : devices_to_activate) {
        if (not this->is_device_active(device_id)) {
            this->activate_device(device_id);
        }
    }
    this->using_fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);
    if (this->using_fast_dispatch) {
        populate_fd_kernels(devices_to_activate, this->num_hw_cqs);
    }
}

void DevicePool::register_worker_thread_for_device(v1::DeviceHandle device, std::thread::id worker_thread_id) {
    TT_FATAL(
        std::this_thread::get_id() == this->device_pool_creation_thread_id,
        "Worker threads can only be registered in the thread where the Device(s) were created");
    auto worker_thread_handle = this->device_to_worker_thread_id.find(device);
    if (worker_thread_handle != this->device_to_worker_thread_id.end()) {
        TT_FATAL(
            worker_thread_handle->second == worker_thread_id,
            "Cannot register more than one worker thread per device.");
        ;
    } else {
        TT_FATAL(
            this->worker_thread_ids.find(worker_thread_id) == this->worker_thread_ids.end(),
            "Cannot register a single worker thread on multiple devices");
    }

    this->device_to_worker_thread_id.insert({device, worker_thread_id});
    this->worker_thread_ids.insert(worker_thread_id);
}

void DevicePool::unregister_worker_thread_for_device(v1::DeviceHandle device) {
    TT_FATAL(
        std::this_thread::get_id() == this->device_pool_creation_thread_id,
        "Worker threads can only be unregistered in the thread where the Device(s) were created");
    auto worker_thread_handle = this->device_to_worker_thread_id.find(device);
    if (worker_thread_handle != this->device_to_worker_thread_id.end()) {
        this->worker_thread_ids.erase(worker_thread_handle->second);
        this->device_to_worker_thread_id.erase(device);
    }
}

const std::unordered_set<std::thread::id>& DevicePool::get_worker_thread_ids() const { return this->worker_thread_ids; }

void DevicePool::init_firmware_on_active_devices() const {
    for (const auto& handle : this->get_all_active_devices()) {
        const auto dev = this->devices[handle.key.index()].get();
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
        this->initialize_device(handle);
        if (not this->skip_remote_devices) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_debug(tt::LogMetal, "Tunnel {} Device {} Tunnel Stop: {}", t, mmio_controlled_device_id, ts);
                    this->initialize_device({{mmio_controlled_device_id, 0}});
                }
            }
        }
    }
}

DevicePool::DevicePool() {
    ZoneScoped;
    log_debug(tt::LogMetal, "DevicePool constructor");
    bool use_numa_node_based_thread_binding = parse_env("TT_METAL_NUMA_BASED_AFFINITY", false);
    std::vector<chip_id_t> all_device_ids;
    for (int i = 0; i < tt::Cluster::instance().number_of_devices(); i++) {
        all_device_ids.emplace_back((chip_id_t)i);
    }
    std::unordered_set<uint32_t> free_cores = {};
    this->worker_thread_to_cpu_core_map = device_cpu_allocator::get_device_id_to_core_map(
        all_device_ids,
        free_cores,
        use_numa_node_based_thread_binding,
        num_hw_cqs,
        this->completion_queue_reader_to_cpu_core_map);
    if (use_numa_node_based_thread_binding) {
        // Bind main thread to cores not being used by workers
        device_cpu_allocator::bind_current_thread_to_free_cores(free_cores);
    }
}

v1::DeviceHandle DevicePool::get_active_device(chip_id_t device_id) const {
    TT_ASSERT(this->is_device_active(device_id), "DevicePool does not contain active device {}", device_id);
    return {{device_id, 0}};
}

std::vector<v1::DeviceHandle> DevicePool::get_all_active_devices() const {
    std::vector<v1::DeviceHandle> user_devices;
    for (int id = 0; id < this->devices.size(); id++) {
        if (this->is_device_active(id)) {
            user_devices.push_back({{id, 0}});
        }
    }
    return user_devices;
}

bool DevicePool::close_device(chip_id_t device_id) {
    // Sync and close one device
    // Currently can only call this on mmio chips, once we split dispatch kernel shutdown
    // from device close, we can call this on remote devices too
    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    bool pass = true;
    const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_id);
    for (const auto& mmio_controlled_device_id :
         tt::Cluster::instance().get_devices_controlled_by_mmio_device(mmio_device_id)) {
        if (this->is_device_active(mmio_controlled_device_id)) {
            pass &= this->devices[mmio_controlled_device_id]->close();
            // When a device is closed, its worker thread is joined. Stop tracking this
            // worker thread.
            this->unregister_worker_thread_for_device({{mmio_controlled_device_id, 0}});
        }
    }
    return pass;
}

void DevicePool::close_devices(const std::vector<Device*>& devices) {
    // Ordered, because we need to shutdown tunnels from the farthest to the closest.
    std::vector<chip_id_t> devices_to_close;

    // Loop over all devices and add remote devices to devices_to_close
    // For Galaxy if an mmio device's tunnels are being closed, close the mmio device as well
    std::unordered_set<chip_id_t> mmio_devices_to_close;
    for (const auto& dev : devices) {
        const auto& mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(dev->id());
        if (mmio_devices_to_close.find(mmio_device_id) != mmio_devices_to_close.end()) {
            continue;
        }
        auto mmio_dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
        auto tunnels_from_mmio = tt::Cluster::instance().get_tunnels_from_mmio_device(mmio_device_id);
        // iterate over all tunnels origination from this mmio device
        for (auto t : tunnels_from_mmio) {
            // iterate over all tunneled devices (tunnel stops) in this tunnel
            for (uint32_t ts = t.size() - 1; ts > 0; ts--) {
                if (this->is_device_active(t[ts])) {
                    devices_to_close.push_back(t[ts]);
                }
            }
        }
        devices_to_close.push_back(mmio_device_id);
        mmio_devices_to_close.insert(mmio_device_id);
    }

    // Global Sync across all devices that are being closed
    // We need to ensure that commands sent to each device have been completed
    // before closing any device + modifying routing info.
    // If this is not done, non-blocking CCLs followed by a close will hang, since
    // the main thread will modify device state while the CCL is running on device.
    for (const auto& dev_id : devices_to_close) {
        auto dev = tt::DevicePool::instance().get_active_device(dev_id);
        dev->synchronize();  // Synchronize worker queue
        Synchronize(dev);    // Synchronize device
    }

    tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
    for (const auto& dev_id : devices_to_close) {
        auto dev = tt::DevicePool::instance().get_active_device(dev_id);
        dev->close();
        // When a device is closed, its worker thread is joined. Stop tracking this
        // worker thread.
        this->unregister_worker_thread_for_device(this->get_handle(dev));
    }
}

DevicePool::~DevicePool() {
    log_debug(tt::LogMetal, "DevicePool destructor");
    for (const auto& dev : this->devices) {
        if (dev != nullptr and dev->is_initialized()) {
            // TODO: #13876, Was encountering issues with the dispatch_constants being destroyed before the DevicePool
            // destructor, which leads to device->close() hitting asserts. We need to move the ownership of
            // dispatch_constants to the device, so it doesn't go out of scope before the device is closed.
            dev->close();
        }
    }
    this->devices.clear();
}

v1::DeviceHandle DevicePool::get_handle(Device* device) const {
    for (size_t index = 0; index < this->devices.size(); ++index) {
        if (this->devices[index].get() == device) {
            return {{index, 0}};
        }
    }
    return {};
}

}  // namespace tt
