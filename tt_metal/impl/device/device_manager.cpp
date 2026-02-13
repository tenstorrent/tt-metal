// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_manager.hpp"

#include <numa.h>
#include <pthread.h>
#include <tracy/Tracy.hpp>
#include <unistd.h>  // Warning Linux Only, needed for _SC_NPROCESSORS_ONLN

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_metal.hpp>
#include "fabric/fabric_host_utils.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/context_descriptor.hpp"
#include "firmware/command_queue_initializer.hpp"
#include "firmware/profiler_initializer.hpp"
#include "firmware/fabric_firmware_initializer.hpp"
#include "firmware/dispatch_kernel_initializer.hpp"

#include <experimental/fabric/control_plane.hpp>
#include <experimental/fabric/fabric_types.hpp>
#include <experimental/fabric/fabric.hpp>

#include "dispatch/dispatch_settings.hpp"
#include "dispatch/topology.hpp"
#include "dispatch/system_memory_manager.hpp"

#include <device.hpp>
#include "device_impl.hpp"

using namespace tt::tt_metal;

namespace tt {

namespace device_cpu_allocator {
std::unordered_map<int, std::vector<uint32_t>> get_cpu_cores_per_numa_node(std::unordered_set<uint32_t>& free_cores) {
    std::unordered_map<int, std::vector<uint32_t>> cpu_cores_per_numa_node = {};
    if (numa_available() != -1) {
        // Host has NUMA enabled. Group CPU IDs by the NUMA nodes they belong to.
        for (int cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            int node = numa_node_of_cpu(cpu);
            if (!cpu_cores_per_numa_node.contains(node)) {
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
    int numa_node_for_device =
        tt::tt_metal::MetalContext::instance().get_cluster().get_numa_node_for_device(mmio_controlled_device_id);

    if (numa_available() != -1 and cpu_cores_per_numa_node.contains(numa_node_for_device)) {
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

std::unordered_map<uint32_t, uint32_t> get_device_id_to_core_map(
    const uint8_t num_hw_cqs, std::unordered_map<uint32_t, uint32_t>& completion_queue_reader_to_cpu_core_map) {
    std::vector<ChipId> device_ids;
    for (ChipId device_id : tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids()) {
        device_ids.emplace_back(device_id);
    }
    bool use_numa_node_based_thread_binding =
        tt::tt_metal::MetalContext::instance().rtoptions().get_numa_based_affinity();
    std::unordered_set<uint32_t> free_cores = {};
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

    if (use_numa_node_based_thread_binding) {
        // Bind main thread to cores not being used by workers
        bind_current_thread_to_free_cores(free_cores);
    }

    return worker_thread_to_cpu_core_map;
}
}  // namespace device_cpu_allocator

namespace tt_metal {

void DeviceManager::initialize(
    const std::vector<ChipId>& device_ids,
    bool init_profiler,
    bool initialize_fabric_and_dispatch_fw,
    std::shared_ptr<ContextDescriptor> descriptor) {
    ZoneScoped;
    log_debug(tt::LogMetal, "DeviceManager initialize");

    descriptor_ = std::move(descriptor);
    num_hw_cqs_ = descriptor_->num_cqs();
    l1_small_size_ = descriptor_->l1_small_size();
    trace_region_size_ = descriptor_->trace_region_size();
    worker_l1_size_ = descriptor_->worker_l1_size();
    using_fast_dispatch_ = MetalContext::instance().rtoptions().get_fast_dispatch();
    init_profiler_ = init_profiler;
    initialize_fabric_and_dispatch_fw_ = initialize_fabric_and_dispatch_fw;

    worker_thread_to_cpu_core_map_ =
        device_cpu_allocator::get_device_id_to_core_map(num_hw_cqs_, completion_queue_reader_to_cpu_core_map_);

    l1_bank_remap_.assign(descriptor->l1_bank_remap().begin(), descriptor->l1_bank_remap().end());

    open_devices(device_ids);
    is_initialized_ = true;
}

void DeviceManager::open_devices(const std::vector<ChipId>& device_ids) {
    std::vector<ChipId> device_ids_to_open = device_ids;
    // Never skip for TG Cluster
    bool is_galaxy = tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster();
    bool skip = !is_galaxy;
    bool any_remote_devices = false;

    // Fabric requires all devices to be open even though dispatch
    // TODO: https://github.com/tenstorrent/tt-metal/issues/24413
    if (using_fast_dispatch_) {
        // Check if fabric needs to be enabled (any remote devices).
        // Note, all devices must be open to use fabric. This check will happen in add_devices_to_pool.
        for (auto dev_id : device_ids_to_open) {
            any_remote_devices =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev_id) != dev_id;
            if (any_remote_devices) {
                break;
            }
        }
        // Must launch for TG
        any_remote_devices |= is_galaxy;

        // Must open all devices in cluster to use fabric
        if (any_remote_devices) {
            device_ids_to_open.clear();
            for (int id = 0; id < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); ++id) {
                device_ids_to_open.push_back(id);
            }
        }
    }

    std::vector<ChipId> target_mmio_ids;
    for (const auto& device_id : device_ids_to_open) {
        TT_FATAL(
            tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().contains(device_id),
            "Device index {} out of range. There are {} devices available.",
            device_id,
            tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices());
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
        if (std::find(target_mmio_ids.begin(), target_mmio_ids.end(), mmio_device_id) == target_mmio_ids.end()) {
            target_mmio_ids.push_back(mmio_device_id);
        }
        skip &= (device_id == mmio_device_id);
    }
    if (target_mmio_ids.size() != tt::tt_metal::MetalContext::instance().get_cluster().number_of_pci_devices()) {
        log_warning(
            tt::LogMetal,
            "Opening subset of mmio devices slows down UMD read/write to remote chips. If opening more devices, "
            "consider using CreateDevices API.");
    }

    // Need to reserve eth cores for fabric before we initialize individual devices to maintain consistent state
    // while initializing default sub device state.
    // This call will be a no-op if fabric is disabled.
    // May be called again below
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    // Mock devices don't support fabric operations
    bool is_mock =
        tt::tt_metal::MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Mock;
    if (any_remote_devices && !is_mock) {
        auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
        if (fabric_config == tt::tt_fabric::FabricConfig::DISABLED) {
            fabric_config = tt::tt_fabric::FabricConfig::FABRIC_1D;
            tt::tt_fabric::SetFabricConfig(
                fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
            // Call initialize again because previously it was a no-op
            tt::tt_metal::MetalContext::instance().initialize_fabric_config();
            log_info(
                tt::LogMetal,
                "Enabling {} only for dispatch. If your workload requires fabric, please set the fabric config "
                "accordingly.",
                fabric_config);
        } else {
            // Use the same mode
            tt::tt_fabric::SetFabricConfig(
                fabric_config, tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        }
        log_info(tt::LogMetal, "Dispatch on {} with {} Command Queues\n", fabric_config, num_hw_cqs_);
    }

    skip_remote_devices_ = skip;
    add_devices_to_pool(device_ids_to_open);

    // Initialize fabric tensix datamover config after devices are added to the pool
    tt::tt_metal::MetalContext::instance().initialize_fabric_tensix_datamover_config();

    descriptor_ = tt::tt_metal::MetalContext::instance().create_context_descriptor(
        num_hw_cqs_, l1_small_size_, trace_region_size_, worker_l1_size_);

    init_firmware_on_active_devices();
}

void DeviceManager::activate_device(ChipId id) {
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().contains(id),
        "Device index {} out of range. There are {} devices available.",
        id,
        tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices());
    const std::lock_guard<std::mutex> lock(lock_);
    if (this->devices_.size() < id + 1) {
        this->devices_.reserve(id + 1);
    }
    auto* device = get_device(id);
    if (!device) {
        log_debug(tt::LogMetal, "DeviceManager new device {}", id);
        // For mock devices, these maps may not be populated, use defaults
        int worker_core_thread_core =
            this->worker_thread_to_cpu_core_map_.contains(id) ? this->worker_thread_to_cpu_core_map_.at(id) : -1;
        int completion_queue_reader_core = this->completion_queue_reader_to_cpu_core_map_.contains(id)
                                               ? this->completion_queue_reader_to_cpu_core_map_.at(id)
                                               : -1;
        device = new Device(
            id,
            this->num_hw_cqs_,
            this->l1_small_size_,
            this->trace_region_size_,
            this->l1_bank_remap_,
            false,
            worker_core_thread_core,
            completion_queue_reader_core,
            this->worker_l1_size_);
        devices_.emplace_back(std::unique_ptr<Device>(device));
    } else {
        log_debug(tt::LogMetal, "DeviceManager re-initialize device {}", id);
        if (not device->is_initialized()) {
            device->initialize(
                this->num_hw_cqs_,
                this->l1_small_size_,
                this->trace_region_size_,
                this->worker_l1_size_,
                this->l1_bank_remap_);
        } else {
            TT_THROW("Cannot re-initialize device {}, must first call close()", id);
        }
    }
}

bool DeviceManager::is_device_active(ChipId id) const {
    auto* device = this->get_device(id);
    if (!device) {
        return false;
    }

    return device->is_initialized();
}

Device* DeviceManager::get_device(ChipId id) const {
    auto it = std::find_if(devices_.begin(), devices_.end(), [&id](const auto& device) { return device->id() == id; });
    if (it == devices_.end()) {
        return nullptr;
    }

    return it->get();
}

std::size_t DeviceManager::get_max_num_eth_cores_across_all_devices() const {
    // This API is needed due to Issue #19729:
    // Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // Records the maximum number of active ethernet cores across all devices opened in the cluster.
    // TT-Mesh dispatch assumes that all physical devices in the Mesh have the maximum number of active
    // ethernet cores (uniformity assumption)
    // Dispatch firmware running on each physical device knows how many ethernet cores are actually
    // available and will dispatch to/wait on the correct number of cores (effectively ignoring the
    // value host dispatch provides, if its incorrect).
    std::size_t max_eth_core_count = 0;
    for (const auto& device : this->devices_) {
        max_eth_core_count = std::max(
            MetalContext::instance()
                .get_control_plane()
                .get_active_ethernet_cores(device->id(), /*skip_reserved_cores*/ true)
                .size(),
            max_eth_core_count);
    }
    return max_eth_core_count;
}

void DeviceManager::add_devices_to_pool(const std::vector<ChipId>& device_ids) {
    std::set<ChipId> devices_to_activate;

    if (this->skip_remote_devices_) {
        for (const auto& device_id : device_ids) {
            const auto& mmio_device_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
            TT_ASSERT(device_id == mmio_device_id, "Skipping remote devices is only available for mmio devices");
            devices_to_activate.insert(device_id);
        }
    } else {
        for (const auto& device_id : device_ids) {
            // Get list of all devices in the cluster connected to the passed in device_ids
            const auto& mmio_device_id =
                tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
            for (const auto& mmio_controlled_device_id :
                 tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(
                     mmio_device_id)) {
                devices_to_activate.insert(mmio_controlled_device_id);
            }
        }
    }

    for (const auto& device_id : devices_to_activate) {
        if (not this->is_device_active(device_id)) {
            this->activate_device(device_id);
        }
    }

    // Only can launch Fabric if all devices are active
    tt_fabric::FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config) and
        (tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().size() !=
         tt::tt_metal::MetalContext::instance().get_cluster().all_chip_ids().size())) {
        for (int i = 0; i < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); i++) {
            // Fabric currently requires all devices to be active
            TT_FATAL(
                this->is_device_active(i),
                "Fabric is being used but Device {} is not active. "
                "This may indicate that the fabric was launched on a subset of the devices available in the system, "
                "which is currently not supported. "
                "To launch on a subset of devices, first create a MeshDevice of the full system size, then create "
                "submeshes accordingly.\n"
                "For example, on a 6u system (8x4), if you wanted to run a 2x4 workload you could do:\n"
                "ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)\n"
                "mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))\n"
                "submeshes = mesh_device.create_submeshes(ttnn.MeshShape(2,8))",
                i);
        }
    }

    if (this->using_fast_dispatch_ && !devices_to_activate.empty()) {
        populate_fd_kernels(devices_to_activate, this->num_hw_cqs_);
    }
}

void DeviceManager::initialize_profiler() {
    auto& ctx = tt::tt_metal::MetalContext::instance();
    auto active_devices = this->get_all_active_devices_impl();
    initializers_[ProfilerInitializer::key] =
        std::make_unique<ProfilerInitializer>(descriptor_, skip_remote_devices_, ctx.profiler_state_manager().get());
    initializers_[ProfilerInitializer::key]->init(active_devices, init_done_);
    init_done_.insert(ProfilerInitializer::key);
    initializers_[ProfilerInitializer::key]->configure();
}

void DeviceManager::initialize_fabric_and_dispatch_fw() {
    auto& ctx = tt::tt_metal::MetalContext::instance();

    if (using_fast_dispatch_ && ctx.get_cluster().is_galaxy_cluster()) {
        log_info(
            tt::LogMetal, "Initializing Fabric and Dispatch Firmware for Galaxy cluster (this may take a few minutes)");
    }

    auto active_devices = this->get_all_active_devices_impl();

    initializers_[FabricFirmwareInitializer::key] =
        std::make_unique<FabricFirmwareInitializer>(descriptor_, ctx.get_control_plane());
    initializers_[FabricFirmwareInitializer::key]->init(active_devices, init_done_);
    init_done_.insert(FabricFirmwareInitializer::key);

    initializers_[DispatchKernelInitializer::key] = std::make_unique<DispatchKernelInitializer>(descriptor_);
    initializers_[DispatchKernelInitializer::key]->init(active_devices, init_done_);
    init_done_.insert(DispatchKernelInitializer::key);

    initializers_[DispatchKernelInitializer::key]->configure();
    initializers_[FabricFirmwareInitializer::key]->configure();

    log_trace(tt::LogMetal, "Fabric and Dispatch Firmware initialized");
}

void DeviceManager::initialize_dispatch_firmware() {
    // This function is used by DispatchContext for manual FD setup.
    // It will re initialize the dispatch firmware on the active devices as they were manually
    // disabled
    auto active_devices = this->get_all_active_devices_impl();
    init_done_.erase(DispatchKernelInitializer::key);
    initializers_[DispatchKernelInitializer::key] = std::make_unique<DispatchKernelInitializer>(descriptor_);
    initializers_[DispatchKernelInitializer::key]->init(active_devices, init_done_);
    initializers_[DispatchKernelInitializer::key]->configure();
    init_done_.insert(DispatchKernelInitializer::key);
}

void DeviceManager::init_firmware_on_active_devices() {
    auto active_devices = this->get_all_active_devices_impl();

    initializers_[CommandQueueInitializer::key] =
        std::make_unique<CommandQueueInitializer>(descriptor_, skip_remote_devices_);
    initializers_[CommandQueueInitializer::key]->init(active_devices, init_done_);
    init_done_.insert(CommandQueueInitializer::key);

    if (init_profiler_) {
        this->initialize_profiler();
    }
    if (initialize_fabric_and_dispatch_fw_) {
        this->initialize_fabric_and_dispatch_fw();
    }
}

DeviceManager::DeviceManager() {
    ZoneScoped;
    log_debug(tt::LogMetal, "DeviceManager constructor");
}

Device* DeviceManager::get_active_device_internal(ChipId device_id) const {
    auto* device = get_device(device_id);
    TT_ASSERT(device != nullptr, "DeviceManager does not contain device {}", device_id);
    TT_ASSERT(device->is_initialized(), "Device {} is not initialized", device_id);
    return device;
}

IDevice* DeviceManager::get_active_device(ChipId device_id) const { return get_active_device_internal(device_id); }

std::vector<IDevice*> DeviceManager::get_all_active_devices() const {
    std::vector<IDevice*> user_devices;
    for (const auto& device : this->devices_) {
        if (device && device->is_initialized()) {
            user_devices.push_back(device.get());
        }
    }
    return user_devices;
}

std::vector<Device*> DeviceManager::get_all_active_devices_impl() const {
    std::vector<Device*> user_devices;
    for (const auto& device : this->devices_) {
        if (device && device->is_initialized()) {
            user_devices.push_back(device.get());
        }
    }
    return user_devices;
}

// Get all active device ids
// This function needs to be thread-safe as its called in inspector::data on a different thread
std::vector<ChipId> DeviceManager::get_all_active_device_ids() const {
    std::vector<ChipId> device_ids;
    std::lock_guard<std::mutex> lock(this->lock_);
    device_ids.reserve(this->devices_.size());
    for (const auto& device : devices_) {
        if (device && device->is_initialized()) {
            device_ids.emplace_back(device->id());
        }
    }
    return device_ids;
}

// Get all command queue event infos for all active devices
// The key is the device id and the value is a vector of event ids for each command queue
// This function needs to be thread-safe as its called in inspector::data on a different thread
std::unordered_map<ChipId, std::vector<uint32_t>> DeviceManager::get_all_command_queue_event_infos() const {
    std::unordered_map<ChipId, std::vector<uint32_t>> cq_to_event_by_device;
    std::lock_guard<std::mutex> lock(lock_);
    cq_to_event_by_device.reserve(devices_.size());
    for (const auto& device : devices_) {
        if (device && device->is_initialized()) {
            auto& vec = cq_to_event_by_device[device->id()];
            const auto num_hw_cqs = device->num_hw_cqs();
            vec.resize(num_hw_cqs);
            for (size_t cq_id = 0; cq_id < num_hw_cqs; cq_id++) {
                const auto event_id = device->sysmem_manager().get_last_event(static_cast<uint8_t>(cq_id));
                vec[cq_id] = event_id;
            }
        }
    }
    return cq_to_event_by_device;
}

bool DeviceManager::is_dispatch_firmware_active() const {
    auto it = initializers_.find(DispatchKernelInitializer::key);
    return it != initializers_.end() && it->second->is_initialized();
}

bool DeviceManager::close_device(ChipId device_id) {
    // Sync and close one device
    // Currently can only call this on mmio chips, once we split dispatch kernel shutdown
    // from device close, we can call this on remote devices too
    ZoneScoped;
    const auto& mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    std::vector<IDevice*> devices_to_close;
    for (const auto& mmio_controlled_device_id :
         tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id)) {
        auto* device = this->get_device(mmio_controlled_device_id);
        if (device && device->is_initialized()) {
            devices_to_close.push_back(device);
        }
    }
    return this->close_devices(devices_to_close);
}

bool DeviceManager::close_devices(const std::vector<IDevice*>& devices, bool /*skip_synchronize*/) {
    ZoneScoped;

    // Ordered, because we need to shutdown tunnels from the farthest to the closest.
    std::vector<ChipId> devices_to_close;

    // Loop over all devices and add remote devices to devices_to_close
    // For Galaxy if an mmio device's tunnels are being closed, close the mmio device as well
    std::unordered_set<ChipId> mmio_devices_to_close;
    for (const auto& dev : devices) {
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_devices_to_close.contains(mmio_device_id)) {
            continue;
        }
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
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

    // Order matters
    initializers_[DispatchKernelInitializer::key]->teardown();
    initializers_[FabricFirmwareInitializer::key]->teardown();
    initializers_[ProfilerInitializer::key]->teardown();
    initializers_[CommandQueueInitializer::key]->teardown();

    initializers_[DispatchKernelInitializer::key]->post_teardown();
    initializers_[FabricFirmwareInitializer::key]->post_teardown();
    initializers_[ProfilerInitializer::key]->post_teardown();
    initializers_[CommandQueueInitializer::key]->post_teardown();

    init_done_.clear();
    initializers_.clear();

    bool pass = true;
    for (const auto& dev_id : devices_to_close) {
        auto* dev = this->get_active_device(dev_id);
        pass &= dev->close();
    }

    return pass;
}

DeviceManager::~DeviceManager() {
    for (const auto& dev : this->devices_) {
        if (dev != nullptr and dev->is_initialized()) {
            // TODO: #13876, Was encountering issues with the DispatchMemMap being destroyed before the DeviceManager
            // destructor, which leads to device->close() hitting asserts. We need to move the ownership of
            // DispatchMemMap to the device, so it doesn't go out of scope before the device is closed.
            dev->close();
        }
    }
    this->devices_.clear();
    TT_ASSERT(init_done_.empty(), "Init done set is not empty. Devices not properly teared down.");
    init_done_.clear();
    initializers_.clear();
}
}  // namespace tt_metal
}  // namespace tt
