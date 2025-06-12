// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <device_pool.hpp>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <tracy/Tracy.hpp>
#include <tt_metal.hpp>
#include <unistd.h>  // Warning Linux Only, needed for _SC_NPROCESSORS_ONLN
#include <algorithm>
#include <cstdlib>
#include <future>
#include <set>
#include <utility>

#include "assert.hpp"
#include "control_plane.hpp"
#include "core_coord.hpp"
#include "device_impl.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dprint_server.hpp"
#include "env_lib.hpp"
#include "erisc_datamover_builder.hpp"
#include "fabric_edm_packet_header.hpp"
#include "fabric_host_interface.h"
#include "fabric_types.hpp"
#include "hal.hpp"
#include "host_api.hpp"
#include <tt-logger/tt-logger.hpp>
#include "profiler_types.hpp"
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/fabric.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/impl/debug/noc_logging.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/topology.hpp"
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "tt_metal/common/executor.hpp"
#include <umd/device/tt_core_coordinates.h>

using namespace tt::tt_metal;

namespace tt {

namespace llrt {

namespace internal_ {
void wait_until_cores_done(
    chip_id_t device_id, int run_state, std::unordered_set<CoreCoord>& not_done_phys_cores, int timeout_ms);
}  // namespace internal_
}  // namespace llrt

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
    int numa_node_for_device =
        tt::tt_metal::MetalContext::instance().get_cluster().get_numa_node_for_device(mmio_controlled_device_id);

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

void DevicePool::init_profiler() const {
#if defined(TRACY_ENABLE)
    for (const auto& dev : this->get_all_active_devices()) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        detail::InitDeviceProfiler(dev);
        log_info(tt::LogMetal, "Profiler started on device {}", mmio_device_id);
        if (not this->skip_remote_devices) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    auto mmio_device = get_device(mmio_controlled_device_id);
                    detail::InitDeviceProfiler(mmio_device);
                    log_info(
                        tt::LogMetal,
                        "Profiler started on remote device {}",
                        mmio_device->id());
                }
            }
        }
    }
    detail::ProfilerSync(ProfilerSyncState::INIT);
#endif
}

void DevicePool::initialize(
    const std::vector<chip_id_t>& device_ids,
    const uint8_t num_hw_cqs,
    size_t l1_small_size,
    size_t trace_region_size,
    const DispatchCoreConfig& dispatch_core_config,
    tt::stl::Span<const std::uint32_t> l1_bank_remap,
    size_t worker_l1_size,
    bool init_profiler,
    bool use_max_eth_core_count_on_all_devices,
    bool initialize_fabric_and_dispatch_fw) {
    // Issue #19729: use_max_eth_core_count_on_all_devices is a workaround
    // to allow TT-Mesh Workload dispatch to target active ethernet cores.
    ZoneScoped;
    log_debug(tt::LogMetal, "DevicePool initialize");
    tt::tt_metal::MetalContext::instance().initialize(
        dispatch_core_config, num_hw_cqs, {l1_bank_remap.begin(), l1_bank_remap.end()});

    if (_inst == nullptr) {
        static DevicePool device_pool{};
        _inst = &device_pool;
    }
    _inst->l1_small_size = l1_small_size;
    _inst->trace_region_size = trace_region_size;
    _inst->worker_l1_size = worker_l1_size;
    _inst->num_hw_cqs = num_hw_cqs;
    _inst->l1_bank_remap.assign(l1_bank_remap.begin(), l1_bank_remap.end());
    _inst->init_profiler_ = init_profiler;
    _inst->initialize_fabric_and_dispatch_fw_ = initialize_fabric_and_dispatch_fw;

    // Never skip for TG Cluster
    bool skip = not tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster();
    std::vector<chip_id_t> target_mmio_ids;
    for (const auto& device_id : device_ids) {
        TT_FATAL(
            device_id < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(),
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
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();

    _inst->skip_remote_devices = skip;
    _inst->use_max_eth_core_count_on_all_devices_ = use_max_eth_core_count_on_all_devices;
    _inst->add_devices_to_pool(device_ids);
    tt::tt_metal::MetalContext::instance().get_cluster().set_internal_routing_info_for_ethernet_cores(
        true, target_mmio_ids);
    _inst->init_firmware_on_active_devices();
}

void DevicePool::initialize_fabric_and_dispatch_fw() const {
    this->initialize_active_devices();
    this->wait_for_fabric_router_sync();
}

void DevicePool::initialize_host(IDevice* dev) const {
    detail::ClearProfilerControlBuffer(dev);

    // Create system memory writer for this device to have an associated interface to hardware command queue (i.e.
    // hugepage). Need to do this before FW init so we know what dispatch cores to reset.
    if (this->using_fast_dispatch) {
        detail::DispatchStateCheck(true);
        dev->init_command_queue_host();
    } else {
        detail::DispatchStateCheck(false);
        TT_ASSERT(dev->num_hw_cqs() == 1, "num_hw_cqs must be 1 in slow dispatch");
    }

    ClearNocData(dev->id());
    DprintServerAttach(dev->id());
    watcher_init(dev->id());

    // TODO: as optimization, investigate removing all this call for already initialized devivces
    if (!tt_metal::MetalContext::instance().rtoptions().get_skip_reset_cores_on_init()) {
        dev->reset_cores();
    }
    dev->initialize_and_launch_firmware();

    watcher_attach(dev->id());
}

void DevicePool::init_fabric(const std::vector<tt_metal::IDevice*>& active_devices) const {
    std::vector<std::shared_future<void>> events;
    for (uint32_t i = 0; i < active_devices.size(); i++) {
        auto& dev = active_devices[i];
        events.emplace_back(detail::async([dev]() { dev->init_fabric(); }));
    }
    for (const auto& event : events) {
        // Wait for all fabric programs to be initialized
        event.get();
    }
}

void DevicePool::initialize_active_devices() const {
    const auto& active_devices = this->get_all_active_devices();

    // Activate fabric (must be before FD)
    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        log_info(tt::LogMetal, "Initializing Fabric");
        if (tt_fabric::is_2d_fabric_config(fabric_config)) {
            // TODO: need to write routing tables for unified 2d fabric.
            // write routing tables to all ethernet cores
            tt::tt_metal::MetalContext::instance()
                .get_control_plane().write_routing_tables_to_all_chips();
        }

        // Initialize fabric on mmio device
        init_fabric(active_devices);
        log_info(tt::LogMetal, "Fabric Initialized with config {}", fabric_config);
    }

    // Activate FD kernels
    // Remaining steps are for setting up FD
    if (!this->using_fast_dispatch) {
        return;
    }

    for (auto dev : active_devices) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }

        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        populate_cq_static_args(dev);
        dev->init_command_queue_device();
        if (not this->skip_remote_devices) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    auto device = get_device(mmio_controlled_device_id);
                    populate_cq_static_args(device);
                    device->init_command_queue_device();
                }
            }
        }
    }
    _inst->dispatch_firmware_active_ = true;
}

void DevicePool::activate_device(chip_id_t id) {
    TT_FATAL(
        id < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(),
        "Device index {} out of range. There are {} devices available.",
        id,
        tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices());
    const std::lock_guard<std::mutex> lock(this->lock);
    if (this->devices.size() < id + 1) {
        this->devices.reserve(id + 1);
    }
    auto device = get_device(id);
    if (!device) {
        log_debug(tt::LogMetal, "DevicePool new device {}", id);
        int worker_core_thread_core = this->worker_thread_to_cpu_core_map.at(id);
        int completion_queue_reader_core = this->completion_queue_reader_to_cpu_core_map.at(id);
        device = new Device(
            id,
            this->num_hw_cqs,
            this->l1_small_size,
            this->trace_region_size,
            this->l1_bank_remap,
            false,
            worker_core_thread_core,
            completion_queue_reader_core,
            this->worker_l1_size);
        this->devices.emplace_back(std::unique_ptr<IDevice>(device));
    } else {
        log_debug(tt::LogMetal, "DevicePool re-initialize device {}", id);
        if (not device->is_initialized()) {
            device->initialize(
                num_hw_cqs, this->l1_small_size, this->trace_region_size, this->worker_l1_size, this->l1_bank_remap);
        } else {
            TT_THROW("Cannot re-initialize device {}, must first call close()", id);
        }
    }
}

bool DevicePool::is_device_active(chip_id_t id) const {
    auto device = get_device(id);
    if (!device) {
        return false;
    }

    return device->is_initialized();
}

IDevice* DevicePool::get_device(chip_id_t id) const {
    auto it = std::find_if(devices.begin(), devices.end(), [&id](const auto& device) { return device->id() == id; });
    if (it == devices.end()) {
        return nullptr;
    }

    return it->get();
}

std::size_t DevicePool::get_max_num_eth_cores_across_all_devices() const {
    // This API is needed due to Issue #19729:
    // Workaround to allow TT-Mesh Workload dispatch to target active ethernet cores.
    // Records the maximum number of active ethernet cores across all devices opened in the cluster.
    // TT-Mesh dispatch assumes that all physical devices in the Mesh have the maximum number of active
    // ethernet cores (uniformity assumption)
    // Dispatch firmware running on each physical device knows how many ethernet cores are actually
    // available and will dispatch to/wait on the correct number of cores (effectively ignoring the
    // value host dispatch provides, if its incorrect).
    std::size_t max_eth_core_count = 0;
    for (const auto& device : this->devices) {
        max_eth_core_count = std::max(
            MetalContext::instance()
                .get_cluster()
                .get_active_ethernet_cores(device->id(), /*skip_reserved_tunnel_cores*/ true)
                .size(),
            max_eth_core_count);
    }
    return max_eth_core_count;
}

void DevicePool::add_devices_to_pool(const std::vector<chip_id_t>& device_ids) {
    std::set<chip_id_t> devices_to_activate;
    if (this->skip_remote_devices) {
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

    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    // Only can launch Fabric if all devices are active
    if (tt_fabric::is_tt_fabric_config(fabric_config)) {
        for (int i = 0; i < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); i++) {
            if (not _inst->is_device_active(i)) {
                // Fabric currently requires all devices to be active
                log_fatal(tt::LogMetal, "Fabric is being used but {} is not active", i);
            }
        }
    }

    this->using_fast_dispatch = (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr);
    if (this->using_fast_dispatch) {
        populate_fd_kernels(devices_to_activate, this->num_hw_cqs);
    }
}

void DevicePool::wait_for_fabric_router_sync() const {
    FabricConfig fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (!tt::tt_fabric::is_tt_fabric_config(fabric_config)) {
        return;
    }

    const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();

    auto wait_for_handshake = [&](IDevice* dev) {
        if (!dev) {
            TT_THROW("Fabric router sync on null device. All devices must be opened for Fabric.");
        }
        if (fabric_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
            return;
        }

        const auto master_router_chan = fabric_context.get_fabric_master_router_chan(dev->id());
        const auto master_router_logical_core =
            tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(dev->id()).get_eth_core_for_channel(
                master_router_chan, CoordSystem::LOGICAL);

        const auto [router_sync_address, expected_status] = fabric_context.get_fabric_router_sync_address_and_status();
        std::vector<std::uint32_t> master_router_status{0};
        while (master_router_status[0] != expected_status) {
            tt_metal::detail::ReadFromDeviceL1(
                dev, master_router_logical_core, router_sync_address, 4, master_router_status, CoreType::ETH);
        }

        auto ready_address_and_signal = fabric_context.get_fabric_router_ready_address_and_signal();
        if (ready_address_and_signal) {
            std::vector<uint32_t> signal(1, ready_address_and_signal->second);
            tt_metal::detail::WriteToDeviceL1(
                dev, master_router_logical_core, ready_address_and_signal->first, signal, CoreType::ETH);
        }
    };

    for (const auto& dev : this->get_all_active_devices()) {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id()) != dev->id()) {
            continue;
        }

        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(dev->id());
        for (auto i = 0; i < tunnels_from_mmio.size(); i++) {
            // Need to poll on devices from farthest to the closest.
            for (auto j = tunnels_from_mmio[i].size() - 1; j > 0; j--) {
                wait_for_handshake(get_device(tunnels_from_mmio[i][j]));
            }
        }

        wait_for_handshake(dev);
    }
}

void DevicePool::init_firmware_on_active_devices() const {
    const auto& active_devices = this->get_all_active_devices();
    for (const auto& dev : active_devices) {
        // For Galaxy init, we only need to loop over mmio devices
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_device_id != dev->id()) {
            continue;
        }
        log_debug(
            tt::LogMetal,
            "MMIO Device {} Tunnel Count: {}",
            mmio_device_id,
            tt::tt_metal::MetalContext::instance().get_cluster().get_mmio_device_tunnel_count(mmio_device_id));
        log_debug(
            tt::LogMetal,
            "MMIO Device {} Tunnel Depth: {}",
            mmio_device_id,
            tt::tt_metal::MetalContext::instance().get_cluster().get_mmio_device_max_tunnel_depth(mmio_device_id));
        log_debug(
            tt::LogMetal,
            "MMIO Device {} Tunnel Stop: {}",
            mmio_device_id,
            tt::tt_metal::MetalContext::instance().get_cluster().get_device_tunnel_depth(mmio_device_id));

        auto tunnels_from_mmio =
            tt::tt_metal::MetalContext::instance().get_cluster().get_tunnels_from_mmio_device(mmio_device_id);
        this->initialize_host(dev);
        if (not this->skip_remote_devices) {
            for (uint32_t t = 0; t < tunnels_from_mmio.size(); t++) {
                // Need to create devices from farthest to the closest.
                for (uint32_t ts = tunnels_from_mmio[t].size() - 1; ts > 0; ts--) {
                    uint32_t mmio_controlled_device_id = tunnels_from_mmio[t][ts];
                    log_debug(tt::LogMetal, "Tunnel {} Device {} Tunnel Stop: {}", t, mmio_controlled_device_id, ts);
                    auto device = get_device(mmio_controlled_device_id);
                    this->initialize_host(device);
                }
            }
        }
    }

    if (init_profiler_) {
        this->init_profiler();
    }
    if (initialize_fabric_and_dispatch_fw_) {
        this->initialize_fabric_and_dispatch_fw();
    }
}

DevicePool::DevicePool() {
    ZoneScoped;
    log_debug(tt::LogMetal, "DevicePool constructor");
    bool use_numa_node_based_thread_binding = parse_env("TT_METAL_NUMA_BASED_AFFINITY", false);
    std::vector<chip_id_t> all_device_ids;
    for (int i = 0; i < tt::tt_metal::MetalContext::instance().get_cluster().number_of_devices(); i++) {
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

IDevice* DevicePool::get_active_device(chip_id_t device_id) const {
    auto device = get_device(device_id);
    TT_ASSERT(device != nullptr, "DevicePool does not contain device {}", device_id);
    TT_ASSERT(device->is_initialized(), "Device {} is not initialized", device_id);
    return device;
}

std::vector<IDevice* > DevicePool::get_all_active_devices() const {
    std::vector<IDevice*> user_devices;
    for (const auto& device : this->devices) {
        if (device && device->is_initialized()) {
            user_devices.push_back(device.get());
        }
    }
    return user_devices;
}

void DevicePool::teardown_fd(const std::unordered_set<chip_id_t>& devices_to_close) {
    for (const auto& dev_id : devices_to_close) {
        // Device is still active at this point
        auto dev = tt::DevicePool::instance().get_active_device(dev_id);
        if (!dev->using_fast_dispatch()) {
            continue;
        }

        for (int cq_id = 0; cq_id < dev->num_hw_cqs(); cq_id++) {
            auto& cq = dev->command_queue(cq_id);
            if (cq.sysmem_manager().get_bypass_mode()) {
                cq.record_end();
            }
            cq.terminate();
        }
    }
}

bool DevicePool::is_dispatch_firmware_active() const { return this->dispatch_firmware_active_; }

bool DevicePool::close_device(chip_id_t device_id) {
    // Sync and close one device
    // Currently can only call this on mmio chips, once we split dispatch kernel shutdown
    // from device close, we can call this on remote devices too
    ZoneScoped;
    const auto& mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_id);
    std::vector<IDevice*> devices_to_close;
    for (const auto& mmio_controlled_device_id :
         tt::tt_metal::MetalContext::instance().get_cluster().get_devices_controlled_by_mmio_device(mmio_device_id)) {
        auto device = get_device(mmio_controlled_device_id);
        if (device && device->is_initialized()) {
            devices_to_close.push_back(device);
        }
    }
    return close_devices(devices_to_close);
}

bool DevicePool::close_devices(const std::vector<IDevice*>& devices, bool skip_synchronize) {
    // Ordered, because we need to shutdown tunnels from the farthest to the closest.
    std::vector<chip_id_t> devices_to_close;

    ZoneScoped;
    // Loop over all devices and add remote devices to devices_to_close
    // For Galaxy if an mmio device's tunnels are being closed, close the mmio device as well
    std::unordered_set<chip_id_t> mmio_devices_to_close;
    for (const auto& dev : devices) {
        const auto& mmio_device_id =
            tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(dev->id());
        if (mmio_devices_to_close.find(mmio_device_id) != mmio_devices_to_close.end()) {
            continue;
        }
        auto mmio_dev_handle = tt::DevicePool::instance().get_active_device(mmio_device_id);
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

    // Global Sync across all devices that are being closed
    // We need to ensure that commands sent to each device have been completed
    // before closing any device + modifying routing info.
    // If this is not done, non-blocking CCLs followed by a close will hang, since
    // the main thread will modify device state while the CCL is running on device.
    // On TG - this should not be done on MMIO mapped devices, since we don't run
    // any workloads on them
    if (!skip_synchronize) {
        for (const auto& dev_id : devices_to_close) {
            auto dev = tt::DevicePool::instance().get_active_device(dev_id);
            if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster() and dev->is_mmio_capable()) {
                continue;
            }
            Synchronize(dev);    // Synchronize device
        }
    }

    dispatch_firmware_active_ = false;
    teardown_fd(std::unordered_set<chip_id_t>(devices_to_close.begin(), devices_to_close.end()));
    // Terminate sent to each device. Wait for dispatch to finish. MMIO only to prevent clogging SD path.
    // Dispatch kernels internally have a sync at the end to ensure all credits are returned
    for (const auto& dev_id : devices_to_close) {
        auto dev = tt::DevicePool::instance().get_active_device(dev_id);
        if (!dev->is_mmio_capable() || !dev->using_fast_dispatch()) {
            continue;
        }

        auto dispatch_cores = tt::tt_metal::get_virtual_dispatch_cores(dev_id);
        tt::llrt::internal_::wait_until_cores_done(dev_id, RUN_MSG_GO, dispatch_cores, 0);
    }

    // Process registered termination signals from topology
    for (const auto& dev_id : devices_to_close) {
        auto dev = tt::DevicePool::instance().get_active_device(dev_id);
        const auto& info = tt::tt_metal::get_registered_termination_cores(dev_id);
        for (const auto& core_to_terminate : info) {
            std::vector<uint32_t> val{core_to_terminate.val};
            tt_metal::detail::WriteToDeviceL1(
                dev, core_to_terminate.logical_core, core_to_terminate.address, val, core_to_terminate.core_type);
        }
    }

    // Terminate fabric routers
    const auto fabric_config = tt::tt_metal::MetalContext::instance().get_fabric_config();
    if (tt::tt_fabric::is_tt_fabric_config(fabric_config)) {
        const auto& control_plane= tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto& fabric_context = control_plane.get_fabric_context();
        auto [termination_signal_address, signal] = fabric_context.get_fabric_router_termination_address_and_signal();
        std::vector<uint32_t> termination_signal(1, signal);

        for (const auto& dev : this->get_all_active_devices()) {
            if (fabric_context.get_num_fabric_initialized_routers(dev->id()) == 0) {
                continue;
            }

            auto master_router_logical_core =
                tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(dev->id()).get_eth_core_for_channel(
                    fabric_context.get_fabric_master_router_chan(dev->id()), CoordSystem::LOGICAL);
            tt_metal::detail::WriteToDeviceL1(
                dev, master_router_logical_core, termination_signal_address, termination_signal, CoreType::ETH);
        }
    }

    detail::ProfilerSync(ProfilerSyncState::CLOSE_DEVICE);

    tt::tt_metal::MetalContext::instance().get_cluster().set_internal_routing_info_for_ethernet_cores(false);

    bool pass = true;
    for (const auto& dev_id : devices_to_close) {
        auto dev = tt::DevicePool::instance().get_active_device(dev_id);
        pass &= dev->close();
    }

    return pass;
}

DevicePool::~DevicePool() {
    log_debug(tt::LogMetal, "DevicePool destructor");
    for (const auto& dev : this->devices) {
        if (dev != nullptr and dev->is_initialized()) {
            // TODO: #13876, Was encountering issues with the DispatchMemMap being destroyed before the DevicePool
            // destructor, which leads to device->close() hitting asserts. We need to move the ownership of
            // DispatchMemMap to the device, so it doesn't go out of scope before the device is closed.
            dev->close();
        }
    }
    this->devices.clear();
}

}  // namespace tt
