// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <utility>

#include "hostdevcommon/common_values.hpp"
#include "impl/dispatch/work_executor.hpp"
#include "tt_metal/impl/allocator/basic_allocator.hpp"
#include "tt_metal/impl/allocator/l1_banking_allocator.hpp"
#include "tt_metal/impl/kernels/data_types.hpp"
#include "tt_metal/impl/program/program_device_map.hpp"
#include "tt_metal/jit_build/build.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/hal.hpp"
#include "dev_msgs.h"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "program_cache.hpp"

namespace tt {

namespace tt_metal {
// Fwd declares
enum class BufferType;

inline namespace v0 {

class Buffer;
class Program;
class CommandQueue;

}  // namespace v0

class JitBuildEnv;
class HWCommandQueue;
class TraceBuffer;

namespace detail {

class TraceDescriptor;

}

using on_close_device_callback = std::function<void ()>;

// TODO: These should be moved into arch specific host files that get exported here
static constexpr float  EPS_GS = 0.001953125f;
static constexpr float  EPS_WHB0 = 1.19209e-7f;
static constexpr float  EPS_BH = EPS_WHB0;

static constexpr float  NAN_GS = 6.9752e19;
static constexpr float  NAN_WHB0 = 7.0040e+19;
static constexpr float  NAN_BH = NAN_WHB0;

static constexpr float  INF_GS = 1.6948e38;
static constexpr float  INF_WHB0 = 1.7014e+38;
static constexpr float  INF_BH = INF_WHB0;

inline namespace v0 {

// A physical PCIexpress Tenstorrent device
class Device {
   public:
    // friend void tt_gdb(Device* device, int chip_id, const vector<CoreCoord> cores, vector<string> ops);
    Device () = delete;
    Device(
        chip_id_t device_id,
        const uint8_t num_hw_cqs,
        std::size_t l1_small_size,
        std::size_t trace_region_size,
        const std::vector<uint32_t> &l1_bank_remap = {},
        bool minimal = false,
        uint32_t worker_core = 0,
        uint32_t completion_queue_reader_core = 0);

    ~Device();

    // TODO: Add copy/move semantics
    Device(const Device &other) = delete;
    Device& operator=(const Device &other) = delete;

    Device(Device &&other) = default;
    Device& operator=(Device &&other) = default;

    tt::ARCH arch() const;

    chip_id_t id() const { return id_; }

    uint32_t build_key() const { return build_key_; }

    uint8_t num_hw_cqs() const { return num_hw_cqs_; }

    bool is_initialized() const { return this->initialized_; }

    int num_dram_channels() const;

    uint32_t l1_size_per_core() const;
    uint32_t dram_size_per_channel() const;

    CoreCoord grid_size() const;

    CoreCoord logical_grid_size() const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreCoord dram_grid_size() const;

    CoreCoord physical_core_from_logical_core(const CoreCoord &logical_core, const CoreType &core_type) const;
    CoreCoord physical_core_from_logical_core(const CoreDescriptor &logical_core) const;
    CoreType core_type_from_physical_core(const CoreCoord &physical_core) const;

    CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const;
    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;

    CoreCoord dram_core_from_logical_core(const CoreCoord &logical_core) const;
    std::vector<CoreCoord> dram_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;

    // Ethernet API
    CoreCoord ethernet_core_from_logical_core(const CoreCoord &logical_core) const;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord &physical_core) const;

    std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;
    std::vector<uint32_t> get_noc_encoding_for_active_eth_cores(NOC noc_index);

    std::unordered_set<chip_id_t> get_ethernet_connected_device_ids() const {
        return tt::Cluster::instance().get_ethernet_connected_device_ids(this->id_);
    }

    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const;

    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const;

    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const;

    bool is_inactive_ethernet_core(CoreCoord logical_core) const;

    uint32_t num_eth_worker_cores() const;

    uint32_t num_worker_cores() const;

    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const {
        return tt::Cluster::instance().get_connected_ethernet_core(std::make_tuple(this->id_, eth_core));
    }

    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const {
        return tt::Cluster::instance().get_ethernet_sockets(this->id_, connected_chip_id);
    }

    bool is_mmio_capable() const {
        return tt::Cluster::instance().get_associated_mmio_device(this->id_) == this->id_;
    }

    void setup_tunnel_for_remote_devices();

    void update_workers_build_settings(std::vector<std::vector<std::tuple<tt_cxy_pair, dispatch_worker_build_settings_t>>> &device_worker_variants);

    uint32_t num_banks(const BufferType &buffer_type) const;
    uint32_t bank_size(const BufferType &buffer_type) const;

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;

    CoreCoord dram_core_from_dram_channel(uint32_t dram_channel) const;
    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const;

    int32_t bank_offset(BufferType buffer_type, uint32_t bank_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;

    const std::vector<uint32_t> &bank_ids_from_dram_channel(uint32_t dram_channel) const;

    const std::vector<uint32_t> &bank_ids_from_logical_core(
        BufferType buffer_type, const CoreCoord &logical_core) const;

    allocator::Statistics get_memory_allocation_statistics(const BufferType &buffer_type) const;

    uint32_t get_allocator_alignment() const;

    size_t get_l1_small_size() const;

    void dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const;

    // Set of logical storage only core coordinates
    const std::set<CoreCoord> &storage_only_cores() const { return this->storage_only_cores_; }

    // Set of logical dispatch core coordinates

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    const std::set<CoreCoord> &ethernet_cores() const { return this->ethernet_cores_; }

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& physical_core) const;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& physical_cores) const;

    void deallocate_buffers();

    // machine epsilon
    float sfpu_eps() const;

    // machine nan
    float sfpu_nan() const;

    // machine inf
    float sfpu_inf() const;

    void generate_device_headers(const std::string &path) const;
    const JitBuildEnv& build_env() const { return this->build_env_; }
    const string build_firmware_target_path(uint32_t programmable_core, uint32_t processor_class, int i) const;
    const string build_kernel_target_path(uint32_t programmable_core, uint32_t processor_class, int i, const string& kernel_name) const;
    const JitBuildState& build_firmware_state(uint32_t programmable_core, uint32_t processor_class, int i) const;
    const JitBuildState& build_kernel_state(uint32_t programmable_core, uint32_t processor_class, int i) const;
    const JitBuildStateSubset build_kernel_states(uint32_t programmable_core, uint32_t processor_class) const;
    SystemMemoryManager& sysmem_manager() { return *sysmem_manager_; }
    HWCommandQueue& hw_command_queue(size_t cq_id = 0);
    CommandQueue& command_queue(size_t cq_id = 0);

    // Metal trace device capture mode
    void begin_trace(const uint8_t cq_id, const uint32_t tid);
    void end_trace(const uint8_t cq_id, const uint32_t tid);
    void replay_trace(const uint8_t cq_id, const uint32_t tid, const bool blocking);
    void release_trace(const uint32_t tid);
    std::shared_ptr<TraceBuffer> get_trace(const uint32_t tid);

    bool using_slow_dispatch() const;
    void check_allocator_is_initialized() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, const std::vector<uint32_t> &l1_bank_remap = {}, bool minimal = false);
    void initialize_cluster();
    void initialize_allocator(size_t l1_small_size, size_t trace_region_size, const std::vector<uint32_t> &l1_bank_remap = {});
    void initialize_build();
    void build_firmware();
    void initialize_firmware(const HalProgrammableCoreType &core_type, CoreCoord phys_core, launch_msg_t *launch_msg, go_msg_t* go_msg);
    void reset_cores();
    void initialize_and_launch_firmware();
    void init_command_queue_host();
    void init_command_queue_device();
    void initialize_synchronous_sw_cmd_queue();
    void configure_kernel_variant(Program& program, string path, std::vector<uint32_t> compile_args, CoreCoord kernel_core, CoreCoord Kernel_physical_core,
                                  CoreType dispatch_core_type, CoreCoord upstream_physical_core, CoreCoord downstream_physical_core, CoreCoord downstream_slave_physical_core, std::map<string, string> defines_in, NOC my_noc_index, NOC upstream_noc_index, NOC downstream_noc_index, bool is_active_eth_core = false, bool send_to_brisc = false, bool force_watcher_no_inline = false);
    void compile_command_queue_programs();
    void configure_command_queue_programs();
    void clear_l1_state();
    void get_associated_dispatch_phys_cores(
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &my_dispatch_cores,
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &other_dispatch_cores);
    std::pair<int, int> build_processor_type_to_index(uint32_t programmable_core, uint32_t processor_class) const;

    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    // APIs to access this device's work executor
    std::thread::id get_worker_thread_id() const;
    void push_work(std::function<void()>&& work, bool blocking = false);
    void push_work(std::shared_ptr<std::function<void()>> work, bool blocking = false);
    void synchronize();
    void set_worker_mode(const WorkExecutorMode& mode);
    void enable_async(bool enable);
    WorkExecutorMode get_worker_mode() { return work_executor.get_worker_mode(); }
    void set_worker_queue_mode(const WorkerQueueMode& mode) { this->work_executor.set_worker_queue_mode(mode); }
    WorkerQueueMode get_worker_queue_mode() { return this->work_executor.get_worker_queue_mode(); }
    // TODO: Uplift usage of friends. Buffer and Program just need access to allocator
    friend class Buffer;
    friend class Program;
    friend class SystemMemoryManager;

    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;
    chip_id_t id_;
    uint32_t build_key_;
    std::unique_ptr<Allocator> allocator_ = nullptr;
    bool initialized_ = false;
    std::map<uint32_t, std::map<chip_id_t, std::vector<std::vector<std::tuple<tt_cxy_pair, dispatch_worker_build_settings_t>>>>> tunnel_device_dispatch_workers_;
    std::vector<std::vector<chip_id_t>> tunnels_from_mmio_;

    JitBuildEnv build_env_;
    JitBuildStateSet firmware_build_states_;
    JitBuildStateSet kernel_build_states_;
    std::vector<std::vector<std::pair<int, int>>> build_state_indices_;

    std::set<CoreCoord> compute_cores_;
    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> ethernet_cores_;

    // SystemMemoryManager is the interface to the hardware command queue
    std::vector<std::unique_ptr<HWCommandQueue>> hw_command_queues_;
    std::vector<std::unique_ptr<CommandQueue>> sw_command_queues_;
    // Work Executor for this device - can asynchronously process host side work for
    // all tasks scheduled on this device
    WorkExecutor work_executor;
    uint32_t worker_thread_core;
    uint32_t completion_queue_reader_core;
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
    LaunchMessageRingBufferState worker_launch_message_buffer_state;
    uint8_t num_hw_cqs_;

    std::vector<std::unique_ptr<Program>> command_queue_programs;
    bool using_fast_dispatch;
    program_cache::detail::ProgramCache program_cache;
    uint32_t num_worker_cores_;
    uint32_t num_eth_worker_cores_;
    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    void enable_program_cache() {
        log_info(tt::LogMetal, "Enabling program cache on device {}", this->id_);
        this->synchronize();
        program_cache.enable();
    }
    void disable_and_clear_program_cache() {
        log_info(tt::LogMetal, "Disabling and clearing program cache on device {}", this->id_);
        this->synchronize();
        if (this->program_cache.is_enabled()) {
            program_cache.disable();
        }
        program_cache.clear();
    }
    std::size_t num_program_cache_entries() {
        this->synchronize();
        return program_cache.num_entries();
    }

   uint32_t trace_buffers_size = 0;
   void update_dispatch_cores_for_multi_cq_eth_dispatch();

    HalProgrammableCoreType get_programmable_core_type(CoreCoord phys_core) const;
    template <typename T = DeviceAddr>
    T get_dev_addr(CoreCoord phys_core, HalL1MemAddrType addr_type) const;
    // Returns address where allocator starts allocating buffer
    template <typename T = DeviceAddr>
    T get_base_allocator_addr(const HalMemType &mem_type) const;

    template <typename CoreRangeContainer>
    std::vector<std::pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(const CoreRangeContainer& ranges, const CoreType core_type);
    bool dispatch_s_enabled() const;
    bool distributed_dispatcher() const;

   private:
    void MarkAllocationsUnsafe();
    void MarkAllocationsSafe();
    std::unordered_map<uint32_t, std::shared_ptr<TraceBuffer>> trace_buffer_pool_;
};

}  // namespace v0

inline HalProgrammableCoreType Device::get_programmable_core_type(CoreCoord phys_core) const {

    HalProgrammableCoreType programmable_core_type = HalProgrammableCoreType::TENSIX;
    if (tt::llrt::is_ethernet_core(phys_core, this->id_)) {
        // Eth pcores have a different address, but only active ones.
        CoreCoord logical_core = this->logical_core_from_ethernet_core(phys_core);
        if (this->is_active_ethernet_core(logical_core)) {
            programmable_core_type = HalProgrammableCoreType::ACTIVE_ETH;
        } else {
            programmable_core_type = HalProgrammableCoreType::IDLE_ETH;
        }
    }

    return programmable_core_type;
}

template <typename T>
inline T Device::get_dev_addr(CoreCoord phys_core, HalL1MemAddrType addr_type) const {
    return hal.get_dev_addr<T>(this->get_programmable_core_type(phys_core), addr_type);
}

template <typename T>
inline T Device::get_base_allocator_addr(const HalMemType &mem_type) const {
    return allocator::get_unreserved_base_address(*this->allocator_, mem_type);
}

// TODO: Find a better home for this function
template <typename CoreRangeContainer>
std::vector<std::pair<transfer_info_cores, uint32_t>> Device::extract_dst_noc_multicast_info(const CoreRangeContainer& ranges, const CoreType core_type) {
    // This API extracts all the pairs of noc multicast encodings given a set of core ranges
    std::vector<std::pair<transfer_info_cores, uint32_t>> dst_noc_multicast_info;
    dst_noc_multicast_info.reserve(ranges.size());
    for (const CoreRange& core_range : ranges) {
        CoreCoord physical_start = this->physical_core_from_logical_core(core_range.start_coord, core_type);
        CoreCoord physical_end = this->physical_core_from_logical_core(core_range.end_coord, core_type);

        uint32_t num_receivers = core_range.size();
        dst_noc_multicast_info.push_back(std::make_pair(CoreRange(physical_start, physical_end), num_receivers));
    }
    return dst_noc_multicast_info;
}

}  // namespace tt_metal

}  // namespace tt
