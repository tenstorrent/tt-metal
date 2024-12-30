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
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "tt_metal/impl/sub_device/sub_device_manager.hpp"
#include "tt_metal/impl/sub_device/sub_device_types.hpp"
#include "tt_metal/tt_stl/span.hpp"
#include "program_cache.hpp"

namespace tt {

namespace tt_metal {
// Fwd declares
enum class BufferType;

inline namespace v0 {

class Buffer;
class Program;
class CommandQueue;
class SubDevice;

}  // namespace v0

class JitBuildEnv;
class HWCommandQueue;
class TraceBuffer;

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
        tt::stl::Span<const std::uint32_t> l1_bank_remap = {},
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
    CoreCoord dram_grid_size() const;
    CoreType core_type_from_virtual_core(const CoreCoord& virtual_coord) const;

    // Given a Virtual coordinate in noc_index space, get the equivalent coordinate in Virtual NOC0 space
    CoreCoord virtual_noc_coordinate(uint8_t noc_index, CoreCoord coord) const;
    // Given a coordinate in Virtual NOC0 Space, get the equivalent coordinate in Virtual noc_index space
    CoreCoord virtual_noc0_coordinate(uint8_t noc_index, CoreCoord coord) const;

    std::vector<CoreCoord> worker_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;
    std::vector<CoreCoord> ethernet_cores_from_logical_cores(const std::vector<CoreCoord> &logical_cores) const;
    std::vector<CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment();

    CoreCoord virtual_core_from_logical_core(const CoreCoord &logical_coord, const CoreType& core_type) const;
    CoreCoord worker_core_from_logical_core(const CoreCoord &logical_core) const;

    // Ethernet API
    CoreCoord ethernet_core_from_logical_core(const CoreCoord &logical_core) const;
    CoreCoord logical_core_from_ethernet_core(const CoreCoord &ethernet_core) const;
    std::unordered_set<CoreCoord> get_active_ethernet_cores(bool skip_reserved_tunnel_cores=false) const;
    std::unordered_set<CoreCoord> get_inactive_ethernet_cores() const;
    bool is_active_ethernet_core(CoreCoord logical_core, bool skip_reserved_tunnel_cores=false) const;
    std::tuple<chip_id_t, CoreCoord> get_connected_ethernet_core(CoreCoord eth_core) const;
    std::vector<CoreCoord> get_ethernet_sockets(chip_id_t connected_chip_id) const;
    bool is_inactive_ethernet_core(CoreCoord logical_core) const;

    CoreCoord compute_with_storage_grid_size() const;

    CoreRangeSet worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const;
    uint32_t num_worker_cores(HalProgrammableCoreType core_type, SubDeviceId sub_device_id) const;

    const std::unique_ptr<Allocator> &get_initialized_allocator() const;
    const std::unique_ptr<Allocator> &get_initialized_allocator(SubDeviceId sub_device_id) const;

    DeviceAddr get_base_allocator_addr(const HalMemType &mem_type) const;
    DeviceAddr get_base_allocator_addr(const HalMemType &mem_type, SubDeviceId sub_device_id) const;

    uint32_t num_banks(const BufferType &buffer_type) const;
    uint32_t num_banks(const BufferType &buffer_type, SubDeviceId sub_device_id) const;
    uint32_t bank_size(const BufferType &buffer_type) const;
    uint32_t bank_size(const BufferType &buffer_type, SubDeviceId sub_device_id) const;

    uint32_t dram_channel_from_bank_id(uint32_t bank_id) const;
    uint32_t dram_channel_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const;

    CoreCoord logical_core_from_dram_channel(uint32_t dram_channel) const;
    uint32_t dram_channel_from_logical_core(const CoreCoord& logical_core) const;

    int32_t bank_offset(BufferType buffer_type, uint32_t bank_id) const;
    int32_t bank_offset(BufferType buffer_type, uint32_t bank_id, SubDeviceId sub_device_id) const;

    CoreCoord logical_core_from_bank_id(uint32_t bank_id) const;
    CoreCoord logical_core_from_bank_id(uint32_t bank_id, SubDeviceId sub_device_id) const;

    const std::vector<uint32_t> &bank_ids_from_dram_channel(uint32_t dram_channel) const;
    const std::vector<uint32_t> &bank_ids_from_dram_channel(uint32_t dram_channel, SubDeviceId sub_device_id) const;

    const std::vector<uint32_t> &bank_ids_from_logical_core(BufferType buffer_type, const CoreCoord &logical_core) const;
    const std::vector<uint32_t> &bank_ids_from_logical_core(BufferType buffer_type, const CoreCoord &logical_core, SubDeviceId sub_device_id) const;

    allocator::Statistics get_memory_allocation_statistics(const BufferType &buffer_type) const;
    allocator::Statistics get_memory_allocation_statistics(const BufferType &buffer_type, SubDeviceId sub_device_id) const;

    uint32_t get_allocator_alignment() const;
    uint32_t get_allocator_alignment(SubDeviceId sub_device_id) const;

    std::optional<DeviceAddr> lowest_occupied_compute_l1_address() const;
    std::optional<DeviceAddr> lowest_occupied_compute_l1_address(tt::stl::Span<const SubDeviceId> sub_device_ids) const;

    size_t get_l1_small_size() const;
    size_t get_l1_small_size(SubDeviceId sub_device_id) const;

    const std::unordered_set<Buffer *> &get_allocated_buffers() const;
    const std::unordered_set<Buffer *> &get_allocated_buffers(SubDeviceId sub_device_id) const;

    void deallocate_buffers();
    void deallocate_buffers(SubDeviceId sub_device_id);

    void dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out) const;
    void dump_memory_blocks(const BufferType &buffer_type, std::ofstream &out, SubDeviceId sub_device_id) const;

    std::vector<std::unordered_map<std::string, std::string>> get_block_table(const BufferType &buffer_type) const;

    // Set of logical ethernet core coordinates
    // core.x represents connectivity to one other chip, i.e. cores with <x> all connect to same chip
    // core.y represents different channels along one <x>
    const std::set<CoreCoord> &ethernet_cores() const { return this->ethernet_cores_; }

    const std::set<CoreCoord> &storage_only_cores() const { return this->storage_only_cores_; }

    uint32_t get_noc_unicast_encoding(uint8_t noc_index, const CoreCoord& core) const;
    uint32_t get_noc_multicast_encoding(uint8_t noc_index, const CoreRange& cores) const;

    // machine epsilon
    float sfpu_eps() const;

    // machine nan
    float sfpu_nan() const;

    // machine inf
    float sfpu_inf() const;

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
    std::shared_ptr<TraceBuffer> get_trace(uint32_t tid);
    uint32_t get_trace_buffers_size() const { return trace_buffers_size_; }
    void set_trace_buffers_size(uint32_t size) { trace_buffers_size_ = size; }

    bool using_slow_dispatch() const;
    bool using_fast_dispatch() const;

    // Checks that the given arch is on the given pci_slot and that it's responding
    // Puts device into reset
    bool initialize(const uint8_t num_hw_cqs, size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {}, bool minimal = false);
    void build_firmware();
    void reset_cores();
    void initialize_and_launch_firmware();
    void init_command_queue_host();
    void init_command_queue_device();
    void initialize_synchronous_sw_cmd_queue();
    void update_dispatch_cores_for_multi_cq_eth_dispatch();

    // Puts device into reset
    bool close();
    friend bool CloseDevice(Device *device);

    void enable_async(bool enable);
    void synchronize();
    WorkExecutorMode get_worker_mode() { return work_executor_.get_worker_mode(); }
    void set_worker_queue_mode(const WorkerQueueMode& mode) { this->work_executor_.set_worker_queue_mode(mode); }
    WorkerQueueMode get_worker_queue_mode() { return this->work_executor_.get_worker_queue_mode(); }
    bool is_worker_queue_empty() const { return work_executor_.worker_queue.empty(); }
    bool can_use_passthrough_scheduling() const;

    template<typename F>
    void push_work(F&& work, bool blocking = false) {
        this->work_executor_.push_work(std::forward<F>(work), blocking);
    }

    // Program cache interface. Syncrhonize with worker worker threads before querying or
    // modifying this structure, since worker threads use this for compiling ops
    void enable_program_cache();
    void disable_and_clear_program_cache();
    program_cache::detail::ProgramCache& get_program_cache() { return program_cache_; }
    std::size_t num_program_cache_entries();

    HalProgrammableCoreType get_programmable_core_type(CoreCoord virtual_core) const;

    template <typename T = DeviceAddr>
    T get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const;

    std::vector<std::pair<transfer_info_cores, uint32_t>> extract_dst_noc_multicast_info(const std::vector<CoreRange>& ranges, const CoreType core_type);

    bool dispatch_s_enabled() const;
    bool distributed_dispatcher() const;
    NOC dispatch_go_signal_noc() const;
    size_t get_device_kernel_defines_hash();

    uint8_t num_noc_mcast_txns(SubDeviceId sub_device_id) const;
    uint8_t num_noc_unicast_txns(SubDeviceId sub_device_id) const;
    uint8_t noc_data_start_index(SubDeviceId sub_device_id, bool mcast_data=true, bool unicast_data=true) const;

    SubDeviceManagerId get_active_sub_device_manager_id() const;
    SubDeviceManagerId get_default_sub_device_manager_id() const;
    SubDeviceManagerId create_sub_device_manager(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    void remove_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
    void load_sub_device_manager(SubDeviceManagerId sub_device_manager_id);
    void clear_loaded_sub_device_manager();
    LaunchMessageRingBufferState& get_worker_launch_message_buffer_state(SubDeviceId sub_device_id);
    const std::vector<SubDeviceId> &get_sub_device_ids() const;
    uint32_t num_sub_devices() const;

    // TODO #15944: Temporary api until migration to actual fabric is complete
    std::tuple<SubDeviceManagerId, SubDeviceId> create_sub_device_manager_with_fabric(tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size);
    std::optional<SubDeviceId> get_fabric_sub_device_id() const;

    uint32_t get_completion_queue_reader_core() const { return completion_queue_reader_core_; }

    bool is_mmio_capable() const;
    std::vector<std::vector<chip_id_t>> get_tunnels_from_mmio() const { return tunnels_from_mmio_; }

    static constexpr MemoryAllocator allocator_scheme_ = MemoryAllocator::L1_BANKING;

private:
    static_assert(detail::SubDeviceManager::MAX_NUM_SUB_DEVICES <= dispatch_constants::DISPATCH_MESSAGE_ENTRIES, "MAX_NUM_SUB_DEVICES must be less than or equal to dispatch_constants::DISPATCH_MESSAGE_ENTRIES");
    static constexpr uint32_t DEFAULT_NUM_SUB_DEVICES = 1;

    void initialize_cluster();
    std::unique_ptr<Allocator> initialize_allocator(size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap = {});
    void initialize_build();
    void initialize_device_kernel_defines();
    void initialize_device_bank_to_noc_tables(const HalProgrammableCoreType &core_type, CoreCoord virtual_core);
    void initialize_firmware(const HalProgrammableCoreType &core_type, CoreCoord virtual_core, launch_msg_t *launch_msg, go_msg_t* go_msg);

    void initialize_default_sub_device_state(size_t l1_small_size, size_t trace_region_size, tt::stl::Span<const std::uint32_t> l1_bank_remap);

    void compile_command_queue_programs();
    void configure_command_queue_programs();
    void clear_l1_state();
    void get_associated_dispatch_virtual_cores(
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &my_dispatch_cores,
        std::unordered_map<chip_id_t, std::unordered_set<CoreCoord>> &other_dispatch_cores);
    std::pair<int, int> build_processor_type_to_index(uint32_t programmable_core, uint32_t processor_class) const;

    void set_worker_mode(const WorkExecutorMode& mode);

    void generate_device_bank_to_noc_tables();

    SubDeviceManagerId get_next_sub_device_manager_id();
    void reset_sub_devices_state(const std::unique_ptr<detail::SubDeviceManager>& sub_device_manager);

    void mark_allocations_unsafe();
    void mark_allocations_safe();

    CoreCoord physical_worker_core_from_logical_core(const CoreCoord &logical_core) const;
    CoreCoord dram_core_from_dram_channel(uint32_t dram_channel) const;
    CoreType core_type_from_physical_core(const CoreCoord &physical_core) const;
    CoreCoord virtual_core_from_physical_core(const CoreCoord &physical_coord, const CoreType& core_type) const;

    chip_id_t id_;
    uint32_t build_key_ = 0;
    std::vector<std::vector<chip_id_t>> tunnels_from_mmio_;

    // Leaving here for compatibility with current reacharounds
    // TODO: Replace with get_initialized_allocator()
    Allocator* allocator_ = nullptr;
    bool initialized_ = false;

    std::vector<std::unique_ptr<Program>> command_queue_programs_;
    bool using_fast_dispatch_ = false;

    // Work Executor for this device - can asynchronously process host side work for
    // all tasks scheduled on this device
    WorkExecutor work_executor_;
    uint32_t worker_thread_core_ = 0;
    uint32_t completion_queue_reader_core_ = 0;
    std::unique_ptr<SystemMemoryManager> sysmem_manager_;
    uint8_t num_hw_cqs_ = 1;

    // SystemMemoryManager is the interface to the hardware command queue
    std::vector<std::unique_ptr<HWCommandQueue>> hw_command_queues_;
    std::vector<std::unique_ptr<CommandQueue>> sw_command_queues_;

    JitBuildEnv build_env_;
    JitBuildStateSet firmware_build_states_;
    JitBuildStateSet kernel_build_states_;
    std::vector<std::vector<std::pair<int, int>>> build_state_indices_;

    std::set<CoreCoord> compute_cores_;
    std::set<CoreCoord> storage_only_cores_;
    std::set<CoreCoord> ethernet_cores_;
    std::vector<CoreCoord> optimal_dram_bank_to_logical_worker_assignment_;

    std::map<std::string, std::string> device_kernel_defines_;

    std::unordered_map<SubDeviceManagerId, std::unique_ptr<detail::SubDeviceManager>> sub_device_managers_;
    SubDeviceManagerId active_sub_device_manager_id_ = {0};
    detail::SubDeviceManager *active_sub_device_manager_ = nullptr;
    SubDeviceManagerId next_sub_device_manager_id_ = {0};
    SubDeviceManagerId default_sub_device_manager_id_ = {0};
    detail::SubDeviceManager *default_sub_device_manager_ = nullptr;

    std::vector<int32_t> dram_bank_offset_map_;
    std::vector<int32_t> l1_bank_offset_map_;
    std::vector<uint16_t> dram_bank_to_noc_xy_;
    std::vector<uint16_t> l1_bank_to_noc_xy_;

    program_cache::detail::ProgramCache program_cache_;

    uint32_t trace_buffers_size_ = 0;
};

}  // namespace v0

template <typename T>
inline T Device::get_dev_addr(CoreCoord virtual_core, HalL1MemAddrType addr_type) const {
    return hal.get_dev_addr<T>(this->get_programmable_core_type(virtual_core), addr_type);
}

}  // namespace tt_metal

}  // namespace tt
