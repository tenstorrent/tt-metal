// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace tt {

namespace tt_metal {

// Fwd declares

class Buffer;
class Kernel;
class CircularBuffer;
class IDevice;
class Program;
class CircularBufferConfig;
class ProgramTransferInfo;

struct ProgramCommandSequence;

namespace experimental {
class GlobalCircularBuffer;
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer);

}  // namespace experimental

namespace program_dispatch {
template <typename WorkloadType, typename DeviceType>
uint32_t program_base_addr_on_core(
    WorkloadType& workload, DeviceType generic_device, HalProgrammableCoreType core_type);
}  // namespace program_dispatch

namespace distributed {
class MeshWorkload;
class MeshWorkloadImpl;
}  // namespace distributed

class JitBuildOptions;
class EnqueueProgramCommand;
class CommandQueue;
// Must be removed. Only here because its a friend of a Program
class HWCommandQueue;

namespace detail {
class ProgramImpl;

void ValidateCircularBufferRegion(const Program& program, const IDevice* device);
KernelHandle AddKernel(
    Program& program, const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType core_type);
std::shared_ptr<Kernel> GetKernel(const Program& program, KernelHandle kernel_id);
std::shared_ptr<CircularBuffer> GetCircularBuffer(const Program& program, CBHandle id);

class Internal_;
}  // namespace detail

// Represents the status of Program Kernel Binaries in Device DRAM with respect to the dispatcher
enum class ProgramBinaryStatus : uint8_t {
    NotSent = 0,    // Binaries have not been written
    InFlight = 1,   // Fast Dispatch Commands to write the binaries to DRAM has been issued
    Committed = 2,  // Binaries have been commited to DRAM
};

class Program {
public:
    Program();
    explicit Program(const ProgramDescriptor& descriptor);

    Program(const Program& other) = delete;
    Program& operator=(const Program& other) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    void set_runtime_id(uint64_t id);
    ~Program() noexcept;

    uint64_t get_id() const;
    uint64_t get_runtime_id() const;

    size_t num_kernels() const;

    const std::vector<std::shared_ptr<CircularBuffer>>& circular_buffers() const;

    const std::vector<Semaphore>& semaphores() const;

    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    void add_buffer(std::shared_ptr<Buffer> buf);
    void release_buffers();

    size_t num_semaphores() const;
    void init_semaphores(
        const IDevice& device, const CoreCoord& logical_core, uint32_t programmable_core_type_index) const;
    // XXXXX TODO: this should return a const reference
    std::vector<std::vector<CoreCoord>> logical_cores() const;

    void compile(IDevice* device, bool force_slow_dispatch = false);

    void generate_dispatch_commands(IDevice* device);

    void invalidate_circular_buffer_allocation();

    void allocate_circular_buffers(const IDevice* device);

    void finalize_offsets(IDevice* device);
    bool is_finalized() const;
    ProgramBinaryStatus get_program_binary_status(std::size_t device_id) const;
    void set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status);
    void allocate_kernel_bin_buf_on_device(IDevice* device);
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;

    // debug/test
    uint32_t get_sem_base_addr(IDevice* device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_cb_base_addr(IDevice* device, CoreCoord logical_core, CoreType core_type);
    uint32_t get_sem_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const;
    uint32_t get_cb_size(IDevice* device, CoreCoord logical_core, CoreType core_type) const;
    void set_last_used_command_queue_for_testing(CommandQueue* queue);
    CommandQueue* get_last_used_command_queue() const;
    const std::vector<SubDeviceId>& determine_sub_device_ids(const IDevice* device);
    void set_kernels_bin_buffer(const std::shared_ptr<Buffer>& buffer);
    uint32_t get_cb_memory_size() const;
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;

    friend CBHandle CreateCircularBuffer(
        Program& program,
        const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
        const CircularBufferConfig& config);
    friend CBHandle experimental::CreateCircularBuffer(
        Program& program,
        const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);
    friend std::shared_ptr<CircularBuffer> detail::GetCircularBuffer(const Program& program, CBHandle id);
    friend void detail::ValidateCircularBufferRegion(const Program& program, const IDevice* device);

    friend KernelHandle detail::AddKernel(
        Program& program, const std::shared_ptr<Kernel>& kernel, const HalProgrammableCoreType core_type);
    friend std::shared_ptr<Kernel> detail::GetKernel(const Program& program, KernelHandle kernel_id);

    friend uint32_t CreateSemaphore(
        Program& program,
        const std::variant<CoreRange, CoreRangeSet>& core_spec,
        uint32_t initial_value,
        CoreType core_type);

    CBHandle add_circular_buffer(const CoreRangeSet& core_range_set, const CircularBufferConfig& config);
    CBHandle add_circular_buffer(
        const CoreRangeSet& core_range_set,
        const CircularBufferConfig& config,
        const experimental::GlobalCircularBuffer& global_circular_buffer);

    void add_semaphore(const CoreRangeSet& crs, uint32_t semaphore_id, uint32_t init_value, CoreType core_type);

    bool runs_on_noc_unicast_only_cores();
    bool runs_on_noc_multicast_only_cores();
    std::unordered_map<uint64_t, ProgramCommandSequence>& get_cached_program_command_sequences() noexcept;
    bool kernel_binary_always_stored_in_ringbuffer();

    friend HWCommandQueue;
    friend EnqueueProgramCommand;
    friend distributed::MeshWorkload;
    friend distributed::MeshWorkloadImpl;
    friend detail::Internal_;
};

}  // namespace tt_metal

}  // namespace tt
