// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>
#include <unordered_map>

#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace tt {

namespace tt_metal {

// Fwd declares

class Kernel;
class IDevice;
class Program;
class CircularBufferConfig;
class Semaphore;

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

namespace detail {
class ProgramImpl;
}  // namespace detail

// Represents the status of Program Kernel Binaries in Device DRAM with respect to the dispatcher
enum class ProgramBinaryStatus : uint8_t {
    NotSent = 0,    // Binaries have not been written
    InFlight = 1,   // Fast Dispatch Commands to write the binaries to DRAM has been issued
    Committed = 2,  // Binaries have been committed to DRAM
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

    std::vector<std::vector<CoreCoord>> logical_cores() const;

    void compile(IDevice* device, bool force_slow_dispatch = false);

    void generate_dispatch_commands(IDevice* device, bool use_prefetcher_cache);

    void invalidate_circular_buffer_allocation();

    void allocate_circular_buffers(const IDevice* device);

    void finalize_offsets(IDevice* device);
    bool is_finalized() const;
    ProgramBinaryStatus get_program_binary_status(std::size_t device_id) const;
    void set_program_binary_status(std::size_t device_id, ProgramBinaryStatus status);
    void allocate_kernel_bin_buf_on_device(IDevice* device);
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;

    // debug/test
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;
};

}  // namespace tt_metal

}  // namespace tt
