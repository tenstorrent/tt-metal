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
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/semaphore.hpp>

namespace tt {

namespace tt_metal {

// Fwd declares

class Kernel;
class IDevice;
class Program;
class CircularBufferConfig;

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
    using id_t = std::uint64_t;

    Program();
    explicit Program(const ProgramDescriptor& descriptor);
    ~Program() noexcept;

    Program(const Program& other) = delete;
    Program& operator=(const Program& other) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    //////////////////////////////
    // Runtime ID related functions:
    //////////////////////////////

    // Extensively used by tests
    void set_runtime_id(id_t id);
    // Both used in tracing
    id_t get_id() const;
    id_t get_runtime_id() const;

    //////////////////////////////
    // Buffer related functions:
    //////////////////////////////

    // Only used in AssignGlobalBufferToProgram in tt_metal.cpp
    void add_buffer(std::shared_ptr<Buffer> buf);
    // Only used in EnqueueProgram in host_runtime_commands.cpp
    void release_buffers();

    // Used in ops.
    const std::vector<std::shared_ptr<CircularBuffer>>& circular_buffers() const;

    // Only used in UpdateCircularBufferTotalSize in tt_metal.cpp
    void invalidate_circular_buffer_allocation();

    // Always used in conjuction with validate_circular_buffer_region, which is in impl.
    void allocate_circular_buffers(const IDevice* device);

    //////////////////////////////
    // Kernel related functions:
    //////////////////////////////

    // Used in tests
    std::size_t num_kernels() const;
    // Used in test_kernel_compile_cache.cpp, MeshWorkLoadImpl::get_kernels (which is not used anywhere else)
    std::unordered_map<KernelHandle, std::shared_ptr<Kernel>>& get_kernels(uint32_t programmable_core_type_index);
    // Used in EnqueueProgram in hardware_command_queue.cpp
    void allocate_kernel_bin_buf_on_device(IDevice* device);
    // Used in tests, fabric, CaptureCreateKernel, light metal, etc.
    std::shared_ptr<Kernel> get_kernel(KernelHandle kernel_id) const;

    //////////////////////////////
    // Semaphore related functions:
    //////////////////////////////

    // Used by MeshWorkloadImpl::semaphores, which is not used anywhere else
    const std::vector<Semaphore>& semaphores() const;
    // Not used outside of tests
    std::size_t num_semaphores() const;
    // Used in ConfigureDeviceWithProgram in tt_metal.cpp
    void init_semaphores(
        const IDevice& device, const CoreCoord& logical_core, uint32_t programmable_core_type_index) const;

    // Used in ConfigureDeviceWithProgram in tt_metal.cpp
    // Used in Device::init_command_queue_device, Device::configure_fabric in device.cpp
    std::vector<std::vector<CoreCoord>> logical_cores() const;

    //////////////////////////////
    // Program Binary Status related functions:
    //////////////////////////////

    // Both used in hardware_command_queue.cpp and host_runtime_commands.cpp
    ProgramBinaryStatus get_program_binary_status(chip_id_t device_id) const;
    void set_program_binary_status(chip_id_t device_id, ProgramBinaryStatus status);

    // debug/test
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;
};

}  // namespace tt_metal

}  // namespace tt
