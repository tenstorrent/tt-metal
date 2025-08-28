// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

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

namespace detail {
class ProgramImpl;
}  // namespace detail

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
    // ID related functions:
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

    // Used in tests, fabric, CaptureCreateKernel, light metal, etc.
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
