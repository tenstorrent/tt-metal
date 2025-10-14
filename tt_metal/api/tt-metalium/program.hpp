// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace tt::tt_metal {

// Fwd declares
struct ProgramDescriptor;
class CircularBufferMetadata;

namespace detail {
class ProgramImpl;
}  // namespace detail

using ProgramId = std::uint64_t;

class Program {
public:
    Program();
    explicit Program(const ProgramDescriptor& descriptor);
    ~Program() noexcept;

    Program(const Program& other) = delete;
    Program& operator=(const Program& other) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    //////////////////////////////
    // ID related functions:
    // These are often used in tracing and testing.
    //////////////////////////////

    void set_runtime_id(ProgramId id);
    ProgramId get_runtime_id() const;

    //////////////////////////////
    // Buffer related functions:
    //////////////////////////////

    std::vector<CircularBufferMetadata> circular_buffers() const;

    // debug/test/internal usage.
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;
};

// Only Used in op_profiler, we might want to expose this via a tooling interface instead of through here.
class IDevice;
namespace detail {
struct KernelMeta;
// Collects the meta data of kernels in a program, and the metadata of the binaries within the kernel if device is non-null
// Note: device is nullable
std::vector<detail::KernelMeta> collect_kernel_meta(Program const& program, IDevice* device);
}; //namespace detail

}  // namespace tt::tt_metal
