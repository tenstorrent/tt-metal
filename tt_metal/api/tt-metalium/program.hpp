// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace tt::tt_metal {

// Fwd declares
struct ProgramDescriptor;
class CircularBuffer;
class Kernel;

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

    // Used in ops.
    std::span<const std::shared_ptr<CircularBuffer>> circular_buffers() const;

    // Only Used in op_profiler, we might want to expose this via a tooling interface instead of through here.
    // Return type should be any_range
    std::vector<std::shared_ptr<Kernel>> kernels() const;

    // debug/test/internal usage.
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;
};

}  // namespace tt::tt_metal
