// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

// Fwd declares
struct ProgramDescriptor;
class CircularBuffer;
class Program;

namespace detail {
class ProgramImpl;
}  // namespace detail

using ProgramId = std::uint64_t;

class Program {
public:
    Program();

    // Alternative "ProgramDescriptor" API, created for TTNN generic op
    explicit Program(const ProgramDescriptor& descriptor);

    // Internal: construct from an already-built ProgramImpl.
    explicit Program(std::shared_ptr<detail::ProgramImpl> impl);

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
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers() const;

    // Per-core L1 footprint of one dataflow buffer, for consumers that account for L1 usage
    // without needing the buffer itself (graph capture, memory estimation).
    //
    // A narrow view rather than the DataflowBufferImpl: those live in an internal namespace, and
    // the alias semantics below are metal-side knowledge that shouldn't be re-derived per consumer.
    struct DataflowBufferFootprint {
        CoreRangeSet core_ranges;
        // entry_size * num_entries — the per-core L1 cost, analogous to CircularBuffer::size().
        uint32_t total_size = 0;
        // Built on memory owned elsewhere (a tensor's buffer) rather than program-lifetime L1.
        // The DFB analog of CircularBuffer::globally_allocated(): the bytes are already accounted
        // for by whoever owns them, so a consumer summing L1 must not count them a second time.
        bool borrows_memory = false;
    };

    // One entry per distinct L1 region. Aliased DFBs share a single region, so only the alias
    // primary is reported — returning every alias would multiply-count one allocation.
    std::vector<DataflowBufferFootprint> dataflow_buffer_footprints() const;

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
