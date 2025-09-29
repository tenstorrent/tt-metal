// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "hal_types.hpp"
namespace tt::tt_metal {

// Fwd declares
struct ProgramDescriptor;
class CircularBuffer;

namespace detail {
class ProgramImpl;
}  // namespace detail

using ProgramId = std::uint64_t;

using transfer_info_cores = std::variant<CoreCoord, CoreRange>;

struct multicast_transfer_info {
    transfer_info_cores cores;
    uint32_t num_dests;
};

std::vector<multicast_transfer_info> extract_dst_noc_multicast_info(
    IDevice* device, const std::vector<CoreRange>& ranges, CoreType core_type);
struct transfer_info {
    std::uint32_t dst_base_addr;
    std::vector<std::pair<transfer_info_cores, std::uint32_t>> dst_noc_info;  // noc_encoding, num_mcast_dests
    bool linked;
    std::vector<std::uint32_t> data;
};

struct kernel_bins_transfer_info {
    HalProgrammableCoreType core_type;
    HalProcessorClassType processor_class;
    std::vector<std::uint32_t> dst_base_addrs;  // BRISC, NCRISC, TRISC etc..
    std::vector<std::uint32_t> page_offsets;    // offsets into paged buffer in DRAM
    std::vector<std::uint32_t> lengths;         // WriteLinear lengths
    std::vector<std::uint32_t> processor_ids;   // processor ids that each span is targeted for, for binaries
};

struct ProgramTransferInfo {
    std::uint32_t num_active_cores{};
    std::vector<std::tuple<transfer_info_cores, std::uint32_t, kernel_bins_transfer_info>>
        kernel_bins;                         // noc_encoding, num_mcast_dests, transfer_info
    std::vector<std::uint32_t> binary_data;  // Holds binary data for all program kernels
};

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

    // debug/test/internal usage.
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }
    const ProgramTransferInfo& get_program_transfer_info() const noexcept;

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
