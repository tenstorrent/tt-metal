// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {

// Bounds on the interleaved trace buffer page size (see compute_interleaved_trace_buf_page_size).
// Min is bounded by NOC transfer efficiency; max by the prefetcher CmdDatQ size.
inline constexpr uint32_t kMinTraceBufPageSize = 1024;
inline constexpr uint32_t kMaxTraceBufPageSize = 8192;

// Forward decl to avoid including header
class Buffer;

struct TraceWorkerDescriptor {
    uint32_t num_completion_worker_cores = 0;
    uint32_t num_traced_programs_needing_go_signal_multicast = 0;
    uint32_t num_traced_programs_needing_go_signal_unicast = 0;
    bool operator==(const TraceWorkerDescriptor& other) const {
        return num_completion_worker_cores == other.num_completion_worker_cores &&
               num_traced_programs_needing_go_signal_multicast ==
                   other.num_traced_programs_needing_go_signal_multicast &&
               num_traced_programs_needing_go_signal_unicast == other.num_traced_programs_needing_go_signal_unicast;
    }
};

// A kernel config ring range [begin, end) of one programmable core type, in absolute L1 addresses.
struct KernelConfigRegion {
    uint32_t core_type_index = 0;
    uint32_t begin = 0;
    uint32_t end = 0;
};

// The first and last programs of a trace on one device range, for the dispatch at the trace boundary. The trace
// buffer holds the first program's kernel config writes apart from the rest of the trace, as a separate exec buffer.
// When the trace is replayed right after a trace whose last program's kernel config does not overlap this first
// program's, that exec buffer runs before the trace-boundary barrier, so the config is written while the last program
// still runs (see trace_dispatch::issue_trace_commands). Otherwise it runs after the barrier.
// Only traces on a single sub-device in a dynamically allocated trace region are split this way.
struct TraceBoundaryPrograms {
    // The first program's kernel config write commands, ending with CQ_PREFETCH_CMD_EXEC_BUF_END, until they are
    // written to the trace buffer (empty if the trace holds the whole first program).
    std::vector<uint32_t> first_program_config_commands;
    std::vector<KernelConfigRegion> first_program_config_regions;
    // Where the commands sit in the trace buffer, set when it is written: the pages from the page after the trace,
    // as an exec buffer at the trace buffer's address plus first_program_config_address_offset (whole rows of pages
    // across the DRAM banks) from page first_program_config_start_page.
    uint32_t first_program_config_address_offset = 0;
    uint32_t first_program_config_start_page = 0;
    uint32_t first_program_config_num_pages = 0;
    // The last program's worker count and kernel config ranges.
    uint32_t last_program_num_workers = 0;
    std::vector<KernelConfigRegion> last_program_config_regions;
};

struct TraceDescriptor {
    // Mapping of sub_device_id to descriptor
    std::unordered_map<SubDeviceId, TraceWorkerDescriptor> descriptors;
    // Store the keys of the map in a vector after descriptor has finished being populated
    // This is an optimization since we sometimes need to only pass the keys in a container
    std::vector<SubDeviceId> sub_device_ids;
    std::vector<uint32_t> data;
};

struct TraceBuffer {
    std::shared_ptr<TraceDescriptor> desc;
    std::shared_ptr<Buffer> buffer;

    TraceBuffer(std::shared_ptr<TraceDescriptor> desc, std::shared_ptr<Buffer> buffer);
    ~TraceBuffer();
};

}  // namespace tt::tt_metal
