// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include "impl/dataflow_buffer/dataflow_buffer.hpp"

#include <variant>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}

namespace experimental {

// Implementation of the PrefetcherPipe host object declared in
// tt-metalium/experimental/prefetcher_pipe.hpp: it owns the persistent L1 allocations, composes
// each core's config page and stamps those pages onto the device. A PrefetcherPipe holds one of
// these and forwards to it, and the host runtime records an Attach by pointing at one. That
// pointer is only as good as the handle: destroying the PrefetcherPipe destroys this object and
// frees the ring, which is why the handle must outlive every Program attached to it. Moving a
// handle hands this object to the new owner without relocating it, so destruction is the only
// way to get there.
class PrefetcherPipeImpl {
public:
    PrefetcherPipeImpl(
        distributed::MeshDevice* device,
        CoreCoord sender_core,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        BufferType buffer_type = BufferType::L1);

    PrefetcherPipeImpl(const PrefetcherPipeImpl&) = delete;
    PrefetcherPipeImpl& operator=(const PrefetcherPipeImpl&) = delete;
    PrefetcherPipeImpl(PrefetcherPipeImpl&&) = delete;
    PrefetcherPipeImpl& operator=(PrefetcherPipeImpl&&) = delete;
    ~PrefetcherPipeImpl();

    uint32_t buffer_address() const;
    uint32_t config_address() const;
    uint32_t ring_size() const { return ring_size_; }
    // Active pipe-consumer lanes (matches relay num_producers when multi-producer).
    uint32_t num_credit_lanes() const { return active_credit_lanes_; }
    // Slots allocated in the config page (Quasar may reserve headroom above active).
    uint32_t credit_lane_capacity() const { return credit_lane_capacity_; }
    // Set active lanes from consumer geometry. Host state only: dispatch packs the value into
    // every attached program's kernel-config slot (ordered with that program), nothing in
    // persistent L1 is written. May upgrade from the Create-time default of 1, or no-op if
    // already equal. Reprogramming to a different value after arming is rejected.
    void set_active_credit_lanes(uint32_t num_lanes);
    // Lane mode (num_lanes > 1) needs an exact entry ring whose entry count is a multiple of
    // num_lanes; throws otherwise. Call before set_active_credit_lanes so a rejected Attach
    // does not leave the persistent pipe re-armed.
    void validate_lane_geometry(uint32_t entry_size, uint32_t num_lanes) const;

    uint32_t config_page_size() const { return config_page_size_; }
    uint32_t credit_reset_offset() const { return credit_reset_offset_; }
    uint32_t credit_reset_size() const { return credit_reset_size_; }
    const std::vector<uint32_t>& config_page(const CoreCoord& core) const;

    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    CoreCoord sender_core() const { return sender_core_; }
    distributed::MeshDevice* get_device() const { return device_; }

private:
    void setup_buffers(BufferType buffer_type);
    void build_config_pages();
    void write_config_to_device();
    void release_allocations() noexcept;

    uint64_t data_allocation_id_ = 0;
    uint64_t config_allocation_id_ = 0;
    uint32_t data_address_ = 0;
    uint32_t config_address_ = 0;
    distributed::MeshDevice* device_ = nullptr;
    CoreCoord sender_core_;
    CoreRangeSet sender_cores_;
    CoreRangeSet receiver_cores_;
    CoreRangeSet all_cores_;
    uint32_t ring_size_ = 0;
    // Physical lane slots in the config page (Create-time allocation).
    uint32_t credit_lane_capacity_ = 1;
    // Active lanes for striping / wait_front (per-program kernel-config slot); set from Attach
    // num_pipe_consumer_threads / relay num_producers.
    uint32_t active_credit_lanes_ = 1;
    uint32_t config_page_size_ = 0;
    uint32_t credit_reset_offset_ = 0;
    uint32_t credit_reset_size_ = 0;
    std::unordered_map<CoreCoord, std::vector<uint32_t>> config_pages_;
};

/**
 * @brief Create and register the local DFB used to relay a PrefetcherPipe to TRISC.
 *
 * The local DFB borrows the PrefetcherPipe data ring. `prefetcher_pipe_id` must already be
 * Attached on `receiver_core_spec`. Relay entry_size / depth must match this Attach's
 * dense entry_size and `ring_size / entry_size`.
 *
 * Declared here rather than alongside the rest of the host API because it takes a DFB config,
 * which has no public header yet.
 *
 * Multi-thread relay (Quasar): set `config.num_producers` / `config.num_consumers` and
 * `pap` / `cap` (STRIDED or ALL) to match the bound kernels' `num_threads_per_cluster`.
 * Contiguous prefetch pages: `cap=ALL` → every consumer Neo sees every entry;
 * `cap=STRIDED` → Neo i owns entries i, i+C, …. For `num_producers>1`, registering
 * the relay programs PrefetcherPipe lane credits from `num_producers` (must match
 * `AttachPrefetcherPipe(..., num_pipe_consumer_threads)` if that already armed lanes).
 * With `num_producers>1` the relay DFB is serialized lane-interleaved (producer h at entries
 * h, h+P, …) for both `cap` values, so an ALL consumer sees entries in ring order rather than
 * the contiguous per-producer blocks a standalone ALL DFB would use.
 * Programming model: create the pipe first, bind sender and consumer programs, then
 * enqueue in either order. Multi-DM *pipe sender* parallelism is separate: partition
 * receivers (Flow C). Mid-kernel entry-size resize with `num_tcs_to_rr > 1` is
 * unsupported (align snaps cursors only; TC geometry is fixed at DFB init).
 *
 * @return Program-unique host DFB id (distinct from `prefetcher_pipe_id`).
 */
uint32_t CreatePrefetcherPipeRelayDataflowBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& receiver_core_spec,
    const dfb::DataflowBufferConfig& config,
    uint8_t prefetcher_pipe_id);

}  // namespace experimental
}  // namespace tt::tt_metal
