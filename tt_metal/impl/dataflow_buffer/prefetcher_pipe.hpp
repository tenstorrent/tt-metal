// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/hal_types.hpp>
#include "impl/dataflow_buffer/dataflow_buffer.hpp"

#include <variant>

namespace tt::tt_metal {

class DriscL1Allocation;
class IDevice;
class Program;

namespace distributed {
class MeshDevice;
}

namespace experimental {

// Defined in tt-metalium/experimental/global_circular_buffer.hpp; the DRAM-sender flavour of a
// PrefetcherPipe names the same Worker/Dram distinction as a GlobalCircularBuffer.
enum class SenderCoreType : uint8_t;

// Forward declaration for the DRAM-sender extension, defined in
// impl/buffers/prefetcher_pipe_dram_sender_internal.hpp. DRAM-sender mode is opt-in and is not
// part of the public PrefetcherPipe API surface; existing callers see the original interface
// unchanged.
namespace prefetcher_pipe_dram_sender {
struct PrefetcherPipeDramSenderInternals;
}  // namespace prefetcher_pipe_dram_sender

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

    // Worker for a pipe built by CreatePrefetcherPipe, Dram for one whose sender is a
    // programmable DRAM core (CreatePrefetcherPipesForTensorPrefetcher).
    SenderCoreType sender_core_type() const { return sender_core_type_; }

    // The entry size a DRAM-sender pipe is created at: the ring is this many bytes times the
    // requested depth, and it is what word[5] (applied_entry_size) starts out holding, so a
    // consumer attaching at this size skips the resize handshake on the first program. Later
    // Attaches and later requests may use any size the ring can hold; a size the ring does not
    // divide leaves a trailing gap both endpoints credit at the wrap. 0 for a worker-sender
    // pipe, which is sized in bytes and resizes normally.
    uint32_t initial_entry_size() const { return initial_entry_size_; }

    // Bank-local slab index of this DRAM sender's first receiver: local receiver r of this pipe
    // owns slab recv_index_base() + r of its bank. 0 for the only (or leading) pipe of a bank, and
    // for a worker-sender pipe. Host-side bookkeeping only -- CreatePrefetcherPipesForTensorPrefetcher
    // sets it from the mapping it built, and the Tensor prefetcher stamps it into the request pages
    // it routes here, so a caller may hand the factory's pipes on in any order or subset.
    uint32_t recv_index_base() const { return recv_index_base_; }

private:
    // Tag selecting the DRAM-sender constructor. Private so the only way in is
    // CreatePrefetcherPipesForTensorPrefetcher, which owns the bank -> sender-core mapping.
    struct DramSenderTag {};

    friend struct prefetcher_pipe_dram_sender::PrefetcherPipeDramSenderInternals;

    /**
     * DRAM-sender PrefetcherPipe: the sender is a programmable DRAM core (a Blackhole DRISC)
     * rather than a worker core.
     *
     * Differences from the worker-sender ctor above:
     *   - The data ring and the config pages are sharded over receivers only. A DRAM core holds
     *     no ring slice, and its logical coord is not a worker coord, so it cannot share a
     *     persistent-L1 arena allocation with the receivers.
     *   - The sender's config page lives in DRISC L1 on its own core alone, written directly over
     *     NOC, and is never Attached: DRAM cores are not dispatched to, so the DRISC kernel builds
     *     its sender interface from an explicit config-page address instead of a launch-message
     *     slot.
     *   - Credit counters cross L1 address spaces. Both endpoints carry a page-relative delta to
     *     the peer's counter base (word[9]) that the host computes, so that a sender's
     *     `base + r*L1_ALIGNMENT` lands in receiver r's own page, and a receiver's ack lands in
     *     DRISC L1. On a worker-sender pipe, where the two pages share an address, that delta is 0.
     */
    PrefetcherPipeImpl(
        distributed::MeshDevice* mesh_device,
        CoreCoord dram_sender_logical,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        uint32_t initial_entry_size,
        uint32_t recv_index_base,
        BufferType buffer_type,
        DramSenderTag);

    void setup_buffers(BufferType buffer_type);
    void build_config_pages();
    // DRAM-sender flavour of build_config_pages: receiver pages only, and returned rather than
    // cached in config_pages_ because the sender's virtual DRAM coord -- and so the pages -- vary
    // with each device's harvesting.
    std::unordered_map<CoreCoord, std::vector<uint32_t>> build_dram_sender_receiver_config_pages(
        IDevice* target_device) const;
    // Reserve this pipe's DRISC L1 config page on its sender core, then compose and stamp it on
    // every device.
    void initialize_dram_sender_config_page();
    void write_config_to_device();
    // Record the config-page geometry the credit reset works from: both credit blocks sit in the
    // page's tail, so the reset window is everything from the SENT block to the end of the page.
    void set_config_page_geometry(uint32_t page_size, uint32_t credit_reset_offset);
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

    // ---- DRAM-sender state (unset for a worker-sender PrefetcherPipe) ----
    // The opaque-enum declaration above makes SenderCoreType a complete type here, so the member
    // needs no integer stand-in; only the enumerator names need the experimental header, and the
    // constructors that spell them live in the .cpp.
    SenderCoreType sender_core_type_{};
    uint32_t initial_entry_size_ = 0;
    // See recv_index_base(). Never written to the device from here: the request pages the Tensor
    // prefetcher builds carry it, and nothing in the pipe's own config page depends on it.
    uint32_t recv_index_base_ = 0;
    // This pipe's sender config page in the DRISC L1 arena, reserved on its sender core alone so
    // sibling pipes on other banks can hold the same offset. Null for a worker-sender pipe.
    std::shared_ptr<DriscL1Allocation> drisc_config_page_alloc_;
};

/**
 * @brief Create and register the local DFB used to relay a PrefetcherPipe to TRISC.
 *
 * The local DFB borrows the PrefetcherPipe data ring. `prefetcher_pipe_id` must already be
 * Attached on `receiver_core_spec`. Relay entry_size must match this Attach's dense entry_size,
 * and depth must be `ring_size / entry_size` — the whole entries the ring holds. An entry size
 * that does not divide the ring is allowed: the trailing remainder is a gap holding no entry, and
 * the relay leaves it alone just as the receiver does.
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
