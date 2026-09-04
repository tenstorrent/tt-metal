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

class PrefetcherPipe {
public:
    /**
     * Host object for a durable cross-program remote DFB.
     *
     * Lifetime: Create allocates the data ring + config page from persistent L1
     * pages once. Keep this object alive for the entire time any program Attaches / uses
     * it, destroying it frees the ring and config. The runtime does not fence peers;
     * only destroy (or let it go out of scope) after every peer program has Finished.
     *
     * Host programming model:
     *   auto pipe = CreatePrefetcherPipe(device, sender_core, receiver_cores, ring_size);
     *   AttachPrefetcherPipe(program, pipe, sender_cores, entry_size);  // or all receivers
     *   // optional: CreatePrefetcherPipeRelayDataflowBuffer(program, receivers, cfg, id);
     *
     * Device kernel flows (sender / receiver / relay) are documented on the device API:
     *   tt_metal/hw/inc/api/dataflow/prefetcher_pipe.h
     *
     */
    PrefetcherPipe(
        distributed::MeshDevice* device,
        CoreCoord sender_core,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        BufferType buffer_type = BufferType::L1);

    PrefetcherPipe(const PrefetcherPipe&) = delete;
    PrefetcherPipe& operator=(const PrefetcherPipe&) = delete;
    PrefetcherPipe(PrefetcherPipe&&) = delete;
    PrefetcherPipe& operator=(PrefetcherPipe&&) = delete;
    ~PrefetcherPipe();

    uint32_t buffer_address() const;
    uint32_t config_address() const;
    uint32_t ring_size() const { return ring_size_; }

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
     *   - Credit counters cross L1 address spaces. The sender's remote-counter base and each
     *     receiver's ack target are page-relative deltas the host computes so that a sender's
     *     `base + 2*r*L1_ALIGNMENT` lands on receiver r's own page, and a receiver's ack lands
     *     in DRISC L1.
     */
    PrefetcherPipe(
        distributed::MeshDevice* mesh_device,
        CoreCoord dram_sender_logical,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        uint32_t initial_entry_size,
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
    // This pipe's sender config page in the DRISC L1 arena, reserved on its sender core alone so
    // sibling pipes on other banks can hold the same offset. Null for a worker-sender pipe.
    std::shared_ptr<DriscL1Allocation> drisc_config_page_alloc_;
};

/**
 * @brief Create a PrefetcherPipe host object with an arena-backed data ring and config page.
 *
 * Config pages are written to device L1 at Create (safe-point initial write).
 * Caller keeps the object alive for cross-program persistence; Attach wires programs
 * to the same ring/config addresses.
 */
PrefetcherPipe CreatePrefetcherPipe(
    distributed::MeshDevice* device,
    CoreCoord sender_core,
    const CoreRangeSet& receiver_cores,
    uint32_t ring_size,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Attach a PrefetcherPipe to `program` on the given cores (non-owning).
 *
 * `cores` must be a non-empty role-complete subset of the PrefetcherPipe's mapping
 * cores: the sender role is this pipe's one sender, while the receiver role contains
 * every receiver. This prevents one PrefetcherPipe role from being split across Programs.
 * Returns an independent prefetcher_pipe_id in [0, 255).
 *
 * WH/BH: on each sender core, only one DM (BRISC or NCRISC) may own PrefetcherPipe
 * credit / resize / push for that Attach. Both DMs can run on the same physical
 * core, but dual-DM ownership races on local sent counters and the checkpoint
 * cursor. Host binding / kernel placement should pin a single sender DM owner
 * until Attach can enforce this.
 *
 * @param entry_size Dense entry size for this Program execution epoch.
 */
uint8_t AttachPrefetcherPipe(
    Program& program, PrefetcherPipe& prefetcher_pipe, const CoreRangeSet& cores, uint32_t entry_size);

/**
 * @brief Create and register the local DFB used to relay a PrefetcherPipe to TRISC.
 *
 * The local DFB borrows the PrefetcherPipe data ring. `prefetcher_pipe_id` must already be
 * Attached on `receiver_core_spec`. Relay entry_size / depth must match this Attach's
 * dense entry_size and `ring_size / entry_size`.
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
