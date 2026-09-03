// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental: no API-stability guarantee. Everything in this header may change or be removed.

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/global_circular_buffer.hpp>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PrefetcherPipeImpl;

/**
 * Host object for a durable cross-program remote DFB.
 *
 * Lifetime: construction allocates the data ring + config page from persistent L1
 * pages once, and this handle owns them. Keep it alive for the entire time any program
 * Attaches / uses it; destroying it frees the ring and config, and an attached Program
 * holds a non-owning reference that does not keep either alive. The runtime does not
 * fence peers; only destroy (or let it go out of scope) after every peer program has
 * Finished.
 *
 * The handle is a movable value. An Attach records the implementation object behind it, and a
 * move leaves that object where it is, so a pipe can be handed to a new owner or kept in a
 * container without orphaning an attachment. Moving from a pipe empties it, and an empty pipe
 * is only good for destruction or assignment; move-assigning onto a live pipe destroys the pipe
 * that was there, with the same effect on its attachments as letting it go out of scope.
 *
 * Host programming model:
 *   auto pipe = CreatePrefetcherPipe(device, sender_core, receiver_cores, ring_size);
 *   AttachPrefetcherPipe(program, pipe, sender_cores, entry_size);  // or all receivers
 *   // optional relay to TRISC: CreatePrefetcherPipeRelayDataflowBuffer(program, receivers, cfg, id),
 *   //   declared in impl/dataflow_buffer/prefetcher_pipe.hpp with the DFB config it takes.
 *
 * Device kernel flows (sender / receiver / relay) are documented on the device API:
 *   tt_metal/hw/inc/api/dataflow/prefetcher_pipe.h
 */
class PrefetcherPipe {
public:
    PrefetcherPipe(
        distributed::MeshDevice* device,
        CoreCoord sender_core,
        const CoreRangeSet& receiver_cores,
        uint32_t ring_size,
        BufferType buffer_type = BufferType::L1);

    // Movable, not copyable. A handle is a pointer to the implementation object that owns the
    // durable L1, and a Program's attachment points at that object rather than at the handle, so
    // relocating a handle keeps every attachment valid. Copying would leave two handles freeing
    // one ring.
    PrefetcherPipe(const PrefetcherPipe&) = delete;
    PrefetcherPipe& operator=(const PrefetcherPipe&) = delete;
    PrefetcherPipe(PrefetcherPipe&&) noexcept;
    PrefetcherPipe& operator=(PrefetcherPipe&&) noexcept;
    ~PrefetcherPipe();

    // Base address of the data ring. Every core of the pipe holds its slice at this address.
    uint32_t buffer_address() const;
    // Base address of the config pages. Each core of the pipe holds its own page at this address.
    uint32_t config_address() const;
    uint32_t ring_size() const;
    uint32_t config_page_size() const;
    // The credit counters within a config page, as a byte range relative to config_address(). Zeroing
    // it returns the pipe to its just-created credit state.
    uint32_t credit_reset_offset() const;
    uint32_t credit_reset_size() const;

    CoreCoord sender_core() const;
    const CoreRangeSet& sender_cores() const;
    const CoreRangeSet& receiver_cores() const;
    const CoreRangeSet& all_cores() const;
    distributed::MeshDevice* get_device() const;

    // Worker for a pipe built by CreatePrefetcherPipe, Dram for one whose sender is a
    // programmable DRAM core (CreatePrefetcherPipesForTensorPrefetcher).
    SenderCoreType sender_core_type() const;

    // The single entry size a DRAM-sender pipe is stamped with. A DRAM sender never Attaches, so
    // it can neither observe nor answer a receiver-side resize; every Attach must use this size.
    // 0 for a worker-sender pipe, which resizes normally.
    uint32_t fixed_entry_size() const;

    // Internal: adopt an already-built implementation. The DRAM-sender factory uses it; a
    // worker-sender pipe comes from the constructor above.
    explicit PrefetcherPipe(std::unique_ptr<PrefetcherPipeImpl> impl);

    // Internal (host runtime, tests): the implementation object this pipe owns.
    PrefetcherPipeImpl& impl() { return *pimpl_; }
    const PrefetcherPipeImpl& impl() const { return *pimpl_; }

private:
    std::unique_ptr<PrefetcherPipeImpl> pimpl_;
};

/**
 * @brief Create a PrefetcherPipe host object with an arena-backed data ring and config page.
 *
 * Config pages are written to device L1 at Create (safe-point initial write).
 * Caller keeps the returned pipe alive for cross-program persistence; Attach wires programs
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

// ---- DRAM-sender extension ---------------------------------------------------------------------
// A PrefetcherPipe whose sender is a programmable DRAM core (a Blackhole DRISC) rather than a
// worker core. This is the delivery target the Tensor prefetcher streams into as an alternative to
// a DRAM-sender GlobalCircularBuffer.
//
// A PrefetcherPipe has exactly one sender, so one prefetcher target is a list of them -- one per
// DRISC sender core, built from the mapping BuildTensorPrefetcherSenderMapping returns. The list,
// and whatever the caller wants to bundle with it, is the caller's own concept; this header
// supplies the one-pipe pieces.
//
// Consumers are unchanged from an ordinary PrefetcherPipe: the consumer program calls
// AttachPrefetcherPipe on each pipe's receiver cores and its kernels use the device-side
// experimental::PrefetcherPipe (wait_front / get_read_ptr / pop_front). Only the producer side
// differs, and it is owned by the prefetcher.

// Place the Tensor prefetcher's DRAM sender cores for a (bank id -> receivers) request, returning
// one (DRAM-logical sender core, its receivers) pair per pipe to create.
//
// Placement, the receiver split, and slab numbering are the same ones
// CreateGlobalCircularBufferForTensorPrefetcher uses, so a tensor laid out for one transport is
// laid out for the other. With `dual_senders_per_bank`, a bank whose receiver set has more than
// one core is driven by two DRISC senders that split it ceil/floor -- which requires a
// receiver-contiguous layout, where no shard feeds more than one receiver.
//
// The returned order is semantic and must be preserved: a bank's senders are adjacent, and the
// first of them owns the bank's leading receivers (bank-local slab index 0).
std::vector<std::pair<CoreCoord, CoreRangeSet>> BuildTensorPrefetcherSenderMapping(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    bool dual_senders_per_bank);

// Create one PrefetcherPipe driven by the programmable DRAM core `dram_sender_logical`, holding
// `num_entries` entries of `entry_size` bytes per receiver. Pass a (sender, receivers) pair from
// BuildTensorPrefetcherSenderMapping.
//
// `entry_size` is the per-receiver push granularity and must equal the streamed tensor's
// per-receiver page size: a DRAM sender is never dispatched to and so cannot answer a
// receiver-side resize. The pipe is stamped with it, and both AttachPrefetcherPipe and
// QueueTensorPrefetcherRequest reject any other size (with the offending values in the message).
//
// The ring comes from the persistent L1 arena, which refuses a core a live Program has sealed with
// its own local circular buffers. Create the pipes before running any op on the receiver cores --
// under ttnn's program cache a cached op keeps its Program, and its seal, alive.
//
// Keep the returned pipe alive for as long as any program has Attached it or the prefetcher may
// still deliver into it: an attached Program holds a non-owning pointer to it, and destroying it
// frees the ring and config pages.
//
// MeshDevice-only: the DRISC L1 arena backing the sender config page lives on MeshDeviceImpl.
std::shared_ptr<PrefetcherPipe> CreatePrefetcherPipeForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    CoreCoord dram_sender_logical,
    const CoreRangeSet& receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1);

}  // namespace experimental
}  // namespace tt::tt_metal
