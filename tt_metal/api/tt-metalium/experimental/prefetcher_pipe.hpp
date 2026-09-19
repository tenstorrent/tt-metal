// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental: no API-stability guarantee. Everything in this header may change or be removed.

#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PrefetcherPipeImpl;
class PrefetcherPipeSpaceImpl;

/**
 * Host object for a durable cross-program remote DFB.
 *
 * A PrefetcherPipe is carved from a PrefetcherPipeSpace (below): the space owns the persistent
 * L1 (data ring + config page) on every core it was reserved over, and a pipe is one sender ->
 * receivers mapping inside that reservation. Every pipe carved from one space shares the space's
 * ring and config addresses, which is what lets one Program slot (or one relay DFB) serve several
 * pipes on disjoint cores.
 *
 * Lifetime: a pipe keeps its space alive. Keep the pipe alive for the entire time any Program
 * uses it; destroying it returns its cores to the space (the next carve on those cores rewrites
 * the config pages), and a Program bound to it holds a non-owning reference that does not keep
 * it alive. The runtime does not fence peers; only destroy (or let it go out of scope) after
 * every peer program has Finished.
 *
 * The handle is a movable value. A Program records the implementation object behind it, and a
 * move leaves that object where it is, so a pipe can be handed to a new owner or kept in a
 * container without orphaning a binding. Moving from a pipe empties it, and an empty pipe is only
 * good for destruction or assignment; move-assigning onto a live pipe destroys the pipe that was
 * there, with the same effect on its bindings as letting it go out of scope.
 *
 * Host programming model (Metal 2.0):
 *   auto space = CreatePrefetcherPipeSpace(device, {.sender_cores = ..., .receiver_domain = ...,
 *                                                   .ring_size = R, .max_receivers_per_pipe = N});
 *   PrefetcherPipe pipe = space.create_pipe(sender_core, receiver_cores);
 *   // ProgramSpec declares a PrefetcherPipeParameter with the same geometry; kernels bind it via
 *   // KernelSpec::prefetcher_pipe_bindings; ProgramRunArgs::prefetcher_pipe_args supplies `pipe`.
 *
 * Device kernel flows (sender / receiver / relay) are documented on the device API:
 *   tt_metal/hw/inc/api/dataflow/prefetcher_pipe.h
 */
class PrefetcherPipe {
public:
    // Internal (PrefetcherPipeSpace): wrap a carved implementation object.
    explicit PrefetcherPipe(std::unique_ptr<PrefetcherPipeImpl> impl);

    // Movable, not copyable. A handle is a pointer to the implementation object that owns the
    // claim on the space, and a Program's binding points at that object rather than at the
    // handle, so relocating a handle keeps every binding valid. Copying would leave two handles
    // releasing one claim.
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
    // Size of a config page (the space's page: sized for max_receivers_per_pipe).
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

    // Internal (host runtime, tests): the implementation object this pipe owns.
    PrefetcherPipeImpl& impl() { return *pimpl_; }
    const PrefetcherPipeImpl& impl() const { return *pimpl_; }

private:
    std::unique_ptr<PrefetcherPipeImpl> pimpl_;
};

/**
 * Reservation geometry for a PrefetcherPipeSpace.
 *
 * The space pins persistent L1 on `sender_cores ∪ receiver_domain` before any Program places
 * program-local L1 on those cores. `receiver_domain` is NOT the 1:N map: it is every core that
 * MAY later be carved as a receiver (typically the compute/worker grid or a sub-device), because
 * whoever creates the space (the model / a prefetcher) generally does not know the consumer's
 * receiver layout. The consumer carves pipes once it does.
 */
struct PrefetcherPipeSpaceConfig {
    // Worker cores that may act as pipe senders. Never DRAM coordinates.
    CoreRangeSet sender_cores;
    // Capacity for DRAM-sender endpoints (0 = worker-only space). Names no cores; exact DRAM
    // sender cores are bound through impl-only helpers, not through this public surface. Only 0
    // is accepted until DRAM-sender pipes land (tt-metal#55285).
    uint32_t num_dram_senders = 0;
    // Worker cores that may act as pipe receivers. Non-empty: every pipe has at least one receiver.
    CoreRangeSet receiver_domain;
    // Per-core data ring size in bytes, shared by every pipe carved here. Multiple of L1 alignment.
    uint32_t ring_size = 0;
    // Largest receiver count of any pipe carved here; sizes every core's config page. In
    // [1, receiver_domain.num_cores()].
    uint32_t max_receivers_per_pipe = 0;
    BufferType buffer_type = BufferType::L1;
};

/**
 * A reservation of persistent L1 from which PrefetcherPipes are carved.
 *
 * Creating a space takes ONE ring allocation and ONE config-page allocation over every worker
 * core in `sender_cores ∪ receiver_domain`, so every pipe carved from it shares
 * `buffer_address()` / `config_address()` by contract (not by first-fit accident), and a model
 * can re-carve pipes for a different consumer layout with no reallocation. Every domain core is
 * written a zeroed template page at create; carving writes the real pages for that pipe's cores.
 *
 * Cost: every domain core holds `ring_size + config_page_size()` bytes, used or not, and the page
 * is sized for `max_receivers_per_pipe`. Keep `receiver_domain` to the grid the consumer can
 * actually use.
 *
 * Lifetime: a space outlives its pipes (each pipe holds a reference); the L1 is released when the
 * last handle - space or pipe - goes away.
 */
class PrefetcherPipeSpace {
public:
    // Internal (CreatePrefetcherPipeSpace): wrap the implementation object.
    explicit PrefetcherPipeSpace(std::shared_ptr<PrefetcherPipeSpaceImpl> impl);

    PrefetcherPipeSpace(const PrefetcherPipeSpace&) = delete;
    PrefetcherPipeSpace& operator=(const PrefetcherPipeSpace&) = delete;
    PrefetcherPipeSpace(PrefetcherPipeSpace&&) noexcept;
    PrefetcherPipeSpace& operator=(PrefetcherPipeSpace&&) noexcept;
    ~PrefetcherPipeSpace();

    // Addresses every pipe carved from this space shares.
    uint32_t buffer_address() const;
    uint32_t config_address() const;
    uint32_t ring_size() const;
    uint32_t config_page_size() const;
    uint32_t max_receivers_per_pipe() const;

    const CoreRangeSet& sender_cores() const;
    uint32_t num_dram_senders() const;
    const CoreRangeSet& receiver_domain() const;
    // sender_cores ∪ receiver_domain: the cores holding this space's L1.
    const CoreRangeSet& reservation_cores() const;
    // Reservation cores not currently claimed by a live pipe.
    CoreRangeSet unclaimed_cores() const;
    distributed::MeshDevice* get_device() const;

    /**
     * Carve one pipe. `sender` must be one of `sender_cores` (worker only; DRAM coordinates are
     * rejected), `receivers` a non-empty subset of `receiver_domain` with at most
     * `max_receivers_per_pipe` cores that does not contain `sender`, and none of those cores may
     * be claimed by a live pipe. Writes this pipe's config pages; allocates nothing.
     */
    PrefetcherPipe create_pipe(CoreCoord sender, const CoreRangeSet& receivers);

    /**
     * Carve several disjoint pipes in one call. Validated as a whole before anything is claimed
     * (a core claimed twice within the batch is rejected); create_pipe is this with M = 1.
     */
    std::vector<PrefetcherPipe> create_pipes(std::span<const std::pair<CoreCoord, CoreRangeSet>> pipes);

    // Internal (host runtime, tests): the implementation object this space owns.
    PrefetcherPipeSpaceImpl& impl() { return *pimpl_; }
    const PrefetcherPipeSpaceImpl& impl() const { return *pimpl_; }

private:
    std::shared_ptr<PrefetcherPipeSpaceImpl> pimpl_;
};

/**
 * @brief Reserve persistent L1 for PrefetcherPipes on `config.sender_cores ∪ config.receiver_domain`.
 *
 * Must run before any Program places program-local L1 on those cores (the persistent arena seals
 * a core at that point). Quasar config pages reserve PREFETCHER_PIPE_MAX_CREDIT_LANES credit
 * lanes per receiver slot; the active lane count is a per-Program property set from the receiver
 * kernel's thread count when the pipe is bound.
 */
PrefetcherPipeSpace CreatePrefetcherPipeSpace(distributed::MeshDevice* device, const PrefetcherPipeSpaceConfig& config);

// A Program uses a pipe through the Metal 2.0 host API only: declare a PrefetcherPipeParameter
// with the pipe's geometry in the ProgramSpec, bind it from data-movement kernels via
// KernelSpec::prefetcher_pipe_bindings (optionally aliasing its ring with a relay DFB through
// DataflowBufferSpec::prefetcher_pipe_relays), then supply the PrefetcherPipe object in
// ProgramRunArgs::prefetcher_pipe_args. See metal2_host_api/prefetcher_pipe_parameter.hpp.

}  // namespace experimental
}  // namespace tt::tt_metal
