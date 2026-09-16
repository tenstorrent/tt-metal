// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Experimental: no API-stability guarantee. Everything in this header may change or be removed.

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace experimental {

class PrefetcherPipeImpl;

// Defined in tt-metalium/experimental/global_circular_buffer.hpp; the DRAM-sender flavour of a
// PrefetcherPipe names the same Worker/Dram distinction as a GlobalCircularBuffer. Declared
// opaquely rather than included: naming the enum is all this header needs, and a caller that
// compares against an enumerator includes the GlobalCircularBuffer header anyway.
enum class SenderCoreType : uint8_t;

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

    // The entry size a DRAM-sender pipe is created at: the ring is this many bytes times the
    // requested depth, and it is what a consumer attaching at this size skips the resize handshake
    // for on the first program. Later Attaches and later requests may use any size the ring can
    // hold; a size the ring does not divide leaves a trailing gap both endpoints credit at the
    // wrap. 0 for a worker-sender pipe, which is sized in bytes and resizes normally.
    uint32_t initial_entry_size() const;

    // Reflection for op attribute hashing and serialization. These values identify *this* pipe, not
    // merely a pipe of this shape: a consuming op bakes the config and ring addresses and the
    // receiver placement into its program, so a same-geometry replacement must not hit that op's
    // program cache. A shared_ptr to a pipe reflects as the pipe it points at (ttsl gives a
    // reflective pointee that transparency), so a list of pipes held as an op attribute keys on
    // their identities rather than on where they were allocated.
    //
    // Kept out of aggregate territory by the constructors above: tt_stl's json serializer has an
    // attribute_names specialization and an aggregate one that are unconstrained against each other,
    // and a type satisfying both is an ambiguous partial specialization.
    static constexpr bool ttsl_reflect_through_shared_ptr = true;
    static constexpr auto attribute_names = std::forward_as_tuple(
        "sender_core", "receiver_cores", "config_address", "buffer_address", "initial_entry_size", "ring_size");
    // Spelled out rather than deduced from make_tuple: every accessor but receiver_cores() returns
    // by value, and holding that one by reference keeps a traversal from heap-copying the range set.
    std::tuple<CoreCoord, const CoreRangeSet&, uint32_t, uint32_t, uint32_t, uint32_t> attribute_values() const {
        return {sender_core(), receiver_cores(), config_address(), buffer_address(), initial_entry_size(), ring_size()};
    }

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
// A PrefetcherPipe has exactly one sender, and a bank may be driven by two of them, so one
// prefetcher target is a per-bank group of pipes. CreatePrefetcherPipesForTensorPrefetcher places
// the senders and creates every pipe; the caller only says which receivers each bank feeds.
//
// Consumers are unchanged from an ordinary PrefetcherPipe: the consumer program calls
// AttachPrefetcherPipe on each pipe's receiver cores and its kernels use the device-side
// experimental::PrefetcherPipe (wait_front / scoped_read_lock / pop_front). Only the producer side
// differs, and it is owned by the prefetcher.

// Create the PrefetcherPipes that deliver one Tensor-prefetcher request: one per DRAM sender core,
// bank-major. A bank contributes one pipe, or two when its receivers are split across both of its
// DRISC sender cores, and a bank's pipes stay adjacent and in sender order. That order is the
// caller's to keep -- it is what assigns each sender its bank-local slab base, so an attach id, a
// mapping entry and a pipe share an index. Sender placement, the receiver split, and slab numbering
// are the ones CreateGlobalCircularBufferForTensorPrefetcher uses, so a tensor laid out for one
// transport is laid out for the other.
//
// With `support_multi_receiver_shards` a bank is driven by a single sender, which is what the
// legacy interleaved layout (a shard feeding more than one receiver) requires. Without it — the
// default, matching the receiver-contiguous layout — a bank with more than one receiver gets two
// senders, each pushing roughly half of them: the split is ceil/floor over the bank's ordered
// receivers, so the leading pipe owns ceil(n/2) of them at bank-local slab index 0.
//
// `entry_size` is the per-receiver push granularity a pipe starts life at, and `num_entries` is how
// many of them a receiver's ring holds; together they fix the ring size, which never changes. A
// later Attach and a later queued tensor may use any entry size the ring can hold: the DRAM sender
// snaps its write cursor onto the new grid and publishes the skipped bytes as pad credits, which is
// the same resize handshake a worker sender runs. A size the ring does not divide leaves a trailing
// gap that holds no entry; both endpoints stop at it and credit it as padding at the wrap.
//
// The rings come from the persistent L1 arena, which refuses a core a live Program has sealed with
// its own local circular buffers. Create the pipes before running any op on the receiver cores --
// under ttnn's program cache a cached op keeps its Program, and its seal, alive.
//
// Keep the returned pipes alive for as long as any program has Attached them or the prefetcher may
// still deliver into them: an attached Program holds a non-owning pointer to each pipe, and
// destroying one frees its ring and config pages.
//
// MeshDevice-only: the DRISC L1 arena backing the sender config pages lives on MeshDeviceImpl.
std::vector<std::shared_ptr<PrefetcherPipe>> CreatePrefetcherPipesForTensorPrefetcher(
    distributed::MeshDevice& mesh_device,
    const std::vector<std::pair<uint32_t, CoreRangeSet>>& bank_to_receivers,
    uint32_t entry_size,
    uint32_t num_entries,
    BufferType buffer_type = BufferType::L1,
    bool support_multi_receiver_shards = false);

// One (sender core, its receivers) entry per pipe, in list order. Every layer that walks a pipe
// list -- the prefetcher request path, a consumer op's cache key, a test -- reads the sender
// topology through this, so that they cannot disagree about which sender owns which receivers.
std::vector<std::pair<CoreCoord, CoreRangeSet>> prefetcher_pipe_sender_receiver_mapping(
    const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes);

// Every receiver of every pipe: the core set a consumer program attaches and runs its receiver
// kernel on.
CoreRangeSet prefetcher_pipe_receiver_cores(const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes);

// Attach every pipe to `program` on its own receiver cores, at `entry_size` bytes per entry -- the
// size that program's kernels consume, which need not be the size a pipe was created at: a differing
// size makes the device-side constructor run the resize handshake against the sender. Returns one
// program-local pipe id per pipe, positioned alongside `pipes` (and hence alongside the mapping), so
// a receiver core's kernel can be handed the id of the one pipe it belongs to.
std::vector<uint8_t> AttachPrefetcherPipes(
    Program& program, const std::vector<std::shared_ptr<PrefetcherPipe>>& pipes, uint32_t entry_size);

}  // namespace experimental
}  // namespace tt::tt_metal
