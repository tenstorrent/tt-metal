// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The row-packet relay: receive a packet, hand it to the compute kernel, and
// forward it to the next consumer. Algorithm 3 of main.tex, barrier variant.
//
// The paper's two receive slots are the compute kernel's input buffers with
// room for two packets. That is not a coincidence to be worked around but the
// thing that makes this simple: the circular-buffer protocol already is the
// credit-and-readiness protocol, one core wide, and the multicast receivers
// elsewhere in tt-train use it the same way. A receiver reserves its slot,
// credits the producer, waits for the payload, and pushes; a producer waits
// for the credit, writes the payload, completes it, and publishes readiness.
//
// The packet for destination timestep u occupies slot u mod 2, which is also
// where the receiver's write pointer stands after u pushes, so a producer can
// compute the destination address without tracking the receiver: every core
// has the same buffer layout, so its own base address is the receiver's too.
//
// This kernel both receives and forwards, because forwarding needs the
// packet's immutable fields -- which live in its slots -- and the updated
// dQ, which the compute kernel produces. Splitting those across two RISCs
// would need a handoff of its own (O10 in overlaps.md). The write kernel is
// left with the column gradients and the barrier.
//
// The credit is granted where the multicast receivers elsewhere in tt-train
// grant it: immediately after the reserve that frees the slot, at the start
// of the destination timestep. The producer, still finishing the previous
// timestep, then wakes and sends. That looks late -- the packet arrives just
// in time rather than early -- but it cannot deadlock, because a receiver
// reaching its reserve never needs anything from the producer's *next*
// timestep: its column state comes from DRAM and its barrier release depends
// on the producer's write kernel, which does not wait on the relay.
//
// Granting a timestep earlier, at the release, is what the paper does and
// what the transport probe does. It is not available here: the uses of a slot
// are spread over two RISCs, and the compute kernel reads Q again for dK
// *after* it has published the updated dQ, so nothing this kernel can observe
// says the slot is free any sooner.
//
// The self-transition is the exception, and it has to be. There the producer
// and the receiver are the same core, so a credit it grants itself at the
// start of timestep t + 1 would be waited for during timestep t -- the same
// iteration, in the wrong order, which deadlocks at C = 1 immediately. It
// needs no credit: it reserves the destination slot itself, which is the same
// wait expressed locally.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

// The forward is split, because only one of the packet's five fields depends
// on this core's arithmetic. Q, dO, L and D are immutable -- the packet
// carries them unchanged -- so they go out the moment they arrive, before any
// compute. dQ is the accumulator each consumer adds to, so it goes last, with
// its own readiness tag. A consumer can then get as far as dS on the
// immutable fields alone, which is exactly how long it has before it needs
// dQ: the compute kernel does not touch it until after dS.
//
// ENDPOINT_SYNC selects Algorithm 4: no chip-wide barrier. Within a streak
// the packet is itself the ordering token -- a consumer cannot update dQ_i
// before the previous consumer forwards it. Across a gap, two endpoint
// counters order a reload after the preceding streak's spill: every streak
// followed by a later streak ends on core 1 or core 2, so those two cores
// publish a monotone progress value after an inter-streak spill and a
// consumer waits for it.
//
// The threshold is t_prev + 1, never t. The row is inactive at t - 1 at every
// later streak start and progress is published only for spill events, so
// waiting for t waits for a publication that never comes.
//
// Progress is pulled, not pushed, and that is a departure from the paper
// worth explaining. The paper has each endpoint publish its value into a
// local copy on every participating core, and each consumer poll its own L1.
// Measured, that is what made this variant slower than the barrier it
// replaces: a publication is C writes, the number of publications grows with
// C too, so the traffic is quadratic in C while the work per core is not --
// 12096 semaphore writes at C = 64, and a 1.53x deficit against Algorithm 3.
// Multicast would fix the constant but cannot be issued from this RISC; it
// hangs in its write barrier, on this NoC and on NoC 0 with the rectangle
// named there, while the identical multicast from the write kernel works.
//
// So an endpoint now writes only its own word -- one write per inter-streak
// spill -- and a consumer that needs the value reads that word remotely,
// which is what the cross-core barrier in tt-metal's own debug checkpoint
// does for its non-coordinators. The traffic becomes one write per spill plus
// a few reads per wait, and only the cores that actually wait pay anything:
// 189 wait events at C = 64 against 12096 writes.
//
// The safety property is unchanged, which is what matters: the endpoint sets
// its word only after its spill has completed, so observing a value at least
// t_prev + 1 still certifies that spill is visible. What is lost is the
// paper's local-polling property, and with it any need for ordered
// publication -- there is only one writer of each word now.
#ifndef ENDPOINT_SYNC
#define ENDPOINT_SYNC 0
#endif

// SEED_COLUMN_GRADIENTS: read the column gradients from DRAM on every visit,
// not only on a revisit, so the compute kernel accumulates into whatever the
// caller passed as the outputs. See the compute kernel.
#ifndef SEED_COLUMN_GRADIENTS
#define SEED_COLUMN_GRADIENTS 0
#endif

// DENSE_MODE selects the unmasked schedule: every block pair is live, which
// is what a ring-attention step needs when the visiting key/value chunk is
// earlier in the sequence than the local query chunk. It changes the schedule
// (2T timesteps in two passes rather than T + 1) and, in the compute kernel,
// removes the intra-block mask; nothing else about the relay changes.
#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

// Under NoC event tracing the zones below are noise, and at a whole launch
// they overflow the per-core marker buffer, which breaks the trace's zone
// pairing; compiled out there, unchanged in an ordinary profiling build.
#if defined(PROFILE_NOC_EVENTS)
#undef DeviceZoneScopedN
#define DeviceZoneScopedN(name)
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    // Which (batch, head) slice this core's group is working on. The grid
    // holds one independent schedule per group -- the snake never leaves a
    // group, so nothing crosses between them -- and a group touches only its
    // own slice of every tensor.
    const uint32_t first_slice = get_arg_val<uint32_t>(arg++);
    const uint32_t query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t u_scalar_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);
    // Snake neighbours, the only cores a packet ever moves between.
    const uint32_t prev_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t prev_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_y = get_arg_val<uint32_t>(arg++);
    // Every participating core's coordinates, for publishing endpoint
    // progress: (x, y) per core, cores 1..C in order.
    const uint32_t core_coords_arg = arg;

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    // How many (batch, head) slices this group runs, one after another, and
    // the stride between them: slices are dealt round-robin to groups, so the
    // group's k-th slice is first_slice + k * stride. This is what lets a
    // launch carry more slices than the grid has rectangles for.
    const uint32_t slice_count = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores);
    const uint32_t slice_stride = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 1u);
    // Sub-problems as chunk pairs. Every tensor's sequence is `chunks` equal
    // chunks of T = 2C blocks, and slice sl is (batch x head) sl / pairs
    // running pair sl % pairs: its query-side tensors (Q, dO, L, D, dQ) come
    // from chunk row_chunk, its key-side ones (K, V, dK, dV) from col_chunk.
    // With one chunk and one pair this is the op as it was.
    const uint32_t chunks = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 2u);
    const uint32_t pairs = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 3u);
    const uint32_t heads = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 4u);
    // Grouped-query attention: K, V, dK and dV have kv_heads heads per batch,
    // heads_per_group query heads each, kv_slices = batch x kv_heads slices in
    // all. A slice index idx (below) is decoded as idx = sub x kv_slices + bg:
    // bg is the (batch, key head) slice the key-side tensors are addressed by,
    // sub which of the group's query heads this is. With one query head per
    // key head kv_slices = heads and the decode is the identity.
    const uint32_t kv_slices = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 5u);
    const uint32_t q_heads = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 6u);
    const uint32_t kv_heads = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 7u);
    const uint32_t heads_per_group = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 8u);
    const uint32_t pair_table_arg = core_coords_arg + 2u * kCores + 9u;
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(3);
    // Two readiness words per slot, each carrying the destination timestep's
    // tag rather than a count: a slot's uses are u, u + 2, ... so the tags
    // u + 1, u + 3, ... increase, and a stale tag from the slot's previous
    // use cannot satisfy the wait.
    //
    // Two rather than one because the packet's four immutable fields and its
    // dQ become available at different times, and a consumer needs them at
    // different times too.
    constexpr uint32_t ready_imm_sem_id[2] = {
        get_compile_time_arg_val(4), get_compile_time_arg_val(5)};
    constexpr uint32_t ready_dq_sem_id[2] = {
        get_compile_time_arg_val(6), get_compile_time_arg_val(7)};
    constexpr uint32_t credit_prev_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t credit_next_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t credit_self_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t endpoint1_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t endpoint2_sem_id = get_compile_time_arg_val(12);
    // Row-tiles per block: B = Bt * 32. Every packet field and every column
    // buffer is that many times taller, and the row-packet strides with it.
    constexpr uint32_t Bt = get_compile_time_arg_val(13);
    constexpr uint32_t row_tiles = Bt * qWt;  // Q, dQ, K, dK per block
    constexpr uint32_t val_tiles = Bt * vWt;  // dO, V, dV per block
    constexpr auto query_args = TensorAccessorArgs<14>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto grad_output_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<lse_args.next_compile_time_args_offset()>();
    constexpr auto grad_query_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    // The packet: the compute kernel's inputs, two slots deep.
    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_lse = tt::CBIndex::c_4;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
    // The same statistics gathered into row 0 of a tile, for the compute
    // kernel's row broadcast (it forms S^T); see cyclic_dataflow_utils.hpp.
    // The column-layout tiles above still travel with the packet unchanged.
    constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
    // Column state, still per timestep from DRAM, gradients included.
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
    constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
    // The compute kernel's updated dQ, which travels on with the packet.
    constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
    // The compute kernel's release token: one page per timestep, published
    // once it has popped that timestep's packet slot.
    constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;
    // A local word to publish readiness tags and endpoint progress from.
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;
    constexpr uint32_t cb_column_progress = tt::CBIndex::c_26;

    using namespace ttml::metal::ops::cyclic_sdpa_bw;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();
    const auto neighbors = snake_neighbors(kCores, my_core);

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const uint32_t interm_bytes = get_tile_size(cb_u_scalar);  // the L buffer holds the statistic block now
    const uint32_t grad_bytes = get_tile_size(cb_grad_query_seed);

    const auto query = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto grad_output = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto lse = TensorAccessor(lse_args, lse_addr, interm_bytes);
    const auto u_scalar = TensorAccessor(u_args, u_scalar_addr, interm_bytes);
    const auto grad_query = TensorAccessor(grad_query_args, grad_query_addr, grad_bytes);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);

    // Buffer bases, captured before any reserve so they are the slot-0
    // addresses -- and, because every core has the same layout, also the
    // receiver's.
    const uint32_t base_query = get_write_ptr(cb_query);
    const uint32_t base_grad_output = get_write_ptr(cb_grad_output);
    // L and D as loaded from DRAM at a streak start are scratch here (the D
    // buffer, two slots each; its pages double as the writer's "block made"
    // signal), and the packet carries the statistic block instead (the L
    // buffer, 512 bytes a row tile; see cyclic_dataflow_utils.hpp).
    const uint32_t base_lse = get_write_ptr(cb_u_scalar);
    const uint32_t base_u_scalar = base_lse + 2u * Bt * interm_bytes;
    // The block is memory only, no buffer protocol: its slot is reused two
    // timesteps later, after the release that follows the compute kernel's
    // pop, and the writer has expanded it before the compute kernel starts.
    const uint32_t base_block = get_write_ptr(cb_lse);
    const uint32_t stride_block = Bt * cyclic_dataflow::kStatBlockBytes;
    const uint32_t base_grad_query = get_write_ptr(cb_grad_query_seed);
    const uint32_t stride_query = row_tiles * tile_bytes;
    const uint32_t stride_grad_output = val_tiles * tile_bytes;
    const uint32_t stride_interm = Bt * interm_bytes;
    const uint32_t stride_grad_query = row_tiles * grad_bytes;
    // T = 2C blocks of Bt tiles each, so a chunk is 2 * kCores * Bt tile rows.
    // Per slice; set at the top of each slice below. The row bases address
    // the query-side tensors, the column bases the key-side ones.
    uint32_t row_base = 0;
    uint32_t val_base = 0;
    uint32_t stat_base = 0;
    uint32_t col_row_base = 0;
    uint32_t col_val_base = 0;

    volatile tt_l1_ptr uint32_t* release_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
    volatile tt_l1_ptr uint32_t* ready_imm_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_imm_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_imm_sem_id[1]))};
    volatile tt_l1_ptr uint32_t* ready_dq_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_dq_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_dq_sem_id[1]))};
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* credit_from_prev =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_prev_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_next =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_next_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_self =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_self_sem_id));

#if ENDPOINT_SYNC
    // Every core holds a local copy of both endpoint counters, so a consumer
    // polls its own L1 rather than a remote word.
    volatile tt_l1_ptr uint32_t* endpoint_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(endpoint1_sem_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(endpoint2_sem_id))};
    const uint32_t endpoint_sem_ids[2] = {endpoint1_sem_id, endpoint2_sem_id};
    // A dedicated word to read a remote endpoint's progress into, clear of the
    // one used to publish readiness tags.
    const uint32_t endpoint_read_l1 = scratch_l1 + 16u;
    volatile tt_l1_ptr uint32_t* endpoint_read =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(endpoint_read_l1);
    // This core's write kernel says when the column gradients have landed.
    volatile tt_l1_ptr uint32_t* column_progress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_column_progress));
#endif

    // Which columns this core owns, and whether each has been resident.
    const auto owned = sched.owned_columns(my_core);
    const uint32_t owned_column[2] = {owned.first, owned.second};
    bool visited[2] = {false, false};  // reset per slice

    uint32_t sent_to_prev = 0;
    uint32_t sent_to_next = 0;
    uint32_t sent_to_self = 0;

    const auto noc_xy_of = [&](uint32_t core, uint32_t& x, uint32_t& y) {
        if (core == neighbors.prev) {
            x = prev_noc_x;
            y = prev_noc_y;
        } else {
            x = next_noc_x;
            y = next_noc_y;
        }
    };

    // A slot release grants permission to whoever produces for it next. Seen
    // from that producer, this core is its snake successor when it is this
    // core's predecessor, and the other way round.
    const auto grant_credit_to = [&](uint32_t p) {
        if (p == my_core) {
            noc_semaphore_inc(get_noc_addr(get_semaphore(credit_self_sem_id)), 1u);
            return;
        }
        uint32_t x = 0;
        uint32_t y = 0;
        noc_xy_of(p, x, y);
        const uint32_t sem_id = (p == neighbors.prev) ? credit_next_sem_id : credit_prev_sem_id;
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1u);
    };

    const auto await_credit = [&](uint32_t r) {
        WAYPOINT("CRDW");
        if (r == my_core) {
            const uint32_t want = ++sent_to_self;
            do {
                invalidate_l1_cache();
            } while ((*credit_from_self) < want);
        } else if (r == neighbors.prev) {
            const uint32_t want = ++sent_to_prev;
            do {
                invalidate_l1_cache();
            } while ((*credit_from_prev) < want);
        } else {
            const uint32_t want = ++sent_to_next;
            do {
                invalidate_l1_cache();
            } while ((*credit_from_next) < want);
        }
        WAYPOINT("CRDD");
    };

    // Initial permissions, for destination timesteps 0 and 1: there is no
    // release two timesteps before those, so the receivers grant them up
    // front. Increments are atomic and order-independent, so this needs no
    // synchronisation of its own.
    for (uint32_t s = 0; s < slice_count; ++s) {
    // Slices are pair-major: all heads of pair 0, then of pair 1 (see the host).
    const uint32_t sl = first_slice + s * slice_stride;
    const uint32_t idx = sl % heads;
    const uint32_t pair = sl / heads;
    // Which (batch, key head) the key-side tensors belong to, and which query
    // head of that group this slice is (see the header above). The query-side
    // tensors are addressed by the (batch, query head) slice bh.
    const uint32_t bg = idx % kv_slices;
    const uint32_t sub = idx / kv_slices;
    const uint32_t bh = (bg / kv_heads) * q_heads + (bg % kv_heads) * heads_per_group + sub;
    const uint32_t row_chunk = get_arg_val<uint32_t>(pair_table_arg + 2u * pair);
    const uint32_t col_chunk = get_arg_val<uint32_t>(pair_table_arg + 2u * pair + 1u);
    row_base = (bh * chunks + row_chunk) * 2u * kCores * row_tiles;
    val_base = (bh * chunks + row_chunk) * 2u * kCores * val_tiles;
    stat_base = (bh * chunks + row_chunk) * 2u * kCores * Bt;
    col_row_base = (bg * chunks + col_chunk) * 2u * kCores * row_tiles;
    col_val_base = (bg * chunks + col_chunk) * 2u * kCores * val_tiles;
    visited[0] = false;
    visited[1] = false;

    // Initial permissions for this slice's destination timesteps 0 and 1.
    // Safe to grant here even though the compute kernel may still be on the
    // previous slice's last pair: this reader only reaches the top of a
    // slice after that pair's slot-release token, and pops are in order, so
    // both slots are free. Credits are monotone counters per (producer,
    // receiver), and the sequence of grants and forwards between a pair
    // repeats identically per slice, so the counts stay matched.
    for (uint32_t u = 0; u < 2u && u < kTimesteps; ++u) {
        const auto initial = sched.producer(my_core, u);
        if (initial.internal && initial.core != my_core) {
            grant_credit_to(initial.core);
        }
    }
    noc_async_atomic_barrier();

    // Whether the next timestep's constant packet fields (Q, dO, L, D) were
    // already read from DRAM during this one's dQ wait; see the prefetch below.
    bool prefetched_next = false;
    bool prefetched_dq_next = false;
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const bool prefetched = prefetched_next;
        const bool prefetched_dq = prefetched_dq_next;
        prefetched_next = false;
        prefetched_dq_next = false;
        DeviceZoneScopedN("RELAY-READER-STEP");
        const auto pair = sched.pair(my_core, t);
        const uint32_t i = pair.i;
        const uint32_t j = pair.j;
        // The global timestep across slices. Everything that must stay
        // monotone or consistent with a circular buffer's write pointer --
        // slot parity, readiness tags, endpoint values, progress words --
        // is derived from it. The schedule itself is per slice and uses t.
        // With T + 1 odd, a slice boundary would otherwise land two
        // consecutive uses on the same slot while the receiver's pointer
        // had moved on.
        const uint32_t g = s * kTimesteps + t;
        const uint32_t slot = g % 2u;

        // Column state, resident for a whole residency interval: K_j and V_j
        // are read only when the column changes, which the schedule says
        // happens exactly twice per core over T + 1 timesteps, always at a
        // diagonal block. The reserve waits for the compute kernel to have
        // popped the previous column, which is the paper's rule that all
        // operations using that storage complete before it is reused.
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != j);
        if (column_changed) {
            DeviceZoneScopedN("LOAD-COL");
            read_tiles_by_row(cb_key, key, col_row_base + (j - 1u) * row_tiles, row_tiles, tile_bytes, row_tiles);
            read_tiles_by_row(cb_value, value, col_val_base + (j - 1u) * val_tiles, val_tiles, tile_bytes, val_tiles);
        }

        // Accumulated column gradients are needed only where an interval
        // starts on a column this core has been resident on before -- one
        // revisit per core, its third interval. They were stored by this
        // core's own write kernel at the end of the earlier interval, so the
        // wait is on that kernel's progress, not on anything chip-wide.
        if (column_changed) {
            const uint32_t owned_slot = (j == owned_column[0]) ? 0u : 1u;
            // On a first visit that is seeded, the DRAM value is the caller's
            // incoming accumulator, which nothing on this core wrote; the wait
            // below is then trivially satisfied and harmless.
            if (visited[owned_slot] || SEED_COLUMN_GRADIENTS) {
                WAYPOINT("COLW");
#if ENDPOINT_SYNC
                do {
                    invalidate_l1_cache();
                } while ((*column_progress) < g);
#else
                do {
                    invalidate_l1_cache();
                } while ((*release_sem) < g);
#endif
                WAYPOINT("COLD");
                DeviceZoneScopedN("LOAD-COL-SEEDS");
                read_tiles_by_row(
                    cb_grad_key_seed, grad_key, col_row_base + (j - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
                read_tiles_by_row(
                    cb_grad_value_seed, grad_value, col_val_base + (j - 1u) * val_tiles, val_tiles, grad_bytes,
                    val_tiles);
            }
            visited[owned_slot] = true;
        }
        // Reserve this timestep's slot in every packet buffer, which is the
        // release of t - 2 becoming a credit for whoever fills it.
        {
            DeviceZoneScopedN("SLOT-RESERVE");
            if (!prefetched) {
                cb_reserve_back(cb_query, row_tiles);
                cb_reserve_back(cb_grad_output, val_tiles);
            }
            cb_reserve_back(cb_grad_query_seed, row_tiles);
        }

        const uint32_t qs = base_query + slot * stride_query;
        const uint32_t os = base_grad_output + slot * stride_grad_output;
        const uint32_t ls = base_lse + slot * stride_interm;
        const uint32_t ds = base_u_scalar + slot * stride_interm;
        const uint32_t gs = base_grad_query + slot * stride_grad_query;

        const auto producer = sched.producer(my_core, t);
        if (producer.internal) {
            // Inside a streak: Q, dO, L and D arrive first, and they are all
            // the compute kernel needs to get as far as dS. The credit for
            // this slot was granted at its release, two timesteps back.
            DeviceZoneScopedN("RECV-IMM");
            WAYPOINT("RDYW");
            do {
                invalidate_l1_cache();
            } while ((*ready_imm_sem[slot]) < g + 1u);
            WAYPOINT("RDYD");
        } else {
            // A streak start: load the packet from DRAM. dQ_i must carry
            // every earlier update.
#if ENDPOINT_SYNC
            if (sched.is_later_streak_start(i, t)) {
                // Order this reload after the preceding streak's spill. The
                // threshold certifies that actual spill, not the inactive
                // timestep t - 1.
                const uint32_t e = sched.spill_endpoint(i, t);
                const uint32_t want = s * kTimesteps + sched.endpoint_threshold(i, t);
                const uint32_t endpoint_x = get_arg_val<uint32_t>(core_coords_arg + 2u * (e - 1u));
                const uint32_t endpoint_y =
                    get_arg_val<uint32_t>(core_coords_arg + 2u * (e - 1u) + 1u);
                const uint64_t endpoint_addr =
                    get_noc_addr(endpoint_x, endpoint_y, get_semaphore(endpoint_sem_ids[e - 1u]));
                WAYPOINT("ENDW");
                if (e == my_core) {
                    // Its own word: a later streak of this row can be consumed
                    // on the endpoint itself, which is the case the paper's
                    // remark works through.
                    do {
                        invalidate_l1_cache();
                    } while ((*endpoint_sem[e - 1u]) < want);
                } else {
                    do {
                        noc_async_read(endpoint_addr, endpoint_read_l1, sizeof(uint32_t));
                        noc_async_read_barrier();
                        invalidate_l1_cache();
                    } while ((*endpoint_read) < want);
                }
                WAYPOINT("ENDD");
            }
#endif
            DeviceZoneScopedN("LOAD-IMM-DRAM");
            if (!prefetched) {
                for (uint32_t k = 0; k < row_tiles; ++k) {
                    noc_async_read_page(row_base + (i - 1u) * row_tiles + k, query, qs + k * tile_bytes);
                }
                for (uint32_t k = 0; k < val_tiles; ++k) {
                    noc_async_read_page(val_base + (i - 1u) * val_tiles + k, grad_output, os + k * tile_bytes);
                }
                for (uint32_t k = 0; k < Bt; ++k) {
                    noc_async_read_page(stat_base + (i - 1u) * Bt + k, lse, ls + k * interm_bytes);
                    noc_async_read_page(stat_base + (i - 1u) * Bt + k, u_scalar, ds + k * interm_bytes);
                }
            }
            noc_async_read_barrier();
            // The writer makes the statistic block from L and D, into this
            // slot; it then travels with the packet, so this happens only
            // here, where the row entered from DRAM. A prefetched packet was
            // signalled when its reads completed, a timestep ago.
            if (!prefetched) {
                cb_reserve_back(cyclic_dataflow::kStatsReadyCb, 1);
                cb_push_back(cyclic_dataflow::kStatsReadyCb, 1);
            }
            cb_wait_front(cyclic_dataflow::kStatsDoneCb, 1);
            cb_pop_front(cyclic_dataflow::kStatsDoneCb, 1);
        }

        // The compute kernel can start on S, P, dP and dS now. It does not
        // touch dQ until after dS, so the rest of the packet has that long to
        // arrive.
        cb_push_back(cb_query, row_tiles);
        cb_push_back(cb_grad_output, val_tiles);

        // Forward the immutable fields straight away, before this core has
        // computed anything: the packet carries them unchanged, so they never
        // depend on the arithmetic. Only dQ does.
        const uint32_t receiver = sched.next_consumer(i, t);
        const uint32_t u = g + 1u;  // destination global timestep
        const uint32_t dst = u % 2u;
        const uint32_t dst_query = base_query + dst * stride_query;
        const uint32_t dst_grad_output = base_grad_output + dst * stride_grad_output;
        const uint32_t dst_block = base_block + dst * stride_block;
        const uint32_t src_block = base_block + slot * stride_block;
        const uint32_t dst_grad_query = base_grad_query + dst * stride_grad_query;
        uint32_t receiver_x = 0;
        uint32_t receiver_y = 0;
        if (receiver != kNoCore && receiver != my_core) {
            noc_xy_of(receiver, receiver_x, receiver_y);
        }

        {
            DeviceZoneScopedN("SEND-IMM");
        if (receiver == my_core) {
            // The self-transition: reserving the destination slot is the
            // credit, and the copies are local.
            cb_reserve_back(cb_query, row_tiles);
            cb_reserve_back(cb_grad_output, val_tiles);
            cb_reserve_back(cb_grad_query_seed, row_tiles);
            noc_async_write(qs, get_noc_addr(dst_query), stride_query);
            noc_async_write(os, get_noc_addr(dst_grad_output), stride_grad_output);
            noc_async_write(src_block, get_noc_addr(dst_block), stride_block);
            noc_async_write_barrier();
            noc_semaphore_set(ready_imm_sem[dst], u + 1u);
        } else if (receiver != kNoCore) {
            await_credit(receiver);
            noc_async_write(qs, get_noc_addr(receiver_x, receiver_y, dst_query), stride_query);
            noc_async_write(
                os, get_noc_addr(receiver_x, receiver_y, dst_grad_output), stride_grad_output);
            noc_async_write(src_block, get_noc_addr(receiver_x, receiver_y, dst_block), stride_block);
            // Payload complete before readiness, as the contract requires.
            noc_async_write_barrier();
            // An inline write carries the value in the command itself, so
            // there is no 4-byte source word in L1 to keep alive and no
            // trailing write barrier to wait for the ack of. The payload
            // barrier above is what orders payload before readiness; this
            // write is fire-and-forget, and the consumer polls for it.
            noc_inline_dw_write(
                get_noc_addr(receiver_x, receiver_y, get_semaphore(ready_imm_sem_id[dst])), u + 1u);
        }
        }

        // If the next timestep's packet comes from DRAM, read its constant
        // fields now, into the slot the compute kernel released two timesteps
        // ago, while this timestep's dQ is awaited. The snake's first core does
        // this every timestep of a dense pass; without it that core, and so
        // the whole ring, waited on the read.
        if (t + 1u < kTimesteps && !sched.producer(my_core, t + 1u).internal) {
            DeviceZoneScopedN("PREFETCH-IMM");
            const auto next_pair = sched.pair(my_core, t + 1u);
            const uint32_t ni = next_pair.i;
            const uint32_t nslot = (g + 1u) % 2u;
            cb_reserve_back(cb_query, row_tiles);
            cb_reserve_back(cb_grad_output, val_tiles);
            const uint32_t nqs = base_query + nslot * stride_query;
            const uint32_t nos = base_grad_output + nslot * stride_grad_output;
            const uint32_t nls = base_lse + nslot * stride_interm;
            const uint32_t nds = base_u_scalar + nslot * stride_interm;
            for (uint32_t k = 0; k < row_tiles; ++k) {
                noc_async_read_page(row_base + (ni - 1u) * row_tiles + k, query, nqs + k * tile_bytes);
            }
            for (uint32_t k = 0; k < val_tiles; ++k) {
                noc_async_read_page(val_base + (ni - 1u) * val_tiles + k, grad_output, nos + k * tile_bytes);
            }
            for (uint32_t k = 0; k < Bt; ++k) {
                noc_async_read_page(stat_base + (ni - 1u) * Bt + k, lse, nls + k * interm_bytes);
                noc_async_read_page(stat_base + (ni - 1u) * Bt + k, u_scalar, nds + k * interm_bytes);
            }
            // Complete the reads now and hand L and D to the writer, which
            // has the rest of this timestep to make the statistic tiles.
            noc_async_read_barrier();
            cb_reserve_back(cyclic_dataflow::kStatsReadyCb, 1);
            cb_push_back(cyclic_dataflow::kStatsReadyCb, 1);
            prefetched_next = true;
            // The dQ seed too, where the row enters for the first time: the
            // op's input accumulator, which nothing in flight writes. (A later
            // streak start reloads a spill and must wait for it; that load
            // stays where it was.) Into the seed slot the compute kernel
            // released with t - 1, whose spill or forward this thread has
            // completed. Issued after the hand-off above and left in flight --
            // the next step's dQ path completes it -- so it holds up neither
            // the writer nor this step's dQ send. Without it the snake's head,
            // which loads every row's dQ in a dense pass, waited on the read
            // in its dQ update and set the whole snake's pace.
            // Dense passes only: there the head loads a row every timestep
            // and paced the snake; in a causal launch the starts are spread
            // over the cores and the early reads only got in the way of the
            // packet traffic (the full grid at Bt = 1 lost 19% with them).
            if (sched.dense() && !sched.is_later_streak_start(ni, t + 1u)) {
                DeviceZoneScopedN("PREFETCH-DQ");
                const uint32_t ngs = base_grad_query + nslot * stride_grad_query;
                for (uint32_t k = 0; k < row_tiles; ++k) {
                    noc_async_read_page(row_base + (ni - 1u) * row_tiles + k, grad_query, ngs + k * grad_bytes);
                }
                prefetched_dq_next = true;
            }
        }

        // Now dQ, which is the one field that has to wait for arithmetic --
        // it is the accumulator each consumer adds to.
        if (producer.internal) {
            DeviceZoneScopedN("RECV-DQ");
            WAYPOINT("DQRW");
            do {
                invalidate_l1_cache();
            } while ((*ready_dq_sem[slot]) < g + 1u);
            WAYPOINT("DQRD");
        } else {
            DeviceZoneScopedN("LOAD-DQ-DRAM");
            if (!prefetched_dq) {
                for (uint32_t k = 0; k < row_tiles; ++k) {
                    noc_async_read_page(row_base + (i - 1u) * row_tiles + k, grad_query, gs + k * grad_bytes);
                }
            }
            // Completes this read, or the one prefetched a timestep ago.
            noc_async_read_barrier();
        }
        cb_push_back(cb_grad_query_seed, row_tiles);

        // This core's contribution closes the packet.
        {
            DeviceZoneScopedN("WAIT-COMPUTE-DQ");
            cb_wait_front(cb_grad_query_out, row_tiles);
        }
        const uint32_t dq_out = get_read_ptr(cb_grad_query_out);

        {
            DeviceZoneScopedN("SEND-DQ");
        if (receiver == my_core) {
            noc_async_write(dq_out, get_noc_addr(dst_grad_query), stride_grad_query);
            noc_async_write_barrier();
            noc_semaphore_set(ready_dq_sem[dst], u + 1u);
        } else if (receiver != kNoCore) {
            noc_async_write(
                dq_out, get_noc_addr(receiver_x, receiver_y, dst_grad_query), stride_grad_query);
            noc_async_write_barrier();
            // An inline write carries the value in the command itself, so
            // there is no 4-byte source word in L1 to keep alive and no
            // trailing write barrier to wait for the ack of. The payload
            // barrier above is what orders payload before readiness; this
            // write is fire-and-forget, and the consumer polls for it.
            noc_inline_dw_write(
                get_noc_addr(receiver_x, receiver_y, get_semaphore(ready_dq_sem_id[dst])), u + 1u);
        } else {
            // Streak end: spill dQ_i and complete the write, so the reload at
            // the next streak start observes it.
            for (uint32_t k = 0; k < row_tiles; ++k) {
                noc_async_write_page(row_base + (i - 1u) * row_tiles + k, grad_query, dq_out + k * grad_bytes);
            }
            noc_async_write_barrier();
#if ENDPOINT_SYNC
            // An inter-streak spill certifies itself for the later streak's
            // reload. By the endpoint spill property this core is 1 or 2, and
            // it writes its own copy too, since a later streak of this row may
            // well be consumed here.
            if (sched.has_later_active(i, t)) {
                // The spill above has completed, so this value certifies it.
                // One local write: consumers read it from here.
                noc_semaphore_set(endpoint_sem[my_core - 1u], g + 1u);
            }
#endif
        }

        }

        cb_pop_front(cb_grad_query_out, row_tiles);

        // The compute kernel has popped this timestep's slot, so it is free
        // for destination t + 2 and its producer can be told now rather than
        // when this core reaches t + 2. That is the paper's release timing,
        // and it is what lets a producer forward as soon as its payload is
        // ready instead of waiting for its receiver to arrive.
        {
            DeviceZoneScopedN("RELEASE");
            cb_wait_front(cb_slot_release, 1);
            cb_pop_front(cb_slot_release, 1);
            const uint32_t released_for = t + 2u;
            if (released_for < kTimesteps) {
                const auto next_producer = sched.producer(my_core, released_for);
                if (next_producer.internal && next_producer.core != my_core) {
                    grant_credit_to(next_producer.core);
                }
                // A self-transition needs no credit: it reserves the
                // destination slot itself. And a DRAM load needs none, the
                // slot being its own.
            }
        }
    }
    }  // slices

    // Drain what is still in flight before the kernel ends: the readiness and
    // endpoint writes are issued without waiting for their acks, and a
    // kernel must not finish with outstanding NoC transactions.
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
