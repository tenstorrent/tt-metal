// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP hub (BRISC, one core per chip): pure data movement between the streamers and the fabric.
// Per layer l (slot parity p = l & 1):
//   1. wait until every streamer has written its down columns into slot[p][chip] (gather semaphore),
//   2. line-multicast slot[p][chip] over fabric into slot[p][chip] of every peer hub (fused increment
//      of the peers' CCL semaphore of parity p), wait for the peers' slots. A peer that already has
//      layer l's slots runs ahead and sends layer l + 1's while a slower peer's layer l packets are still
//      in flight, so the two parities count on separate semaphores (a peer cannot get two layers ahead:
//      layer l + 2 needs this chip's layer l + 1 slot, sent after its layer l wait),
//   3. multicast slot[p][0 .. num_chips) to the streamers' partial slots and bump their slots semaphore,
//      one multicast per rectangle of a cover of exactly the streamer cores: the slots are allocated on the
//      streamers only, so any other core under a multicast could hold a CB at that address.
// The last layer's slots stay on the hub for the host to read back (and are also multicast when
// mcast_last is set, for a stage that runs after the last layer).
//
// Compile-time args: 0 num_chips, 1 chip, 2 vector_bytes, 3 packet_bytes, 4 layers, 5 num_streamers,
//   6 sem_gather id, 7 sem_slots id, 8 sem_flag id (local word holding the value multicast into the
//   streamers' slots semaphore), 9 mcast_last, 10 header CB (holds the packet headers when they exceed the
//   per-RISC header pool), 11 published, 12 sem_addr
// Runtime args: 0 slots_addr, 1, 2 ccl semaphore addresses of parity 0, 1 (global semaphores), 3 streamer
// slots addr, 4 rectangles, then per rectangle noc x0, y0, x1, y1, cores, then FabricConnectionManager
// args (fwd, bwd).
// published: the slots are this core's CB 0 (the same address on every chip); a streamer sends the
// streamers' x address into sem_addr and the hub multicasts its slots address into the streamers'
// sem_addr. The hub's compute sums each round's slots into CB 2 and only x is multicast. Arg 0 is then a
// DRAM buffer (0 = none) that gets the last layer's x, arg 3 is unused.
//
// feed (batch 1, with the lm_head): after the last layer the hub writes the next step's token state, so
// steps can run back to back with no host in between. It waits for the streamers' argmax records (one
// more gather count each), reduces them to (order key, vocab id), exchanges that 16 B record with every
// peer hub over fabric and picks the same token on every chip (largest key, then smallest id, the host's
// rule). Then into tok (DRAM): ring + 1 and pos + 1, the token's embedding row as x0 and the rope tiles of
// pos + 1 (expanded from the rope table's compact entry, see expand_rope); into out: (token, pos + 1).
// Counts on the global semaphores are monotonic: a peer can start the next step and count into this chip
// before this hub ends, so nothing is reset here. A per-hub launch counter gives each launch its base; the
// host zeroes all of them while the device is idle.
// Compile-time args 13 feed, 14 max_pos, 15 tok page bytes, 16 rope offset in tok, 17 rope bytes, 18 feed
// kind, 19 rope offset in the peer's tok, 20 embedding row bytes.
// Runtime args (feed), after the rectangles: argmax records, candidate slots (L1, num_chips x 16 B, same
// address on every chip), candidate semaphore, launch counter, tok, embedding table, rope table (a page per
// position: cos then sin of its ROT / 2 rotation angles as bf16, kRopeEntry bytes), out, sampling
// parameters, then the peer's tok (verify kinds).
//
// Compile-time arg 21 sample (feed kind 0): the records are the streamers' top-k lists and the candidate slots
// the chips' (sampling::kListBytes each); the token is drawn from them with the sampling parameters (DRAM, 16 words,
// the first 4 used, see sampling.hpp) instead of taken as the argmax.
//
// Feed kinds 1 and 2 run MTP speculative decode with no host between steps: the main model's verify step
// (kind 1, rows t at p and draft d at p + 1) and the draft model's (kind 2, see mtp.py) alternate, each
// writing the other's token state (words: 0 ring, 1 / 2 positions, 4 GDN state slot, 5 draft (main) or
// accept (draft), 6 t (main) or a0 (draft), 7 a1, 8 / 9 the main model's ring and slot, 10 base: the
// draft model's positions are the main model's minus base, so its KV cache starts where its entries do).
// Both rows' greedy tokens are reduced (the 16 B candidate record holds both).
//   kind 1: accept = a0 == d; the draft model's entries are (a0, x row 0) at p and (a1, x row 1) at p + 1
//     (x0 = both embedding rows, then this step's x); out = (accept, a0, a1, p).
//   kind 2: d' = its row-accept token; the main model continues at p' = p + 1 + accept with rows
//     (accept ? a1 : a0, d'), ring + 1 + accept and slot ^ accept; out = (d', t', p').

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "sampling.hpp"

constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t chip = get_compile_time_arg_val(1);
constexpr uint32_t vector_bytes = get_compile_time_arg_val(2);
constexpr uint32_t packet_bytes = get_compile_time_arg_val(3);
constexpr uint32_t layers = get_compile_time_arg_val(4);
constexpr uint32_t num_streamers = get_compile_time_arg_val(5);
constexpr uint32_t sem_gather = get_compile_time_arg_val(6);
constexpr uint32_t sem_slots = get_compile_time_arg_val(7);
constexpr bool mcast_last = get_compile_time_arg_val(9) != 0;
constexpr uint32_t sem_flag = get_compile_time_arg_val(8);
constexpr uint32_t cb_headers = get_compile_time_arg_val(10);
constexpr bool published = get_compile_time_arg_val(11) != 0;
constexpr uint32_t sem_addr = get_compile_time_arg_val(12);
constexpr bool feed = get_compile_time_arg_val(13) != 0;
constexpr uint32_t max_pos = get_compile_time_arg_val(14);
constexpr uint32_t tok_bytes = get_compile_time_arg_val(15);
constexpr uint32_t rope_off = get_compile_time_arg_val(16);
constexpr uint32_t rope_bytes = get_compile_time_arg_val(17);
constexpr uint32_t feed_kind = get_compile_time_arg_val(18);
constexpr uint32_t peer_rope_off = get_compile_time_arg_val(19);
constexpr uint32_t row_bytes = get_compile_time_arg_val(20);
constexpr uint32_t rows = feed_kind > 0 ? 2 : 1;  // greedy tokens per launch
constexpr bool sample = get_compile_time_arg_val(21) != 0;
static_assert(!sample || (feed && feed_kind == 0), "sampling feeds batch 1");
constexpr uint32_t cand_bytes = sample ? sampling::kListBytes : 16;
constexpr uint32_t packets_per_vector = (vector_bytes + packet_bytes - 1) / packet_bytes;
constexpr uint32_t kRopeEntry = 128;  // compact rope table page: cos[32] | sin[32] bf16
static_assert(rope_bytes == 3 * 2048, "the token state's rope tiles: cos, sin, -sin");
constexpr uint32_t num_headers = 2 * packets_per_vector * 2 + (feed ? 2 : 0);
constexpr bool pooled = num_headers <= NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;

// row `row` (row_bytes, in L1) as row u of the 2 x 32 tiles at `dst` (face rows of 32 B)
void scatter_row(uint32_t row, uint32_t dst, uint32_t u) {
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t t = 0; t < row_bytes / 64; t++) {
        for (uint32_t f = 0; f < 2; f++) {
            for (uint32_t w = 0; w < 8; w++) {
                out[t * 32 + (2 * f + u) * 8 + w] = src[t * 16 + f * 8 + w];
            }
        }
    }
}

// the compact rope entry at `src` (cos[32] | sin[32] bf16) as the token state's cos, sin and -sin tiles at `dst`:
// bf16 32x32 tiles whose rows all equal the vector (faces 0 and 2 hold its first 16 values, faces 1 and 3 the
// rest; model.py tile_bytes_rows_replicated). -sin flips the sign bits, which is exact
void expand_rope(uint32_t src, uint32_t dst) {
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    uint32_t v[16];
    for (uint32_t t = 0; t < 3; t++) {
        const uint32_t flip = t == 2 ? 0x80008000u : 0;
        for (uint32_t w = 0; w < 16; w++) {
            v[w] = in[(t == 0 ? 0 : 16) + w] ^ flip;
        }
        for (uint32_t f = 0; f < 4; f++) {
            const uint32_t* half = v + (f & 1) * 8;
            for (uint32_t r = 0; r < 16; r++) {
                for (uint32_t w = 0; w < 8; w++) {
                    out[t * 512 + f * 128 + r * 8 + w] = half[w];
                }
            }
        }
    }
}

// feed kinds 1 and 2 (see above); h: this launch's token state head (64 B, in L1 at `stage`); rope_c: L1 scratch
// for the two compact rope entries
void feed_verify(
    volatile tt_l1_ptr uint32_t* h,
    const uint32_t* best,
    uint32_t stage,
    uint32_t embed_addr,
    uint32_t rope_addr,
    uint32_t peer_tok_addr,
    uint32_t out_addr,
    uint32_t x_addr,
    uint32_t rope_c) {
    // stage: head | peer head | x0 (kind 1: 2 vectors) | rope (2 rows) | 2 embedding rows
    const uint32_t ph = stage + 64, x0 = stage + 128;
    const uint32_t x0_bytes = (feed_kind == 1 ? 2 : 1) * vector_bytes;
    const uint32_t rope = x0 + x0_bytes, rows_l1 = rope + 2 * rope_bytes;
    volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ph);
    const uint32_t a0 = best[0], a1 = best[1];
    uint32_t p, toks[2];
    for (uint32_t i = 0; i < 16; i++) {
        g[i] = 0;
    }
    uint32_t rope_p;  // the peer's position of row 0
    if constexpr (feed_kind == 1) {
        p = h[1];
        rope_p = p - h[10];
        const uint32_t accept = a0 == h[5];
        toks[0] = a0;
        toks[1] = a1;
        g[1] = rope_p;
        g[2] = rope_p + 1;
        g[10] = h[10];
        g[5] = accept;
        g[6] = a0;
        g[7] = a1;
        g[8] = h[0];
        g[9] = h[4];
        // out: (accept, a0, a1, p), written below from the head block
        h[0] = accept;
        h[1] = a0;
        h[2] = a1;
        h[3] = p;
    } else {
        const uint32_t accept = h[5];
        p = h[1] + h[10] + 1 + accept;
        rope_p = p;
        g[10] = h[10];
        const uint32_t d = accept ? a1 : a0, t = accept ? h[7] : h[6];
        toks[0] = t;
        toks[1] = d;
        g[0] = h[8] + 1 + accept;
        g[1] = p;
        g[2] = p + 1;
        g[4] = h[9] ^ accept;
        g[5] = d;
        g[6] = t;
        h[0] = d;
        h[1] = t;
        h[2] = p;
        h[3] = 0;
    }
    const uint32_t r0 = rope_p < max_pos ? rope_p : max_pos - 1;
    const uint32_t r1 = rope_p + 1 < max_pos ? rope_p + 1 : max_pos - 1;
    const InterleavedAddrGen<true> emb{.bank_base_address = embed_addr, .page_size = row_bytes};
    const InterleavedAddrGen<true> rtab{.bank_base_address = rope_addr, .page_size = kRopeEntry};
    noc_async_read(emb.get_noc_addr(toks[0]), rows_l1, row_bytes);
    noc_async_read(emb.get_noc_addr(toks[1]), rows_l1 + row_bytes, row_bytes);
    noc_async_read(rtab.get_noc_addr(r0), rope_c, kRopeEntry);
    noc_async_read(rtab.get_noc_addr(r1), rope_c + kRopeEntry, kRopeEntry);
    if constexpr (feed_kind == 1) {
        noc_async_read(get_noc_addr(x_addr), x0 + vector_bytes, vector_bytes);  // this step's x (row 0, 1)
    }
    noc_async_read_barrier();
    expand_rope(rope_c, rope);
    expand_rope(rope_c + kRopeEntry, rope + rope_bytes);
    scatter_row(rows_l1, x0, 0);
    scatter_row(rows_l1 + row_bytes, x0, 1);
    const InterleavedAddrGen<true> peer{.bank_base_address = peer_tok_addr, .page_size = 64};
    noc_async_write(ph, peer.get_noc_addr(0), 64);
    noc_async_write(x0, peer.get_noc_addr(0, 64), x0_bytes);
    noc_async_write(rope, peer.get_noc_addr(0, peer_rope_off), 2 * rope_bytes);
    const InterleavedAddrGen<true> out{.bank_base_address = out_addr, .page_size = 64};
    noc_async_write(stage, out.get_noc_addr(0), 64);
    noc_async_write_barrier();
}

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t arg0 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ccl_sem_addr[2] = {get_arg_val<uint32_t>(arg_idx), get_arg_val<uint32_t>(arg_idx + 1)};
    arg_idx += 2;
    const uint32_t arg3 = get_arg_val<uint32_t>(arg_idx++);
    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t slots = published ? cb_base(0) : arg0;
    const uint32_t rects = get_arg_val<uint32_t>(arg_idx++);
    const size_t rect_args = arg_idx;
    arg_idx += 5 * rects;
    uint32_t argmax_addr = 0, cand_addr = 0, cand_sem_addr = 0, launch_addr = 0;
    uint32_t tok_addr = 0, embed_addr = 0, rope_addr = 0, out_addr = 0, params_addr = 0, peer_tok_addr = 0;
    if constexpr (feed) {
        argmax_addr = get_arg_val<uint32_t>(arg_idx++);
        cand_addr = get_arg_val<uint32_t>(arg_idx++);
        cand_sem_addr = get_arg_val<uint32_t>(arg_idx++);
        launch_addr = get_arg_val<uint32_t>(arg_idx++);
        tok_addr = get_arg_val<uint32_t>(arg_idx++);
        embed_addr = get_arg_val<uint32_t>(arg_idx++);
        rope_addr = get_arg_val<uint32_t>(arg_idx++);
        out_addr = get_arg_val<uint32_t>(arg_idx++);
        params_addr = get_arg_val<uint32_t>(arg_idx++);
        if constexpr (feed_kind > 0) {
            peer_tok_addr = get_arg_val<uint32_t>(arg_idx++);
        }
    }
    const uint32_t flag_addr = get_semaphore(sem_flag);
    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    // one header per (parity, packet, direction), built once: the header writes to the routers are
    // non-blocking, so a header is never rewritten while a send of it may still be reading it
    PacketHeaderPool::reset();
    uint32_t next_hdr = cb_base(cb_headers);
    auto allocate = [&]() -> volatile PACKET_HEADER_TYPE* {
        if constexpr (pooled) {
            return PacketHeaderPool::allocate_header();
        }
        auto* h = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(next_hdr);
        next_hdr += (sizeof(PACKET_HEADER_TYPE) + 15) & ~15u;
        return h;
    };
    volatile PACKET_HEADER_TYPE* hdr[2][packets_per_vector][2];
    for (uint32_t par = 0; par < 2; par++) {
        for (uint32_t p = 0; p < packets_per_vector; p++) {
            const uint32_t off = p * packet_bytes;
            const uint32_t bytes = off + packet_bytes <= vector_bytes ? packet_bytes : vector_bytes - off;
            const uint64_t dst = get_noc_addr(my_x[0], my_y[0], slots + (par * num_chips + chip) * vector_bytes + off);
            const uint64_t sem = get_noc_addr(my_x[0], my_y[0], ccl_sem_addr[par]);
            for (uint32_t dir = 0; dir < 2; dir++) {
                volatile PACKET_HEADER_TYPE* h = allocate();
                h->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                    1, static_cast<uint8_t>(dir == 0 ? num_chips - 1 - chip : chip)});
                h->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, bytes);
                hdr[par][p][dir] = h;
            }
        }
    }
    volatile PACKET_HEADER_TYPE* cand_hdr[2] = {nullptr, nullptr};
    if constexpr (feed) {
        const uint64_t dst = get_noc_addr(my_x[0], my_y[0], cand_addr + chip * cand_bytes);
        const uint64_t sem = get_noc_addr(my_x[0], my_y[0], cand_sem_addr);
        for (uint32_t dir = 0; dir < 2; dir++) {
            volatile PACKET_HEADER_TYPE* h = allocate();
            h->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                1, static_cast<uint8_t>(dir == 0 ? num_chips - 1 - chip : chip)});
            h->to_noc_fused_unicast_write_atomic_inc(
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, cand_bytes);
            cand_hdr[dir] = h;
        }
    }
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }

    volatile tt_l1_ptr uint32_t* gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
    volatile tt_l1_ptr uint32_t* ccl_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccl_sem_addr[0]),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ccl_sem_addr[1])};
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    auto rect = [&](uint32_t r, uint32_t i) { return get_arg_val<uint32_t>(rect_args + 5 * r + i); };

    uint32_t streamer_slots = arg3;
    if constexpr (published) {
        // our slots address to the streamers (from a word past the headers), theirs from a streamer
        volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(next_hdr);
        *word = slots;
        for (uint32_t r = 0; r < rects; r++) {
            noc_semaphore_set_multicast(
                next_hdr,
                get_noc_multicast_addr(rect(r, 0), rect(r, 1), rect(r, 2), rect(r, 3), get_semaphore(sem_addr)),
                rect(r, 4));
        }
        volatile tt_l1_ptr uint32_t* addr_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_addr));
        while (*addr_sem == 0) {
            invalidate_l1_cache();
        }
        streamer_slots = *addr_sem;
        noc_async_write_barrier();
    }

    uint32_t expected[2] = {0, 0};
    uint32_t launch = 0;
    if constexpr (feed) {
        volatile tt_l1_ptr uint32_t* launch_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_addr);
        invalidate_l1_cache();
        launch = *launch_word;
        // (an odd round count puts the extra round on parity 0)
        expected[0] = launch * ((layers + 1) / 2) * (num_chips - 1) * packets_per_vector;
        expected[1] = launch * (layers / 2) * (num_chips - 1) * packets_per_vector;
    }
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t par = l & 1;
        const uint32_t base = slots + par * num_chips * vector_bytes;
        noc_semaphore_wait_min(gather, num_streamers * (l + 1));
        if constexpr (num_chips > 1) {
            for (uint32_t p = 0; p < packets_per_vector; p++) {
                const uint32_t src = base + chip * vector_bytes + p * packet_bytes;
                const uint32_t bytes =
                    (p + 1) * packet_bytes <= vector_bytes ? packet_bytes : vector_bytes - p * packet_bytes;
                if (fabric.has_forward_connection()) {
                    perform_payload_send(fabric.get_forward_connection(), src, bytes, hdr[par][p][0]);
                }
                if (fabric.has_backward_connection()) {
                    perform_payload_send(fabric.get_backward_connection(), src, bytes, hdr[par][p][1]);
                }
            }
            expected[par] += (num_chips - 1) * packets_per_vector;
            noc_semaphore_wait_min(ccl_sem[par], expected[par]);
        }
        // published: the compute sums the slots into x
        uint32_t src = base, bytes = num_chips * vector_bytes;
        if constexpr (published) {
            constexpr uint32_t views = vector_bytes / 2048;
            cb_reserve_back(0, num_chips * views);
            cb_push_back(0, num_chips * views);
            cb_wait_front(2, views);
            src = get_read_ptr(2);
            bytes = vector_bytes;
            if (l + 1 == layers && arg0 != 0) {
                const InterleavedAddrGen<true> dump{.bank_base_address = arg0, .page_size = vector_bytes};
                noc_async_write(src, dump.get_noc_addr(0), vector_bytes);
            }
        }
        if (l + 1 < layers || mcast_last) {
            for (uint32_t r = 0; r < rects; r++) {
                noc_async_write_multicast(
                    src,
                    get_noc_multicast_addr(rect(r, 0), rect(r, 1), rect(r, 2), rect(r, 3), streamer_slots),
                    bytes,
                    rect(r, 4));
            }
            noc_async_write_barrier();
            *flag = l + 1;
            for (uint32_t r = 0; r < rects; r++) {
                noc_semaphore_set_multicast(
                    flag_addr,
                    get_noc_multicast_addr(rect(r, 0), rect(r, 1), rect(r, 2), rect(r, 3), get_semaphore(sem_slots)),
                    rect(r, 4));
            }
            noc_async_write_barrier();
        }
        if constexpr (published) {
            noc_async_write_barrier();
            cb_pop_front(2, vector_bytes / 2048);
        }
    }
    noc_async_write_barrier();
    if constexpr (feed) {
        // this step's records: one more gather count per streamer after its argmax write
        noc_semaphore_wait_min(gather, num_streamers * (layers + 1));
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(argmax_addr);
        volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_addr);
        auto better = [](uint32_t k, uint32_t i, uint32_t bk, uint32_t bi) { return k > bk || (k == bk && i < bi); };
        // per row u: record of streamer s at rec[4 (s rows + u)]; candidate words 2u, 2u + 1
        uint32_t best_ks[rows] = {}, best_is[rows] = {};
        if constexpr (sample) {
            sampling::merge(rec, num_streamers, cand + chip * 2 * sampling::kTopK);
        }
        for (uint32_t u = 0; !sample && u < rows; u++) {
            uint32_t bk = rec[4 * u], bi = rec[4 * u + 1];
            for (uint32_t s = 1; s < num_streamers; s++) {
                if (better(rec[4 * (s * rows + u)], rec[4 * (s * rows + u) + 1], bk, bi)) {
                    bk = rec[4 * (s * rows + u)];
                    bi = rec[4 * (s * rows + u) + 1];
                }
            }
            best_ks[u] = bk;
            best_is[u] = bi;
            cand[4 * chip + 2 * u] = bk;
            cand[4 * chip + 2 * u + 1] = bi;
        }
        if constexpr (num_chips > 1) {
            if (fabric.has_forward_connection()) {
                perform_payload_send(
                    fabric.get_forward_connection(), cand_addr + chip * cand_bytes, cand_bytes, cand_hdr[0]);
            }
            if (fabric.has_backward_connection()) {
                perform_payload_send(
                    fabric.get_backward_connection(), cand_addr + chip * cand_bytes, cand_bytes, cand_hdr[1]);
            }
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_sem_addr), (launch + 1) * (num_chips - 1));
            invalidate_l1_cache();
        }
        for (uint32_t u = 0; !sample && u < rows; u++) {
            for (uint32_t c = 0; c < num_chips; c++) {
                if (better(cand[4 * c + 2 * u], cand[4 * c + 2 * u + 1], best_ks[u], best_is[u])) {
                    best_ks[u] = cand[4 * c + 2 * u];
                    best_is[u] = cand[4 * c + 2 * u + 1];
                }
            }
        }
        uint32_t best_i = best_is[0];

        // staging in the parity-1 slots: a peer writes there only after this chip's next round 0
        const uint32_t stage = (slots + num_chips * vector_bytes + 63) & ~63u;
        const uint32_t head = stage, x0 = stage + 64, rope = x0 + vector_bytes;
        const InterleavedAddrGen<true> tok{.bank_base_address = tok_addr, .page_size = tok_bytes};
        noc_async_read(tok.get_noc_addr(0), head, 64);
        // sample: the parameters, the global list and the draw's weights after the rope tiles
        const uint32_t params = rope + rope_bytes, list = params + 64, weights = list + sampling::kListBytes;
        // the compact rope entries, at the end of the parity-1 slots (after everything staged above, in either kind)
        const uint32_t rope_c = (slots + 2 * num_chips * vector_bytes - 2 * kRopeEntry) & ~63u;
        // (staged bytes, plus the two 64 B alignments and the entries themselves)
        static_assert(
            feed_kind > 0 ||
                128 + vector_bytes + rope_bytes + 64 + 3 * sampling::kListBytes / 2 + 128 + 2 * kRopeEntry <=
                    num_chips * vector_bytes,
            "kind 0 staging overlaps the compact rope entries");
        static_assert(
            feed_kind == 0 || 128 + 2 * vector_bytes + 2 * rope_bytes + 2 * row_bytes + 128 + 2 * kRopeEntry <=
                                  num_chips * vector_bytes,
            "verify staging overlaps the compact rope entries");
        if constexpr (sample) {
            const InterleavedAddrGen<true> par{.bank_base_address = params_addr, .page_size = 64};
            noc_async_read(par.get_noc_addr(0), params, 64);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(head);
        if constexpr (sample) {
            volatile tt_l1_ptr uint32_t* l = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(list);
            sampling::merge(cand, num_chips, l);
            best_i = sampling::draw(
                l,
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(params),
                h[1],
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weights));
        }
        if constexpr (feed_kind > 0) {
            feed_verify(h, best_is, stage, embed_addr, rope_addr, peer_tok_addr, out_addr, cb_base(2), rope_c);
        } else {
            const uint32_t pos = h[1] + 1 < max_pos ? h[1] + 1 : max_pos - 1;
            const InterleavedAddrGen<true> emb{.bank_base_address = embed_addr, .page_size = vector_bytes};
            const InterleavedAddrGen<true> rtab{.bank_base_address = rope_addr, .page_size = kRopeEntry};
            noc_async_read(emb.get_noc_addr(best_i), x0, vector_bytes);
            noc_async_read(rtab.get_noc_addr(pos), rope_c, kRopeEntry);
            noc_async_read_barrier();
            expand_rope(rope_c, rope);
            h[0] = h[0] + 1;
            h[1] = pos;
            noc_async_write(head, tok.get_noc_addr(0), 64);
            noc_async_write(x0, tok.get_noc_addr(0, 64), vector_bytes);
            noc_async_write(rope, tok.get_noc_addr(0, rope_off), rope_bytes);
            noc_async_write_barrier();
            // out: (token, pos + 1), from the head block once its write to tok has left
            h[0] = best_i;
            h[1] = pos;
            const InterleavedAddrGen<true> out{.bank_base_address = out_addr, .page_size = 64};
            noc_async_write(head, out.get_noc_addr(0), 64);
            noc_async_write_barrier();
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_addr) = launch + 1;
    } else {
        noc_semaphore_set(ccl_sem[0], 0);
        noc_semaphore_set(ccl_sem[1], 0);
    }
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
