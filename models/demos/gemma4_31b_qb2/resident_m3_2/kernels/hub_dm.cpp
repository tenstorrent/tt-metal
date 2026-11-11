// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident hub (BRISC, one core per chip): the chip's fabric endpoint. Adapted from changh95/qwen3.8-27b-p300x2
// code/qwen36_resident/kernels/mlp_hub.cpp (Apache-2.0): the line multicast of every chip's partial into slot
// [parity][chip] of every peer, with a fused atomic increment of the receiver's semaphore of that parity.
// Here the receivers are the K slice cores of the peer chips (slice k owns hidden tiles [k T, (k + 1) T)): each
// all-reduce round the streamers write their partials into this hub's staging [parity] (and into their own slice's
// local slot); once all have, the hub sends each slice's columns to the same slice of every peer chip.
// Then the argmax: the streamers' per-row records (writer and reader halves), the chip's per-row best exchanged with
// the peers over fabric, the same per-row token on every chip (largest key, then smallest index) into `out`.
// Serving (M1_INTERFACE.md): at the end of every launch the hub applies the control word's feed mode to the engine's
// tokens / positions (argmax: tokens <- ids, positions + 1; external: positions + 1, the sampler writes tokens; none:
// tokens <- ids) and updates the status words: the launch counter, the block's position / slot flags (the attention
// cores' launch_of test) and every attention core's table errors (A_ERR words). A flagged or inert block's ids are 0
// and its positions do not move. The slices gather each launch's x0 from `tokens` themselves.
// Counts on the global semaphores are monotonic (a peer can start the next step first), so a per-hub launch counter
// gives each launch its base; the host zeroes them while idle.
//
// Compile-time args: 0 num_chips, 1 chip, 2 R (tile rows), 3 M (rows), 4 Ht, 5 K (slices), 6 T (tiles per slice),
// 7 packet_bytes, 8 rounds, 9 num_streamers, 10 sem_gather, 11 sem_rec, 12 links (1 / 2), 13 attention cores
// Runtime args: 0 sem_stage id (the streamers' mailbox for the staging address), 1 sem_slotaddr id (this core's
// mailbox for the slices' slots address, from slice 0), 2 / 3 slices' slot semaphores (parity 0 / 1), 4 slot bytes (one chip's slot on a slice), 5 records (L1, num_streamers x R x 8 B), 6 candidates (L1,
// num_chips x R x 8 B, the same address on every chip), 7 candidate semaphore, 8 launch counter word, 9 tokens,
// 10 positions, 11 slots (DRAM [1, 1, 1, 32] rows), 12 out (DRAM: per row (token, key), then at word 64 (launch,
// position, M); 80 words), 13 control, 14 status (DRAM [1, 1, 1, 32] uint32 rows), 15 capacity, 16 max_users,
// 17 timeline (L1, 0 = off), 18 rectangles of the streamers, then per rectangle x0, y0, x1, y1, cores, then K x (x, y)
// of the slices, then the attention cores (x | y << 16 each), then the fabric connection args (forward, backward)
// per link. The staging is CB 0 (2 parities x Ht tiles), its address published to the streamers.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t chip = get_compile_time_arg_val(1);
constexpr uint32_t R = get_compile_time_arg_val(2);
constexpr uint32_t M = get_compile_time_arg_val(3);
constexpr uint32_t Ht = get_compile_time_arg_val(4);
constexpr uint32_t K = get_compile_time_arg_val(5);
constexpr uint32_t T = get_compile_time_arg_val(6);
constexpr uint32_t packet_bytes = get_compile_time_arg_val(7);
constexpr uint32_t rounds = get_compile_time_arg_val(8);
constexpr uint32_t num_streamers = get_compile_time_arg_val(9);
constexpr uint32_t sem_gather = get_compile_time_arg_val(10);
constexpr uint32_t sem_rec = get_compile_time_arg_val(11);
constexpr uint32_t links = get_compile_time_arg_val(12);
constexpr uint32_t num_attn = get_compile_time_arg_val(13);
static_assert(links == 1 || links == 2, "one or two fabric links");
// status words / flags and the attention cores' error semaphores (M1_INTERFACE.md s5, attn_common.hpp)
constexpr uint32_t ST_LAUNCHES = 0, ST_FLAGS = 1, ST_ROW = 2, ST_POS = 3, ST_PAGE = 4, ST_ENTRY = 5;
constexpr uint32_t F_POSITION = 0x1, F_SLOT = 0x2;
constexpr uint32_t A_ERR = 6;
constexpr uint32_t FEED_ARGMAX = 0, FEED_EXTERNAL = 1, FEED_NONE = 2;

constexpr uint32_t kTileBytes = 64 * R;
constexpr uint32_t slice_bytes = T * kTileBytes;
constexpr uint32_t staging_bytes = Ht * kTileBytes;
constexpr uint32_t P = (slice_bytes + packet_bytes - 1) / packet_bytes;  // packets per slice per round
constexpr uint32_t kHdr = 160;                                          // header stride in cb_headers
constexpr uint32_t cb_staging = 0, cb_stage = 6, cb_headers = 7;
constexpr uint32_t kTs = 2;  // timeline words per round
constexpr uint32_t kOutMeta = 64;
constexpr uint32_t kRecWords = (R * 8 + 15) / 16 * 4;  // a streamer's records (16 B aligned)

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t stage_box = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    volatile tt_l1_ptr uint32_t* slots_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));
    const uint32_t slice_sem[2] = {get_arg_val<uint32_t>(arg_idx), get_arg_val<uint32_t>(arg_idx + 1)};
    arg_idx += 2;
    const uint32_t slot_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cand_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cand_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t launch_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t tokens_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t positions_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t slots_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t control_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t status_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t capacity = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t max_users = get_arg_val<uint32_t>(arg_idx++);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t rects = get_arg_val<uint32_t>(arg_idx++);
    const size_t rect_args = arg_idx;
    arg_idx += 5 * rects;
    const size_t slice_args = arg_idx;
    arg_idx += 2 * K;
    const size_t attn_args = arg_idx;
    arg_idx += num_attn;
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t w) {
        if (ts != nullptr) {
            ts[w] = *clk;
        }
    };
    mark(0);
    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    auto rect = [&](uint32_t q, uint32_t i) { return get_arg_val<uint32_t>(rect_args + 5 * q + i); };
    // publish the staging address to every streamer (a word past the headers holds it for the multicast)
    const uint32_t staging = cb_base(cb_staging);
    {
        const uint32_t word = cb_base(cb_stage) + 4 * 120;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = staging;
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_set_multicast(
                word, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), stage_box), rect(q, 4));
        }
    }
    // the slices' slots address (the same on every slice and every chip), from slice 0
    uint32_t slice_slots = 0;
    while (slice_slots == 0) {
        invalidate_l1_cache();
        slice_slots = *slots_box;
    }

    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);
    auto fabric1 =
        links > 1
            ? FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
                  arg_idx)
            : FabricConnectionManager{};

    // one header per (parity, slice, packet, direction), built once: the header writes to the routers are
    // non-blocking, so a header is never rewritten while a send of it may still be reading it
    const uint32_t hdr_base = cb_base(cb_headers);
    auto hdr = [&](uint32_t par, uint32_t k, uint32_t p, uint32_t dir) {
        return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_base + (((par * K + k) * P + p) * 2 + dir) * kHdr);
    };
    static_assert(sizeof(PACKET_HEADER_TYPE) <= kHdr, "header stride");
    for (uint32_t par = 0; par < 2; par++) {
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t sx = get_arg_val<uint32_t>(slice_args + 2 * k), sy = get_arg_val<uint32_t>(slice_args + 2 * k + 1);
            for (uint32_t p = 0; p < P; p++) {
                const uint32_t off = p * packet_bytes;
                const uint32_t bytes = off + packet_bytes <= slice_bytes ? packet_bytes : slice_bytes - off;
                const uint64_t dst = get_noc_addr(sx, sy, slice_slots + (par * num_chips + chip) * slot_bytes + off);
                const uint64_t sem = get_noc_addr(sx, sy, slice_sem[par]);
                for (uint32_t dir = 0; dir < 2; dir++) {
                    volatile PACKET_HEADER_TYPE* h = hdr(par, k, p, dir);
                    h->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{
                        1, static_cast<uint8_t>(dir == 0 ? num_chips - 1 - chip : chip)});
                    h->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, sem, 1, true}, bytes);
                }
            }
        }
    }
    constexpr uint32_t cand_bytes = kRecWords * 4;  // a chip's per-row (key, index), 16 B aligned
    volatile PACKET_HEADER_TYPE* cand_hdr[2];
    {
        const uint64_t dst = get_noc_addr(my_x[0], my_y[0], cand_addr + chip * cand_bytes);
        const uint64_t sem = get_noc_addr(my_x[0], my_y[0], cand_sem_addr);
        for (uint32_t dir = 0; dir < 2; dir++) {
            volatile PACKET_HEADER_TYPE* h = hdr(2, 0, 0, dir);  // after the round headers
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
    if (links > 1 && fabric1.is_logically_connected()) {
        fabric1.open_finish();
    }

    volatile tt_l1_ptr uint32_t* gather = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_gather));
    volatile tt_l1_ptr uint32_t* launch_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_addr);
    invalidate_l1_cache();
    const uint32_t launch = *launch_word;

    // the all-reduce rounds: each slice's columns of this chip's partial to the same slice of every peer chip
    for (uint32_t ar = 0; ar < rounds; ar++) {
        const uint32_t par = ar & 1;
        noc_semaphore_wait_min(gather, num_streamers * (ar + 1));
        mark(2 + kTs * ar);
        if constexpr (num_chips > 1) {
            for (uint32_t k = 0; k < K; k++) {
                const uint32_t base = staging + par * staging_bytes + k * slice_bytes;
                for (uint32_t p = 0; p < P; p++) {
                    const uint32_t bytes =
                        (p + 1) * packet_bytes <= slice_bytes ? packet_bytes : slice_bytes - p * packet_bytes;
                    auto& f = links > 1 && ((k * P + p) & 1) ? fabric1 : fabric;
                    if (f.has_forward_connection()) {
                        perform_payload_send(f.get_forward_connection(), base + p * packet_bytes, bytes, hdr(par, k, p, 0));
                    }
                    if (f.has_backward_connection()) {
                        perform_payload_send(f.get_backward_connection(), base + p * packet_bytes, bytes, hdr(par, k, p, 1));
                    }
                }
            }
        }
        mark(2 + kTs * ar + 1);
    }

    // argmax: per row the best of the streamers' records (writer and reader halves), then of the chips over fabric
    volatile tt_l1_ptr uint32_t* recs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rec_addr);
    volatile tt_l1_ptr uint32_t* cand = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_addr);
    auto better = [](uint32_t k, uint32_t i, uint32_t bk, uint32_t bi) { return k > bk || (k == bk && i < bi); };
    noc_semaphore_wait_min(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rec)), 2 * num_streamers);
    mark(2 + kTs * rounds);
    invalidate_l1_cache();
    for (uint32_t r = 0; r < M; r++) {
        uint32_t bk = recs[2 * r], bi = recs[2 * r + 1];
        for (uint32_t s = 1; s < num_streamers; s++) {
            const uint32_t k = recs[s * kRecWords + 2 * r], i = recs[s * kRecWords + 2 * r + 1];
            if (better(k, i, bk, bi)) {
                bk = k;
                bi = i;
            }
        }
        cand[chip * kRecWords + 2 * r] = bk;
        cand[chip * kRecWords + 2 * r + 1] = bi;
    }
    if constexpr (num_chips > 1) {
        if (fabric.has_forward_connection()) {
            perform_payload_send(fabric.get_forward_connection(), cand_addr + chip * cand_bytes, cand_bytes, cand_hdr[0]);
        }
        if (fabric.has_backward_connection()) {
            perform_payload_send(fabric.get_backward_connection(), cand_addr + chip * cand_bytes, cand_bytes, cand_hdr[1]);
        }
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cand_sem_addr), (launch + 1) * (num_chips - 1));
        invalidate_l1_cache();
    }
    const uint32_t stage = cb_base(cb_stage);
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    for (uint32_t r = 0; r < M; r++) {
        uint32_t bk = cand[2 * r], bi = cand[2 * r + 1];
        for (uint32_t c = 1; c < num_chips; c++) {
            if (better(cand[c * kRecWords + 2 * r], cand[c * kRecWords + 2 * r + 1], bk, bi)) {
                bk = cand[c * kRecWords + 2 * r];
                bi = cand[c * kRecWords + 2 * r + 1];
            }
        }
        out[2 * r] = bi;
        out[2 * r + 1] = bk;
    }
    mark(2 + kTs * rounds + 1);
    // serving: control, feed rows and status (the stage CB holds 128 words: out [0, 80), these from 80)
    volatile tt_l1_ptr uint32_t* ctl = out + 80;
    auto rd = [&](uint32_t addr, volatile tt_l1_ptr uint32_t* dst) {
        const InterleavedAddrGen<true> g{.bank_base_address = addr, .page_size = 128};
        noc_async_read(g.get_noc_addr(0), reinterpret_cast<uint32_t>(dst), 128);
    };
    auto wr = [&](uint32_t addr, volatile tt_l1_ptr uint32_t* src) {
        const InterleavedAddrGen<true> g{.bank_base_address = addr, .page_size = 128};
        noc_async_write(reinterpret_cast<uint32_t>(src), g.get_noc_addr(0), 128);
    };
    // the staging (parity 0, idle after the last round) holds tokens / positions / slots / status / attention errors
    volatile tt_l1_ptr uint32_t* toks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(staging);
    volatile tt_l1_ptr uint32_t* poss = toks + 32;
    volatile tt_l1_ptr uint32_t* slts = toks + 64;
    volatile tt_l1_ptr uint32_t* stat = toks + 96;
    volatile tt_l1_ptr uint32_t* errs = toks + 128;  // num_attn x 8 words (A_ERR, A_ERR2 16 B apart)
    rd(control_addr, ctl);
    rd(tokens_addr, toks);
    rd(positions_addr, poss);
    rd(slots_addr, slts);
    rd(status_addr, stat);
    const uint32_t err_addr = get_semaphore(A_ERR);
    for (uint32_t c = 0; c < num_attn; c++) {
        const uint32_t xy = get_arg_val<uint32_t>(attn_args + c);
        noc_async_read(get_noc_addr(xy & 0xffff, xy >> 16, err_addr), reinterpret_cast<uint32_t>(errs + 8 * c), 32);
    }
    noc_async_read_barrier();
    // the block's flags, exactly as the attention cores' launch_of
    uint32_t flags = 0, err_word = 0, err_entry = 0;
    bool inert = true;
    {
        const int32_t p0 = static_cast<int32_t>(poss[0]);
        if (p0 >= 0) {
            if (slts[0] >= max_users) {
                flags = F_SLOT;
            } else if (static_cast<uint32_t>(p0) + M > capacity) {
                flags = F_POSITION;
            } else {
                inert = false;
            }
        }
    }
    for (uint32_t c = 0; c < num_attn; c++) {
        const uint32_t e = errs[8 * c];
        if ((e & 0xf) && (err_word & 0xf) == 0) {
            err_word = e;
            err_entry = errs[8 * c + 4];
        }
        flags |= e & 0xf;
    }
    const uint32_t mode = ctl[0];
    const bool ok = !inert && flags == 0;
    for (uint32_t r = 0; r < M; r++) {
        const uint32_t id = ok ? out[2 * r] : 0;
        if (mode == FEED_ARGMAX || mode == FEED_NONE) {
            toks[r] = id;
        }
        if (ok && (mode == FEED_ARGMAX || mode == FEED_EXTERNAL)) {
            poss[r] = poss[r] + 1;
        }
    }
    if (flags && (stat[ST_FLAGS] & 0xf) == 0) {
        const bool table = (flags & (F_POSITION | F_SLOT)) == 0;
        stat[ST_ROW] = table ? (err_word >> 4) & 0x1f : 0;
        stat[ST_POS] = poss[0];
        stat[ST_PAGE] = table ? (err_word >> 9) : 0xffffffff;
        stat[ST_ENTRY] = table ? err_entry : slts[0];
    }
    stat[ST_FLAGS] = stat[ST_FLAGS] | flags;
    stat[ST_LAUNCHES] = stat[ST_LAUNCHES] + 1;
    if (mode == FEED_ARGMAX || mode == FEED_NONE) {
        wr(tokens_addr, toks);
    }
    if (ok && (mode == FEED_ARGMAX || mode == FEED_EXTERNAL)) {
        wr(positions_addr, poss);
    }
    wr(status_addr, stat);
    out[kOutMeta] = launch;
    out[kOutMeta + 1] = poss[0];
    out[kOutMeta + 2] = M;
    noc_async_write_barrier();
    const InterleavedAddrGen<true> outg{.bank_base_address = out_addr, .page_size = 4 * (kOutMeta + 16)};
    noc_async_write(stage, outg.get_noc_addr(0), 4 * (kOutMeta + 16));
    noc_async_write_barrier();
    *launch_word = launch + 1;
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
    if (links > 1 && fabric1.is_logically_connected()) {
        fabric1.close();
    }
    mark(1);
}
