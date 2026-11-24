// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK program B (pmk/moe2/README.md sec 12): the EXPERT STREAMER phase on a bank-adjacent core (BRISC, NoC0; 24 per
// chip, 3 per DRAM bank: 7a's traced bench 13:46Z measured 493.5 GB/s summed over these 24 cores vs 304 GB/s for
// all-core rotating reads). Expert slices live in ONE interleaved bfp4 tile tensor in the mk-stream-v1 manner:
// expert e's gate|up (H_T x GU_T tiles, K-row-major) then down (W_T x H_T tiles) are CONSECUTIVE pages of bank
// e mod 8 (interleaved page p sits in bank p mod 8 at in-bank index p / 8), so a chunk is one contiguous in-bank
// range: gate|up chunk kc = KC rows x GU_T tiles from in-bank byte (m0[e] + kc KC GU_T) WTB, down chunk nc = W_T
// rows x NC tiles (strided by H_T tiles per row: NC-tile pieces). Streamer s serves the experts e with
// e mod 8 == s mod 8 and (e / 8) mod 3 == s / 8 (12 each, ascending), each with >= 1 M group, as streams of G groups
// (ceil(ng / G) streams of a hot expert) to its rectangle of G compute cores (the PMK_MCAST receiver path of
// ffn_reader.hpp with PMK_MCAST_EXT): per chunk, [ring slot free: the previous multicast out of it flushed] -> DRAM
// read (KC x one GU_T-tile row per transaction, <= NOC_MAX_BURST) -> barrier -> wait READY_x >= G x n (the
// rectangle's credits, one word per ring on THIS core) -> multicast into the receivers' slot (their CB_WGU / CB_WD
// base, published by the rectangle's first receiver into PARK_GU / PARK_D words here, + (n - 1) mod 2 x chunk bytes)
// -> VALID_x SET n on the receivers (semaphores 10 / 11), on NoC0 behind the data. Counts are monotonic per launch.
// The chunks of a launch form ONE sequence (every stream: H_T / KC gate|up chunks then H_T / NC down chunks; both
// even) over the 2-slot ring: chunk n sits in slot n mod 2 (= the per-ring parity the receivers use).
// PMK_STREAM_PREFETCH (perf knob, moe2/PERF.md): the DRAM read of chunk n + 1 is issued BEFORE the credit wait of
// chunk n (its slot held chunk n - 1, whose multicast has been flushed), so a credit-bound streamer always has the
// next chunk landing under the wait and a DRAM-bound one overlaps the read with the multicast. Default (off): the
// original order (read n -> barrier -> credits n -> multicast n -> read n + 1).
// XFFN_PROBE (the bench's "probe": true): per-streamer counters accumulated in registers, written ONCE at the end
// as row (RT0 + 12) of the probe tensor (RT0 + 11; kernels/prefill/moe_xffn_probe.hpp's [rows, 16] layout):
//   0 S_GU_READ / 1 S_DN_READ: cycles from the chunk's first DRAM read issue to its read barrier (PREFETCH: only
//     the residual barrier wait after the credits + multicast of the previous chunk);
//   2 S_GU_CREDIT / 3 S_DN_CREDIT: cycles waiting READY >= G n (the members have not freed the slot: credit-bound);
//   4 S_STREAMS: streams served | (PMK_STREAM_PREFETCH << 16); 5 S_KB: KB read from DRAM; 6 S_MC_ISSUE: cycles
//   INSIDE the multicast + VALID issue calls (the NIU queue full: the multicast TRANSMIT is on the critical path --
//   the 18:xxZ "0.53 unattributed"); 7 S_MC_FLUSH: cycles in noc_async_writes_flushed (the previous multicast still
//   leaving the NIU); 8 / 9: gate|up / down chunks; 10 S_PLAN_WAIT: the plan barrier + the table reads; 11 S_MC_ACK:
//   the final write barrier (multicast acks); 12 S_RD_ISSUE: cycles inside the DRAM read issue calls (12-32 per
//   chunk); 13 span; 14 / 15 start / end (low 32 bits of the wall clock). Every cycle of the span is in one slot.
// PMK_MCAST_LINKED (knob): a chunk's multicast packets are LINKED (the NoC keeps the multicast path reserved across
// the chunk's 16-KB packets; ttnn's mcast matmul convention) -- the last packet unlinked.
// PMK_MCAST_DUAL (knob, PERF.md sec 10): the streamer core's idle NCRISC (exp_streamer_nc.hpp, NoC1) multicasts the
// SECOND half of every chunk while this BRISC sends the first half on NoC0 -- two NIUs per streamer, the per-packet
// issue cost (~700-1,800 cycles per 16-KB packet on silicon) split in two. Hand-off through the sync words: GO
// (96: n, 100: slot, 104: receivers' dst, 108: bytes) written here after the credits, DONE (112) by the NCRISC after
// its half left L1 (so a ring slot is reused only after both halves of its previous chunk are out). The NCRISC SETs
// a second VALID per ring (semaphores 12 / 13) behind its half on NoC1; receivers wait both.
// PMK_MCAST_UNICAST (knob): no multicast at all -- g unicast writes of the chunk (one per member of the exact box)
// and a 4-B unicast write of the VALID count to each member's semaphore on the same route (ordered behind its data):
// g x the NoC bytes, but no multicast path reservation (24 streamers reserving routes into the member block at once
// is the suspected 6 GB/s transmit of the 20-replay scoreboard, PERF.md sec 9).
// Compile-time (CT0 + i): H_T, W_T, KC, NC, MG, E, EP, WTB, G, GU_BB, DN_BB. Runtime (RT0 + i): cnt addr, start addr,
// m0 table addr ([E] uint32 in-bank tile index of expert e's slice), s (streamer index), sync words L1 address
// (READY_GU, READY_D, PARK_GU, PARK_D, VAL_GU, VAL_D at 16-B stride), lx, ly, hx, hy (the rectangle's NoC box),
// P_compute (the plan barrier count), the stream tensor's address, [XFFN_PROBE: the probe tensor's address, the row],
// g (RT0 + 13: THIS rectangle's member count -- PERF.md sec 7 cover placement: rectangles of 1..gmax members cover
// every compute core, so g varies per streamer; CT0 + 8 G is the largest).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/moe_common.hpp"

using namespace pmk_moe;

namespace exps {
constexpr uint32_t SW_READY_GU = 0, SW_READY_D = 16, SW_PARK_GU = 32, SW_PARK_D = 48, SW_VAL_GU = 64, SW_VAL_D = 80;
// PMK_MCAST_DUAL hand-off words (perf5, 16 2026-10-22 22:23Z watcher: "NCRISC noc1 multicast write 4 bytes from L1
// ..f4 (4-byte aligned, not 16)" = the NCRISC's VALID source word sat at SW_DONE + 4; a multicast SOURCE must be 16-B
// aligned on Blackhole): GO (n, slot, dst, bytes, DOWN: the ring kind -- GU_BB == DN_BB at (1, 4, 8), so the kind
// cannot be told from the byte count), DONE, and the NCRISC's two 16-B-aligned VALID source words. The sync shard is
// 256 B.
constexpr uint32_t SW_GO = 96, SW_GO_SLOT = 100, SW_GO_DST = 104, SW_GO_BYTES = 108, SW_DONE = 112, SW_GO_DOWN = 116;
constexpr uint32_t SW_VAL1_GU = 128, SW_VAL1_D = 144;
constexpr uint32_t SYNC_B = 256;
// the streamer probe row's slots (XFFN_PROBE)
constexpr uint32_t S_GU_READ = 0, S_DN_READ = 1, S_GU_CREDIT = 2, S_DN_CREDIT = 3, S_STREAMS = 4, S_KB = 5,
                   S_MC_ISSUE = 6, S_MC_FLUSH = 7, S_GU_CHUNKS = 8, S_DN_CHUNKS = 9, S_PLAN_WAIT = 10, S_MC_ACK = 11,
                   S_RD_ISSUE = 12, S_SPAN = 13, S_START = 14, S_END = 15;
}  // namespace exps

#ifdef XFFN_PROBE
#define SP_TIC() sp_t0 = get_timestamp()
#define SP_TOC(acc) acc += static_cast<uint32_t>(get_timestamp() - sp_t0)
#define SP_ADD(acc, v) acc += (v)
#else
#define SP_TIC() \
    do {         \
    } while (0)
#define SP_TOC(acc) \
    do {            \
    } while (0)
#define SP_ADD(acc, v) \
    do {               \
    } while (0)
#endif

FORCE_INLINE void phase_main() {
    constexpr uint32_t H_T = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t W_T = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t KC = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t NC = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t MG = get_compile_time_arg_val(CT0 + 4);
    constexpr uint32_t E = get_compile_time_arg_val(CT0 + 5);
    constexpr uint32_t EP = get_compile_time_arg_val(CT0 + 6);
    constexpr uint32_t WTB = get_compile_time_arg_val(CT0 + 7);
    constexpr uint32_t G = get_compile_time_arg_val(CT0 + 8);
    (void)G;  // the largest rectangle (the runtime g below is this one's)
    constexpr uint32_t GU_T = 2 * W_T;
    constexpr uint32_t GU_BB = KC * GU_T * WTB;  // one gate|up chunk
    constexpr uint32_t DN_BB = W_T * NC * WTB;   // one down chunk
    // a ring slot holds either chunk: the larger of the two (16, 2026-10-22 15:59Z watcher overflow: TP2's lever-2
    // blocks (1, 1, 4) have GU_BB 36,864 < DN_BB 73,728 -- the down reads ran past a gate|up-sized ring); the host
    // sizes the ring 2 x SLOT_BB (pmk_moe2.ProgramB / program_b_plan)
    constexpr uint32_t SLOT_BB = GU_BB > DN_BB ? GU_BB : DN_BB;
#ifdef PMK_RING_SLOTS
    // perf5 (moe2/PERF.md sec 11 lever 3): S-chunk rings on both sides -- the streamer runs S - 1 chunks ahead of the
    // receivers' compute, so the credit round trip + transmit leave the per-chunk chain (T = max(C, M) instead of
    // max(C, (C + delta + M) / 2)); the receivers' slot of chunk n is (n - 1) mod S past the published base
    constexpr uint32_t SLOTS = PMK_RING_SLOTS;
#else
    constexpr uint32_t SLOTS = 2;
#endif
    constexpr uint32_t GU_ROW = GU_T * WTB;  // one K row of gate|up: one read
    constexpr uint32_t DN_PIECE = NC * WTB;  // NC tiles of one down row: one read
    static_assert(DN_PIECE <= NOC_MAX_BURST_SIZE, "one transaction per down row piece");
    // a gate|up K row (GU_T tiles: 9,216 B at TP8, 36,864 B on the TP2 pair) in <= NOC_MAX_BURST_SIZE transactions
    auto read_row = [&](uint64_t src, uint32_t dst, uint32_t bytes) {
        while (bytes > 0) {
            const uint32_t n = bytes < NOC_MAX_BURST_SIZE ? bytes : NOC_MAX_BURST_SIZE;
            noc_async_read(src, dst, n);
            src += n;
            dst += n;
            bytes -= n;
        }
    };
    static_assert(
        (H_T / KC) % SLOTS == 0 && (H_T / NC) % SLOTS == 0,
        "chunks per unit = whole rings (the receivers' S-slot rings)");
#ifdef PMK_STREAM_PREFETCH
    constexpr bool PREFETCH = true;
#else
    constexpr bool PREFETCH = false;
#endif

    const uint32_t cnt_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t start_addr = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t m0_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t s = get_arg_val<uint32_t>(RT0 + 3);
    const uint32_t sync = get_arg_val<uint32_t>(RT0 + 4);
    const uint32_t lx = get_arg_val<uint32_t>(RT0 + 5), ly = get_arg_val<uint32_t>(RT0 + 6);
    const uint32_t hx = get_arg_val<uint32_t>(RT0 + 7), hy = get_arg_val<uint32_t>(RT0 + 8);
    const uint32_t p_compute = get_arg_val<uint32_t>(RT0 + 9);
    // the packed stream tensor's base (each bank's copy of an interleaved buffer starts at the same bank offset);
    // m0 counts tiles from 0 within the bank (16, 2026-10-22 review: without it every chunk came from bank offset 0)
    const uint32_t w_base = get_arg_val<uint32_t>(RT0 + 10);
    const uint32_t g = get_arg_val<uint32_t>(RT0 + 13);  // this rectangle's members (<= G)
    const uint32_t bank = s & 7u;
    const uint32_t ring = get_write_ptr(CB_WGU);  // the streamer's ring: 2 x SLOT_BB (gate|up and down chunks)
#ifdef XFFN_PROBE
    uint64_t sp_t0 = 0;
    const uint64_t sp_start = get_timestamp();
    uint32_t c_gu_read = 0, c_dn_read = 0, c_gu_cred = 0, c_dn_cred = 0, c_streams = 0, c_kb = 0, c_mc_issue = 0,
             c_flush = 0, c_gu_n = 0, c_dn_n = 0, c_plan = 0, c_ack = 0, c_rd_issue = 0;
#endif

    // the sync words (this core's) start at 0 for the launch
    for (uint32_t k = 0; k < exps::SYNC_B / 4; ++k) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync)[k] = 0;
    }
    // the plan (on the compute cores) is done: they incremented this core's SEM_PLAN3 copy too
    SP_TIC();
    wait_min(SEM_PLAN3, p_compute);
    SP_TOC(c_plan);
    // tables: counts | starts | m0
    const uint32_t tab = get_write_ptr(CB_TAB) + SYNC_WORDS_B;
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    volatile tt_l1_ptr uint32_t* start = cnt + EP;
    volatile tt_l1_ptr uint32_t* m0 = start + EP;
    {
        SP_TIC();
        const InterleavedAddrGen<true> cnt_g{.bank_base_address = cnt_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> m0_g{.bank_base_address = m0_addr, .page_size = EP * 4};
        noc_async_read(cnt_g.get_noc_addr(0), reinterpret_cast<uint32_t>(cnt), EP * 4);
        noc_async_read(start_g.get_noc_addr(0), reinterpret_cast<uint32_t>(start), EP * 4);
        noc_async_read(m0_g.get_noc_addr(0), reinterpret_cast<uint32_t>(m0), EP * 4);
        noc_async_read_barrier();
        SP_TOC(c_plan);
    }
    volatile tt_l1_ptr uint32_t* ready_gu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_READY_GU);
    volatile tt_l1_ptr uint32_t* ready_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_READY_D);
    volatile tt_l1_ptr uint32_t* park_gu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_PARK_GU);
    volatile tt_l1_ptr uint32_t* park_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_PARK_D);
    volatile tt_l1_ptr uint32_t* val_gu = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_VAL_GU);
    volatile tt_l1_ptr uint32_t* val_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_VAL_D);
    const uint32_t vgu = get_semaphore(SEM_VALID_GU), vd = get_semaphore(SEM_VALID_D);
    uint32_t n_gu = 0, n_d = 0;
#ifdef PMK_STREAM_LPT
    // the live expert -> streamer table (moe_common.hpp stream_lpt; the receivers compute the same one): the g table
    // at RT0 + 14 (n, then g per streamer); scratch past the three tables (the host sized CB_TAB for it)
    SP_TIC();
    // RT0 + 14: n, then g per streamer; RT0 + 15 + n: the per-streamer cost weights (PMK_LPT_W)
    const uint32_t n_s = get_arg_val<uint32_t>(RT0 + 14);
    volatile tt_l1_ptr uint8_t* owner = stream_lpt(
        cnt,
        E,
        EP,
        MG,
        n_s,
        RT0 + 15,
        PMK_STREAM_LPT,
        (reinterpret_cast<uint32_t>(m0 + EP) + 63) & ~63u,
#ifdef PMK_LPT_W
        RT0 + 15 + n_s);
#else
        0);
#endif
    SP_TOC(c_plan);
#else
    (void)bank;
#endif
#ifdef PMK_MCAST_DUAL
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_GO);
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync + exps::SW_DONE);
    uint32_t n_go = 0;
#endif
    auto mcast = [&](uint32_t l1, uint32_t dst_l1, uint32_t bytes, uint32_t down) {
        (void)down;
#ifdef PMK_MCAST_DUAL
        // the NCRISC takes the second half (params then GO, program order on this core's L1); the ring kind travels
        // in SW_GO_DOWN (the halves of a gate|up and a down chunk have the same byte count at (1, 4, 8))
        const uint32_t half = bytes / 2;
        go[1] = l1 + half;
        go[2] = dst_l1 + half;
        go[3] = bytes - half;
        go[5] = down;
        go[0] = ++n_go;
        bytes = half;
#endif
#ifdef PMK_MCAST_UNICAST
        for (uint32_t y = ly; y <= hy; ++y) {
            for (uint32_t x = lx; x <= hx; ++x) {
                noc_async_write(l1, get_noc_addr(x, y, dst_l1), bytes);
            }
        }
        return;
#endif
#ifdef PMK_MCAST_LINKED
        if (bytes > NOC_MAX_BURST_SIZE) {  // all packets but the last linked: one path reservation per chunk
            const uint32_t head = bytes - NOC_MAX_BURST_SIZE;
            noc_async_write_multicast(l1, get_noc_multicast_addr(lx, ly, hx, hy, dst_l1), head, g, true);
            l1 += head;
            dst_l1 += head;
            bytes -= head;
        }
#endif
        noc_async_write_multicast(l1, get_noc_multicast_addr(lx, ly, hx, hy, dst_l1), bytes, g, false);
    };
    auto set_valid = [&](volatile tt_l1_ptr uint32_t* src, uint32_t sem_l1, uint32_t v) {
        *src = v;
#ifdef PMK_MCAST_UNICAST
        for (uint32_t y = ly; y <= hy; ++y) {
            for (uint32_t x = lx; x <= hx; ++x) {
                noc_async_write(reinterpret_cast<uint32_t>(src), get_noc_addr(x, y, sem_l1), 4);
            }
        }
#else
        noc_semaphore_set_multicast(
            reinterpret_cast<uint32_t>(src), get_noc_multicast_addr(lx, ly, hx, hy, sem_l1), g, false);
#endif
    };

    // ---- the chunk sequence: (served expert e, stream st < ns(e), gate|up chunks 0 .. H_T / KC - 1, then down chunks
    // 0 .. H_T / NC - 1); one stream = one (expert, group of G M groups)
    struct Chunk {
        uint32_t gu0;   // in-bank byte of the slice's gate|up (its down follows at + H_T GU_T WTB)
        uint32_t idx;   // kc or nc
        uint32_t down;  // 0 = gate|up chunk, 1 = down chunk
        uint32_t bank;  // the expert's DRAM bank (e mod 8: this streamer's own unless PMK_STREAM_LPT mode 2)
        uint32_t e;     // the expert (PMK_B_INPLACE: its shard bases)
    };
#ifdef PMK_STREAM_LPT
    auto served = [&](uint32_t e) { return owner[e] == s; };
#else
    auto served = [&](uint32_t e) { return (e & 7u) == bank && ((e >> 3) % 3) == (s >> 3) && cnt[e] != 0; };
#endif
#ifdef PMK_GU_SPLIT
    // perf5 PMK_GU_SPLIT: a stream = split_rows(g) row tiles (every member computes a column slice of all of them)
    const uint32_t rows_per_stream =
        split_rows(g, (2 * W_T) / PMK_GU_SPLIT_SUBA, PMK_GU_SPLIT_SUBA, PMK_GU_SPLIT_GUPT, PMK_GU_SPLIT_RMAX);
#else
    const uint32_t rows_per_stream = g;
#endif
    auto streams_of = [&](uint32_t e) {
        const uint32_t ng = ((cnt[e] + 31) / 32 + MG - 1) / MG;
        return (ng + rows_per_stream - 1) / rows_per_stream;
    };
    uint32_t en_e = 0, en_st = 0, en_ns = 0, en_idx = 0, en_down = 0, en_live = 0;
    auto next = [&](Chunk& c) -> bool {
        if (!en_live) {
            while (en_e < E && !served(en_e)) {
                ++en_e;
            }
            if (en_e >= E) {
                return false;
            }
            en_ns = streams_of(en_e);
            en_st = en_idx = en_down = 0;
            en_live = 1;
        } else {
            ++en_idx;
            if (!en_down && en_idx == H_T / KC) {
                en_down = 1;
                en_idx = 0;
            } else if (en_down && en_idx == H_T / NC) {
                en_down = 0;
                en_idx = 0;
                if (++en_st == en_ns) {
                    ++en_e;
                    while (en_e < E && !served(en_e)) {
                        ++en_e;
                    }
                    if (en_e >= E) {
                        return false;
                    }
                    en_ns = streams_of(en_e);
                    en_st = 0;
                }
            }
        }
        c.gu0 = m0[en_e] * WTB;
        c.idx = en_idx;
        c.down = en_down;
        c.bank = en_e & 7u;
        c.e = en_e;
        return true;
    };
#ifdef PMK_B_INPLACE
    // PMK_B_INPLACE (moe2/PERF.md sec 13, Stage 12's ruling: no stream copy): the chunks come from pl.xsl's RESIDENT
    // expert tensors -- gate_up [H, 2 W] / down [W, H] ND-sharded [rows, 64] over the 8 banks (fused_decoder._nd_dram):
    // shard j = tile-column pair j, in bank j mod 8, its tiles (k, c) contiguous at base_j + (2 k + c) WTB. A chunk is
    // therefore KC x W_T (gate|up) or W_T x NC / 2 (down) reads of ONE PAIR (1,152 B) landing in the ring's row-major
    // tile order (the receivers' compute is untouched: bitwise); the shard bases of the current expert come from the
    // [E, 2] address table (RT0 + 2, every expert's tensor shares the layout: ffn_reader.hpp XFFN_SHARD_BASES) once
    // per expert. Non-adjacent banks: the stream rate is measured, not assumed (sec 13).
    constexpr auto gu_args = TensorAccessorArgs<CT0 + 9>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    volatile tt_l1_ptr uint32_t* tbl;
    uint64_t* gu_base;
    uint64_t* dn_base;
    {
        // the table past the tables + the LPT scratch (the host sized CB_TAB: pmk_moe2.ProgramB.parts tab_b)
        const uint32_t lpt0 = (reinterpret_cast<uint32_t>(m0 + EP) + 63) & ~63u;
        const uint32_t tbl0 = (lpt0 + lpt_scratch_bytes(EP) + 63) & ~63u;
        tbl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tbl0);
        gu_base = reinterpret_cast<uint64_t*>((tbl0 + 2 * E * 4 + 63) & ~63u);
        dn_base = gu_base + W_T;
        SP_TIC();
        const InterleavedAddrGen<true> tbl_g{
            .bank_base_address = m0_addr, .page_size = 2 * E * 4};  // RT0 + 2 = the table
        noc_async_read(tbl_g.get_noc_addr(0), tbl0, (2 * E * 4 + 63) / 64 * 64);
        noc_async_read_barrier();
        SP_TOC(c_plan);
    }
    uint32_t base_e = 0xFFFFFFFFu;
    auto issue = [&](const Chunk& c, uint32_t slot) {
        SP_TIC();
        if (c.e != base_e) {  // a new expert: its shard bases (W_T gate|up pairs, H_T / 2 down pairs)
            base_e = c.e;
            const auto gu = TensorAccessor(gu_args, tbl[2 * c.e], WTB);
            const auto dn = TensorAccessor(dn_args, tbl[2 * c.e + 1], WTB);
            for (uint32_t j = 0; j < W_T; ++j) {
                gu_base[j] = gu.get_noc_addr(2 * j);
            }
            for (uint32_t sh = 0; sh < H_T / 2; ++sh) {
                dn_base[sh] = dn.get_noc_addr(2 * sh);
            }
        }
        if (!c.down) {
            for (uint32_t k = 0; k < KC; ++k) {
                const uint32_t off = (c.idx * KC + k) * 2 * WTB;
                for (uint32_t j = 0; j < W_T; ++j) {
                    noc_async_read(gu_base[j] + off, slot + (k * GU_T + 2 * j) * WTB, 2 * WTB);
                }
            }
            SP_ADD(c_kb, GU_BB / 1024);
        } else {
            for (uint32_t k = 0; k < W_T; ++k) {
                for (uint32_t jj = 0; jj < NC; jj += 2) {
                    noc_async_read(dn_base[(c.idx * NC + jj) / 2] + k * 2 * WTB, slot + (k * NC + jj) * WTB, 2 * WTB);
                }
            }
            SP_ADD(c_kb, DN_BB / 1024);
        }
        SP_TOC(c_rd_issue);
    };
#else
    // the chunk's DRAM reads into a ring slot (no barrier)
    auto issue = [&](const Chunk& c, uint32_t slot) {
        SP_TIC();
        const uint64_t dram = get_noc_addr_from_bank_id<true>(c.bank, w_base);
        if (!c.down) {
            for (uint32_t k = 0; k < KC; ++k) {
                read_row(dram + c.gu0 + (c.idx * KC + k) * GU_ROW, slot + k * GU_ROW, GU_ROW);
            }
            SP_ADD(c_kb, GU_BB / 1024);
        } else {
            const uint32_t dn0 = c.gu0 + H_T * GU_T * WTB;
            for (uint32_t k = 0; k < W_T; ++k) {
                noc_async_read(dram + dn0 + (k * H_T + c.idx * NC) * WTB, slot + k * DN_PIECE, DN_PIECE);
            }
            SP_ADD(c_kb, DN_BB / 1024);
        }
        SP_TOC(c_rd_issue);
    };
#endif
    // the chunk's hand-off: credits (the receivers' slot n is free) -> multicast -> VALID SET n (NoC0, ordered)
    auto send = [&](const Chunk& c, uint32_t slot) {
        if (!c.down) {
            ++n_gu;
            SP_ADD(c_gu_n, 1);
            SP_TIC();
            noc_semaphore_wait_min(ready_gu, g * n_gu);
            SP_TOC(c_gu_cred);
            invalidate_l1_cache();
            SP_TIC();
            mcast(slot, *park_gu + ((n_gu - 1) % SLOTS) * GU_BB, GU_BB, 0);
            set_valid(val_gu, vgu, n_gu);
            SP_TOC(c_mc_issue);
        } else {
            ++n_d;
            SP_ADD(c_dn_n, 1);
            SP_TIC();
            noc_semaphore_wait_min(ready_d, g * n_d);
            SP_TOC(c_dn_cred);
            invalidate_l1_cache();
            SP_TIC();
            mcast(slot, *park_d + ((n_d - 1) % SLOTS) * DN_BB, DN_BB, 1);
            set_valid(val_d, vd, n_d);
            SP_TOC(c_mc_issue);
        }
    };

    Chunk cur, nxt;
    uint32_t n = 0;  // chunks of the launch so far: chunk n in ring slot n mod SLOTS
    if (next(cur)) {
        issue(cur, ring);
        SP_TIC();
        noc_async_read_barrier();
        SP_TOC(c_gu_read);
        while (true) {
            const bool more = next(nxt);
            const uint32_t slot = ring + (n % SLOTS) * SLOT_BB, slot_next = ring + ((n + 1) % SLOTS) * SLOT_BB;
            if (!cur.down && cur.idx == 0) {
                SP_ADD(c_streams, 1);
            }
            if (PREFETCH) {
                if (more) {
                    SP_TIC();
                    noc_async_writes_flushed();  // chunk n - 1's multicast left slot_next
#ifdef PMK_MCAST_DUAL
                    // DONE counts the NCRISC halves fully out of L1 (chunk k = count k + 1): slot_next last held chunk
                    // n + 1 - SLOTS, so DONE >= n + 2 - SLOTS (S = 2: DONE >= n)
                    while (n + 2 > SLOTS && *done < n + 2 - SLOTS) {
                        invalidate_l1_cache();
                    }
#endif
                    SP_TOC(c_flush);
                    issue(nxt, slot_next);  // lands under the credit wait / multicast of chunk n
                }
                send(cur, slot);
                if (!more) {
                    break;
                }
                SP_TIC();
                noc_async_read_barrier();  // the residual wait for chunk n + 1
                if (nxt.down) {
                    SP_TOC(c_dn_read);
                } else {
                    SP_TOC(c_gu_read);
                }
            } else {
                send(cur, slot);
                if (!more) {
                    break;
                }
                SP_TIC();
                noc_async_writes_flushed();  // the multicast out of slot_next (chunk n - 1) left
#ifdef PMK_MCAST_DUAL
                while (n + 2 > SLOTS && *done < n + 2 - SLOTS) {
                    invalidate_l1_cache();
                }
#endif
                SP_TOC(c_flush);
                issue(nxt, slot_next);
                SP_TIC();
                noc_async_read_barrier();
                if (nxt.down) {
                    SP_TOC(c_dn_read);
                } else {
                    SP_TOC(c_gu_read);
                }
            }
            cur = nxt;
            ++n;
        }
    }
    SP_TIC();
    noc_async_write_barrier();  // every multicast and VALID set acknowledged
#ifdef PMK_MCAST_DUAL
    go[1] = go[2] = go[3] = go[5] = 0;
    go[0] = ++n_go;  // the NCRISC's end mark (bytes 0)
    while (*done < n_go) {
        invalidate_l1_cache();
    }
#endif
    SP_TOC(c_ack);
#ifdef XFFN_PROBE
    {
        // one 64-B row at the end: the counts table (done) is the scratch
        const uint64_t sp_end = get_timestamp();
        volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>((tab + 63) & ~63u);
        row[exps::S_GU_READ] = c_gu_read;
        row[exps::S_DN_READ] = c_dn_read;
        row[exps::S_GU_CREDIT] = c_gu_cred;
        row[exps::S_DN_CREDIT] = c_dn_cred;
        row[exps::S_STREAMS] = c_streams | ((PREFETCH ? 1u : 0u) << 16);
        row[exps::S_KB] = c_kb;
        row[exps::S_MC_ISSUE] = c_mc_issue;
        row[exps::S_MC_FLUSH] = c_flush;
        row[exps::S_GU_CHUNKS] = c_gu_n;
        row[exps::S_DN_CHUNKS] = c_dn_n;
        row[exps::S_PLAN_WAIT] = c_plan;
        row[exps::S_MC_ACK] = c_ack;
        row[exps::S_RD_ISSUE] = c_rd_issue;
        row[exps::S_SPAN] = static_cast<uint32_t>(sp_end - sp_start);
        row[exps::S_START] = static_cast<uint32_t>(sp_start);
        row[exps::S_END] = static_cast<uint32_t>(sp_end);
        const InterleavedAddrGen<true> probe{.bank_base_address = get_arg_val<uint32_t>(RT0 + 11), .page_size = 64};
        noc_async_write(reinterpret_cast<uint32_t>(row), probe.get_noc_addr(get_arg_val<uint32_t>(RT0 + 12)), 64);
        noc_async_write_barrier();
    }
#endif
}
