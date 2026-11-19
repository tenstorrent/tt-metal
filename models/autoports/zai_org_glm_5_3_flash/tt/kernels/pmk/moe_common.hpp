// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill megakernel (PMK) MoE phase set (tt/pmk_moe.py; doc/ttnn/16-prefill-production/pmk/moe/README.md): the
// EXISTING Stage 16 prefill MoE kernels (kernels/prefill/moe_plan_mc_*, moe_xffn_*, moe_combine_rm_*) run back to back
// as PHASES of one program on the MM cores -- plan (3 steps), expert-slice FFN (xffn: one expert M group per core at a
// time, the slice weights streamed by the BRISC into double-buffered CBs under the compute), combine (rm: the weighted
// top-k sum per token in fp32, slot order) -- with the hand-offs of prefill_megakernel.md sec 4b:
//   wait   H8 (the x-hat_f store + the routing ids): a wait_min on a semaphore the family names (PMK_MOE_WAIT_SEM);
//   signal H9 (the FFN partial, per 32-row block): +1 per combine unit on the consumer's per-block counter words
//          (PMK_MOE_H9: consumer NoC (x, y), base address, 16-B stride; a block is complete at >= H_SPLIT).
// Barriers between the phases are program-wide counters on semaphores 5..8 (the ids the experts reuse after the
// attention half, sec 4b-ids): every core increments the id on every MM core, then waits >= P (wait_min, never
// equality; no resets inside a launch; the next barrier uses the next id).
// CB indices: PMK_MOE_CB0 .. +12 (default 44: the MM cores' ring range of Stage 14's map; the family may alias them
// onto idle phase CBs). The phase kernels keep the ORIGINAL kernels' arithmetic and CB roles; only the arg bases
// (CT0 / RT0 per phase) and the CB numbering moved.
#pragma once

#include <cstdint>

#ifndef PMK_MOE_CB0
#define PMK_MOE_CB0 44
#endif

// GLM53_PMK_ARGS=table on STANDALONE program B (tt/pmk_moe2.ProgramB._arg_table; Stage 14, trace-light PMK 2026-10-23):
// the launch carries only [table address, row] per core; the row = this kernel's complete runtime-arg vector in the
// program's DRAM table (written once at the compile launch). One NoC read into this RISC's slot of CB PMK_B_ARG_CB,
// then rta_l1_base points at it: every get_arg_val after reads the row (the pmk_common.hpp load_arg_table pattern). Its
// own define (PMK_B_ARG_TABLE): B's kernels are also phase bodies of the family's composites, which use PMK_ARG_TABLE.
// THE DYNAMIC BLOCK (2026-10-23 18:43:37Z): the launch args are [table, row, n_dyn] + n_dyn (position, value) pairs:
// the row's words that are caller-tensor addresses (they move between calls in the served chain) travel in the launch
// and overwrite the copied row before rta_l1_base moves (get_arg_val still reads the launch args until then).
#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT)
#define PMK_B_RTA_COUNT() \
    (rta_count = PMK_B_ARG_PAGE / 4)  // the watcher bounds get_arg_val by the row (pmk_common.hpp)
#else
#define PMK_B_RTA_COUNT() ((void)0)
#endif
#if defined(PMK_B_ARG_TABLE) && !defined(COMPILE_FOR_TRISC)
#define PMK_B_PROLOGUE()                                                                                     \
    do {                                                                                                     \
        const uint32_t tbl_ = get_arg_val<uint32_t>(0);                                                      \
        const uint32_t row_ = get_arg_val<uint32_t>(1);                                                      \
        const uint32_t ndyn_ = get_arg_val<uint32_t>(2);                                                     \
        const uint32_t dst_ = ((get_write_ptr(PMK_B_ARG_CB) + 63) & ~63u) + PMK_B_ARG_SLOT * PMK_B_ARG_PAGE; \
        const InterleavedAddrGen<true> t_{.bank_base_address = tbl_, .page_size = PMK_B_ARG_PAGE};           \
        noc_async_read(t_.get_noc_addr(row_), dst_, PMK_B_ARG_PAGE);                                         \
        noc_async_read_barrier();                                                                            \
        volatile tt_l1_ptr uint32_t* rw_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_);             \
        for (uint32_t i_ = 0; i_ < ndyn_; i_++) {                                                            \
            rw_[get_arg_val<uint32_t>(3 + 2 * i_)] = get_arg_val<uint32_t>(4 + 2 * i_);                      \
        }                                                                                                    \
        rta_l1_base = reinterpret_cast<uint32_t tt_l1_ptr*>(dst_);                                           \
        PMK_B_RTA_COUNT();                                                                                   \
    } while (0)
#else
#define PMK_B_PROLOGUE() \
    do {                 \
    } while (0)
#endif
// the combine's tilize (ttnn/kernel_lib/tilize_helpers.hpp) takes CB indices < 32: its two CBs sit at PMK_MOE_CB1 + k
// (default 27, 28: the owners' mHC-post range, idle during the FFN half; the family re-bases / aliases)
#ifndef PMK_MOE_CB1
#define PMK_MOE_CB1 27
#endif
// per-CB overrides (Stage 16 M-B, 2026-10-24: program B as a window of the family's fused layer program reuses the
// family's dead CB indices of the same data format -- tt/pmk_moe.PmkMoe.cb_remap -> PMK_MOE_CBX_<NAME>); default = the
// contiguous map from PMK_MOE_CB0 / PMK_MOE_CB1
#ifndef PMK_MOE_CBX_CTL
#define PMK_MOE_CBX_CTL (PMK_MOE_CB0 + 0)
#endif
#ifndef PMK_MOE_CBX_WGU
#define PMK_MOE_CBX_WGU (PMK_MOE_CB0 + 1)
#endif
#ifndef PMK_MOE_CBX_XRM
#define PMK_MOE_CBX_XRM (PMK_MOE_CB0 + 2)
#endif
#ifndef PMK_MOE_CBX_XT
#define PMK_MOE_CBX_XT (PMK_MOE_CB0 + 3)
#endif
#ifndef PMK_MOE_CBX_WD
#define PMK_MOE_CBX_WD (PMK_MOE_CB0 + 4)
#endif
#ifndef PMK_MOE_CBX_WCTL
#define PMK_MOE_CBX_WCTL (PMK_MOE_CB0 + 5)
#endif
#ifndef PMK_MOE_CBX_TAB
#define PMK_MOE_CBX_TAB (PMK_MOE_CB0 + 6)
#endif
#ifndef PMK_MOE_CBX_WTAB
#define PMK_MOE_CBX_WTAB (PMK_MOE_CB0 + 7)
#endif
#ifndef PMK_MOE_CBX_GUP
#define PMK_MOE_CBX_GUP (PMK_MOE_CB0 + 8)
#endif
#ifndef PMK_MOE_CBX_ACT
#define PMK_MOE_CBX_ACT (PMK_MOE_CB0 + 9)
#endif
#ifndef PMK_MOE_CBX_YRM
#define PMK_MOE_CBX_YRM (PMK_MOE_CB0 + 10)
#endif
#ifndef PMK_MOE_CBX_PLAN
#define PMK_MOE_CBX_PLAN (PMK_MOE_CB0 + 11)
#endif
#ifndef PMK_MOE_CBX_CIN
#define PMK_MOE_CBX_CIN (PMK_MOE_CB0 + 12)
#endif
#ifndef PMK_MOE_CBX_CW
#define PMK_MOE_CBX_CW (PMK_MOE_CB0 + 13)
#endif
#ifndef PMK_MOE_CBX_CIDX
#define PMK_MOE_CBX_CIDX (PMK_MOE_CB0 + 14)
#endif
#ifndef PMK_MOE_CBX_COUT
#define PMK_MOE_CBX_COUT (PMK_MOE_CB1 + 0)
#endif
#ifndef PMK_MOE_CBX_CRM
#define PMK_MOE_CBX_CRM (PMK_MOE_CB1 + 1)
#endif
#ifndef PMK_MOE_CBX_CSH
#define PMK_MOE_CBX_CSH (PMK_MOE_CB0 + 15)
#endif
#ifndef PMK_MOE_CBX_ACTS
#define PMK_MOE_CBX_ACTS (PMK_MOE_CB0 + 16)
#endif
#ifndef PMK_MOE_CBX_CIN2
#define PMK_MOE_CBX_CIN2 (PMK_MOE_CB0 + 17)
#endif
#ifndef PMK_MOE_CBX_CW2
#define PMK_MOE_CBX_CW2 (PMK_MOE_CB0 + 18)
#endif
// W7 step 5 (G: 14's gang EP expert FFN, kernels/pmk/gang_*.hpp; the RM_SKIP combine's zero part)
#ifndef PMK_MOE_CBX_RIN
#define PMK_MOE_CBX_RIN (PMK_MOE_CB0 + 19)
#endif
#ifndef PMK_MOE_CBX_CZERO
#define PMK_MOE_CBX_CZERO (PMK_MOE_CB0 + 20)
#endif

namespace pmk_moe {

// ---- FFN (xffn) CBs: the moe_xffn_common.hpp roles at PMK_MOE_CB0 + k
constexpr uint32_t CB_CTL = PMK_MOE_CBX_CTL;    // reader -> compute: [mt rows of the group] (uint32), 0 = no more work
constexpr uint32_t CB_WGU = PMK_MOE_CBX_WGU;    // gate|up weight chunk: KC x 2W tiles (bfp4), 2 slots
constexpr uint32_t CB_XRM = PMK_MOE_CBX_XRM;    // x rows of one tile row, one K chunk: 32 rows x KC tiles ROW_MAJOR
constexpr uint32_t CB_XT = PMK_MOE_CBX_XT;      // tilized x chunk
constexpr uint32_t CB_WD = PMK_MOE_CBX_WD;      // down weight chunk: W x NC tiles (bfp4), 2 slots
constexpr uint32_t CB_WCTL = PMK_MOE_CBX_WCTL;  // reader -> writer: [mt, valid rows, first dispatch row]
constexpr uint32_t CB_TAB = PMK_MOE_CBX_TAB;  // reader scratch: counts, region starts, weight address table, disp rows
constexpr uint32_t CB_WTAB = PMK_MOE_CBX_WTAB;  // writer scratch: the group's disp rows
constexpr uint32_t CB_GUP = PMK_MOE_CBX_GUP;    // fp32 gate|up partials: MG x 2W tiles (packer L1 accumulation)
constexpr uint32_t CB_ACT = PMK_MOE_CBX_ACT;    // activation tiles A: MG x W (bf16)
constexpr uint32_t CB_YRM = PMK_MOE_CBX_YRM;    // y rows of one tile row, one N chunk: 32 rows x NC tiles ROW_MAJOR
// ---- plan scratch (BRISC only; the 3 plan steps run before the FFN: it ALIASES CB_WGU's bytes on the host side)
constexpr uint32_t CB_PLAN = PMK_MOE_CBX_PLAN;
// ---- combine (rm) CBs, after the FFN (the host aliases them onto the FFN rings: both phases never overlap in time)
constexpr uint32_t CB_CIN = PMK_MOE_CBX_CIN;    // the K gathered row parts of one token (bf16, K x PT pages)
constexpr uint32_t CB_CW = PMK_MOE_CBX_CW;      // the K weights (fp32 tiles, datum 0)
constexpr uint32_t CB_CIDX = PMK_MOE_CBX_CIDX;  // reader scratch: dest words + the weight tile
constexpr uint32_t CB_COUT = PMK_MOE_CBX_COUT;  // the chunk's 32 PT output tiles (bf16) -- index < 32 (tilize helper)
constexpr uint32_t CB_CRM = PMK_MOE_CBX_CRM;    // fp32 L1-accumulated rows (32 PT tiles) -- index < 32
constexpr uint32_t CB_CSH = PMK_MOE_CBX_CSH;    // v2 PMK_SHARED on the row-bfp8 path: the chunk's y_s tiles (bf16)
// perf5 PMK_GU_SPLIT (moe2/PERF.md sec 12): this member's activation COLUMN SLICE of the stream's R row tiles (bf16
// tiles (r, j), j over the slice's pairs), sent by the NCRISC to the row owners' CB_ACT; ONE block (the base never
// moves)
constexpr uint32_t CB_ACTS = PMK_MOE_CBX_ACTS;
// perf5 round 3 PMK_COMB_SPLIT (moe2/PERF.md sec 15): the combine's odd slots are rebuilt by the BRISC into CIN2 / CW2
// (the NCRISC keeps the even slots, the y_s term and -- under the split -- the y write); the compute alternates
constexpr uint32_t CB_CIN2 = PMK_MOE_CBX_CIN2;
constexpr uint32_t CB_CW2 = PMK_MOE_CBX_CW2;
// W7 step 5 (G, tt/pmk_gang.py): the gang's in-gang y ring -- the predecessor's partial blocks (bf16 tiles, 2 MG slots
// of NC tiles). Contract (w8/w7_gang_interface.md (a)): the SAME L1 address on every core of a gang (the predecessor
// writes at its own get_write_ptr(CB_RIN) offset) -- the host lays it first in every role's window
constexpr uint32_t CB_RIN = PMK_MOE_CBX_RIN;
// the RM_SKIP combine (comb_reader.hpp PMK_RM_SKIP): one zeroed part, the slot source of every weight-0 pair
constexpr uint32_t CB_CZERO = PMK_MOE_CBX_CZERO;
constexpr uint32_t N_CBS = 21;  // from PMK_MOE_CB0, + 2 from PMK_MOE_CB1
// ---- v2 (tt/pmk_moe2.py; pmk/moe2/README.md): the x-hat_f store read over the NoC (lever 2, XFFN_X_TILED): the
// family's XS 35 (H2 / H8: block G = rows [32 G, 32 G + 32) x H bf16 TILE, tile k at base + 2048 k, the same L1
// address on every holder); standalone (the mock / bench) the NCRISC of holder G fills it from x [C, H] TILE first
#ifndef PMK_MOE_CB_XS
#define PMK_MOE_CB_XS 35
#endif
constexpr uint32_t CB_XS = PMK_MOE_CB_XS;
// v2 lever 3 (PMK_MCAST): the units of an expert with >= 2 M groups run on the G cores of ONE rectangle; the
// rectangle's first core streams the slice from DRAM once and multicasts every chunk into the others' ring slot
// (identical CB addresses and ring positions: every unit pushes an even number of chunks on the 2-slot rings), 7a's
// PfStream credit protocol (kernels/pfs): receivers credit a slot AHEAD (READY word on the streamer, +1 per receiver
// per block, after cb_reserve_back), the streamer waits READY >= (G - 1) x blocks, then data + a VALID semaphore SET
// on the same NoC (ordered behind the data). READY / VAL words: the first 64 B of CB_TAB (zeroed by the BRISC before
// the plan barriers; receivers credit only after barrier 3). VALID: program semaphores 10 / 11 (sec 4b-ids: the
// weight multicast's ids), one per ring (gate|up, down), monotonic within the launch.
constexpr uint32_t SYNC_WORDS_B = 128;  // CB_TAB prefix: READY_GU, READY_D, VAL_GU, VAL_D (16-B stride), then the
constexpr uint32_t W_READY_GU = 0, W_READY_D = 16, W_VAL_GU = 32, W_VAL_D = 48;  // push probe words (below)
// the PUSH PROBE (Stage 16, 13:44:48Z watcher assert `fifo_wr_ptr <= fifo_limit` in a cb_push_back whose CB the dump
// did not name): before every block push a RISC pins (cb index << 16 | pages) at CB_TAB + W_PUSH_<risc>, and the
// first word of the L1 PROBE CB 31 when the family allocates it (PMK_PUSH_PROBE_L1 = that address, else only TAB)
constexpr uint32_t W_PUSH_BRISC = 64, W_PUSH_NCRISC = 80, W_PUSH_TAB_BYTES = 96;
constexpr uint32_t CTL_SKIP = 0x100;  // CB_CTL / CB_WCTL word 0: mt | CTL_SKIP = receive the stream, no M group
// perf5 PMK_GU_SPLIT: a stream = (expert, R consecutive row tiles); EVERY member of the rectangle computes gate|up for
// the SLICE of columns [c0, c0 + cw) of all R rows (CTL words 1 / 2), the activation of its pairs goes to the row
// owners (member pos r owns row r of the stream: CB_ACT filled over the NoC by all g members, SEM_ACT counts the
// senders), and the down phase runs on the owners only. CTL word 0 = R | CTL_NOROW (this member owns no row of the
// stream).
constexpr uint32_t CTL_NOROW = 0x200;
constexpr uint32_t SEM_ACT =
    14;  // PMK_GU_SPLIT: +1 per sender per stream on a row owner (cumulative: g per owned stream)
// the column slice of member `pos` of g: the GU_T / SUBA subblock units dealt evenly (units [pos U / g, (pos + 1) U /
// g))
FORCE_INLINE constexpr uint32_t split_c0(uint32_t pos, uint32_t g, uint32_t units, uint32_t suba) {
    return (pos * units / g) * suba;
}
FORCE_INLINE constexpr uint32_t split_cw(uint32_t pos, uint32_t g, uint32_t units, uint32_t suba) {
    return ((pos + 1) * units / g - pos * units / g) * suba;
}
// rows per stream for a rectangle of g: min(g, RMAX, GUP tiles / the widest slice) -- the host mirrors (split_rows)
FORCE_INLINE constexpr uint32_t split_rows(uint32_t g, uint32_t units, uint32_t suba, uint32_t gupt, uint32_t rmax) {
    const uint32_t cwmax = ((units + g - 1) / g) * suba;
    uint32_t r = g < rmax ? g : rmax;
    const uint32_t cap = gupt / cwmax;
    return r < cap ? r : cap;
}
// v2 PMK_OUT_BFP8 (pmk/moe2/README.md sec 9): the expert output rows go to out_rm as "row-bfp8" instead of bf16: the
// down phase packs bfp8 TILES (shared exponent per 16-element face row), the writer re-lays every row of a tile row
// per N chunk as [NC exponent PAIRS (2 B per tile: faces (t/16)*2, +1), padded to 64 B][NC x 32 B mantissas (16 B per
// face)] = 64 + 32 NC bytes (320 B at NC = 8; 64-B multiples: the DRAM rule), so a row = (H_T / NC) x that (5,120 B at
// NC = 8, 62.5 % of bf16). The combine reader gathers the 32 tokens' row parts of a slot and REBUILDS bfp8 tiles in L1
// (token t -> tile row t), and the compute multiplies each tile by the tokens' weights with a COL broadcast.
constexpr uint32_t BFP8_TILE_B = 1088;  // 64 exponent bytes + 1,024 mantissa bytes
constexpr uint32_t BFP8_EXP_B = 64;
constexpr uint32_t ROWB_EXP_PAD = 64;  // the exponent pairs of one N chunk, padded
FORCE_INLINE constexpr uint32_t rowb_chunk_bytes(uint32_t nc) { return ROWB_EXP_PAD + 32 * nc; }

// ---- semaphores (prefill_megakernel.md sec 4b-ids: the experts reuse 5..9 after the attention half)
constexpr uint32_t SEM_PLAN1 = 5;      // plan step 1 (counts) landed
constexpr uint32_t SEM_PLAN2 = 6;      // plan step 2 (dest / starts / staging) landed
constexpr uint32_t SEM_PLAN3 = 7;      // plan step 3 (disp) landed: the FFN may start
constexpr uint32_t SEM_FFN = 8;        // every core's FFN output rows landed: the combine may start
constexpr uint32_t SEM_M = 9;          // H9 (when the consumer is a program semaphore instead of counter words)
constexpr uint32_t SEM_XS = 9;         // v2 standalone: every holder's XS block landed (the family: FS_ARRIVE 15 >= 16)
constexpr uint32_t SEM_VALID_GU = 10;  // v2 PMK_MCAST: gate|up chunks multicast so far (receivers wait_min)
constexpr uint32_t SEM_VALID_D = 11;   // v2 PMK_MCAST: down chunks multicast so far
// W7 step 5 (G): the gang ring's counters (ids >= 10, off the fabric connection words; G runs no multicast, so 10 / 11
// stay B's; 14 is the A -> B / G rendezvous): +1 per block landed in CB_RIN / per CB_RIN slot freed, growing (>= waits)
constexpr uint32_t SEM_RING_VALID = 12;
constexpr uint32_t SEM_RING_CREDIT = 13;

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t BF16_TILE = 2048;

}  // namespace pmk_moe

// GLM53_PMK_PHASE_PROBE (tt/pmk_layer.PHASE_PROBE): a body's own wall-clock mark; the fused composite defines it before
// the bodies when the probe is on (its part declares the labels: pmk_b_window.register_b's probe_marks), else nothing
#ifndef PMK_PP_MARK
#define PMK_PP_MARK() \
    do {              \
    } while (0)
#endif

#if !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/dataflow_api.h"

namespace pmk_moe {
// program-wide barrier k (k = 1, 2, ...) on semaphore `sem`: +1 on every MM core (NoC coordinates at runtime args
// [arg0, arg0 + 2 P)), then wait until this core's copy is >= P (wait_min: a late core's increments for the NEXT use
// of the same id would only raise it further -- every id is used once per launch here)
FORCE_INLINE void barrier_all(uint32_t sem, uint32_t arg0, uint32_t p) {
    for (uint32_t r = 0; r < p; ++r) {
        const uint32_t x = get_arg_val<uint32_t>(arg0 + 2 * r), y = get_arg_val<uint32_t>(arg0 + 2 * r + 1);
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem)), 1);
    }
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
    noc_semaphore_wait_min(s, p);
}
// the push probe: pin (cb << 16 | pages) before a block push (see W_PUSH_*); `tab` = get_write_ptr(CB_TAB)
FORCE_INLINE void push_mark(uint32_t tab, uint32_t word, uint32_t cb, uint32_t pages) {
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab + word) = (cb << 16) | (pages & 0xFFFFu);
#ifdef PMK_PUSH_PROBE_L1
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(PMK_PUSH_PROBE_L1) = (cb << 16) | (pages & 0xFFFFu);
#endif
}
// wait_min on a hand-off semaphore (no reset: the producer side owns the epoch)
FORCE_INLINE void wait_min(uint32_t sem, uint32_t count) {
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
    noc_semaphore_wait_min(s, count);
}

// ---- PMK_STREAM_LPT (program B, moe2/PERF.md sec 11): the expert -> streamer assignment from the LIVE counts,
// computed identically on every streamer and every receiver right after the plan barrier (the routing is a device
// result, so no host table can carry it; the static map e -> (e mod 8) + 8 ((e / 8) mod 3) left the phase span at the
// SLOWEST streamer, 16.0 ms vs a 11.6 ms mean on the perf4 cover cell). Longest-processing-time first: experts in
// descending ng (ties: ascending e), each to the admissible streamer with the least load + cost, cost = ceil(ng / g_s)
// streams (ties: the lowest s); admissible = the 3 streamers of the expert's bank (mode 1: s & 7 == e & 7, 7a's
// adjacency kept, every read stays bank-local) or every streamer (mode 2: a streamer reads any bank). Pure integer
// arithmetic in a fixed order: every core gets the same table. Scratch (64-B aligned): gtab[32] u32 | ng[EP] u16 |
// order[EP] u16 | hist[LPT_NG_MAX + 2] u16 | load[32] u16 | owner[EP] u8 (0xFF = unserved: cnt 0).
constexpr uint32_t LPT_NG_MAX = 1024;  // ng = ceil(ceil(cnt / 32) / MG) <= rows / 32: 8,192 rows -> 256; room for 32K
FORCE_INLINE constexpr uint32_t lpt_scratch_bytes(uint32_t ep) {
    return 256 + 2 * ep + 2 * ep + 2 * (LPT_NG_MAX + 2) + 128 + ep;
}
// perf5 round 3 (PERF.md sec 15): w_arg0 = per-streamer cost WEIGHTS (uint16 fixed point, 256 = 1.0: the measured
// us per chunk of each rectangle from the previous row, the same bytes run 1.4x slower on some boxes) -- the cost of
// an expert on streamer s is ceil(ng / g_s) x w_s and the loads are in weighted units; w_arg0 = 0: every weight 256
FORCE_INLINE volatile tt_l1_ptr uint8_t* stream_lpt(
    volatile tt_l1_ptr uint32_t* cnt,
    uint32_t e_n,
    uint32_t ep,
    uint32_t mg,
    uint32_t ns,
    uint32_t g_arg0,
    uint32_t mode,
    uint32_t scratch,
    uint32_t w_arg0 = 0) {
    volatile tt_l1_ptr uint32_t* gtab =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);  // [32] g | [32] w (uint16 as u32)
    volatile tt_l1_ptr uint16_t* ng = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + 256);
    volatile tt_l1_ptr uint16_t* order = ng + ep;
    volatile tt_l1_ptr uint16_t* hist = order + ep;
    volatile tt_l1_ptr uint32_t* load =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hist + LPT_NG_MAX + 2);                     // weighted: u32
    volatile tt_l1_ptr uint8_t* owner = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(load + 32);  // (128 B of loads)
    for (uint32_t s = 0; s < ns; ++s) {
        gtab[s] = get_arg_val<uint32_t>(g_arg0 + s);
        gtab[32 + s] = w_arg0 ? get_arg_val<uint32_t>(w_arg0 + s) : 256u;
        load[s] = 0;
    }
    for (uint32_t v = 0; v <= LPT_NG_MAX + 1; ++v) {
        hist[v] = 0;
    }
    uint32_t vmax = 0;
    for (uint32_t e = 0; e < e_n; ++e) {
        const uint32_t m = cnt[e];
        uint32_t v = m ? (((m + 31) >> 5) + mg - 1) / mg : 0;
        v = v > LPT_NG_MAX ? LPT_NG_MAX : v;
        ng[e] = v;
        owner[e] = 0xFF;
        if (v) {
            hist[v] = hist[v] + 1;
            vmax = v > vmax ? v : vmax;
        }
    }
    uint32_t at = 0;  // counting sort, descending ng, ascending e within a bucket
    for (uint32_t v = vmax; v >= 1; --v) {
        const uint32_t t = hist[v];
        hist[v] = at;
        at += t;
    }
    for (uint32_t e = 0; e < e_n; ++e) {
        const uint32_t v = ng[e];
        if (v) {
            order[hist[v]] = e;
            hist[v] = hist[v] + 1;
        }
    }
    for (uint32_t k = 0; k < at; ++k) {
        const uint32_t e = order[k], v = ng[e];
        uint32_t best = 0xFFFFFFFFu, bs = 0;
        for (uint32_t s = 0; s < ns; ++s) {
            if (mode == 1 && (s & 7u) != (e & 7u)) {
                continue;
            }
            const uint32_t gs = gtab[s];
            const uint32_t c = load[s] + ((v + gs - 1) / gs) * gtab[32 + s];
            if (c < best) {
                best = c;
                bs = s;
            }
        }
        owner[e] = bs;
        load[bs] = best;
    }
    return owner;
}
}  // namespace pmk_moe
#endif
