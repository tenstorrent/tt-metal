// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader: one core = (user u, value head h). Reads the ctrl page (parity, mi[u], initial ring block of (u,h); HOLD
// sentinel -> block bh, mi 0), the constants, the head's 12 whole window tiles (win[par], user u) + 12 whole raw
// projection tiles + the 12 tap tiles (rows 0..K-1 only), z, the a|b tile(s), the norm weight row, builds the bf16
// one-hot selector tiles, then reads the 64 KiB initial state last (compute's conv/pre overlap it). No per-token reads.
// a|b: a[h] is column h and b[h] column Nv + h past tile ab_page. With 2*Nv <= 32 both sit in tile ab_page (one read,
// ab_in[0]); with Nv = 24 (TP = 2) the a|b block spans two tiles and b's tile is read into ab_in[1] when it differs
// from a's (b_idx = 1, decided by the host per core), so the compute gathers b's row from ab_in[b_idx].
// GDN_FAST order: ctrl, constants, z + a|b, bulk reads + selectors (pushed), then dt_bias / nea, the state read
// (issued), the software gate math and beta_t / dec fills, state push (compute needs dec / beta_t only inside the token
// loop). GDN_FAST (fast_mode): the a|b tile(s) are staging only. The reader extracts a[u*T+t, h] / b[u*T+t, h] for the
// T tokens, computes beta = sigmoid(b) and decay = exp(-exp(A) * softplus(a + dt_bias)) in software fp32 (the very
// functions of reader_gdn_decode_step_conv_fast.cpp, gdn_step_gate_math.hpp) and pushes them as T all-equal fp32
// scalar tiles beta_t[t] / dec[t]; the gate selector tiles (csel, rsel) and the SFPU gate path do not exist.
// VB > 1 (value blocks): core (u, h, s) reads only state column tiles [s*Vs, (s+1)*Vs) of each of the Kt tile rows of
// the initial block (KVs = Kt*Vs tiles, row-major (i, j) at i*Vs + j); every other read is the full head's, as the
// conv / l2norm / gate prefix runs in full on every block.
// GDN_GATE_SCALARS (kernel_opts bit 1, fast_mode + fp32_state): the gate math moves to the writer RISC and the compute
// builds the scalar tiles in DST (fill_tile_bitcast of the same fp32 bits), so this reader skips the a|b staging read,
// the dt_bias / -exp(A) reads, the software gate math and the 2*T full-tile fills (1024 RISC-V stores each), and pushes
// the initial state as soon as its read lands. GDN_EARLY_SRC (kernel_opts bit 2): src_in and the selector tiles are
// pushed right after their reads / build, before the taps, the norm weight row and the row masks (which the compute
// needs only from phase A3 on), so phase A starts earlier. Both move no value: every tile holds the same bytes.
// GDN_LAZY (kernel_opts K4 lazy): the initial state is D, ring slot 0 block (u, h), whatever the ctrl page's block word
// says (that word only carries HOLD), plus the previous verify's stash from ring slot 1 + par: its Kt K^T tiles
// (ktp_in) and this block's Vs U tiles (up_in), read with the state. A ctrl mi >= T is FRESH (the slot-0 state was just
// seeded: the compute replays nothing) and rebuilds the conv window at mi 0. No e_t row masks (no token loop).
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "api/debug/kernel_profiler.hpp"  // zones compile to nothing unless the build defines PROFILE_KERNEL

#ifndef GDN_SP_RD
#define GDN_SP_RD 0  // engine GDN role (BATCH + SP_LATE): S' is written by this reader (NOC1) after the kmq stream, not by
                     // the writer (whose queue then reaches the next layer's wr_rcoef without the S' writes in front)
#endif
#if GDN_SP_RD
static uint32_t g_rd_sp_lay = 0;  // GDN layers done by this reader (sp ring offset at B < 8)
#endif
#ifndef GDN_OUT_RD
#define GDN_OUT_RD 0  // engine GDN role (BATCH, nb * T == 32): this reader (NOC1) writes the gated out tiles and raises the
                      // out signal, not the writer: at +36 us the writer's NOC0 at physical x 1-4 is saturated (weight
                      // stream), the 2 KB out write took 20-55 us there (v1.91); the reader is idle from the kmq stream's
                      // end to the next layer's rd_const (blocked on the compute's scaler pop anyway)
#endif
#if GDN_OUT_RD
void gdn_out_signal();  // the including TU (out_ready +1 / the aggregator, issued on this kernel's NOC)
#endif
#ifndef GDN_OUT_RD_KEEP
#define GDN_OUT_RD_KEEP 0
#endif
#if GDN_OUT_RD == 2
void gdn_out_local(uint32_t src_l1, uint32_t t0, uint32_t n);  // engine gdn_out: the out tiles into every GDN core's A (TU)
#endif
#ifndef GDN_OUT_PRE
#define GDN_OUT_PRE 0  // engine gdn_out (E1_GO_RING0=hook): gdn_out_pre() (TU) runs right before rd_owait (L0 ring read)
#endif
#if GDN_OUT_PRE
void gdn_out_pre();
#endif
#if defined(GSTEP_NO_ATTHOLD) && defined(GSTEP_ATTHOLD)
#undef GSTEP_ATTHOLD  // AR-sender corners / the Sigma reducer (engine program.py): never hold (their post-body work feeds the attention layer)
#endif
#ifndef GSTEP_ATTHOLD
#define GSTEP_ATTHOLD 0  // engine GDN role (E1_ATTHOLD): gdn_att_hold() (the including TU) before rd_sp
#endif
#if GSTEP_ATTHOLD
void gdn_att_hold();
#endif
#ifndef GDN_PRE_SP_HOOK
#define GDN_PRE_SP_HOOK 0  // engine AR-sender corner cores (MK_CGAR_EARLY): gdn_pre_sp_hook() (TU) after the out / gdn_out tail,
                           // before the S' write-back -- the hosted GDN-AR sender iteration runs there instead of after the body
#endif
#if GDN_PRE_SP_HOOK
void gdn_pre_sp_hook();
#endif
#ifndef GDN_HOLD_SKIP
#define GDN_HOLD_SKIP 0  // engine GDN role (BATCH, ST_RING 0; RC v1.89): HOLD (idle) users get no state / stash reads and no
                         // masked K_u / Q_u copies; the compute skips their replay, K S' / Q S' terms and K^T transposes
                         // (block-diagonal per user: live rows bit-identical; HOLD rows finite, their ring untouched)
#endif
#if GDN_HOLD_SKIP
void gdn_hold_publish(uint32_t live);  // the including TU: this layer's live-user mask -> the compute (L1 word + seq)
#endif
#ifndef GDN_STNARROW
#define GDN_STNARROW 0  // engine GDN role (BATCH, ST_RING 0; E1_STNARROW): the previous verify's per-user stash pages are read
                        // narrow: a user's K^T page has only its T token columns live (faces {0,2} or {1,3} read, 2 x 1 KB) and
                        // its U page only its T rows (2 x T*64 B); the rest of each slot is +0 like the page itself (the
                        // writer's masked copies). Zeroed once per slot base, never overwritten elsewhere -> same L1 bytes as
                        // the full-page read (layout of the ring unchanged). 160 -> 68 KB of DRAM reads per core at u8
#endif
#ifndef GDN_KTDD
#define GDN_KTDD 0  // engine GDN role (STNARROW, VB > 1; E1_KTDD): the 4 value-block cores of a head need the SAME K^T stash tiles;
                    // block s reads only tiles i = s mod VB (narrow) and writes them into its VB - 1 peers' ktp_in slots
                    // (same L1 address on every GDN core). Two argblock counters per core (the including TU): [0] = peers
                    // whose ktp_in slot is reserved + zeroed for this layer, [1] = peers whose K^T share has landed here;
                    // cumulative (VB - 1) per layer, reset by the TU at program end (every increment awaited in-program)
#endif
#if GDN_KTDD
uint32_t gdn_ktdd_peer(uint32_t p);  // the including TU: NoC (x << 16 | y) of value block p of this core's head
volatile tt_l1_ptr uint32_t* gdn_ktdd_sem();  // the including TU: the 2 counter words (same address on every GDN core)
static uint32_t g_ktdd_l = 0;  // GDN layers done by this reader (reset with gdn_stn_reset)
// +1 on counter word `w` (0 / 1) of every peer except block s (one call site: code size)
__attribute__((noinline)) void gdn_ktdd_bump(uint32_t s, uint32_t vb, uint32_t w) {
    for (uint32_t p = 0; p < vb; ++p) {
        if (p != s) {
            const uint32_t xy = gdn_ktdd_peer(p);
            noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, (uint32_t)(gdn_ktdd_sem() + w)), 1);
        }
    }
}
// the 1 KB at L1 `a` -> the same address of every peer except block s
__attribute__((noinline)) void gdn_ktdd_fwd(uint32_t s, uint32_t vb, uint32_t a) {
    for (uint32_t p = 0; p < vb; ++p) {
        if (p != s) {
            const uint32_t xy = gdn_ktdd_peer(p);
            noc_async_write(a, get_noc_addr(xy >> 16, xy & 0xFFFFu, a), 1024);
        }
    }
}
#endif
#if GDN_STNARROW
static uint32_t g_stn_kb = 0;  // ktp_in / up_in write pointers whose slots hold zeros outside the live parts
static uint32_t g_stn_ub = 0;
inline void gdn_stn_reset() {
    g_stn_kb = 0;
    g_stn_ub = 0;
#if GDN_KTDD
    g_ktdd_l = 0;
#endif
}  // once per program (the including TU's kernel_main)
#endif
#ifndef GDN_RPX
#define GDN_RPX 0  // REPLAY bench-only stash-read timing diagnostics (E1_RPX; never in production)
#endif
#ifndef GDN_CUT_FAST
#define GDN_CUT_FAST 0  // engine GDN role (BATCH): masked K_u / Q_u slots (kts, kmq) keep their zeros across uses: only the
                        // previous occupant's T rows are re-zeroed (no full-tile zero fill + barrier per tile)
#endif
#if GDN_CUT_FAST
static uint32_t g_cut_addr[32];
static uint32_t g_cut_r0[32];
static uint32_t g_cut_n = 0;
// once per program (the including TU's kernel_main): every slot's contents are unknown
inline void gdn_cut_reset() { g_cut_n = 0; }
#endif
#ifndef GDN_NO_EPS_L2
#define GDN_NO_EPS_L2 0  // engine GDN role (PREFIXED): the l2 eps tile is neither generated nor waited (its CB id is freed)
#endif
#ifndef GDN_FAST
#define GDN_FAST 0
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0
#endif
#ifndef GDN_EARLY_SRC
#define GDN_EARLY_SRC 0
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0  // kernel_opts bit 4 (VB > 1): this block reads only its q/k share + its v chunks
#endif
#ifndef GDN_LAZY
#define GDN_LAZY 0  // kernel_opts bit 8 (K4): D from ring slot 0 + the previous verify's stash
#endif
#ifndef GDN_RING_SPLIT
#define GDN_RING_SPLIT 0  // kernel_opts bit 18: this reader writes the odd tokens' post-token states to the ring (NoC 1)
#endif
#ifndef GDN_RT_BM
#define GDN_RT_BM 0  // engine GDN role: B / M from gdn_rt_B() / gdn_rt_M() (defined by the including TU), one binary for
                     // type 11 (B = 1) and type 4 (B = 8); 0 = the template values (the op)
#endif
#ifndef GDN_BATCH
#define GDN_BATCH 0  // batched type 4: all users of head h on core (h, s) (see the compute)
#endif
#if GDN_BATCH
constexpr uint32_t kBatchMax = 8;
#endif
#ifndef GDN_STATE_FIRST
#define GDN_STATE_FIRST 0  // engine GDN role (BATCH): the state / stash reads go out before rd_pre (ring + ctrl only)
#endif
#ifndef GDN_ST_RING
#define GDN_ST_RING 0  // BATCH: state / stash DFBs hold GDN_ST_RING users (streamed per user; 0 = all kBatchMax at once)
#endif
#ifndef GSTEP_WYPAIR
#define GSTEP_WYPAIR 0  // engine GDN role (BATCH, T*kBatchMax == 32): kmq tiles carry K_u AND Q_u (Q rows in the other face row)
#endif
#ifndef GSTEP_WYORD
#define GSTEP_WYORD 0  // WYPAIR: user order 0 1 4 5 2 3 6 7 in the kmq stream / c_wy (compute must match)
#endif
#ifndef GSTEP_KTSLATE
#define GSTEP_KTSLATE 0  // WYPAIR + KTLATE: the kts (stash K^T) cuts after the kmq stream instead of before it
#endif
#if GSTEP_KTSLATE && !GSTEP_WYPAIR
#error "GSTEP_KTSLATE needs GSTEP_WYPAIR"
#endif
#ifndef GSTEP_CZ
#define GSTEP_CZ 0  // bench: reader zone rd_kmq
#endif
#ifndef GSTEP_CUTST
#define GSTEP_CUTST 0  // WYPAIR: the kmq cut reads through one stateful local read command (noc_async_read_one_packet_with_state)
#endif
#ifndef GSTEP_CUTNZ
#define GSTEP_CUTNZ 0  // BENCH ONLY (wrong numerics): WYPAIR cuts skip the previous occupant's zeroing (reader-bound probe)
#endif
#ifndef GSTEP_WPRE
#define GSTEP_WPRE 0  // engine GDN role (GDN_STATE_FIRST + GDN_PREFIXED): w_in (norm weight rows) read before gdn_in_wait
#endif
#if defined(GB_TS) && GB_TS  // E1_GB_TS bench probe: wall-clock stamps into the arg block (gbts slots, 32 per GDN layer)
#define GBTS(n) (g_ab[GB_TS_OFF + g_l * 32u + (n)] = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#else
#define GBTS(n) ((void)0)
#endif
#ifndef GDN_IN_WAIT
#define GDN_IN_WAIT 0  // engine GDN role: gdn_in_wait() (the including TU) gates the reads of `pre`
#endif
#if GDN_IN_WAIT
void gdn_in_wait();
#endif
#ifndef GDN_GB_DIRECT
#define GDN_GB_DIRECT 0  // G-b: the prefix writes zs / kc / qc / vc straight into this core's DFB slots (no `pre` reads)
#endif
#if GDN_GB_DIRECT
void gdn_gb_wait(uint32_t which);  // G-b (the including TU): 1 = vc / zs / gate rows landed
void gdn_gb_knq(uint32_t knq_l1);  // G-b: the normalized k / q into knq (pulled from the L2 roots)
#endif
#ifndef GDN_RAWKQ
#define GDN_RAWKQ 0  // G-b (E1_GB): `pre` holds raw conv'd k / q; the compute's l2norm (L2-root fallback)
#endif
#ifndef GDN_PREFIXED
#define GDN_PREFIXED 0  // kernel_opts bit 21 (K4): knq / vc / zs from the u1 prefix's output tensor
#endif
#ifndef GDN_NARROW_SRC
#define GDN_NARROW_SRC 0  // kernel_opts bit 13: row-narrow src / z reads into zero-filled slots (bit-identical)
#endif
#if GDN_FAST && !GDN_GATE_SCALARS
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_step/device/kernels/dataflow/gdn_step_gate_math.hpp"
#endif

using namespace gdn_step_df;
#if GDN_BATCH
namespace {
// zero `n` tiles of a reserved DFB from tile `t0` (FRESH / HOLD users' stash slots only)
inline void zero_tiles_at(DataflowBuffer& dfb, Noc&, uint32_t t0, uint32_t n) {
    // NoC reads of the core's zero region (covered by the caller's read barrier)
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t base = dfb.get_write_ptr() + t0 * entry;
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < n * entry; o += MEM_ZEROS_SIZE) {
        noc_async_read(z, base + o, MEM_ZEROS_SIZE);
    }
}
}  // namespace
#endif

namespace {
template <typename P>
FORCE_INLINE void set_one(P& p, uint32_t tile, uint32_t row, uint32_t col) {
    p[tile * 1024 + tile_elem_index(row, col)] = 0x3F80;  // bf16 1.0 (p: the lock's CoreLocalMem<volatile uint16_t>)
}
#if GDN_NARROW_SRC
// NARROW_SRC: rows [r0, r0 + n) of bf16 tile `page` into the same rows of slot `dst_tile` (already zero-filled): per
// face-row group (rows 0..15 = faces 0 / 1, rows 16..31 = faces 2 / 3) one contiguous read per face of the rows that
// fall in it (a face stores its 16 rows of 16 elements row-major, 32 B per row). Source / destination offsets are equal.
template <typename Accessor>
inline void read_rows_bf16(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t dst_tile, uint32_t r0, uint32_t n) {
    const uint32_t base = dst_tile * dfb.get_entry_size();
    for (uint32_t fr = 0; fr < 2; ++fr) {
        const uint32_t lo = r0 > fr * 16 ? r0 : fr * 16;
        const uint32_t hi = (r0 + n) < (fr * 16 + 16) ? (r0 + n) : (fr * 16 + 16);
        if (lo >= hi) {
            continue;
        }
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t off = ((2 * fr + half) * 256u + (lo - fr * 16) * 16u) * 2u;
            noc.async_read(acc, dfb, (hi - lo) * 32u, {.page_id = page, .offset_bytes = off}, {.offset_bytes = base + off});
        }
    }
}
#endif
}  // namespace

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Nk,
    uint32_t Nv,
    uint32_t T,
    uint32_t B,
    uint32_t M,  // ring/ctrl layout width (>= B)
    uint32_t K,
    uint32_t Lw,
    uint32_t Ct,
    uint32_t z_tile0,
    uint32_t ab_page,
    uint32_t w_tiles,
    uint32_t dtb_fp32,
    uint32_t nea_fp32,
    uint32_t l2_eps_bits,
    uint32_t norm_eps_bits,
    uint32_t ctrl_bytes,
    uint32_t hold_sentinel,
    uint32_t AB2,
    uint32_t VB>
TT_KERNEL void reader(uint32_t u, uint32_t h, uint32_t b_idx, uint32_t s) {
#if GDN_RT_BM
    const uint32_t Mr = gdn_rt_M();
#else
    constexpr uint32_t Mr = M;
#endif
    const auto qkv_acc = TensorAccessor(tensor::qkv);
    const auto wa_acc = TensorAccessor(tensor::win_a);
    const auto wb_acc = TensorAccessor(tensor::win_b);
    const auto ring_acc = TensorAccessor(tensor::ring);
    const auto ctrl_acc = TensorAccessor(tensor::ctrl);
    const auto taps_acc = TensorAccessor(tensor::taps);
    const auto dtb_acc = TensorAccessor(tensor::dtb);
    const auto nea_acc = TensorAccessor(tensor::nea);
    const auto w_acc = TensorAccessor(tensor::weight);
    DataflowBuffer src_in(dfb::src_in);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer z_in(dfb::z_in);
#if !GDN_GATE_SCALARS
    DataflowBuffer ab_in(dfb::ab_in);
#endif
    DataflowBuffer sel(dfb::sel);
    DataflowBuffer mask_T(dfb::mask_T);
#if GDN_LAZY
    DataflowBuffer ktp_in(dfb::ktp_in);  // the previous verify's K^T tiles (stash pages 0..Kt-1)
    DataflowBuffer up_in(dfb::up_in);    // ... and this block's U tiles (stash pages Kt + s*Vs ..)
#else
    DataflowBuffer e_t(dfb::e_t);
#endif
#if GDN_GATE_SCALARS
    // the writer computes the gates; the compute builds beta_t / dec in DST
    (void)b_idx;
    (void)dtb_acc;
    (void)nea_acc;
#elif GDN_FAST
    DataflowBuffer beta_t(dfb::beta_t);
    DataflowBuffer dec(dfb::dec);
#else
    DataflowBuffer rsel(dfb::rsel);
    DataflowBuffer csel(dfb::csel);
#endif
    DataflowBuffer w_in(dfb::w_in);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer eps_l2(dfb::eps_l2);
    DataflowBuffer eps_norm(dfb::eps_norm);
#if !GDN_GATE_SCALARS
    DataflowBuffer dtb_s(dfb::dtb_s);
    DataflowBuffer nea_s(dfb::nea_s);
#endif
    DataflowBuffer state_in(dfb::state_in);
    DataflowBuffer ctrl_r(dfb::ctrl_r);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ch = 2 * Kt + Vt;
    static_assert(VB >= 1 && Vt % VB == 0, "value blocks must divide Vt");
    constexpr uint32_t Vs = Vt / VB;  // this block's state column tiles
    constexpr uint32_t KVs = Kt * Vs;
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // the user's rows live in qkv tile row tr ...
    const uint32_t r0 = (u * T) % 32;  // ... at rows r0 .. r0+T-1 of that tile row
    const uint32_t pg = tr * w_tiles;  // page of tile column 0 of that tile row
    // QKSHARE: this block's ChA chunks, in block order: q/k chunks [s*Qk, (s+1)*Qk) (Qk = 2*Kt / VB), then its own v
    // chunks [2*Kt + s*Vs, ..); otherwise ChA = Ch and the block order is the head order
    constexpr uint32_t Qk = GDN_QKSHARE ? (2 * Kt) / VB : 0u;
    constexpr uint32_t ChA = GDN_QKSHARE ? Qk + Vs : Ch;
    auto blk_chunk = [&](uint32_t ci) -> uint32_t {
        if constexpr (GDN_QKSHARE) {
            return ci < Qk ? s * Qk + ci : 2 * Kt + s * Vs + (ci - Qk);
        } else {
            return ci;
        }
    };
    // tile column of this head's chunk c within a [.., C]-wide row: q(hk) | k(hk) | v(h)
    auto chunk_col = [&](uint32_t c) -> uint32_t {
        if (c < Kt) {
            return hk * Kt + c;
        }
        if (c < 2 * Kt) {
            return Nk * Kt + hk * Kt + (c - Kt);
        }
        return 2 * Nk * Kt + h * Vt + (c - 2 * Kt);
    };

    // ---- ctrl page: parity, mi[u], initial ring block of (u,h)
    uint32_t par, mi, blk;
#if GDN_BATCH
    uint32_t knq_l1 = 0;  // L1 address of the [kn | qn] tiles (the per-user masked copies are cut from them)
    const uint32_t nb = gdn_rt_B();
    uint32_t mi_b[kBatchMax];
    bool hold_b[kBatchMax];
#endif
    {
        DeviceZoneScopedN("rd_ctrl");
        ctrl_r.reserve_back(1);
        noc.async_read(ctrl_acc, ctrl_r, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        {
            auto lock = ctrl_r.scoped_write_lock(1);
            auto p = lock.template get_ptr<volatile uint32_t>();
            par = p[0] & 1u;
            mi = p[1 + u];
            blk = p[1 + Mr + bh];
#if GDN_BATCH
            for (uint32_t ub_ = 0; ub_ < kBatchMax; ++ub_) {
                if (ub_ < nb) {
                    mi_b[ub_] = p[1 + ub_];
                    hold_b[ub_] = p[1 + Mr + ub_ * Nv + h] == hold_sentinel;
                }
            }
#endif
        }
        ctrl_r.push_back(1);
        ctrl_r.wait_front(1);
        ctrl_r.pop_front(1);
#if GDN_HOLD_SKIP && GDN_BATCH
        {
            uint32_t live = 0;
            for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
                live |= hold_b[ub_] ? 0u : (1u << ub_);
            }
            gdn_hold_publish(live);
        }
#endif
    }
#if GDN_LAZY
    // no replay for HOLD or FRESH (the writer's r = 0): zero stash tiles, never the stale stash (the same rule as the
    // writer's r_rep)
    const bool lazy_no_replay = (blk == hold_sentinel) || (mi >= T);
#endif
    const bool hold = (blk == hold_sentinel);  // RING_SPLIT: held heads skip their (odd tokens') ring writes too
    (void)hold;
    if (blk == hold_sentinel) {
        // HOLD: the writer skips this head's ring writes and copies the window through, so any valid block and any
        // mi serve the (discarded) recurrence -- this head's own slot-0 block, mi = 0.
        blk = bh;
        mi = 0;
    }
#if GDN_LAZY
    // K4: the state is always D (slot 0); FRESH (mi >= T: a seeded / restored row, nothing to replay) rebuilds the conv
    // window at mi 0 like any first verify of a row. The stash of the previous verify sits in slot 1 + par.
    blk = bh;
    if (mi >= T) {
        mi = 0;
    }
    const uint32_t stash_blk = ((1 + par) * Mr + u) * Nv + h;
#endif
    if (mi > T - 1) {
        mi = T - 1;  // ctrl contents are data (not validated by the host): an out-of-range mi is clamped, not reported
    }
    // the initial state of this block: KVs tiles of ring block blk (VB = 1: the whole 16-tile block in one run)
    auto read_state = [&]() {
        if constexpr (VB == 1) {
            (void)s;
            read_tiles_at(ring_acc, state_in, noc, blk * KV, KV, 0);
        } else {
            for (uint32_t i = 0; i < Kt; ++i) {
                read_tiles_at(ring_acc, state_in, noc, blk * KV + i * Vt + s * Vs, Vs, i * Vs);
            }
        }
    };

    // ---- constants and per-head scalars (each scalar is a small read with its own barrier)
    {
        DeviceZoneScopedN("rd_const");
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
#if !GDN_NO_EPS_L2
        generate_bcast_col_scalar(eps_l2, l2_eps_bits);
#else
        (void)eps_l2;
        (void)l2_eps_bits;
#endif
        generate_bcast_col_scalar(eps_norm, norm_eps_bits);
#if !GDN_FAST
        load_head_scalar<dtb_fp32 != 0>(dtb_acc, dtb_s, noc, h);
        load_head_scalar<nea_fp32 != 0>(nea_acc, nea_s, noc, h);
#endif
    }

    // ---- phase 1 (small): z, the a|b tile(s), the gates -> compute starts silu(z) (and, plain mode, the T gate pairs)
#if !GDN_PREFIXED
    z_in.reserve_back(Vt);
#if GDN_NARROW_SRC
    // only the user's T rows of the head's z tiles (rows outside them are zero: silu(0) = 0, and they never reach the
    // user's output rows -- every phase-D op is per row)
    zero_reserved(z_in, noc, Vt);
    for (uint32_t t = 0; t < Vt; ++t) {
        read_rows_bf16(qkv_acc, z_in, noc, pg + z_tile0 + h * Vt + t, t, r0, T);
    }
#else
    read_tiles_at(qkv_acc, z_in, noc, pg + z_tile0 + h * Vt, Vt, 0);
#endif
#if GDN_GATE_SCALARS
    {
        DeviceZoneScopedN("rd_zwait");  // gov gdn-v2 profiling zone (no code without PROFILE_KERNEL)
        noc.async_read_barrier();
    }
    z_in.push_back(Vt);
#else
    // ab_in[0] = a's tile (ab_page + h/32); ab_in[1] = b's tile (ab_page + (Nv+h)/32) only when it differs (b_idx = 1;
    // AB2 = 0 -> one tile, one entry, exactly the 2*Nv <= 32 read). AB2 is the factory's compile-time arg (it also
    // sizes ab_in and g1), not re-derived here.
    ab_in.reserve_back(1 + AB2);
    read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page + (h >> 5), 1, 0);
    if constexpr (AB2 != 0) {
        if (b_idx != 0) {
            read_tiles_at(qkv_acc, ab_in, noc, pg + ab_page + ((Nv + h) >> 5), 1, 1);
        }
    } else {
        (void)b_idx;
    }
#if GDN_FAST
    noc.async_read_barrier();
    z_in.push_back(Vt);
#else
    // csel[0]: row h & 31 all ones (column extractor of a within its tile), csel[1]: row (Nv + h) & 31 (of b within
    // its)
    csel.reserve_back(2);
    zero_reserved(csel, noc, 2);
    {
        auto lock = csel.scoped_write_lock(2);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t j = 0; j < 32; ++j) {
            set_one(p16, 0, h & 31u, j);
            set_one(p16, 1, (Nv + h) & 31u, j);
        }
    }
    csel.push_back(2);
    // rsel[t]: 1.0 at column r0 + t in every row (rsel[t] @ X = every row equal to row r0 + t of X)
    rsel.reserve_back(T);
    zero_reserved(rsel, noc, T);
    {
        auto lock = rsel.scoped_write_lock(T);
        auto p16 = lock.template get_ptr<volatile uint16_t>();
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t i = 0; i < 32; ++i) {
                set_one(p16, t, i, r0 + t);
            }
        }
    }
    rsel.push_back(T);
    noc.async_read_barrier();
    z_in.push_back(Vt);
    // With AB2 = 1 and b_idx = 0 (a and b share a tile: heads h < 8 at Nv = 24) entry 1 is pushed UNWRITTEN; compute
    // gathers b from ab_in[b_idx] = entry 0, so it is never indexed.
    ab_in.push_back(1 + AB2);
#endif
#endif  // GDN_GATE_SCALARS

    // ---- phase 2 (bulk): window + raw projection tiles, taps, weight row; the conv selectors built meanwhile
    auto build_sel = [&]() {
        // sel[0] = WO_W: row i (i < K-1) picks window row mi+1+i of E_prev; sel[1] = WO_N: row i (K-1 <= i < Lw) picks
        // raw row r0 + i - (K-1); sel[2+j] = SH_j: row r0+t picks W row t+j (the tap-j operand of output token t);
        // sel[2+K] = P: diagonal one-hot on rows r0..r0+T-1 (the conv tile passes through a matmul as the plain op's
        // does)
        sel.reserve_back(3 + K);
        zero_reserved(sel, noc, 3 + K);
        {
            auto lock = sel.scoped_write_lock(3 + K);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            for (uint32_t i = 0; i < K - 1; ++i) {
                set_one(p16, 0, i, mi + 1 + i);
            }
            for (uint32_t i = K - 1; i < Lw; ++i) {
                set_one(p16, 1, i, r0 + i - (K - 1));
            }
            for (uint32_t j = 0; j < K; ++j) {
                for (uint32_t t = 0; t < T; ++t) {
                    set_one(p16, 2 + j, r0 + t, t + j);
                }
            }
            for (uint32_t t = 0; t < T; ++t) {
                set_one(p16, 2 + K, r0 + t, r0 + t);
            }
        }
        sel.push_back(3 + K);
    };
    {
        // (x7: this line keeps the rd_bulk zone off the 16-bit profiler hash of brisck.cc:77 under /tmp/x7sf/tree)
        DeviceZoneScopedN("rd_bulk");
        // src_in[c] = whole window tile of user u, chunk c (win[par]); src_in[Ch + c] = whole raw projection tile
        src_in.reserve_back(2 * ChA);
#if GDN_NARROW_SRC
        // window rows [0, Lw) (the only rows WO_W selects: mi + 1 + i <= Lw - 1) and the user's rows [r0, r0 + T) of
        // the raw qkv tile (the only rows WO_N selects); every other row is a finite zero, so the one-hot matmuls
        // (0 * x = 0 for finite x) give the same W_c. Lw > 16 falls back to whole window tiles.
        zero_reserved(src_in, noc, 2 * ChA);
        for (uint32_t c = 0; c < ChA; ++c) {
            const uint32_t col = chunk_col(blk_chunk(c));
            if constexpr (Lw <= 16) {
                if (par) {
                    read_rows_bf16(wb_acc, src_in, noc, u * Ct + col, c, 0, Lw);
                } else {
                    read_rows_bf16(wa_acc, src_in, noc, u * Ct + col, c, 0, Lw);
                }
            } else {
                if (par) {
                    read_tiles_at(wb_acc, src_in, noc, u * Ct + col, 1, c);
                } else {
                    read_tiles_at(wa_acc, src_in, noc, u * Ct + col, 1, c);
                }
            }
            read_rows_bf16(qkv_acc, src_in, noc, pg + col, ChA + c, r0, T);
        }
#else
        for (uint32_t c = 0; c < ChA; ++c) {
            const uint32_t col = chunk_col(blk_chunk(c));
            if (par) {
                read_tiles_at(wb_acc, src_in, noc, u * Ct + col, 1, c);
            } else {
                read_tiles_at(wa_acc, src_in, noc, u * Ct + col, 1, c);
            }
            read_tiles_at(qkv_acc, src_in, noc, pg + col, 1, ChA + c);
        }
#endif
#if GDN_EARLY_SRC
        // phase A1 / A2 need only src_in and the selectors: hand them over before the taps / weight row / masks
        build_sel();
        noc.async_read_barrier();
        src_in.push_back(2 * ChA);
#endif
        // taps[c]: rows 0..K-1 (tap j in row j) of tile column chunk_col(c); the rest of the tile is zero
        taps.reserve_back(ChA);
        zero_reserved(taps, noc, ChA);
        {
            const uint32_t entry = taps.get_entry_size();
            const uint32_t nbytes = K * 32;  // K face rows of bf16
            for (uint32_t c = 0; c < ChA; ++c) {
                const uint32_t page = chunk_col(blk_chunk(c));
                const uint32_t base = c * entry;
                noc.async_read(taps_acc, taps, nbytes, {.page_id = page, .offset_bytes = 0}, {.offset_bytes = base});
                noc.async_read(
                    taps_acc,
                    taps,
                    nbytes,
                    {.page_id = page, .offset_bytes = entry / 4},
                    {.offset_bytes = base + entry / 4});
            }
        }
        w_in.reserve_back(Vt);
        zero_reserved(w_in, noc, Vt);
        for (uint32_t t = 0; t < Vt; ++t) {
            read_row0_at(w_acc, w_in, noc, t, t);  // the norm weight lives in row 0 (row-broadcast operand)
        }
#if !GDN_EARLY_SRC
        build_sel();
#endif
        // mask_T: 1.0 at (r0 + t, 0) for t < T; e_t[t]: 1.0 at (r0 + t, 0)
        mask_T.reserve_back(1);
        zero_reserved(mask_T, noc, 1);
        {
            auto lock = mask_T.scoped_write_lock(1);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            for (uint32_t t = 0; t < T; ++t) {
                set_one(p16, 0, r0 + t, 0);
            }
        }
        mask_T.push_back(1);
#if !GDN_LAZY
        e_t.reserve_back(T);
        zero_reserved(e_t, noc, T);
        {
            auto lock = e_t.scoped_write_lock(T);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            for (uint32_t t = 0; t < T; ++t) {
                set_one(p16, t, r0 + t, 0);
            }
        }
        e_t.push_back(T);
#endif
        noc.async_read_barrier();
#if !GDN_EARLY_SRC
        src_in.push_back(2 * ChA);
#endif
        taps.push_back(ChA);
        w_in.push_back(Vt);
    }
#else  // GDN_PREFIXED
    // GDN_STEP_U1 prototype: [kn | qn] (knq), this block's v tiles (vc) and silu(z) (zs) come from the prefix's output
    // tensor (per value head h: tiles h*S + [kn 0..Kt) [qn Kt..2Kt) [vc 2Kt..2Kt+Vt) [zs 2Kt+Vt..2Kt+2Vt)), no window /
    // projection / tap reads, no selectors, no l2norm mask
    (void)blk_chunk;
    (void)chunk_col;
    (void)mi;
    (void)pg;
    (void)qkv_acc;
    (void)wa_acc;
    (void)wb_acc;
    (void)taps_acc;
#if GDN_BATCH && GDN_GATE_SCALARS
    // all users of head h, value block s: D_u (slot 0 block (u, h), this block's columns), the previous stash (K^T
    // tiles, this block's U tiles) or zeros (FRESH / HOLD), user-major. Needs only ring + ctrl: with GDN_STATE_FIRST
    // (engine GDN role) it is issued before the in_ready wait, i.e. while the previous layer's compute still runs.
    // GDN_ST_RING: users [u0, u1) only (the ring holds GDN_ST_RING users; the compute pops them in its replay, which
    // comes after the knq transposes, so at most GDN_ST_RING users may go out before rd_pre); else all users, u0 / u1
    // unused
    auto rd_state_b = [&](uint32_t u0, uint32_t u1) {
        DeviceZoneScopedN("rd_state");
#if GDN_ST_RING
        // GDN_ST_RING: one user at a time through a ring of GDN_ST_RING users (the compute pops per user)
        for (uint32_t ub_ = u0; ub_ < u1; ++ub_) {
            state_in.reserve_back(KVs);
            ktp_in.reserve_back(Kt);
            up_in.reserve_back(Vs);
            const uint32_t blk_u = ub_ * Nv + h;
            for (uint32_t i = 0; i < Kt; ++i) {
                read_tiles_at(ring_acc, state_in, noc, blk_u * KV + i * Vt + s * Vs, Vs, i * Vs);
            }
            const uint32_t stash_u = ((1 + par) * Mr + ub_) * Nv + h;
            if (hold_b[ub_] || mi_b[ub_] >= T) {
                zero_tiles_at(ktp_in, noc, 0, Kt);
                zero_tiles_at(up_in, noc, 0, Vs);
            } else {
                read_tiles_at(ring_acc, ktp_in, noc, stash_u * KV, Kt, 0);
                read_tiles_at(ring_acc, up_in, noc, stash_u * KV + Kt + s * Vs, Vs, 0);
            }
            noc.async_read_barrier();
            state_in.push_back(KVs);
            ktp_in.push_back(Kt);
            up_in.push_back(Vs);
        }
#else
        (void)u0;
        (void)u1;
        state_in.reserve_back(nb * KVs);
        ktp_in.reserve_back(nb * Kt);
        up_in.reserve_back(nb * Vs);
#if GDN_STNARROW
        if (ktp_in.get_write_ptr() != g_stn_kb || up_in.get_write_ptr() != g_stn_ub) {
            zero_tiles_at(ktp_in, noc, 0, nb * Kt);  // slot base moved (first layer of the program, or B < 8): all +0
            zero_tiles_at(up_in, noc, 0, nb * Vs);
            noc.async_read_barrier();  // the zeros land before the narrow reads below overwrite the live parts
            g_stn_kb = ktp_in.get_write_ptr();
            g_stn_ub = up_in.get_write_ptr();
        }
#if GDN_KTDD
        // this core's ktp_in slots are reserved (the compute popped the previous layer's) and +0 outside the live parts:
        // the peers may write their K^T shares into them
        gdn_ktdd_bump(s, VB, 0);
#endif
#endif
        for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
            if (GDN_HOLD_SKIP && hold_b[ub_]) {
                continue;  // HOLD_SKIP: the compute never reads this user's state / stash slots
            }
            const uint32_t blk_u = ub_ * Nv + h;
            for (uint32_t i = 0; i < Kt; ++i) {
                read_tiles_at(ring_acc, state_in, noc, blk_u * KV + i * Vt + s * Vs, Vs, ub_ * KVs + i * Vs);
            }
            const uint32_t stash_u = ((1 + par) * Mr + ub_) * Nv + h;
#if GDN_RPX  // REPLAY bench-only timing diagnostics (wrong numerics): 1 = never read the stash (zeros), 2 = read it even
             // when nothing is replayed, 4 = K^T from DRAM on value block 0 only (zeros elsewhere)
            const bool rd_st = (GDN_RPX & 1) ? false : ((GDN_RPX & 2) ? !hold_b[ub_] : !(hold_b[ub_] || mi_b[ub_] >= T));
            if (!rd_st) {
                zero_tiles_at(ktp_in, noc, ub_ * Kt, Kt);
                zero_tiles_at(up_in, noc, ub_ * Vs, Vs);
            } else {
                if ((GDN_RPX & 4) && s != 0) {
                    zero_tiles_at(ktp_in, noc, ub_ * Kt, Kt);
                } else {
                    read_tiles_at(ring_acc, ktp_in, noc, stash_u * KV, Kt, ub_ * Kt);
                }
                read_tiles_at(ring_acc, up_in, noc, stash_u * KV + Kt + s * Vs, Vs, ub_ * Vs);
            }
#else
            if (hold_b[ub_] || mi_b[ub_] >= T) {
                zero_tiles_at(ktp_in, noc, ub_ * Kt, Kt);
                zero_tiles_at(up_in, noc, ub_ * Vs, Vs);
            } else {
#if GDN_STNARROW
                // user ub_'s tokens = rows / columns r0u .. r0u + T - 1 (16 % T == 0: inside one face row / column)
                const uint32_t r0u = ub_ * T;
                const uint32_t kf = (r0u >> 4) * 1024u;                      // K^T: faces 0 / 2 (cols < 16) or 1 / 3
                const uint32_t uo = (r0u >> 4) * 2048u + (r0u & 15u) * 64u;  // U: the T rows in faces 0-1 or 2-3
                const uint32_t te = ktp_in.get_entry_size();
                for (uint32_t i = GDN_KTDD ? s : 0; i < Kt; i += GDN_KTDD ? VB : 1) {
                    const uint32_t d = (ub_ * Kt + i) * te;
                    noc.async_read(ring_acc, ktp_in, 1024, {.page_id = stash_u * KV + i, .offset_bytes = kf},
                                   {.offset_bytes = d + kf});
                    noc.async_read(ring_acc, ktp_in, 1024, {.page_id = stash_u * KV + i, .offset_bytes = kf + 2048},
                                   {.offset_bytes = d + kf + 2048});
                }
                for (uint32_t j = 0; j < Vs; ++j) {
                    const uint32_t d = (ub_ * Vs + j) * te;
                    const uint32_t pg = stash_u * KV + Kt + s * Vs + j;
                    noc.async_read(ring_acc, up_in, T * 64, {.page_id = pg, .offset_bytes = uo}, {.offset_bytes = d + uo});
                    noc.async_read(ring_acc, up_in, T * 64, {.page_id = pg, .offset_bytes = uo + 1024},
                                   {.offset_bytes = d + uo + 1024});
                }
#else
                read_tiles_at(ring_acc, ktp_in, noc, stash_u * KV, Kt, ub_ * Kt);
                read_tiles_at(ring_acc, up_in, noc, stash_u * KV + Kt + s * Vs, Vs, ub_ * Vs);
#endif
            }
#endif
        }
        noc.async_read_barrier();
#if GDN_KTDD
        {
            // own K^T share (tiles i = s mod VB, the user's 2 faces) -> the same slots of the VB - 1 peers, once each
            // peer's slots are ready; then wait for the peers' shares to land here
            volatile tt_l1_ptr uint32_t* ks = gdn_ktdd_sem();
            const uint32_t want = (VB - 1) * (g_ktdd_l + 1);
            noc_semaphore_wait_min(ks, want);
            const uint32_t kb = ktp_in.get_write_ptr(), te = ktp_in.get_entry_size();
            for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
                if (hold_b[ub_] || mi_b[ub_] >= T) {
                    continue;  // HOLD: no slot use; FRESH: every block zero-fills locally (ctrl is per user / head)
                }
                const uint32_t kf = ((ub_ * T) >> 4) * 1024u;
                for (uint32_t i = s; i < Kt; i += VB) {
                    const uint32_t a = kb + (ub_ * Kt + i) * te + kf;
                    gdn_ktdd_fwd(s, VB, a);
                    gdn_ktdd_fwd(s, VB, a + 2048);
                }
            }
            noc_async_write_barrier();
            gdn_ktdd_bump(s, VB, 1);
            noc_semaphore_wait_min(ks + 1, want);
            ++g_ktdd_l;
        }
#endif
        state_in.push_back(nb * KVs);
        ktp_in.push_back(nb * Kt);
        up_in.push_back(nb * Vs);
#endif
    };
#if GDN_STATE_FIRST
    {
        DeviceZoneScopedN("rd_state_first");
        rd_state_b(0, (GDN_ST_RING != 0 && GDN_ST_RING < nb) ? GDN_ST_RING : nb);
    }
#endif
#endif
#if GSTEP_WPRE
    // GSTEP_WPRE: the gated-norm weight rows (static per layer) go out before the in_ready wait, off the rd_pre path
    {
        w_in.reserve_back(Vt);
        zero_reserved(w_in, noc, Vt);
        for (uint32_t t = 0; t < Vt; ++t) {
            read_row0_at(w_acc, w_in, noc, t, t);
        }
        noc.async_read_barrier();
        w_in.push_back(Vt);
    }
#endif
#if GDN_IN_WAIT
    // engine GDN role: this layer's prefix output (pre) is complete once in_ready reaches the layer's target
    GBTS(0);
    gdn_in_wait();
    GBTS(8);
#endif
    {
        DeviceZoneScopedN("rd_pre");
        const auto pre_acc = TensorAccessor(tensor::pre);
        DataflowBuffer knq(dfb::knq);
        DataflowBuffer vc(dfb::vc);
        DataflowBuffer zs(dfb::zs);
        constexpr uint32_t S = GDN_PRE_S;  // 2*Kt + 2*Vt (+ 9 gate tiles with PREFIXED_GATES)
        const uint32_t base = tr * GDN_PRE_W + h * S;
        zs.reserve_back(Vt);
#if !GDN_GB_DIRECT  // G-b direct: zs / kc / qc / vc were written into these DFBs' slots by the prefix (done barrier passed)
        read_tiles_at(pre_acc, zs, noc, base + 2 * Kt + Vt, Vt, 0);
#endif
#if GDN_RAWKQ
        // G-b L2-root fallback: [k | q] arrive raw (conv + SiLU) -> kc / qc; the compute normalizes them into knq
        // (its own producer), knq_l1 = knq's base (2 Kt entries = the whole DFB: every layer lands at the base)
        DataflowBuffer kq(dfb::kq);  // [k | q] raw, Kt tiles each
        knq_l1 = knq.get_write_ptr();
        kq.reserve_back(2 * Kt);
#if !GDN_GB_DIRECT
        read_tiles_at(pre_acc, kq, noc, base, 2 * Kt, 0);
#else
        gdn_gb_wait(0);  // the 2 Kt raw k / q tiles have landed: the compute's l2 starts while vc / zs / gates arrive
        kq.push_back(2 * Kt);
#endif
#else
        knq.reserve_back(2 * Kt);
#if GDN_BATCH
        knq_l1 = knq.get_write_ptr();
#endif
#if GDN_GB_DIRECT
        gdn_gb_knq(knq.get_write_ptr());  // G-b: pull the L2 roots' 2 Kt normalized k / q tiles into knq
#else
        read_tiles_at(pre_acc, knq, noc, base, 2 * Kt, 0);
#endif
#endif
        vc.reserve_back(Vs);
#if !GDN_GB_DIRECT
        read_tiles_at(pre_acc, vc, noc, base + 2 * Kt + s * Vs, Vs, 0);
#endif
#if !GSTEP_WPRE
        w_in.reserve_back(Vt);
        zero_reserved(w_in, noc, Vt);
        for (uint32_t t = 0; t < Vt; ++t) {
            read_row0_at(w_acc, w_in, noc, t, t);
        }
#endif
        noc.async_read_barrier();
#if GDN_GB_DIRECT
        gdn_gb_wait(1);  // vc / zs / the head's gate rows have landed
#endif
        GBTS(11);
        zs.push_back(Vt);
#if GDN_RAWKQ
#if !GDN_GB_DIRECT
        kq.push_back(2 * Kt);
#endif
#else
        knq.push_back(2 * Kt);
#endif
        vc.push_back(Vs);
#if !GSTEP_WPRE
        w_in.push_back(Vt);
#endif
    }
#endif  // GDN_PREFIXED

#if GDN_GATE_SCALARS
    // ---- GATE_SCALARS: the gates are the writer's; the initial state goes out as soon as it lands
#if GDN_BATCH
    {
        // all users of head h, value block s: D_u (slot 0 block (u, h), this block's columns), the previous stash
        // (K^T tiles, this block's U tiles) or zeros (FRESH / HOLD), user-major; then the user masks
#if !GDN_STATE_FIRST
        rd_state_b(0, nb);
#elif GDN_ST_RING
        if (GDN_ST_RING < nb) {
            rd_state_b(GDN_ST_RING, nb);  // the rest of the users, behind the compute's replay
        }
#endif
        // per-user masked operands: knq tiles with only user u's T rows (exact fp32 bits, +0 elsewhere; T consecutive rows
        // of one face pair = two contiguous T*64 B runs per tile, local NoC copies)
        static_assert(16 % T == 0, "a user's rows inside one face row");
        const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
        auto cut = [&](uint32_t dst_tile, uint32_t src_tile, uint32_t te, uint32_t r0u) {
#if GDN_CUT_FAST
            uint32_t e = 0;
            while (e < g_cut_n && g_cut_addr[e] != dst_tile) {
                ++e;
            }
            if (e == g_cut_n) {  // first use in this program: full zero fill, then the rows
                for (uint32_t o = 0; o < te; o += MEM_ZEROS_SIZE) {
                    noc_async_read(z, dst_tile + o, MEM_ZEROS_SIZE);
                }
                noc_async_read_barrier();
                g_cut_addr[g_cut_n++] = dst_tile;
            } else if (g_cut_r0[e] != r0u) {  // only the previous occupant's rows are non-zero
                const uint32_t pof = ((g_cut_r0[e] >> 4) * 2 * 256 + (g_cut_r0[e] & 15) * 16) * 4;
                noc_async_read(z, dst_tile + pof, T * 64);
                noc_async_read(z, dst_tile + pof + 1024, T * 64);
            }
            g_cut_r0[e] = r0u;
#else
            for (uint32_t o = 0; o < te; o += MEM_ZEROS_SIZE) {
                noc_async_read(z, dst_tile + o, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
#endif
            const uint32_t off = ((r0u >> 4) * 2 * 256 + (r0u & 15) * 16) * 4;
            noc_async_read(get_noc_addr(src_tile + off), dst_tile + off, T * 64);
            noc_async_read(get_noc_addr(src_tile + off + 1024), dst_tile + off + 1024, T * 64);
        };
#if GDN_RAWKQ
        {   // the normalized [kn | qn] are the compute's (read in place below; the compute pops them at its layer end)
            DeviceZoneScopedN("rd_knq_wait");
            DataflowBuffer knq_w(dfb::knq);
            knq_w.wait_front(2 * Kt);
        }
#endif
        // kts: K_u tile i = s + q*VB for this block's share of the stash K^T pages (kpb per user). GSTEP_KTSLATE (with
        // KTLATE: the compute transposes them in c_D): cut after the kmq stream, so c_wy's first users land earlier
        auto kts_cuts = [&]() {
        {
            DataflowBuffer kts(dfb::kts);
            constexpr uint32_t kpb = (Kt + VB - 1) / VB;
            kts.reserve_back(kBatchMax * kpb);
            const uint32_t te = kts.get_entry_size(), kb = kts.get_write_ptr();
            for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
                if (GDN_HOLD_SKIP && hold_b[ub_]) {
                    continue;
                }
                for (uint32_t q = 0; q < kpb; ++q) {
                    const uint32_t i = s + q * VB;
                    if (i < Kt) {
                        cut(kb + (ub_ * kpb + q) * te, knq_l1 + i * te, te, ub_ * T);
                    }
                }
            }
            noc_async_read_barrier();
            kts.push_back(kBatchMax * kpb);
        }
        };
#if !GSTEP_KTSLATE
        kts_cuts();
#endif
#if GSTEP_WYPAIR
        // GSTEP_WYPAIR: one masked tile per (j, u, i) holds K_u's T rows at row r = (u T) mod 16 (face row 0) and Q_u's at
        // 16 + r (face row 1): the compute's DST tile (u T >= 16) gets [K S' | Q S'] rows for 4 users, then a DST half swap
        static_assert(kBatchMax * T == 32, "WYPAIR: users fill one 32-row tile");
        {
#if GSTEP_CZ
            DeviceZoneScopedN("rd_kmq");  // bench: the paired kmq cut stream
#endif
            DataflowBuffer kmq(dfb::kmq);
            const uint32_t te = kmq.get_entry_size();
            constexpr uint32_t rb = T * 64;  // one user's rows in one face (fp32)
            auto cut_rd = [&](uint32_t src, uint32_t dst) {
#if GSTEP_CUTST
                noc_async_read_one_packet_with_state(src, dst);
#else
                noc_async_read(get_noc_addr(src), dst, rb);
#endif
            };
            auto cut_pair = [&](uint32_t dst_tile, uint32_t src_k, uint32_t src_q, uint32_t r0u) {
                const uint32_t rr = r0u & 15, soff = (r0u >> 4) * 2048 + rr * 64, top = rr * 64, bot = 2048 + rr * 64;
#if GDN_CUT_FAST
                uint32_t e = 0;
                while (e < g_cut_n && g_cut_addr[e] != dst_tile) {
                    ++e;
                }
                if (e == g_cut_n) {
                    for (uint32_t o = 0; o < te; o += MEM_ZEROS_SIZE) {
                        noc_async_read(z, dst_tile + o, MEM_ZEROS_SIZE);
                    }
                    noc_async_read_barrier();
                    g_cut_addr[g_cut_n++] = dst_tile;
#if GSTEP_CUTST
                    noc_async_read_one_packet_set_state(z, rb);  // the plain reads above reprogrammed the read cmd buffer
#endif
                } else if (!GSTEP_CUTNZ && g_cut_r0[e] != rr) {  // the previous occupant's K and Q rows (disjoint)
                    const uint32_t pt = g_cut_r0[e] * 64;
                    cut_rd(MEM_ZEROS_BASE, dst_tile + pt);
                    cut_rd(MEM_ZEROS_BASE, dst_tile + pt + 1024);
                    cut_rd(MEM_ZEROS_BASE, dst_tile + 2048 + pt);
                    cut_rd(MEM_ZEROS_BASE, dst_tile + 2048 + pt + 1024);
                }
                g_cut_r0[e] = rr;
#else
                for (uint32_t o = 0; o < te; o += MEM_ZEROS_SIZE) {
                    noc_async_read(z, dst_tile + o, MEM_ZEROS_SIZE);
                }
                noc_async_read_barrier();
#if GSTEP_CUTST
                noc_async_read_one_packet_set_state(z, rb);
#endif
#endif
                cut_rd(src_k + soff, dst_tile + top);
                cut_rd(src_k + soff + 1024, dst_tile + top + 1024);
                cut_rd(src_q + soff, dst_tile + bot);
                cut_rd(src_q + soff + 1024, dst_tile + bot + 1024);
            };
#if GSTEP_CUTST
            // GSTEP_CUTST: every cut read is local (this core's L1 -> L1, rb bytes): one stateful read command, only the two
            // local addresses change per read (same reads, same bytes)
            noc_async_read_one_packet_set_state(z, rb);
#endif
#pragma GCC unroll 1
            for (uint32_t j = 0; j < Vs; ++j) {
#pragma GCC unroll 1
                for (uint32_t k_ = 0; k_ < kBatchMax; ++k_) {
                    // GSTEP_WYORD: users 0 1 4 5 2 3 6 7 -> each kmq ring slot sees row offsets 0 0 8 8 / 4 4 12 12 (fewer
                    // zeroings); DST 0 / 1 still accumulate 0..3 / 4..7 in order (the compute uses the same order)
                    const uint32_t ub_ = GSTEP_WYORD ? ((k_ & 1u) + ((k_ >> 1) & 1u) * 4u + (k_ >> 2) * 2u) : k_;
                    if (ub_ >= nb || (GDN_HOLD_SKIP && hold_b[ub_])) {
                        continue;
                    }
                    kmq.reserve_back(Kt);
                    const uint32_t kb = kmq.get_write_ptr();
#pragma GCC unroll 1
                    for (uint32_t i = 0; i < Kt; ++i) {
                        cut_pair(kb + i * te, knq_l1 + i * te, knq_l1 + (Kt + i) * te, ub_ * T);
                    }
                    noc_async_read_barrier();
                    kmq.push_back(Kt);
                }
            }
        }
#if GSTEP_KTSLATE
        kts_cuts();
#endif
#else
        // kmq stream in the compute's order: for each output o of [K S' | Q S'] (o < Vs: K rows, else Q rows), for each
        // user u, the Kt masked tiles of K_u / Q_u (a 4-chunk ring)
        {
            DataflowBuffer kmq(dfb::kmq);
            const uint32_t te = kmq.get_entry_size();
            for (uint32_t o = 0; o < 2 * Vs; ++o) {
                const uint32_t ro = o >= Vs ? 1u : 0u;
                for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
                    if (GDN_HOLD_SKIP && hold_b[ub_]) {
                        continue;  // the compute skips this user's K S' / Q S' term
                    }
                    kmq.reserve_back(Kt);
                    const uint32_t kb = kmq.get_write_ptr();
                    for (uint32_t i = 0; i < Kt; ++i) {
                        cut(kb + i * te, knq_l1 + (ro * Kt + i) * te, te, ub_ * T);
                    }
                    noc_async_read_barrier();
                    kmq.push_back(Kt);
                }
            }
        }
#endif  // GSTEP_WYPAIR
#if GDN_OUT_RD
        // ---- the gated out tiles (all 32 rows live: whole tiles, as the writer's GDN_OUT_TILE) + the out signal, on NOC1
        //      ahead of any S' write (the barrier must not wait for those)
        {
            const auto out_acc = TensorAccessor(tensor::out_r);
            DataflowBuffer out(dfb::out);
#if GDN_OUT_PRE
            gdn_out_pre();
#endif
            {
                DeviceZoneScopedN("rd_owait");  // the compute's gated output tiles
                out.wait_front(Vs);
            }
            GBTS(12);
            DeviceZoneScopedN("rd_owr");  // the out tile write + its barrier + the out signal
            const uint32_t entry = out.get_entry_size();
            const uint32_t page0 = tr * (Nv * Vt) + h * Vt + s * Vs;
#if GDN_OUT_RD == 2
            // engine gdn_out (MKA_GDNOUT): the out tiles go to every GDN core's A (multicast + barrier + gdna), not `out`
            gdn_out_local(out.get_read_ptr(), page0 - tr * (Nv * Vt), Vs);
#if GDN_OUT_RD_KEEP  // + the out tensor rows (gdn24 check / gdn_out PCC gold), off the handoff path
            for (uint32_t t = 0; t < Vs; ++t) {
                noc.async_write(out, out_acc, entry, {.offset_bytes = t * entry}, {.page_id = page0 + t});
            }
            noc.async_write_barrier();
#else
            (void)out_acc;
            (void)entry;
#endif
#else
            for (uint32_t t = 0; t < Vs; ++t) {
                noc.async_write(out, out_acc, entry, {.offset_bytes = t * entry}, {.page_id = page0 + t});
            }
            noc.async_write_barrier();
#endif
            out.pop_front(Vs);
            gdn_out_signal();
            GBTS(13);
        }
#endif
#if GDN_PRE_SP_HOOK
        gdn_pre_sp_hook();
#endif
#if GSTEP_ATTHOLD
        gdn_att_hold();  // E1_ATTHOLD: a pre-attention GDN layer's S' write-back waits for the attention layer's SDPA end
#endif
#if GDN_SP_RD
        // ---- S'_u -> ring slot 0 block (u, h) (this block's columns) from the compute's sp (waited on, never popped
        //      here: the compute pops it at the next layer's replay after `spd`), as the writer's wr_sp did
        {
            DeviceZoneScopedN("rd_sp");
            DataflowBuffer sp(dfb::sp);
            DataflowBuffer spd(dfb::spd);
            const uint32_t entry = sp.get_entry_size();
            sp.wait_front(nb * KVs);
            const uint32_t base_t = (g_rd_sp_lay * nb * KVs) % (kBatchMax * KVs);
            const uint32_t v0 = s * Vs;
            for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
                if (hold_b[ub_]) {
                    continue;
                }
                const uint32_t bh_u = ub_ * Nv + h;
                for (uint32_t i = 0; i < Kt; ++i) {
                    for (uint32_t j = 0; j < Vs; ++j) {
                        noc.async_write(sp, ring_acc, entry, {.offset_bytes = (base_t + ub_ * KVs + i * Vs + j) * entry},
                                        {.page_id = bh_u * KV + v0 + i * Vt + j});
                    }
                }
            }
            noc.async_writes_flushed();
            spd.reserve_back(1);
            spd.push_back(1);
            ++g_rd_sp_lay;
        }
#endif
    }
    (void)read_state;
    (void)lazy_no_replay;
    (void)stash_blk;
#else
    {
        DeviceZoneScopedN("rd_state");
        state_in.reserve_back(KVs);
        read_state();
#if GDN_LAZY
        // the stash of the previous verify; zeros when nothing is replayed (FRESH / HOLD: the slots may hold a stale
        // stash, even non-finite values, and the compute's replay multiplies them by W = 0)
        ktp_in.reserve_back(Kt);
        up_in.reserve_back(Vs);
        if (lazy_no_replay) {
            zero_reserved(ktp_in, noc, Kt);
            zero_reserved(up_in, noc, Vs);
        } else {
            read_tiles_at(ring_acc, ktp_in, noc, stash_blk * KV, Kt, 0);
            read_tiles_at(ring_acc, up_in, noc, stash_blk * KV + Kt + s * Vs, Vs, 0);
        }
#endif
        noc.async_read_barrier();
        state_in.push_back(KVs);
#if GDN_LAZY
        ktp_in.push_back(Kt);
        up_in.push_back(Vs);
#endif
    }
#endif  // GDN_BATCH
#elif GDN_FAST
    // ---- FAST tail: the compute is busy with phase A / B (about 25 us), so the reader's serial software gate math
    // runs now instead of ahead of the bulk reads. dt_bias / -exp(A) first (small, own barriers), then the 64 KiB
    // initial state read is issued, then the T gate pairs of user u are computed while that read is in flight; the
    // compute needs dec / beta_t only inside the token loop.
    uint32_t dtb_bits, nea_bits;
    {
        DeviceZoneScopedN("rd_scalars");
        dtb_bits = read_head_scalar_bits < dtb_fp32 != 0 > (dtb_acc, dtb_s, noc, h);
        nea_bits = read_head_scalar_bits < nea_fp32 != 0 > (nea_acc, nea_s, noc, h);
    }
    state_in.reserve_back(KVs);
    read_state();
    // The T gate pairs of user u from rows r0 .. r0+T-1 of the a|b tile(s): a[h] at column h & 31 of entry 0,
    // b[h] at column (Nv + h) & 31 of entry b_idx (AB2) / entry 0. bf16 -> fp32 bits is exact (<< 16); the gate math
    // is reader_gdn_decode_step_conv_fast.cpp's, bit for bit. The ab_in slot stays reserved (staging, never pushed).
    {
        DeviceZoneScopedN("rd_gates");
        uint32_t beta_bits[T];
        uint32_t dec_bits[T];
        {
            auto lock = ab_in.scoped_write_lock(1 + AB2);
            auto p16 = lock.template get_ptr<volatile uint16_t>();
            const uint32_t b_tile = (AB2 != 0) ? b_idx : 0u;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t a_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, h & 31u)]) << 16;
                const uint32_t b_bits =
                    static_cast<uint32_t>(p16[b_tile * 1024 + tile_elem_index(r0 + t, (Nv + h) & 31u)]) << 16;
                gate_beta_decay_bits(a_bits, b_bits, dtb_bits, nea_bits, beta_bits[t], dec_bits[t]);
            }
        }
        beta_t.reserve_back(T);
        fill_scalar_tiles_fast(beta_t, beta_bits, T);
        beta_t.push_back(T);
        dec.reserve_back(T);
        fill_scalar_tiles_fast(dec, dec_bits, T);
        dec.push_back(T);
    }
    {
        DeviceZoneScopedN("rd_state");
        noc.async_read_barrier();
        state_in.push_back(KVs);
    }
#else
    // ---- the initial state, last: 16 tiles from ring block blk (the writer's block-0 write of this core is ordered
    // after this read completes, since compute cannot produce a state before it is pushed)
    {
        DeviceZoneScopedN("rd_state");
        state_in.reserve_back(KVs);
        read_state();
        noc.async_read_barrier();
        state_in.push_back(KVs);
    }
#endif
#if GDN_RING_SPLIT
    // ---- RING_SPLIT: the odd tokens' post-token states (hnew2, KVs tiles each) to ring block t*BH + bh at this block's
    //      column tiles -- the writer's ring loop for those tokens, on this RISC's NoC
    {
        DataflowBuffer hnew2(dfb::hnew2);
        constexpr uint32_t BH = M * Nv;
        const uint32_t v0 = s * Vs;
        const uint32_t entry = hnew2.get_entry_size();
        for (uint32_t t = 1; t < T; t += 2) {
            hnew2.wait_front(KVs);
            DeviceZoneScopedN("rd_ring");
            if (!hold) {
                const uint32_t page0 = (t * BH + bh) * KV + v0;
                for (uint32_t i = 0; i < Kt; ++i) {
                    for (uint32_t j = 0; j < Vs; ++j) {
                        noc.async_write(
                            hnew2,
                            ring_acc,
                            entry,
                            {.offset_bytes = (i * Vs + j) * entry},
                            {.page_id = page0 + i * Vt + j});
                    }
                }
                noc.async_write_barrier();
            }
            hnew2.pop_front(KVs);
        }
    }
#endif
}
