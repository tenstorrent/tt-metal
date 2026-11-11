// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Writer: one core = (user u, value head h). Reads the ctrl page (parity, HOLD). Window: rows 0..Lw-1 (rounded up
// to an even count; the extra row is an exact zero) of the compute's rebuilt window tiles go to win[1-par] (q/k
// chunks only from heads with h % rf == 0, v chunks from every head); a held user's tiles are copied through
// win[par] -> win[1-par] via a bounce buffer and the compute's tiles are drained. Ring: per token the post-token
// state (16 fp32 tiles) from the buffered `hnew` to block (t*BH + bh), skipped for held heads. Then the user's T
// output rows (whole face-row spans when T is even, 32 B single rows at T = 1). The output tensor is not zero-filled
// at creation, so the cores of the last user (u = B-1) also write exact zeros into rows [B*T % 32, 32) of the last
// tile row (before the ring stream, while the writer is idle) -- with R <= round_up(B*T, 32) enforced by the host,
// every output row outside [u*T, u*T + T) is then exactly 0. HOLD is decided per (u,h) ctrl word; q/k window chunk
// ownership is per key-head group (h % rf == 0), so a caller must hold ALL Nv heads of a user (the op cannot check
// ctrl contents).
// VB > 1 (value blocks, GDN_VSPLIT): core (u, h, s) owns state / output column tiles [s*Vs, (s+1)*Vs). Block 0 alone
// writes (or copies through) the window; each block writes its KVs ring tiles (i, s*Vs + j) per token, zeroes / writes
// only its Vs output tiles, and before that swaps the last token's pre-norm output tiles with its head's other blocks:
// acc_mine (Vs fp32 tiles, exact bytes) -> acc_x[s*Vs ..] on every block (itself included) by unicast NoC writes.
// Handshake (both semaphores start at 0 on every launch): each block raises `ready` on every block once it runs (its
// acc_x is reserved), waits for VB before writing, write-barriers, then raises `valid` on every block and waits for VB
// before handing acc_x to the compute. No multicast; at most 2 * VB atomics and VB writes per core.
// GDN_GATE_SCALARS (kernel_opts bit 1, fast_mode + fp32_state): this writer, idle until the compute's phase A1 hands it
// the window tiles, computes the T beta / decay gate pairs of (u, h) instead of the reader -- the a|b tile(s) and the
// head's dt_bias / -exp(A) read here, then reader_gdn_decode_step_conv_fast.cpp's software-fp32 gate math
// (gdn_step_gate_math.hpp, bit for bit) -- and hands the compute the 2*T fp32 bit patterns in ONE small `gsc` entry
// (beta[t] at word t, decay[t] at word T + t) instead of 2*T broadcast tiles: the compute fills its DST tiles from the
// same bits (fill_tile_bitcast). The extra qkv / dt_bias / -exp(A) geometry arrives as defines (GDN_AB_PAGE,
// GDN_W_TILES, GDN_DTB_FP32, GDN_NEA_FP32), so the named-argument signature is the same in every mode.
// GDN_LAZY (kernel_opts K4 lazy, implies GATE_SCALARS): after the gate pairs (split over the head's blocks and swapped
// with GDN_GSC_SPLIT) the writer builds the eight WY coefficient tiles (coef: -beta G, G, E, BETA, -beta E, EYE, W, C)
// from all T pairs and the previous stash's decay page: G[t][i] = prod dec[i+1..t] and E[t] = prod dec[0..t] are plain
// fp32 products of the decays (no extra exp); the replay of the previous verify's accepted prefix (r = mi + 1 tokens; 0
// for a FRESH row, ctrl mi >= T, or HOLD) arrives as W (row-broadcast weights prod dec_prev(i..r)) and C (the decay of
// D, prod dec_prev[0..r), as a whole tile). Instead of the T ring blocks per token: S' (dnew) -> ring slot 0 block
// (u, h); this verify's stash -> slot 1 + (1 - par): K^T tiles (block 0), this block's U tiles, the decay page (block
// 0); HOLD writes none.
// GDN_SFPU_GATE (kernel_opts bit 9, K4 only): no software gate math here. The writer pushes the raw a|b tile(s) (ab_w)
// and one gsc entry (word 0 = dt_bias bits, word 1 = -exp(A) bits, word 2 = the replay decay C bits) to the compute,
// whose SFPU returns dec over a's tile and beta over b's tile (gt); token t's pair sits at (r0 + t, h) / (r0 + t, Nv + h)
// of those tiles. The coefficient tiles are built from it as above, except C (tile 7, not written: the compute fills it
// in DST from gsc word 2).
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_decode_step/device/kernels/dataflow/gdn_step_dataflow_helpers.hpp"
#include "api/debug/kernel_profiler.hpp"  // zones compile to nothing unless the build defines PROFILE_KERNEL

#ifndef GDN_VSPLIT
#define GDN_VSPLIT 0
#endif
#ifndef GDN_GATE_SCALARS
#define GDN_GATE_SCALARS 0
#endif
#ifndef GDN_QKSHARE
#define GDN_QKSHARE 0  // kernel_opts bit 4 (VB > 1): per-block window ownership + the q/k conv tile exchange
#endif
#ifndef GDN_GSC_SPLIT
#define GDN_GSC_SPLIT 0  // kernel_opts bit 6 (GATE_SCALARS, VB > 1): the head's blocks split the T gate computations
#endif
#ifndef GDN_R16_EARLY_COEF
#define GDN_R16_EARLY_COEF 0
#endif
#ifndef GDN_RT_BM
#define GDN_RT_BM 0  // engine GDN role: B / M from gdn_rt_B() / gdn_rt_M() (defined by the including TU), one binary for
                     // type 11 (B = 1) and type 4 (B = 8); 0 = the template values (the op)
#endif
#ifndef GDN_LOOP
// engine GDN role: the writer entry runs once per GDN layer inside one kernel; the head's exchange semaphores count
// cumulatively (wait for VB * (layer + 1)), so nothing is reset between layers. gdn_loop_k() = layer + 1 is defined by
// the including TU. 0 = one call (the op).
#define GDN_LOOP 0
#endif
#if GDN_LOOP
uint32_t gdn_loop_k();
#define GDN_SEM_K gdn_loop_k()
#else
#define GDN_SEM_K 1u
#endif
#ifndef GDN_BATCH
#define GDN_BATCH 0  // batched type 4: all users of head h on core (h, s) (see the compute)
#endif
#if GDN_BATCH
constexpr uint32_t kBatchMax = 8;
constexpr uint32_t kCoefN = 7 + kBatchMax;  // NGB G E BETA NBE EYE W + C_u
#endif
#ifndef GDN_PREFIXED_GATES
#define GDN_PREFIXED_GATES 0  // kernel_opts bit 22: coefficient tiles + gate page from `prefixed`
#endif
#ifndef GDN_SP_LATE
#define GDN_SP_LATE 0  // engine GDN role (BATCH): S' written after the output rows (see the compute's sp pop)
#endif
#ifndef GDN_RCOEF_FAST
#define GDN_RCOEF_FAST 0  // engine GDN role (EARLY_REPLAY): batched decay-row reads + NoC-doubled C_u fill in wr_rcoef
#endif
#ifndef GDN_EARLY_REPLAY
#define GDN_EARLY_REPLAY 0  // engine GDN role (BATCH + PREFIXED_GATES): W / C_u built here before in_ready (see wr_rcoef)
#endif
#ifndef GDN_W_IN_WAIT
#define GDN_W_IN_WAIT 0  // engine GDN role: gdn_in_wait() (the including TU) gates this writer's reads of `pre`
#endif
#if GDN_W_IN_WAIT
void gdn_in_wait();
#endif
#if GDN_EARLY_REPLAY
constexpr uint32_t kCoefRep = 1 + kBatchMax;  // W | C_0..C_7, pushed first; then NGB G E BETA NBE EYE
inline float gdn_f32(uint32_t b) {
    union {
        uint32_t u;
        float f;
    } x;
    x.u = b;
    return x.f;
}
inline uint32_t gdn_bits(float f) {
    union {
        uint32_t u;
        float f;
    } x;
    x.f = f;
    return x.u;
}
#endif
#ifndef GDN_PREFIXED
#define GDN_PREFIXED 0  // kernel_opts bit 21 (K4): no window write (the u1 prefix owns it)
#endif
#ifndef GDN_LAZY
#define GDN_LAZY 0  // kernel_opts bit 8 (K4): coefficient tiles + S' / stash writes, no per-token ring stream
#endif
#if GDN_LAZY && !GDN_GATE_SCALARS
#error "GDN_LAZY needs GDN_GATE_SCALARS"
#endif
#ifndef GDN_SFPU_GATE
#define GDN_SFPU_GATE 0  // kernel_opts bit 9 (K4 only): the compute's SFPU evaluates the gates (no soft-float exp here)
#endif
#ifndef GDN_NARROW_GR
#define GDN_NARROW_GR 0  // kernel_opts bit 10 (GATE_SCALARS): gate reads fetch only the face rows they use (bit-identical)
#endif
#ifndef GDN_EARLY_GR
#define GDN_EARLY_GR 0  // kernel_opts bit 12 (x8, with SFPU_GATE + NARROW_GR): the a|b / dt_bias / -exp(A) gate reads are
                        // issued with the ctrl read (they do not depend on ctrl), ahead of the reader's bulk reads
#endif
#if GDN_EARLY_GR && !(GDN_SFPU_GATE && GDN_NARROW_GR)
#error "GDN_EARLY_GR needs GDN_SFPU_GATE and GDN_NARROW_GR"
#endif
#if GDN_SFPU_GATE && (!GDN_LAZY || GDN_GSC_SPLIT)
#error "GDN_SFPU_GATE needs GDN_LAZY and no GDN_GSC_SPLIT"
#endif
#ifndef GDN_NARROW_GR2
#define GDN_NARROW_GR2 0  // kernel_opts bit 15 (with NARROW_GR): one contiguous read per face holding a / b (bit-identical)
#endif
#if GDN_GATE_SCALARS
#if !GDN_SFPU_GATE
#define GDN_GATE_NOINLINE 1  // code size; see gdn_step_gate_math.hpp (SFPU_GATE: exp_acc / log1p_acc unused, inline)
#endif
#include "ttnn/cpp/ttnn/operations/experimental/kda/gdn_spec_step/device/kernels/dataflow/gdn_step_gate_math.hpp"
#endif
#if GDN_VSPLIT
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#ifndef GDN_EXT_SEM
#define GDN_EXT_SEM 0  // engine GDN role: ready / valid are device-local global semaphores at gdn_xsem_addr(0 / 1)
#endif
#ifndef GDN_OUT_SIGNAL
#define GDN_OUT_SIGNAL 0  // engine GDN role: gdn_out_signal() (the including TU) after the output rows are written
#endif
#ifndef GDN_XVC
#define GDN_XVC 0  // engine GDN role (E1_XVC): static unicast VC of the head exchange (lz_xchg data + increments); 0 = default VC
#endif
#ifndef GDN_LAZY_GATE
#define GDN_LAZY_GATE 0  // engine GDN role (E1_LAZY_GATE): gdn_lazy_gate() (the including TU) before the batched stash writes
#endif
#if GDN_LAZY_GATE
void gdn_lazy_gate();
#endif
#if GDN_EXT_SEM
uint32_t gdn_xsem_addr(uint32_t i);
// the subset of Semaphore the head exchange uses, on a global semaphore (same L1 address on every GDN core); counts are
// cumulative over the layers of one program (wait >= target), the program's last wrapper step resets them
struct GdnXSem {
    uint32_t a;
    explicit GdnXSem(uint32_t addr) : a(addr) {}
#if GDN_XVC
    // GDN_XVC: the head exchange's increments on static unicast VC GDN_XVC (not the default VC 1 that the other heads'
    // stash / S' DRAM writes fill), so a late head's exchange is not queued behind them
    void up(Noc&, uint32_t x, uint32_t y, uint32_t v) { noc_semaphore_inc(get_noc_addr(x, y, a), v, noc_index, GDN_XVC); }
#else
    void up(Noc&, uint32_t x, uint32_t y, uint32_t v) { noc_semaphore_inc(get_noc_addr(x, y, a), v); }
#endif
    void wait(uint32_t v) const { noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a), v); }
};
#endif
#ifndef GDN_OUT_LOCAL
#define GDN_OUT_LOCAL 0  // engine gdn_out: 1 = hand this core's out tiles to gdn_out_local() (the including TU) before the
#endif               // DRAM out rows; 2 = and skip the DRAM out rows (the out tensor is then not written)
#if GDN_OUT_LOCAL
void gdn_out_local(uint32_t src_l1, uint32_t t0, uint32_t n);
#endif
#if GDN_OUT_SIGNAL
void gdn_out_signal();
#endif
#endif

using namespace gdn_step_df;

namespace {
// Rows [r0, r0 + nr) of `count` L1 tiles to the same rows of the destination tiles, ONE face-row (32 B for bf16,
// 64 B for fp32) per NoC write; source and destination offsets are equal, so both sit in the same alignment class
// whatever the row parity (validated in the M0a scratch for fp32 outputs; bf16 is tested here).
template <typename Accessor>
inline void write_rows_single(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t first_page, uint32_t count, uint32_t r0, uint32_t nr) {
    dfb.wait_front(count);
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t esz = entry / 1024;
    const uint32_t seg = entry / 64;
    for (uint32_t t = 0; t < count; ++t) {
        const uint32_t base = t * entry;
        for (uint32_t r = r0; r < r0 + nr; ++r) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t off = tile_elem_index(r, half * 16) * esz;
                noc.async_write(
                    dfb, acc, seg, {.offset_bytes = base + off}, {.page_id = first_page + t, .offset_bytes = off});
            }
        }
    }
    noc.async_write_barrier();
    dfb.pop_front(count);
}

template <typename Accessor>
inline void write_whole_tile(const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t src_tile) {
    const uint32_t entry = dfb.get_entry_size();
    noc.async_write(dfb, acc, entry, {.offset_bytes = src_tile * entry}, {.page_id = page});
}

#if GDN_NARROW_GR
// NARROW_GR: rows [r0, r0 + n) of tile `page` into the same rows of slot `dst_tile`, one face-row segment (16 elements)
// per row and tile half (left: columns 0..15, right: 16..31). The gate reads of every core hit the same few DRAM pages
// (the users' a|b tile row, tile 0 of dt_bias / -exp(A)): fetching 4 KB pages for a handful of elements serialized on
// one bank. Source / destination offsets are equal (same alignment class, as write_rows does); esz is the source's
// element size (dt_bias / -exp(A) may be bf16 in a 4 KB staging slot, as the whole-page reads leave them).
template <typename Accessor>
inline void read_rows_at(
    const Accessor& acc,
    DataflowBuffer& dfb,
    Noc& noc,
    uint32_t page,
    uint32_t dst_tile,
    uint32_t r0,
    uint32_t n,
    bool left,
    bool right,
    uint32_t esz) {  // bytes per element of the SOURCE tile (the staging slot keeps the source layout)
    const uint32_t entry = dfb.get_entry_size();
    const uint32_t seg = 16 * esz;
    for (uint32_t r = r0; r < r0 + n; ++r) {
        for (uint32_t half = 0; half < 2; ++half) {
            if ((half == 0 && !left) || (half == 1 && !right)) {
                continue;
            }
            const uint32_t off = tile_elem_index(r, half * 16) * esz;
            noc.async_read(acc, dfb, seg, {.page_id = page, .offset_bytes = off}, {.offset_bytes = dst_tile * entry + off});
        }
    }
}
#endif

#if GDN_LAZY && GDN_VSPLIT
// K4 (code size: the program fills the kernel config buffer): the head's three NoC exchanges -- gate slots, q/k conv
// tiles, pre-norm output tiles -- share one body: wait until every block runs this program (`ready`), `bytes` of src at
// src_off to `dst` on every block of the head (itself included), the write barrier, `sem` up on every block, and wait
// until all VB blocks' data is in
template <uint32_t VB, typename Sem>
__attribute__((noinline)) void lz_xchg(
    Noc& noc,
    DataflowBuffer& src,
    uint32_t src_off,
    uint32_t bytes,
    uint32_t dst,
    Sem& ready,
    Sem& sem,
    const uint32_t* px,
    const uint32_t* py) {
    ready.wait(VB * GDN_SEM_K);
    if (bytes != 0) {
        for (uint32_t p = 0; p < VB; ++p) {
#if GDN_XVC
            noc_async_write(src.get_read_ptr() + src_off, get_noc_addr(px[p], py[p], dst), bytes, noc_index, GDN_XVC);
#else
            noc.async_write(
                src,
                UnicastEndpoint{},
                bytes,
                {.offset_bytes = src_off},
                {.noc_x = px[p], .noc_y = py[p], .addr = dst});
#endif
        }
        noc.async_write_barrier();  // the data landed before `sem` is raised
    }
    for (uint32_t p = 0; p < VB; ++p) {
        sem.up(noc, px[p], py[p], 1);
    }
    sem.wait(VB * GDN_SEM_K);
}
#endif
#if GDN_NARROW_GR2
// NARROW_GR2: rows [r0, r0 + n) of the tile half `right` (columns 16..31) or left (0..15) of bf16 tile `page`, one
// contiguous read per face group (a face stores its 16 rows of 16 elements row-major, 32 B per row), into the same
// offsets of slot `dst_tile`. Every core reads the same a|b page: 1 request per core instead of 2 * n (NARROW_GR).
template <typename Accessor>
inline void read_rows_half(
    const Accessor& acc, DataflowBuffer& dfb, Noc& noc, uint32_t page, uint32_t dst_tile, uint32_t r0, uint32_t n, bool right) {
    const uint32_t base = dst_tile * dfb.get_entry_size();
    for (uint32_t fr = 0; fr < 2; ++fr) {
        const uint32_t lo = r0 > fr * 16 ? r0 : fr * 16;
        const uint32_t hi = (r0 + n) < (fr * 16 + 16) ? (r0 + n) : (fr * 16 + 16);
        if (lo >= hi) {
            continue;
        }
        const uint32_t off = ((2 * fr + (right ? 1u : 0u)) * 256u + (lo - fr * 16) * 16u) * 2u;
        noc.async_read(acc, dfb, (hi - lo) * 32u, {.page_id = page, .offset_bytes = off}, {.offset_bytes = base + off});
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
    uint32_t Lw,
    uint32_t Ct,
    uint32_t out_w_tiles,
    uint32_t ctrl_bytes,
    uint32_t hold_sentinel,
    uint32_t VB>
TT_KERNEL void writer(
    uint32_t u,
    uint32_t h,
    uint32_t s,
    uint32_t peer_x0,
    uint32_t peer_y0,
    uint32_t peer_x1,
    uint32_t peer_y1,
    uint32_t peer_x2,
    uint32_t peer_y2,
    uint32_t peer_x3,
    uint32_t peer_y3) {
#if GDN_RT_BM
    const uint32_t Br = gdn_rt_B(), Mr = gdn_rt_M();
#else
    constexpr uint32_t Br = B, Mr = M;
#endif
    const auto ring_acc = TensorAccessor(tensor::ring_out);
    const auto out_acc = TensorAccessor(tensor::out);
    const auto wa_acc = TensorAccessor(tensor::win_a_w);
    const auto wb_acc = TensorAccessor(tensor::win_b_w);
    const auto ctrl_acc = TensorAccessor(tensor::ctrl_w);
#if !GDN_LAZY
    DataflowBuffer hnew(dfb::hnew);
#endif
    DataflowBuffer out(dfb::out);
    DataflowBuffer wout(dfb::wout);
    DataflowBuffer ctrl_w(dfb::ctrl_w);
    DataflowBuffer bounce(dfb::bounce);
    Noc noc;
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t Ch = 2 * Kt + Vt;
    static_assert(VB >= 1 && VB <= 4 && Vt % VB == 0, "value blocks must divide Vt (at most 4 peers)");
    static_assert((VB > 1) == (GDN_VSPLIT != 0), "GDN_VSPLIT must be defined exactly when VB > 1");
    constexpr uint32_t Vs = Vt / VB;  // this block's state / output column tiles
    constexpr uint32_t KVs = Kt * Vs;
    const uint32_t v0 = s * Vs;  // first column tile of this block
    // QKSHARE: every block writes the window rows of its own ChA chunks (q/k share [s*Qk, (s+1)*Qk), then its v
    // chunks); otherwise block 0 writes the head's window rows (ChA = Ch, block order = head order)
    constexpr uint32_t Qk = GDN_QKSHARE ? (2 * Kt) / VB : 0u;
    constexpr uint32_t ChA = GDN_QKSHARE ? Qk + Vs : Ch;
    const bool window_writer = GDN_QKSHARE ? true : (s == 0);
    auto blk_chunk = [&](uint32_t ci) -> uint32_t {
        if constexpr (GDN_QKSHARE) {
            return ci < Qk ? s * Qk + ci : 2 * Kt + v0 + (ci - Qk);
        } else {
            return ci;
        }
    };
    constexpr uint32_t rf = Nv / Nk;
    const uint32_t BH = Mr * Nv;
    constexpr uint32_t Lw_w = (Lw + 1) & ~1u;  // even row count (row Lw is an exact zero when Lw is odd)
    const uint32_t hk = h / rf;
    const uint32_t bh = u * Nv + h;
    const uint32_t tr = (u * T) / 32;  // output tile row of the user's rows
    const uint32_t r0 = (u * T) % 32;
    const bool write_qk = (h % rf) == 0;  // the rf value heads of a key head share its q/k chunks: one writer
    auto chunk_col = [&](uint32_t c) -> uint32_t {
        if (c < Kt) {
            return hk * Kt + c;
        }
        if (c < 2 * Kt) {
            return Nk * Kt + hk * Kt + (c - Kt);
        }
        return 2 * Nk * Kt + h * Vt + (c - 2 * Kt);
    };
    auto owns = [&](uint32_t c) -> bool { return c >= 2 * Kt || write_qk; };

    // ---- ctrl page: parity and HOLD
    ctrl_w.reserve_back(1);
    noc.async_read(ctrl_acc, ctrl_w, ctrl_bytes, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
#if GDN_EARLY_GR
    // EARLY_GR: the ctrl-independent gate reads go out with the ctrl read (the ctrl barrier covers them), before the
    // reader's bulk window / state reads queue on the same DRAM banks; only the replay decay rows (they need mi) wait
    // for ctrl. Same reads, same staging slots as NARROW_GR below (bit-identical).
    const auto qkv_acc = TensorAccessor(tensor::qkv_w);
    const auto dtb_acc = TensorAccessor(tensor::dtb_w);
    const auto nea_acc = TensorAccessor(tensor::nea_w);
    DataflowBuffer ab_w(dfb::ab_w);
    DataflowBuffer dtb_ws(dfb::dtb_ws);
    DataflowBuffer nea_ws(dfb::nea_ws);
    constexpr uint32_t AB2 = (2 * Nv > 32) ? 1u : 0u;  // the factory's rule: the a|b block spans two tiles
    const uint32_t b_tile = (AB2 != 0 && ((Nv + h) >> 5) != (h >> 5)) ? 1u : 0u;
    const uint32_t pg = tr * GDN_W_TILES;
    ab_w.reserve_back(1 + AB2);
    read_rows_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + (h >> 5), 0, r0, T, true, true, 2);
    if (b_tile != 0) {
        read_rows_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + ((Nv + h) >> 5), 1, r0, T, true, true, 2);
    }
    dtb_ws.reserve_back(1);
    nea_ws.reserve_back(1);
    read_rows_at(dtb_acc, dtb_ws, noc, 0, 0, 0, 1, (h & 16u) == 0, (h & 16u) != 0, GDN_DTB_FP32 ? 4 : 2);
    read_rows_at(nea_acc, nea_ws, noc, 0, 0, 0, 1, (h & 16u) == 0, (h & 16u) != 0, GDN_NEA_FP32 ? 4 : 2);
#endif
    noc.async_read_barrier();
    uint32_t par, blk;
#if GDN_BATCH
    const uint32_t nb = Br;  // users on this core (all of head h)
    bool hold_b[kBatchMax];
#if GDN_EARLY_REPLAY
    uint32_t mi_b[kBatchMax];
#endif
#endif
#if GDN_LAZY
    uint32_t mi_raw;
#endif
    {
        auto lock = ctrl_w.scoped_write_lock(1);
        auto p = lock.template get_ptr<volatile uint32_t>();
        par = p[0] & 1u;
#if GDN_LAZY
        mi_raw = p[1 + u];
#endif
        blk = p[1 + Mr + bh];
#if GDN_BATCH
        for (uint32_t ub_ = 0; ub_ < kBatchMax; ++ub_) {
            if (ub_ < nb) {
                hold_b[ub_] = p[1 + Mr + ub_ * Nv + h] == hold_sentinel;
#if GDN_EARLY_REPLAY
                mi_b[ub_] = p[1 + ub_];
#endif
            }
        }
#endif
    }
    ctrl_w.push_back(1);
    ctrl_w.wait_front(1);
    ctrl_w.pop_front(1);
    const bool hold = (blk == hold_sentinel);
#if GDN_LAZY
    // replay count: the previous verify's accepted prefix mi + 1 (mi >= T: FRESH, nothing to replay; HOLD: discarded)
    const uint32_t r_rep = (hold || mi_raw >= T) ? 0u : mi_raw + 1;
    const uint32_t stash_rd = ((1 + par) * Mr + u) * Nv + h;        // the previous verify's stash block
    const uint32_t stash_wr = ((1 + (1 - par)) * Mr + u) * Nv + h;  // this verify's
    DataflowBuffer gnew(dfb::gnew);  // block 0: this verify's decay page, built with the gates, written with the stash
#endif

#if GDN_VSPLIT
    // acc_x is this program's buffer from launch on: reserve it now (same L1 address on every block: same spec, first
    // use) and tell every block of the head it may write into it
    const uint32_t peer_x[4] = {peer_x0, peer_x1, peer_x2, peer_x3};
    const uint32_t peer_y[4] = {peer_y0, peer_y1, peer_y2, peer_y3};
    DataflowBuffer acc_mine(dfb::acc_mine);
    DataflowBuffer acc_x(dfb::acc_x);
#if GDN_EXT_SEM
    GdnXSem ready(gdn_xsem_addr(0));
    GdnXSem valid(gdn_xsem_addr(1));
#else
    Semaphore ready(sem::ready);
    Semaphore valid(sem::valid);
#endif
    acc_x.reserve_back(Vt);
    const uint32_t acc_x_base = acc_x.get_write_ptr();
#if GDN_GSC_SPLIT
    // gsc is filled by the head's blocks (each writes its tokens' 64 B slots into every block): reserved before `ready`
    DataflowBuffer gsc_x(dfb::gsc);
    gsc_x.reserve_back(1);
    const uint32_t gsc_base = gsc_x.get_write_ptr();
#endif
#if GDN_QKSHARE
    // qc / kc are filled by the head's blocks (each writes its Qk conv tiles into every block): reserved before
    // `ready` goes up, at the same L1 address on every block (same spec, first use), like acc_x
    DataflowBuffer qc_x(dfb::qc);
    DataflowBuffer kc_x(dfb::kc);
    qc_x.reserve_back(Kt);
    kc_x.reserve_back(Kt);
    const uint32_t qc_base = qc_x.get_write_ptr();
    const uint32_t kc_base = kc_x.get_write_ptr();
#endif
    for (uint32_t p = 0; p < VB; ++p) {
        ready.up(noc, peer_x[p], peer_y[p], 1);
    }
#else
    (void)peer_x0;
    (void)peer_y0;
    (void)peer_x1;
    (void)peer_y1;
    (void)peer_x2;
    (void)peer_y2;
    (void)peer_x3;
    (void)peer_y3;
#endif

#if GDN_LAZY
    uint32_t beta_all[T], dec_all[T];  // the T gate pairs (fp32 bits) the coefficient tiles are built from
#endif
#if GDN_SFPU_GATE
    float wrep[T];  // the replay weights / decay, from the previous stash's decay page (gate block)
    float crep = 1.0f;
#endif
#if GDN_PREFIXED_GATES
    // ---- lever 1: the head's 8 WY coefficient tiles (C included) and its gate page come from `prefixed` (the prefix
    //      side built them with this writer's op sequence); this verify's decay page (block 0) is built from the gate
    //      page exactly as below. No gate reads, no SFPU gates, no coefficient build on this core.
#if GDN_EARLY_REPLAY
    // ---- engine GDN role: the replay tiles from ctrl + the previous stash's decay pages, BEFORE the layer's in_ready
    //      (the tg producer's op sequence, bit for bit): W (row r0_u + i = w_i for i < r_u, else 0) and C_u (the
    //      replay decay over the whole tile); the compute's replay starts on them while the prefix still runs
    {
        DeviceZoneScopedN("wr_rcoef");
        DataflowBuffer coef(dfb::coef);
        DataflowBuffer gprev(dfb::gprev);  // staging for the decay rows (writer-local)
        coef.reserve_back(kCoefN);
        const uint32_t cb0 = coef.get_write_ptr();
        const uint64_t zr = get_noc_addr(MEM_ZEROS_BASE);
#if GDN_RCOEF_FAST
        // GDN_RCOEF_FAST: only W needs zeros (every C_u tile is filled whole below); each user's T decay rows (column
        // 0 lives in the left face: rows r0u .. r0u + T - 1 = T * 64 contiguous bytes) are read at their own offset of
        // ONE staging tile (users' rows don't overlap), all under one barrier; C_u = one 256 B RISC seed + 4 local NoC
        // doublings (256 B -> 4 KB). Same bits as the per-user path.
        for (uint32_t o = 0; o < 4096u; o += MEM_ZEROS_SIZE) {
            noc_async_read(zr, cb0 + o, MEM_ZEROS_SIZE);
        }
        gprev.reserve_back(1);
        const uint32_t gpa = gprev.get_write_ptr();
        for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
            const uint32_t r0u = ub_ * T;
            const uint32_t rr = (hold_b[ub_] || mi_b[ub_] >= T) ? 0u : mi_b[ub_] + 1;
            if (rr > 0) {
                const uint32_t sr = ((1 + par) * Mr + ub_) * Nv + h;
                const uint32_t off = ((r0u >> 4) * 512 + (r0u & 15) * 16) * 4;
                noc.async_read(ring_acc, gprev, T * 64, {.page_id = sr * KV + Kt + Vt, .offset_bytes = off},
                               {.offset_bytes = off});
            }
        }
        noc.async_read_barrier();
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* c32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb0);
        volatile tt_l1_ptr uint32_t* gp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gpa);
        for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
            const uint32_t r0u = ub_ * T;
            const uint32_t rr = (hold_b[ub_] || mi_b[ub_] >= T) ? 0u : mi_b[ub_] + 1;
            float wrp[T];
            float crp = 1.0f;
            if (rr > 0) {
                float acc = 1.0f;
                for (uint32_t i = rr; i-- > 0;) {
                    wrp[i] = acc;
                    acc = acc * gdn_f32(gp[tile_elem_index(r0u + i, 0)]);
                }
                crp = acc;
            }
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t wb = (t < rr) ? gdn_bits(wrp[t]) : 0u;
                for (uint32_t c = 0; c < 32; ++c) {
                    c32[tile_elem_index(r0u + t, c)] = wb;
                }
            }
            const uint32_t cbits = gdn_bits(crp);
            for (uint32_t i = 0; i < 64; ++i) {
                c32[(1 + ub_) * 1024 + i] = cbits;
            }
        }
        // users >= nb: C tiles stay +0 as with the full zero fill (never read: the replay runs nb users)
        for (uint32_t ub_ = nb; ub_ < kBatchMax; ++ub_) {
            for (uint32_t i = 0; i < 64; ++i) {
                c32[(1 + ub_) * 1024 + i] = 0u;
            }
        }
        asm volatile("fence" ::: "memory");  // the RISC seeds reach L1 before the NoC reads them
        for (uint32_t sz = 256; sz < 4096u; sz <<= 1) {
            for (uint32_t ub_ = 0; ub_ < kBatchMax; ++ub_) {
                const uint32_t t0 = cb0 + (1 + ub_) * 4096u;
                noc_async_read(get_noc_addr(t0), t0 + sz, sz);
            }
            noc_async_read_barrier();
        }
#else
        for (uint32_t o = 0; o < kCoefRep * 4096u; o += MEM_ZEROS_SIZE) {
            noc_async_read(zr, cb0 + o, MEM_ZEROS_SIZE);
        }
        gprev.reserve_back(1);
        const uint32_t gpa = gprev.get_write_ptr();
        noc.async_read_barrier();
        volatile tt_l1_ptr uint32_t* c32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb0);
        for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
            const uint32_t r0u = ub_ * T;
            const uint32_t rr = (hold_b[ub_] || mi_b[ub_] >= T) ? 0u : mi_b[ub_] + 1;
            float wrp[T];
            float crp = 1.0f;
            if (rr > 0) {
                const uint32_t sr = ((1 + par) * Mr + ub_) * Nv + h;
                noc.async_read(ring_acc, gprev, 4096, {.page_id = sr * KV + Kt + Vt}, {.offset_bytes = 0});
                noc.async_read_barrier();
                invalidate_l1_cache();
                volatile tt_l1_ptr uint32_t* gp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gpa);
                float acc = 1.0f;
                for (uint32_t i = rr; i-- > 0;) {
                    wrp[i] = acc;
                    acc = acc * gdn_f32(gp[tile_elem_index(r0u + i, 0)]);
                }
                crp = acc;
            }
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t wb = (t < rr) ? gdn_bits(wrp[t]) : 0u;
                for (uint32_t c = 0; c < 32; ++c) {
                    c32[tile_elem_index(r0u + t, c)] = wb;
                }
            }
            const uint32_t cbits = gdn_bits(crp);
            for (uint32_t i = 0; i < 1024; ++i) {
                c32[(1 + ub_) * 1024 + i] = cbits;
            }
        }
#endif
        coef.push_back(kCoefRep);
    }
#endif
#if GDN_W_IN_WAIT
    gdn_in_wait();
#endif
    {
        DeviceZoneScopedN("wr_pgates");
        const auto pre_acc = TensorAccessor(tensor::pre_w);
        DataflowBuffer coef(dfb::coef);
        DataflowBuffer gprev(dfb::gprev);  // staging for the gate page (writer-local)
        const uint32_t pbase = tr * GDN_PRE_W + h * GDN_PRE_S + 2 * Kt + 2 * Vt;
#if GDN_BATCH
        constexpr uint32_t ncoef = kCoefN;
#else
        constexpr uint32_t ncoef = 8;
#endif
#if GDN_EARLY_REPLAY
        // the 6 WY tiles (pre's NGB G E BETA NBE EYE = its coefficient tiles 0..5) behind W / C_u
        constexpr uint32_t nwy = kCoefN - kCoefRep;
        coef.reserve_back(nwy);
        read_tiles_at(pre_acc, coef, noc, pbase, nwy, 0);
        gprev.reserve_back(1);
        read_tiles_at(pre_acc, gprev, noc, pbase + ncoef, 1, 0);
        noc.async_read_barrier();
        coef.push_back(nwy);
#else
        coef.reserve_back(ncoef);
        read_tiles_at(pre_acc, coef, noc, pbase, ncoef, 0);
        gprev.reserve_back(1);
        read_tiles_at(pre_acc, gprev, noc, pbase + ncoef, 1, 0);
        noc.async_read_barrier();
        coef.push_back(ncoef);
#endif
#if GDN_BATCH
        (void)gnew;  // the per-user decay pages are built at the stash write (wr_lazy), from the gate page
    }
    if (false) {
        DataflowBuffer gprev(dfb::gprev);
#endif
        {
            auto gp_lock = gprev.scoped_write_lock(1);
            auto gp = gp_lock.template get_ptr<volatile uint32_t>();
            for (uint32_t t = 0; t < T; ++t) {
                dec_all[t] = gp[tile_elem_index(r0 + t, 0)];
                beta_all[t] = gp[tile_elem_index(r0 + t, 1)];
            }
        }
        if (s == 0) {
            gnew.reserve_back(1);
            {
                auto gn_lock = gnew.scoped_write_lock(1);
                auto gn = gn_lock.template get_ptr<volatile uint32_t>();
                for (uint32_t r = 0; r < 32; ++r) {
                    gn[tile_elem_index(r, 0)] = 0;
                }
                for (uint32_t t = 0; t < T; ++t) {
                    gn[tile_elem_index(r0 + t, 0)] = dec_all[t];
                }
            }
            gnew.push_back(1);
        }
        (void)r_rep;
        (void)wrep;
        (void)crep;
    }
#endif
#if GDN_GATE_SCALARS && !GDN_PREFIXED_GATES
    // ---- the T gate pairs of (u, h) -> gsc (the reader's FAST-mode gate block, moved here: same reads, same math)
    {
        DeviceZoneScopedN("wr_gates");
#if !GDN_EARLY_GR
        const auto qkv_acc = TensorAccessor(tensor::qkv_w);
        const auto dtb_acc = TensorAccessor(tensor::dtb_w);
        const auto nea_acc = TensorAccessor(tensor::nea_w);
        DataflowBuffer ab_w(dfb::ab_w);
        DataflowBuffer dtb_ws(dfb::dtb_ws);
        DataflowBuffer nea_ws(dfb::nea_ws);
#endif
#if GDN_GSC_SPLIT
        // GSC_SPLIT: this block computes tokens [t0, t1) only (contiguous share of ceil(T / VB)) into its send buffer
        // gss (slot t = 16 words at word 16*t: beta, dec), then every block's gsc receives them
        constexpr uint32_t Tb = (T + VB - 1) / VB;
        const uint32_t t0 = (s * Tb < T) ? s * Tb : T;
        const uint32_t t1 = (t0 + Tb < T) ? t0 + Tb : T;
        DataflowBuffer gsc(dfb::gss);
#else
        constexpr uint32_t t0 = 0;
        constexpr uint32_t t1 = T;
        DataflowBuffer gsc(dfb::gsc);
#endif
#if !GDN_EARLY_GR
        constexpr uint32_t AB2 = (2 * Nv > 32) ? 1u : 0u;  // the factory's rule: the a|b block spans two tiles
        // b_idx: head h's b column lies in a different tile than its a column (the factory's per-core rule)
        const uint32_t b_tile = (AB2 != 0 && ((Nv + h) >> 5) != (h >> 5)) ? 1u : 0u;
        const uint32_t pg = tr * GDN_W_TILES;
        // staging only (never pushed): the a|b tile(s) and the head scalars' tile 0, all issued under one barrier
        ab_w.reserve_back(1 + AB2);
#endif
#if GDN_EARLY_GR
        // (issued with the ctrl read)
#elif GDN_NARROW_GR
        // only the user's T rows of the a|b tile(s) (both halves: a at column h, b at column Nv + h), tile 0's face row
        // of dt_bias / -exp(A) holding column h, and the r replay rows of the decay page's column 0 (left half)
#if GDN_NARROW_GR2
        {
            // a[r0 + t] sits in column h & 31 of tile 0, b[r0 + t] in column (Nv + h) & 31 of tile b_tile: read the
            // user's rows of only the tile half (face) holding each -- the very bytes the math below reads
            const bool a_right = (h & 31u) >= 16;
            const bool b_right = ((Nv + h) & 31u) >= 16;
            read_rows_half(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + (h >> 5), 0, r0, T, a_right);
            if (b_tile != 0) {
                read_rows_half(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + ((Nv + h) >> 5), 1, r0, T, b_right);
            } else if (b_right != a_right) {
                read_rows_half(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + (h >> 5), 0, r0, T, b_right);
            }
        }
#else
        read_rows_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + (h >> 5), 0, r0, T, true, true, 2);
        if (b_tile != 0) {
            read_rows_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + ((Nv + h) >> 5), 1, r0, T, true, true, 2);
        }
#endif
        dtb_ws.reserve_back(1);
        nea_ws.reserve_back(1);
        read_rows_at(dtb_acc, dtb_ws, noc, 0, 0, 0, 1, (h & 16u) == 0, (h & 16u) != 0, GDN_DTB_FP32 ? 4 : 2);
        read_rows_at(nea_acc, nea_ws, noc, 0, 0, 0, 1, (h & 16u) == 0, (h & 16u) != 0, GDN_NEA_FP32 ? 4 : 2);
#else
        read_tiles_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + (h >> 5), 1, 0);
        if (b_tile != 0) {
            read_tiles_at(qkv_acc, ab_w, noc, pg + GDN_AB_PAGE + ((Nv + h) >> 5), 1, 1);
        }
        dtb_ws.reserve_back(1);
        nea_ws.reserve_back(1);
        noc.async_read(dtb_acc, dtb_ws, GDN_DTB_FP32 ? 4096u : 2048u, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read(nea_acc, nea_ws, GDN_NEA_FP32 ? 4096u : 2048u, {.page_id = 0}, {.offset_bytes = 0});
#endif
#if GDN_LAZY
        // the previous verify's decay page (its stash page Kt + Vt), only when there is a prefix to replay
        DataflowBuffer gprev(dfb::gprev);
        gprev.reserve_back(1);
        if (r_rep > 0) {
#if GDN_NARROW_GR
            read_rows_at(ring_acc, gprev, noc, stash_rd * KV + Kt + Vt, 0, r0, r_rep, true, false, 4);
#else
            read_tiles_at(ring_acc, gprev, noc, stash_rd * KV + Kt + Vt, 1, 0);
#endif
        }
#endif
        {
            DeviceZoneScopedN("wr_grd");
            noc.async_read_barrier();
        }
        uint32_t dtb_bits, nea_bits;
        {
            auto lock = dtb_ws.scoped_write_lock(1);
            if constexpr (GDN_DTB_FP32 != 0) {
                dtb_bits = lock.template get_ptr<volatile uint32_t>()[tile_elem_index(0, h)];
            } else {
                const uint32_t b16 = lock.template get_ptr<volatile uint16_t>()[tile_elem_index(0, h)];
                dtb_bits = b16 << 16;
            }
        }
        {
            auto lock = nea_ws.scoped_write_lock(1);
            if constexpr (GDN_NEA_FP32 != 0) {
                nea_bits = lock.template get_ptr<volatile uint32_t>()[tile_elem_index(0, h)];
            } else {
                const uint32_t b16 = lock.template get_ptr<volatile uint16_t>()[tile_elem_index(0, h)];
                nea_bits = b16 << 16;
            }
        }
#if GDN_SFPU_GATE
        // the replay's decay c = prod dec_prev[0..r) and weights w[i] = prod dec_prev[i+1..r) first (gprev landed with
        // the gate reads): c goes to the compute in gsc word 2 with the head scalars, ahead of the SFPU gates
        if (r_rep > 0) {
            auto gp_lock = gprev.scoped_write_lock(1);
            auto gp = gp_lock.template get_ptr<volatile uint32_t>();
            float acc = 1.0f;
            for (uint32_t i = r_rep; i-- > 0;) {
                wrep[i] = acc;  // prod over j in (i, r)
                acc = acc * f_from_bits(gp[tile_elem_index(r0 + i, 0)]);
            }
            crep = acc;
        }
        gsc.reserve_back(1);
        {
            auto g_lock = gsc.scoped_write_lock(1);
            auto g32 = g_lock.template get_ptr<volatile uint32_t>();
            g32[0] = dtb_bits;
            g32[1] = nea_bits;
            g32[2] = bits_from_f(crep);
        }
        gsc.push_back(1);
        ab_w.push_back(1 + AB2);  // the compute's SFPU gates read the raw a|b tile(s)
#else
        gsc.reserve_back(1);
        {
            auto ab_lock = ab_w.scoped_write_lock(1 + AB2);
            auto p16 = ab_lock.template get_ptr<volatile uint16_t>();
            auto g_lock = gsc.scoped_write_lock(1);
            auto g32 = g_lock.template get_ptr<volatile uint32_t>();
            for (uint32_t t = t0; t < t1; ++t) {
                const uint32_t a_bits = static_cast<uint32_t>(p16[tile_elem_index(r0 + t, h & 31u)]) << 16;
                const uint32_t b_bits =
                    static_cast<uint32_t>(p16[b_tile * 1024 + tile_elem_index(r0 + t, (Nv + h) & 31u)]) << 16;
                uint32_t beta_bits, dec_bits;
                gate_beta_decay_bits(a_bits, b_bits, dtb_bits, nea_bits, beta_bits, dec_bits);
#if GDN_GSC_SPLIT
                g32[16 * t] = beta_bits;
                g32[16 * t + 1] = dec_bits;
#else
                g32[t] = beta_bits;
                g32[T + t] = dec_bits;
#endif
            }
#if GDN_LAZY && !GDN_GSC_SPLIT
            for (uint32_t t = 0; t < T; ++t) {  // every token is this block's: keep the gate bits for the coefficients
                beta_all[t] = g32[t];
                dec_all[t] = g32[T + t];
            }
#endif
        }
        gsc.push_back(1);
#endif  // GDN_SFPU_GATE
#if GDN_GSC_SPLIT
        // slots [t0, t1) (64 B aligned) into every block's gsc (itself included); `gsv` up on every block; once all VB
        // blocks' slots are in, gsc goes to the compute
        {
            DeviceZoneScopedN("wr_gsw");  // gov gdn-v2 profiling zone: the head's gate-slot swap
            Semaphore gsv(sem::gsv);
            gsc.wait_front(1);
#if GDN_LAZY
            lz_xchg<VB>(noc, gsc, t0 * 64, (t1 - t0) * 64, gsc_base + t0 * 64, ready, gsv, peer_x, peer_y);
            gsc.pop_front(1);
#else
            ready.wait(VB);  // every block of the head runs this program and has reserved its gsc
            if (t1 > t0) {
                for (uint32_t p = 0; p < VB; ++p) {
                    noc.async_write(
                        gsc,
                        UnicastEndpoint{},
                        (t1 - t0) * 64,
                        {.offset_bytes = t0 * 64},
                        {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = gsc_base + t0 * 64});
                }
                noc.async_write_barrier();  // the slots landed before `gsv` is raised
            }
            gsc.pop_front(1);
            for (uint32_t p = 0; p < VB; ++p) {
                gsv.up(noc, peer_x[p], peer_y[p], 1);
            }
            gsv.wait(VB);
#endif
#if GDN_LAZY
            {
                invalidate_l1_cache();  // the peers' slots arrived over the NoC: no stale RISC cache line
                auto gx_lock = gsc_x.scoped_write_lock(1);
                auto gx = gx_lock.template get_ptr<volatile uint32_t>();
                for (uint32_t t = 0; t < T; ++t) {
                    beta_all[t] = gx[16 * t];
                    dec_all[t] = gx[16 * t + 1];
                }
            }
#endif
            gsc_x.push_back(1);
        }
#endif
#if !GDN_SFPU_GATE  // SFPU_GATE: after the q/k exchange (below)
#if GDN_LAZY
        // ---- K4: the WY coefficient tiles and this verify's decay page, from all T gate pairs (with GDN_GSC_SPLIT the
        //      head's blocks computed a share each and swapped the slots above)
        {
            float beta[T], dec[T];
            for (uint32_t t = 0; t < T; ++t) {
                beta[t] = f_from_bits(beta_all[t]);
                dec[t] = f_from_bits(dec_all[t]);
            }
            // replay of the previous verify's accepted prefix: c = prod dec_prev[0..r), w[i] = prod dec_prev[i+1..r)
#if !GDN_SFPU_GATE  // (SFPU_GATE: computed above, before the gates)
            float wrep[T];
            float crep = 1.0f;
            if (r_rep > 0) {
                auto gp_lock = gprev.scoped_write_lock(1);
                auto gp = gp_lock.template get_ptr<volatile uint32_t>();
                float acc = 1.0f;
                for (uint32_t i = r_rep; i-- > 0;) {
                    wrep[i] = acc;  // prod over j in (i, r)
                    acc = acc * f_from_bits(gp[tile_elem_index(r0 + i, 0)]);
                }
                crep = acc;
            }
#endif
            // the compute only multiplies and adds (no SFPU subtract / fill routine in its binary): the subtracted
            // terms come negated (-beta G, -beta E) and the replay's decay c as a whole tile
            DataflowBuffer coef(dfb::coef);
            coef.reserve_back(8);
            zero_reserved(coef, noc, GDN_SFPU_GATE ? 7 : 8);  // SFPU_GATE: C (tile 7) is filled by the compute
            {
                auto c_lock = coef.scoped_write_lock(8);
                auto c32 = c_lock.template get_ptr<volatile uint32_t>();
                constexpr uint32_t kNGB = 0, kG = 1024, kE = 2048, kBeta = 3072, kNBE = 4096, kEye = 5120, kW = 6144,
                                   kC = 7168;
                constexpr uint32_t kSign = 0x80000000u;  // exact negation of an fp32 bit pattern
                float e_cum = 1.0f;  // E[t] = prod dec[0..t]
                for (uint32_t t = 0; t < T; ++t) {
                    e_cum = e_cum * dec[t];
                    const uint32_t row = r0 + t;
                    // G[t][i] = prod dec[i+1..t] for i < t (descending i), G[t][t] = 1
                    float g = 1.0f;
                    c32[kG + tile_elem_index(row, row)] = 0x3F800000u;
                    for (uint32_t i = t; i-- > 0;) {
                        g = g * dec[i + 1];
                        c32[kG + tile_elem_index(row, r0 + i)] = bits_from_f(g);
                        c32[kNGB + tile_elem_index(row, r0 + i)] = bits_from_f(beta[t] * g) ^ kSign;
                    }
                    c32[kEye + tile_elem_index(row, row)] = 0x3F800000u;
                    const uint32_t eb = bits_from_f(e_cum);
                    const uint32_t bb = beta_all[t];
                    const uint32_t nbe = bits_from_f(beta[t] * e_cum) ^ kSign;
                    const uint32_t wb = (t < r_rep) ? bits_from_f(wrep[t]) : 0u;
                    for (uint32_t c = 0; c < 32; ++c) {
                        const uint32_t idx = tile_elem_index(row, c);
                        c32[kE + idx] = eb;
                        c32[kBeta + idx] = bb;
                        c32[kNBE + idx] = nbe;
                        c32[kW + idx] = wb;
                    }
                }
#if !GDN_SFPU_GATE
                const uint32_t cb = bits_from_f(crep);
                for (uint32_t i = 0; i < 1024; ++i) {
                    c32[kC + i] = cb;
                }
#else
                (void)kC;
#endif
            }
            coef.push_back(8);
            if (s == 0) {
                gnew.reserve_back(1);
                {
                    auto gn_lock = gnew.scoped_write_lock(1);
                    auto gn = gn_lock.template get_ptr<volatile uint32_t>();
                    // only column 0 is ever read (the next verify's writer and the host park replay take rows r0 ..
                    // r0 + r - 1 of it): zero that column, not the page (a 1024-store fill sat on block 0's
                    // critical path)
                    for (uint32_t r = 0; r < 32; ++r) {
                        gn[tile_elem_index(r, 0)] = 0;
                    }
                    for (uint32_t t = 0; t < T; ++t) {
                        gn[tile_elem_index(r0 + t, 0)] = dec_all[t];
                    }
                }
                gnew.push_back(1);  // writer-local: consumed (written to the stash) after the compute's S'
            }
        }
#endif
#endif
    }
#endif

#if GDN_SFPU_GATE && GDN_R16_EARLY_COEF && !GDN_PREFIXED_GATES
    // r16 EARLY_COEF: the compute evaluates the gates right after silu(z) (before phase A), so the coefficient tiles
    // are built here, before the window write and the q/k exchange -- off the replay's critical path. Same code.
    // ---- SFPU_GATE: the compute's gate tiles (it evaluates them after its phase A) and the coefficient tiles, after the
    //      window write and the q/k exchange (which phase B' waits on); the coefficients are first read at the replay
    {
        DeviceZoneScopedN("wr_coef");
        {
            DeviceZoneScopedN("wr_gwait");
            DataflowBuffer gt(dfb::gt);
            gt.wait_front(2);
            invalidate_l1_cache();  // packed by the compute: no stale RISC cache line
            {
                auto gt_lock = gt.scoped_read_lock(2);
                auto q32 = gt_lock.template get_ptr<volatile uint32_t>();
                for (uint32_t t = 0; t < T; ++t) {
                    dec_all[t] = q32[tile_elem_index(r0 + t, h & 31u)];
                    beta_all[t] = q32[1024 + tile_elem_index(r0 + t, (Nv + h) & 31u)];
                }
            }
            gt.pop_front(2);
        }
#if GDN_LAZY
        // ---- K4: the WY coefficient tiles and this verify's decay page, from all T gate pairs (with GDN_GSC_SPLIT the
        //      head's blocks computed a share each and swapped the slots above)
        {
            float beta[T], dec[T];
            for (uint32_t t = 0; t < T; ++t) {
                beta[t] = f_from_bits(beta_all[t]);
                dec[t] = f_from_bits(dec_all[t]);
            }
            // replay of the previous verify's accepted prefix: c = prod dec_prev[0..r), w[i] = prod dec_prev[i+1..r)
#if !GDN_SFPU_GATE  // (SFPU_GATE: computed above, before the gates)
            float wrep[T];
            float crep = 1.0f;
            if (r_rep > 0) {
                auto gp_lock = gprev.scoped_write_lock(1);
                auto gp = gp_lock.template get_ptr<volatile uint32_t>();
                float acc = 1.0f;
                for (uint32_t i = r_rep; i-- > 0;) {
                    wrep[i] = acc;  // prod over j in (i, r)
                    acc = acc * f_from_bits(gp[tile_elem_index(r0 + i, 0)]);
                }
                crep = acc;
            }
#endif
            // the compute only multiplies and adds (no SFPU subtract / fill routine in its binary): the subtracted
            // terms come negated (-beta G, -beta E) and the replay's decay c as a whole tile
            DataflowBuffer coef(dfb::coef);
            coef.reserve_back(8);
            zero_reserved(coef, noc, GDN_SFPU_GATE ? 7 : 8);  // SFPU_GATE: C (tile 7) is filled by the compute
            {
                auto c_lock = coef.scoped_write_lock(8);
                auto c32 = c_lock.template get_ptr<volatile uint32_t>();
                constexpr uint32_t kNGB = 0, kG = 1024, kE = 2048, kBeta = 3072, kNBE = 4096, kEye = 5120, kW = 6144,
                                   kC = 7168;
                constexpr uint32_t kSign = 0x80000000u;  // exact negation of an fp32 bit pattern
                float e_cum = 1.0f;  // E[t] = prod dec[0..t]
                for (uint32_t t = 0; t < T; ++t) {
                    e_cum = e_cum * dec[t];
                    const uint32_t row = r0 + t;
                    // G[t][i] = prod dec[i+1..t] for i < t (descending i), G[t][t] = 1
                    float g = 1.0f;
                    c32[kG + tile_elem_index(row, row)] = 0x3F800000u;
                    for (uint32_t i = t; i-- > 0;) {
                        g = g * dec[i + 1];
                        c32[kG + tile_elem_index(row, r0 + i)] = bits_from_f(g);
                        c32[kNGB + tile_elem_index(row, r0 + i)] = bits_from_f(beta[t] * g) ^ kSign;
                    }
                    c32[kEye + tile_elem_index(row, row)] = 0x3F800000u;
                    const uint32_t eb = bits_from_f(e_cum);
                    const uint32_t bb = beta_all[t];
                    const uint32_t nbe = bits_from_f(beta[t] * e_cum) ^ kSign;
                    const uint32_t wb = (t < r_rep) ? bits_from_f(wrep[t]) : 0u;
                    for (uint32_t c = 0; c < 32; ++c) {
                        const uint32_t idx = tile_elem_index(row, c);
                        c32[kE + idx] = eb;
                        c32[kBeta + idx] = bb;
                        c32[kNBE + idx] = nbe;
                        c32[kW + idx] = wb;
                    }
                }
#if !GDN_SFPU_GATE
                const uint32_t cb = bits_from_f(crep);
                for (uint32_t i = 0; i < 1024; ++i) {
                    c32[kC + i] = cb;
                }
#else
                (void)kC;
#endif
            }
            coef.push_back(8);
            if (s == 0) {
                gnew.reserve_back(1);
                {
                    auto gn_lock = gnew.scoped_write_lock(1);
                    auto gn = gn_lock.template get_ptr<volatile uint32_t>();
                    // only column 0 is ever read (the next verify's writer and the host park replay take rows r0 ..
                    // r0 + r - 1 of it): zero that column, not the page (a 1024-store fill sat on block 0's
                    // critical path)
                    for (uint32_t r = 0; r < 32; ++r) {
                        gn[tile_elem_index(r, 0)] = 0;
                    }
                    for (uint32_t t = 0; t < T; ++t) {
                        gn[tile_elem_index(r0 + t, 0)] = dec_all[t];
                    }
                }
                gnew.push_back(1);  // writer-local: consumed (written to the stash) after the compute's S'
            }
        }
#endif
    }
#endif


#if !GDN_PREFIXED
    // ---- window: win[1 - par] <- rebuilt rows (or the held user's rows copied through)
    {
        DeviceZoneScopedN("wr_win");
        if (!window_writer) {
            wout.wait_front(ChA);  // value blocks > 0: block 0 writes the head's window rows
            wout.pop_front(ChA);
        } else if (hold) {
            bounce.reserve_back(Ch);  // the whole ring (only the first ChA slots are used): keeps its pointer aligned
            for (uint32_t c = 0; c < ChA; ++c) {
                if (!owns(blk_chunk(c))) {
                    continue;
                }
                const uint32_t page = u * Ct + chunk_col(blk_chunk(c));
                if (par) {
                    read_tiles_at(wb_acc, bounce, noc, page, 1, c);
                } else {
                    read_tiles_at(wa_acc, bounce, noc, page, 1, c);
                }
            }
            noc.async_read_barrier();
            bounce.push_back(Ch);
            bounce.wait_front(Ch);
            for (uint32_t c = 0; c < ChA; ++c) {
                if (!owns(blk_chunk(c))) {
                    continue;
                }
                const uint32_t page = u * Ct + chunk_col(blk_chunk(c));
                if (par) {
                    write_whole_tile(wa_acc, bounce, noc, page, c);
                } else {
                    write_whole_tile(wb_acc, bounce, noc, page, c);
                }
            }
            noc.async_write_barrier();
            bounce.pop_front(Ch);
            wout.wait_front(ChA);  // drain the compute's window tiles without writing them
            wout.pop_front(ChA);
        } else {
            for (uint32_t c = 0; c < ChA; ++c) {
                if (!owns(blk_chunk(c))) {
                    wout.wait_front(1);
                    wout.pop_front(1);
                    continue;
                }
                const uint32_t page = u * Ct + chunk_col(blk_chunk(c));
                if (par) {
                    write_rows(wa_acc, wout, noc, page, 1, 0, Lw_w);
                } else {
                    write_rows(wb_acc, wout, noc, page, 1, 0, Lw_w);
                }
            }
        }
    }
#else
    // GDN_STEP_U1 prototype: the window rows are owned by the prefix phase (no wout from the compute)
    (void)wa_acc;
    (void)wb_acc;
    (void)wout;
    (void)bounce;
    (void)window_writer;
    (void)blk_chunk;
    (void)owns;
    (void)chunk_col;
    (void)hold;
    (void)Lw_w;
    (void)par;
#endif  // !GDN_PREFIXED

#if GDN_QKSHARE
    // ---- the q/k conv tiles: this block's Qk tiles (head q/k chunks [s*Qk, (s+1)*Qk), all in q or all in k) go to the
    //      same tile offset of qc / kc on every block of the head (itself included), exact fp32 bytes, unicast; then
    //      `qkv` goes up on every block and, once all VB blocks' tiles are in, qc / kc are handed to the compute
    {
        DeviceZoneScopedN("wr_qkx");
        DataflowBuffer qkm(dfb::qkm);
        Semaphore qkv(sem::qkv);
        qkm.wait_front(Qk);
        const uint32_t entry = qkm.get_entry_size();
        const uint32_t first = s * Qk;
        const uint32_t dst = (first < Kt) ? qc_base + first * entry : kc_base + (first - Kt) * entry;
#if GDN_LAZY
        lz_xchg<VB>(noc, qkm, 0, Qk * entry, dst, ready, qkv, peer_x, peer_y);
        qkm.pop_front(Qk);
#else
        ready.wait(VB);  // every block of the head runs this program and has reserved its qc / kc
        for (uint32_t p = 0; p < VB; ++p) {
            noc.async_write(
                qkm,
                UnicastEndpoint{},
                Qk * entry,
                {.offset_bytes = 0},
                {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = dst});
        }
        noc.async_write_barrier();  // the tiles landed before `qkv` is raised
        qkm.pop_front(Qk);
        for (uint32_t p = 0; p < VB; ++p) {
            qkv.up(noc, peer_x[p], peer_y[p], 1);
        }
        qkv.wait(VB);  // every block's q/k tiles are in this block's qc / kc
#endif
        qc_x.push_back(Kt);
        kc_x.push_back(Kt);
    }
#endif

#if GDN_SFPU_GATE && !GDN_R16_EARLY_COEF && !GDN_PREFIXED_GATES
    // ---- SFPU_GATE: the compute's gate tiles (it evaluates them after its phase A) and the coefficient tiles, after the
    //      window write and the q/k exchange (which phase B' waits on); the coefficients are first read at the replay
    {
        DeviceZoneScopedN("wr_coef");
        {
            DeviceZoneScopedN("wr_gwait");
            DataflowBuffer gt(dfb::gt);
            gt.wait_front(2);
            invalidate_l1_cache();  // packed by the compute: no stale RISC cache line
            {
                auto gt_lock = gt.scoped_read_lock(2);
                auto q32 = gt_lock.template get_ptr<volatile uint32_t>();
                for (uint32_t t = 0; t < T; ++t) {
                    dec_all[t] = q32[tile_elem_index(r0 + t, h & 31u)];
                    beta_all[t] = q32[1024 + tile_elem_index(r0 + t, (Nv + h) & 31u)];
                }
            }
            gt.pop_front(2);
        }
#if GDN_LAZY
        // ---- K4: the WY coefficient tiles and this verify's decay page, from all T gate pairs (with GDN_GSC_SPLIT the
        //      head's blocks computed a share each and swapped the slots above)
        {
            float beta[T], dec[T];
            for (uint32_t t = 0; t < T; ++t) {
                beta[t] = f_from_bits(beta_all[t]);
                dec[t] = f_from_bits(dec_all[t]);
            }
            // replay of the previous verify's accepted prefix: c = prod dec_prev[0..r), w[i] = prod dec_prev[i+1..r)
#if !GDN_SFPU_GATE  // (SFPU_GATE: computed above, before the gates)
            float wrep[T];
            float crep = 1.0f;
            if (r_rep > 0) {
                auto gp_lock = gprev.scoped_write_lock(1);
                auto gp = gp_lock.template get_ptr<volatile uint32_t>();
                float acc = 1.0f;
                for (uint32_t i = r_rep; i-- > 0;) {
                    wrep[i] = acc;  // prod over j in (i, r)
                    acc = acc * f_from_bits(gp[tile_elem_index(r0 + i, 0)]);
                }
                crep = acc;
            }
#endif
            // the compute only multiplies and adds (no SFPU subtract / fill routine in its binary): the subtracted
            // terms come negated (-beta G, -beta E) and the replay's decay c as a whole tile
            DataflowBuffer coef(dfb::coef);
            coef.reserve_back(8);
            zero_reserved(coef, noc, GDN_SFPU_GATE ? 7 : 8);  // SFPU_GATE: C (tile 7) is filled by the compute
            {
                auto c_lock = coef.scoped_write_lock(8);
                auto c32 = c_lock.template get_ptr<volatile uint32_t>();
                constexpr uint32_t kNGB = 0, kG = 1024, kE = 2048, kBeta = 3072, kNBE = 4096, kEye = 5120, kW = 6144,
                                   kC = 7168;
                constexpr uint32_t kSign = 0x80000000u;  // exact negation of an fp32 bit pattern
                float e_cum = 1.0f;  // E[t] = prod dec[0..t]
                for (uint32_t t = 0; t < T; ++t) {
                    e_cum = e_cum * dec[t];
                    const uint32_t row = r0 + t;
                    // G[t][i] = prod dec[i+1..t] for i < t (descending i), G[t][t] = 1
                    float g = 1.0f;
                    c32[kG + tile_elem_index(row, row)] = 0x3F800000u;
                    for (uint32_t i = t; i-- > 0;) {
                        g = g * dec[i + 1];
                        c32[kG + tile_elem_index(row, r0 + i)] = bits_from_f(g);
                        c32[kNGB + tile_elem_index(row, r0 + i)] = bits_from_f(beta[t] * g) ^ kSign;
                    }
                    c32[kEye + tile_elem_index(row, row)] = 0x3F800000u;
                    const uint32_t eb = bits_from_f(e_cum);
                    const uint32_t bb = beta_all[t];
                    const uint32_t nbe = bits_from_f(beta[t] * e_cum) ^ kSign;
                    const uint32_t wb = (t < r_rep) ? bits_from_f(wrep[t]) : 0u;
                    for (uint32_t c = 0; c < 32; ++c) {
                        const uint32_t idx = tile_elem_index(row, c);
                        c32[kE + idx] = eb;
                        c32[kBeta + idx] = bb;
                        c32[kNBE + idx] = nbe;
                        c32[kW + idx] = wb;
                    }
                }
#if !GDN_SFPU_GATE
                const uint32_t cb = bits_from_f(crep);
                for (uint32_t i = 0; i < 1024; ++i) {
                    c32[kC + i] = cb;
                }
#else
                (void)kC;
#endif
            }
            coef.push_back(8);
            if (s == 0) {
                gnew.reserve_back(1);
                {
                    auto gn_lock = gnew.scoped_write_lock(1);
                    auto gn = gn_lock.template get_ptr<volatile uint32_t>();
                    // only column 0 is ever read (the next verify's writer and the host park replay take rows r0 ..
                    // r0 + r - 1 of it): zero that column, not the page (a 1024-store fill sat on block 0's
                    // critical path)
                    for (uint32_t r = 0; r < 32; ++r) {
                        gn[tile_elem_index(r, 0)] = 0;
                    }
                    for (uint32_t t = 0; t < T; ++t) {
                        gn[tile_elem_index(r0 + t, 0)] = dec_all[t];
                    }
                }
                gnew.push_back(1);  // writer-local: consumed (written to the stash) after the compute's S'
            }
        }
#endif
    }
#endif

    // ---- output padding rows: the last user's cores zero rows [r0 + T, 32) of the head's Vt output tiles from a
    //      zeroed bounce region (4 KiB covers every in-tile offset of an fp32 tile). Equal source / destination
    //      offsets keep both addresses in the same alignment class, as write_rows / write_rows_single do.
    if (u == Br - 1 && r0 + T < 32) {
        bounce.reserve_back(2);
        zero_reserved(bounce, noc, 2);
        bounce.push_back(2);
        bounce.wait_front(2);
        const uint32_t entry = out.get_entry_size();
        const uint32_t esz = entry / 1024;
        const uint32_t seg = entry / 64;
        for (uint32_t t = 0; t < Vs; ++t) {
            const uint32_t page = tr * out_w_tiles + h * Vt + v0 + t;
            for (uint32_t lo = r0 + T; lo < 32;) {
                const uint32_t hi = (lo < 16) ? 16u : 32u;
                for (uint32_t half = 0; half < 2; ++half) {
                    const uint32_t off = tile_elem_index(lo, half * 16) * esz;
                    noc.async_write(
                        bounce,
                        out_acc,
                        seg * (hi - lo),
                        {.offset_bytes = off},
                        {.page_id = page, .offset_bytes = off});
                }
                lo = hi;
            }
        }
        noc.async_write_barrier();
        bounce.pop_front(2);
    }

#if GDN_LAZY
    // ---- K4: S' -> ring slot 0 block (u, h) (this block's columns), then this verify's stash -> slot 1 + (1 - par):
    //      K^T tiles (block 0), this block's U tiles, the decay page dec[t] at (r0 + t, 0) (block 0); HOLD writes none
    // (GDN_BATCH: run after the exchange and the out rows -- the per-user stash writes are long and would hold the
    // head's acc exchange, i.e. the gated norm, behind them; in the engine they trail the layer)
    auto wr_lazy_fn = [&]() {
        DeviceZoneScopedN("wr_lazy");
        DataflowBuffer dnew(dfb::dnew);
        DataflowBuffer kt_out(dfb::kt_out);
        DataflowBuffer u_out(dfb::u_out);
        const uint32_t entry = dnew.get_entry_size();
#if GDN_BATCH
        // every user of head h: S'_u -> slot 0 block (u, h) (this block's columns); the stash -> slot 1 + (1 - par)
        // block (u, h): K^T_u (block 0), U_u (this block's tiles), the decay page (block 0: column 0 = dec of user u's
        // rows, from the gate page; the rest of the page is not written by anyone, as in the per-user program)
        // per (u, h): S'_u -> slot 0 (this block's columns); stash slot 1 + (1 - par): U_u (this block's tiles), K^T_u
        // pages i = s, s + VB, .. (the compute's kmq transposes), the decay page of users u = s mod VB. U_u / decay pages
        // are built in one zeroed staging region (local NoC row copies), then every page goes out with one barrier.
        DataflowBuffer gprev(dfb::gprev);  // the gate page (dec at column 0, all users' rows)
        DataflowBuffer stg(dfb::stg);      // [U_u (kBatchMax * Vs) | decay_u (kBatchMax)] staging pages
        constexpr uint32_t kpb = (Kt + VB - 1) / VB;
        u_out.wait_front(Vs);
        kt_out.wait_front(kBatchMax * kpb);
        // [U_u (kBatchMax * Vs) | the decay pages of users u = s mod VB (kStgDec)]
        constexpr uint32_t kStgDec = (kBatchMax + VB - 1) / VB;
        stg.reserve_back(kBatchMax * Vs + kStgDec);
        const uint32_t sb = stg.get_write_ptr();
        const uint32_t utb = u_out.get_read_ptr();
        {
            const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t o = 0; o < (kBatchMax * Vs + kStgDec) * entry; o += MEM_ZEROS_SIZE) {
                noc_async_read(z, sb + o, MEM_ZEROS_SIZE);
            }
            noc_async_read_barrier();
        }
        invalidate_l1_cache();
        {
            auto gp_lock = gprev.scoped_write_lock(1);
            auto gp = gp_lock.template get_ptr<volatile uint32_t>();
            for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
                const uint32_t r0u = ub_ * T;
                const uint32_t off = ((r0u >> 4) * 512 + (r0u & 15) * 16) * 4;
                for (uint32_t j = 0; j < Vs; ++j) {
                    const uint32_t src = utb + j * entry + off, dst = sb + (ub_ * Vs + j) * entry + off;
                    noc_async_read(get_noc_addr(src), dst, T * 64);
                    noc_async_read(get_noc_addr(src + 1024), dst + 1024, T * 64);
                }
                if (ub_ % VB == s) {
                    volatile uint32_t* gn = reinterpret_cast<volatile uint32_t*>(sb + (kBatchMax * Vs + ub_ / VB) * entry);
                    for (uint32_t t = 0; t < T; ++t) {
                        gn[tile_elem_index(r0u + t, 0)] = gp[tile_elem_index(r0u + t, 0)];
                    }
                }
            }
            noc_async_read_barrier();
        }
        for (uint32_t ub_ = 0; ub_ < nb; ++ub_) {
            if (hold_b[ub_]) {
                continue;
            }
            const uint32_t sw_u = ((1 + (1 - par)) * Mr + ub_) * Nv + h;
            for (uint32_t j = 0; j < Vs; ++j) {
                noc.async_write(stg, ring_acc, entry, {.offset_bytes = (ub_ * Vs + j) * entry},
                                {.page_id = sw_u * KV + Kt + v0 + j});
            }
            for (uint32_t q = 0; q < kpb; ++q) {
                const uint32_t i = s + q * VB;
                if (i < Kt) {
                    noc.async_write(kt_out, ring_acc, entry, {.offset_bytes = (ub_ * kpb + q) * entry},
                                    {.page_id = sw_u * KV + i});
                }
            }
            if (ub_ % VB == s) {
                noc.async_write(stg, ring_acc, entry, {.offset_bytes = (kBatchMax * Vs + ub_ / VB) * entry},
                                {.page_id = sw_u * KV + Kt + Vt});
            }
        }
        noc.async_write_barrier();
        u_out.pop_front(Vs);
        kt_out.pop_front(kBatchMax * kpb);
        (void)gnew;
#else
        dnew.wait_front(KVs);
        if (!hold) {
            const uint32_t page0 = bh * KV + v0;
            for (uint32_t i = 0; i < Kt; ++i) {
                for (uint32_t j = 0; j < Vs; ++j) {
                    noc.async_write(
                        dnew, ring_acc, entry, {.offset_bytes = (i * Vs + j) * entry}, {.page_id = page0 + i * Vt + j});
                }
            }
        }
        if (s == 0) {
            kt_out.wait_front(Kt);
            gnew.wait_front(1);  // built in the gate block
            if (!hold) {
                for (uint32_t i = 0; i < Kt; ++i) {
                    noc.async_write(
                        kt_out, ring_acc, entry, {.offset_bytes = i * entry}, {.page_id = stash_wr * KV + i});
                }
                noc.async_write(gnew, ring_acc, entry, {.offset_bytes = 0}, {.page_id = stash_wr * KV + Kt + Vt});
            }
        }
        u_out.wait_front(Vs);
        if (!hold) {
            for (uint32_t j = 0; j < Vs; ++j) {
                noc.async_write(
                    u_out, ring_acc, entry, {.offset_bytes = j * entry}, {.page_id = stash_wr * KV + Kt + v0 + j});
            }
        }
        noc.async_write_barrier();
        dnew.pop_front(KVs);
        u_out.pop_front(Vs);
        if (s == 0) {
            kt_out.pop_front(Kt);
            gnew.pop_front(1);
        }
#endif  // GDN_BATCH
    };
#if !GDN_BATCH
    wr_lazy_fn();
#endif
#else
    // ---- ring: the post-token state of every token, token-major block (t*BH + bh); skipped for held heads. This
    //      block's tile (i, j) of hnew (index i*Vs + j) is tile (i, v0 + j) of the [Kt, Vt] block (VB = 1: tile i*Vt +
    //      j)
    {
        const uint32_t entry = hnew.get_entry_size();
#ifndef GDN_RING_SPLIT
#define GDN_RING_SPLIT 0
#endif
        // RING_SPLIT (kernel_opts bit 18): the even tokens only; the reader writes the odd tokens' states from hnew2
        constexpr uint32_t t_step = GDN_RING_SPLIT ? 2u : 1u;
        for (uint32_t t = 0; t < T; t += t_step) {
            {
                DeviceZoneScopedN("wr_wait_hnew");
                hnew.wait_front(KVs);
            }
            DeviceZoneScopedN("wr_ring");
            if (!hold) {
                const uint32_t page0 = (t * BH + bh) * KV + v0;
                for (uint32_t i = 0; i < Kt; ++i) {
                    for (uint32_t j = 0; j < Vs; ++j) {
                        noc.async_write(
                            hnew,
                            ring_acc,
                            entry,
                            {.offset_bytes = (i * Vs + j) * entry},
                            {.page_id = page0 + i * Vt + j});
                    }
                }
                noc.async_write_barrier();
            }
            hnew.pop_front(KVs);
        }
    }
#endif

#if GDN_LAZY && GDN_BATCH && !GDN_SP_LATE
    // ---- S'_u -> ring slot 0 block (u, h) (this block's columns) straight out of the compute's sp (no dnew copy): sp
    //      is waited on (not popped: the compute pops it after K S' / Q S'), read at its ring offset for this layer, and
    //      `spd` tells the compute the reads have left L1
    {
        DeviceZoneScopedN("wr_sp");
        DataflowBuffer sp(dfb::sp);
        DataflowBuffer spd(dfb::spd);
        const uint32_t entry = sp.get_entry_size();
        sp.wait_front(nb * KVs);
#if GDN_LOOP
        const uint32_t lay = gdn_loop_k() - 1;
#else
        const uint32_t lay = 0;
#endif
        const uint32_t base_t = (lay * nb * KVs) % (kBatchMax * KVs);
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
    }
#endif
#if GDN_VSPLIT
    // ---- swap the last token's pre-norm output tiles within the head: acc_mine -> acc_x[v0 ..] of every block
    {
        DeviceZoneScopedN("wr_xchg");
        acc_mine.wait_front(Vs);
        const uint32_t entry = acc_mine.get_entry_size();
#if GDN_LAZY
        lz_xchg<VB>(noc, acc_mine, 0, Vs * entry, acc_x_base + v0 * entry, ready, valid, peer_x, peer_y);
        acc_mine.pop_front(Vs);
#else
        ready.wait(VB);  // every block of the head runs this program (its acc_x is ours to write)
        for (uint32_t p = 0; p < VB; ++p) {
            noc.async_write(
                acc_mine,
                UnicastEndpoint{},
                Vs * entry,
                {.offset_bytes = 0},
                {.noc_x = peer_x[p], .noc_y = peer_y[p], .addr = acc_x_base + v0 * entry});
        }
        noc.async_write_barrier();  // the tiles landed before `valid` is raised
        acc_mine.pop_front(Vs);
        for (uint32_t p = 0; p < VB; ++p) {
            valid.up(noc, peer_x[p], peer_y[p], 1);
        }
        valid.wait(VB);  // every block's tiles are in this block's acc_x
#endif
        acc_x.push_back(Vt);
    }
#endif

    // ---- the user's T output rows of the head's Vt output tiles
    DeviceZoneScopedN("wr_out");
#ifndef GDN_OUT_RD
#define GDN_OUT_RD 0  // engine GDN role: the reader (NOC1) writes the out tiles and raises the out signal (reader source)
#endif
#if GDN_OUT_RD
    static_assert(GDN_BATCH && GDN_LAZY && !GDN_OUT_LOCAL, "GDN_OUT_RD: engine role (BATCH + LAZY), not with GDN_OUT_LOCAL yet");
    if (true) {
        // nothing: the reader owns out (wait / write / pop / signal)
    } else
#endif
#if GDN_OUT_LOCAL
    out.wait_front(Vs);
    gdn_out_local(out.get_read_ptr(), h * Vt + v0, Vs);  // engine gdn_out: full out tiles to every GDN core's A, local
#endif
#if GDN_BATCH && GDN_OUT_LOCAL == 2
    if (true) {
        out.pop_front(Vs);
    } else
#endif
#if GDN_BATCH
#ifndef GDN_OUT_TILE
#define GDN_OUT_TILE 0  // engine GDN role: all 32 rows live (nb * T == 32) -> one whole-tile write per out tile, not 4 face-row spans
#endif
    if (GDN_OUT_TILE && nb * T == 32) {
        {
            DeviceZoneScopedN("wr_owait");  // the compute's gated output tiles
            out.wait_front(Vs);
        }
        DeviceZoneScopedN("wr_owr");  // the out tile write + its barrier
        const uint32_t entry = out.get_entry_size();
        for (uint32_t t = 0; t < Vs; ++t) {
            noc.async_write(out, out_acc, entry, {.offset_bytes = t * entry}, {.page_id = tr * out_w_tiles + h * Vt + v0 + t});
        }
        noc.async_write_barrier();
        out.pop_front(Vs);
    } else if (true) {
        write_rows(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, 0, nb * T);
    } else
#endif
    if constexpr ((T % 2) == 0) {
        write_rows(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, r0, T);
    } else {
        write_rows_single(out_acc, out, noc, tr * out_w_tiles + h * Vt + v0, Vs, r0, T);
    }
#if GDN_LAZY && GDN_BATCH
    {
        DeviceZoneScopedN("wr_obar");  // the out rows' write barrier + the out signal (wr_out spans to the end)
#if !GDN_OUT_RD
        noc.async_write_barrier();
#if GDN_OUT_SIGNAL
        gdn_out_signal();  // engine GDN role: this core's output rows have landed (the stash writes trail)
#endif
#endif
    }
#ifndef GDN_SP_RD
#define GDN_SP_RD 0
#endif
#if GDN_SP_LATE && !GDN_SP_RD
    // GDN_SP_LATE (engine GDN role): S' leaves sp only now, behind the output rows and the out signal (the compute pops
    // sp at the next layer's replay / program end, after spd)
    // ---- S'_u -> ring slot 0 block (u, h) (this block's columns) straight out of the compute's sp (no dnew copy): sp
    //      is waited on (not popped: the compute pops it after K S' / Q S'), read at its ring offset for this layer, and
    //      `spd` tells the compute the reads have left L1
    {
        DeviceZoneScopedN("wr_sp");
        DataflowBuffer sp(dfb::sp);
        DataflowBuffer spd(dfb::spd);
        const uint32_t entry = sp.get_entry_size();
        sp.wait_front(nb * KVs);
#if GDN_LOOP
        const uint32_t lay = gdn_loop_k() - 1;
#else
        const uint32_t lay = 0;
#endif
        const uint32_t base_t = (lay * nb * KVs) % (kBatchMax * KVs);
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
    }
#endif
#if GDN_LAZY_GATE
    gdn_lazy_gate();  // E1_LAZY_GATE: the stash writes (NOC0 DRAM) wait until every head's exchange is past
#endif
    wr_lazy_fn();
#endif
#if GDN_VSPLIT
    noc.async_atomic_barrier();  // retire the ready / valid increments before exit
#endif
}
