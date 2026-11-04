// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequence-sharded DSA state + split-K decode over the C chips of a TP mesh (tt/dsa_shard.py, tt/tp_dsa_shard.py; the
// MLA_DSA_SHARD paths of kernels/mla/att_dsa_dm0.cpp, att_dsa_dm1.cpp, att_c.cpp, ix_dm0.cpp, ix_dm1.cpp, lat_dm1.cpp).
// Unit = 128 tokens = one pooled-key page; unit u lives on chip u mod C. The chip count C and this chip's rank r are
// RUNTIME args (one program per mesh; the rank differs per chip). Every chip:
//   writes the new token's latent row / pooled key only when it owns the position (local position / local pool);
//   I4 scores ITS closed pools (local ids), I5 selects the exact local top-512 (the radix select, unchanged), then
//   the group-0 chunk cores write their selected (key, core) records, packed by the coordinator's offsets, into this
//   token's candidate chunk of the all-gather block x_ag; the coordinator (ix_dm0) waits for the gathered y_ag, merges
//   the C local top-512 lists into the exact global top-512 (threshold T_g; ties in (chip, core, local) order) and
//   broadcasts (T_g, k_rem) + per-core tie quotas as a 10th mailbox message; the owners keep their local selection
//   that survives T_g;
//   S1 runs TWICE per token (pass 0: the PEER's heads with q from y_ag; pass 1: this chip's heads with q from q_lat),
//   each pass over the chip's own selected pools (+ the open pool's tail rows on the owner of the open pool), through
//   the group's partial tree as today; the root of pass 0 ships its partial (m, l, O: the tree's fp32 message) to the
//   peer through the reduce-scatter block x_rs; the root of pass 1 merges the peer's partial of ITS heads (combine) and
//   finalises o = O / l.
// Decode rows per launch T <= T_MAX = 32 (rev 60, W: the native B=32 launch and widths 8 / 16), one head group per
// chip. T > 4 runs TOKEN LANES (att_common.hpp LANES = L, tt/mk_mla.py att_lanes): token t belongs to lane t % L, its
// lane's index core (ix_dm0 / ix_c / ix_dm1 on the lane's tokens) coordinates the local select over the lane's C / L
// chunk cores and runs the global merge of the lane's tokens on the lane's merge helpers (its chunk cores 0 ..
// NCHIPS - 1); the lane's root ships / lands the token's cross-chip partials. The all-gather / reduce-scatter blocks
// stay indexed by the launch token t; the owners' mailbox, the kept pairs and CB_ASEL by the LANE-LOCAL token j = t / L
// (< TL = mladsa::T_MAX: 4 at t <= 4, TL_MAX = 8 at t > 4 -- tt/mk_mla.py shard_tl).
// TP8 on the 2x4 mesh (MLA_DSA_SHARD_LINE = 4, tt/tp_dsa_shard.py; doc/ttnn/14-production-tp8-megakernel/dsa/
// shard_tp8_design.md): rank r = 4 row + col (8 heads [8 r, 8 r + 8)); the two 32-head q tiles are the mesh rows
// (tile g = ranks [4 g, 4 g + 4), chip (g, j) = rows 8 j .. 8 j + 7), each assembled from the 4 chips' 8-row q pieces
// of y_ag; pass 0 attends with the OTHER row's tile, pass 1 with this chip's row's tile. The all-gather runs in two
// rounds (axis 0: the column pair; axis 1: the row line of 4 multicasts its 2-block bundle), the reduce-scatter in two
// rounds with a combine after each (axis 0: the pass-0 partial to the column peer, combined with the own pass-1 partial
// -> the column-combined partial of the row's 32 heads; axis 1: each row peer's 8-row slice of it, a per-destination
// unicast scatter; the head owner combines its own slice + the 3 landed ones in column order, then finalises).
// The pair (TP2, MLA_DSA_SHARD_LINE unset = 1): one head tile per chip, one round each (the validated 6043dff8af form).
#pragma once

#include <cstdint>

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/att_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mla/dsa_common.hpp"

namespace mlashard {
constexpr uint32_t UNIT = 128;      // tokens per shard unit (= KPOOL x POOL_PAGE)
constexpr uint32_t POOL_PAGE = 32;  // pools per pooled-key page
constexpr uint32_t TB = 2048;       // bf16 tile bytes (one DRAM page of the exchange tensors)
constexpr uint32_t PPC = 16;        // pages per 32-row chunk of a [., 512] bf16 tensor
constexpr uint32_t REC_BYTES = 16;  // candidate record: key, chunk core, 0, 0 (16-B NoC granularity)
constexpr uint32_t REC_HDR = 16;    // record chunk header: word 0 = this chip's record count (the coordinator's)
constexpr uint32_t RS_CHUNKS = 3;   // 32-row chunks per (destination chip, token) of x_rs / y_rs
constexpr uint32_t RS_PPC = 12;   // rev 59: pages per x_rs chunk ([., 384] bf16): 3 x 12 = 36 pages = the fp32 message
constexpr uint32_t ST_TILES = 4;  // (the first form's bf16 hi + lo stats chunk; unused since the fp32 tree message)
constexpr uint32_t T_MAX = 32;    // decode rows per launch this variant serves (rev 60; tt/mk_mla.py SHARD_T_MAX)
constexpr uint32_t TL_MAX = 8;    // rev 60: tokens per token lane at t > 4 (mailbox slots, kept blocks, CB_ASEL pages)
constexpr uint32_t CAND_MAX = mladsa::TOPK;    // records per chip per token (the local top-512)
constexpr uint32_t MSG_GLOBAL = mladsa::MSGS;  // the 10th mailbox message of a token (T_g, k_rem)
// the shard mailbox serves the lane-local tokens j < TL = mladsa::T_MAX (the host's MLA_DSA_TMAX on a shard build: 4 at
// t <= 4, TL_MAX at t > 4); the kept blocks / the T_g probe words follow it
constexpr uint32_t TL = mladsa::T_MAX;
static_assert(TL == 4 || TL == TL_MAX, "shard: the lane-local token slots (tt/mk_mla.py shard_tl)");
static_assert(mlaatt::LANES >= 1 && (mlaatt::LANES & (mlaatt::LANES - 1)) == 0, "token lanes: a power of two");
// rev 60: the lane-local index of launch token t (token t runs on lane t % LANES) and the lane's token count
inline uint32_t lane_tok(uint32_t t) { return t / mlaatt::LANES; }
inline uint32_t lane_tokens(uint32_t t_rows, uint32_t t0) {
    return t0 < t_rows ? (t_rows - t0 + mlaatt::LANES - 1) / mlaatt::LANES : 0;
}
// mailbox extension (after dsa_common.hpp's MBOX_BYTES): the local tie message carries (quota, offset) pairs
// (QUOTA_W = 2 words per core, dsa_common.hpp), then per token the global message at GMSG_OFF + t 16 and the global
// tie quotas (one word per chunk core) at GQUOTA_OFF + t MAX_CHUNKS 4
constexpr uint32_t GMSG_OFF = mladsa::MBOX_BYTES;
constexpr uint32_t GQUOTA_OFF = GMSG_OFF + mladsa::T_MAX * mladsa::MSG_BYTES;
// rev 59: the global merge's messages to the merge helpers (owner chunk cores c < NCHIPS of group 0; helper q holds
// chip q's candidate list): per token MERGE_MSGS 16-B slots -- 0 start, 1 + r round r's (prefix, k_rem, done), then the
// tie quotas of the NCHIPS helpers (2 slots: 8 words)
constexpr uint32_t MERGE_MSGS = 1 + mladsa::ROUNDS + 2;
constexpr uint32_t MERGE_OFF = GQUOTA_OFF + mladsa::T_MAX * mladsa::MAX_CHUNKS * 4;
constexpr uint32_t MBOX_SHARD_BYTES = MERGE_OFF + mladsa::T_MAX * MERGE_MSGS * mladsa::MSG_BYTES;
inline uint32_t merge_msg(uint32_t t, uint32_t i) { return MERGE_OFF + (t * MERGE_MSGS + i) * mladsa::MSG_BYTES; }

inline uint32_t owner(uint32_t p, uint32_t C) { return (p / UNIT) % C; }
inline uint32_t local_pos(uint32_t p, uint32_t C) { return (p / UNIT / C) * UNIT + p % UNIT; }
// rows of chip r among positions 0..p (the dense-exact path attends them all)
inline uint32_t n_local_rows(uint32_t p, uint32_t r, uint32_t C) {
    const uint32_t n = p + 1, fu = n / UNIT, rem = n % UNIT;
    const uint32_t full = fu > r ? (fu - r + C - 1) / C : 0;
    return full * UNIT + (fu % C == r ? rem : 0);
}
// closed pools (< (p + 1) / 4) that live on chip r
inline uint32_t n_closed_local(uint32_t p, uint32_t r, uint32_t C) {
    const uint32_t n = (p + 1) / mladsa::KPOOL, fu = n / POOL_PAGE, rem = n % POOL_PAGE;
    const uint32_t full = fu > r ? (fu - r + C - 1) / C : 0;
    return full * POOL_PAGE + (fu % C == r ? rem : 0);
}
inline uint32_t local_pool(uint32_t g, uint32_t C) { return (g / POOL_PAGE / C) * POOL_PAGE + g % POOL_PAGE; }
// x_ag / y_ag, the pair (TP8: the compact AG8 format below): a chip's block = 2 T chunks of 32 rows x 512 bf16 (x_ag
// holds ONE block -- the chip's own, blk = 0 on every chip; y_ag the n gathered blocks, blk = the source chip's
// position): chunk t = q_lat's tile row of token
// t (this chip's heads: the q_lat DRAM copy itself), chunk T + t = token t's candidate records (raw bytes over the
// chunk's 16 pages: byte b at page b / TB, offset b % TB; the header at byte 0, record i at REC_HDR + i REC_BYTES).
// y_ag = the n blocks.
inline uint32_t ag_q_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t d) { return (blk * 2 * T + t) * PPC + d; }
inline uint32_t ag_cand_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t byte) {
    return (blk * 2 * T + T + t) * PPC + byte / TB;
}
// x_rs / y_rs: block q (= destination chip position) x token t x RS_CHUNKS chunks of RS_PPC pages (2 KiB): the root's
// final partial of pass 0 as the tree's own fp32 message -- the O tiles (DT x otb bytes, otb / TB pages each) from page
// 0, then the stats tiles (ST x ptb bytes) after them. y_rs = one block (this chip's heads, from the peer).
inline uint32_t rs_base(uint32_t T, uint32_t blk, uint32_t t) { return (blk * T + t) * RS_CHUNKS * RS_PPC; }
constexpr uint32_t RS_PAGES = RS_CHUNKS * RS_PPC;  // 36 pages = 72 KiB = 16 x 4 KiB (O) + 2 x 4 KiB (m, l) exactly

// ---- TP8 (the 2x4 mesh): MLA_DSA_SHARD_LINE = the chips of a mesh row (the axis-1 line); the pair: 1
#if defined(MLA_DSA_SHARD_LINE)
constexpr uint32_t LINE = MLA_DSA_SHARD_LINE;
#else
constexpr uint32_t LINE = 1;
#endif
static_assert(LINE == 1 || LINE == 4, "shard: the pair (1) or the 2x4 mesh rows (4)");
constexpr uint32_t NCHIPS = 2 * LINE;                            // the axis-0 pair x the row line
constexpr uint32_t PIECE_ROWS = 32 / LINE;                       // q / o rows (heads) of one chip in a 32-row head tile
inline uint32_t pair_pos(uint32_t rank) { return rank / LINE; }  // the mesh row (2x4) / the pair position (TP2)
inline uint32_t line_pos(uint32_t rank) { return rank % LINE; }  // the mesh column (2x4) / 0 (TP2)
// y_ag block of chip `rank`: the two chips of a column are adjacent (the round-2 bundle of column c = blocks 2 c,
// 2 c + 1); the pair: block = rank
inline uint32_t ag_blk(uint32_t rank) { return line_pos(rank) * 2 + pair_pos(rank); }
// the TP8 AG block: per token ONE chunk of AG8_PPC pages ([T x 32, 320] bf16 tensors): pages [0, AG8_QPAGES) = the
// chip's q piece (q_lat rows 0..7 of tile d = 2 face-row runs of 256 B at page d / 4, offset (d % 4) 512 (+ 256)),
// then the candidate records (header + CAND_MAX records = 5 pages), then a pad page (page pairs)
constexpr uint32_t AG8_PPC = 10;
constexpr uint32_t AG8_QPAGES = 4;
constexpr uint32_t CAND_PAGES = (REC_HDR + CAND_MAX * REC_BYTES + TB - 1) / TB;
static_assert(AG8_QPAGES + CAND_PAGES <= AG8_PPC && AG8_PPC % 2 == 0, "TP8 AG chunk");
constexpr uint32_t AG_PPC = LINE > 1 ? AG8_PPC : PPC;  // pages per AG chunk (the AG tensors' width / 32)
constexpr uint32_t AG_CHUNKS_PER_TOKEN = LINE > 1 ? 1 : 2;
// (the pair forms of ag_q_page / ag_cand_page above; these pick the format of this build)
inline uint32_t agx_q_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t d) {
    return LINE > 1 ? (blk * T + t) * AG8_PPC + d / 4 : ag_q_page(T, blk, t, d);
}
inline uint32_t agx_q_off(uint32_t d) { return LINE > 1 ? (d % 4) * 512 : 0; }
inline uint32_t agx_cand_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t byte) {
    return LINE > 1 ? (blk * T + t) * AG8_PPC + AG8_QPAGES + byte / TB : ag_cand_page(T, blk, t, byte);
}
// rows 8 j .. 8 j + 7 of a 32x32 tile of tb bytes (4 faces of tb / 4; a face row = tb / 64 bytes): two runs of tb / 8
// bytes, run A at slice_off(j, tb) (face 0 / 2), run B = run A + tb / 4 (face 1 / 3)
inline uint32_t slice_off(uint32_t j, uint32_t tb) { return (j / 2) * (tb / 2) + (j % 2) * (tb / 8); }
// RS round 2 (x_rs2 / y_rs2 [LINE x T x 32, 320] bf16): block q x token t = one chunk of RS2_PPC pages; x_rs2 block q
// = the slice for destination column q (the own column too: the root lands it from there), y_rs2 block s = the slice
// from source column s. A slice = rows 8 q .. of the column-combined partial: O tile d at byte d (otb / 4), the stats
// tile i at DT (otb / 4) + i (ptb / 4) (9 pages with fp32 O, 5 with bf16 O)
constexpr uint32_t RS2_PPC = 10;
inline uint32_t rs2_page(uint32_t T, uint32_t blk, uint32_t t) { return (blk * T + t) * RS2_PPC; }
// rev 59, TP8: q_asm [2 x T x 32, 512] bf16 -- head tile g of token t assembled ONCE per chip (the index core's NCRISC,
// behind the global merge) from the 4 pieces in y_ag; the owners read 16 whole pages per push
inline uint32_t qasm_page(uint32_t T, uint32_t tile, uint32_t t, uint32_t d) { return (tile * T + t) * PPC + d; }
// rev 59: the merge helpers' scratch in CB_AKEY: their chip's candidate pages, its keys, the compacted candidates, the
// per-core tie quotas (helper `rank` only). Rev 60 (review 2 item 1): it overlays the phase-A keys / candidates
// (dead once every lane token's local select is done; the merge runs after the all-gather), so CB_AKEY holds
// max(2 MLA_DSA_KEYS words, MERGE_SCR_BYTES + 64) there instead of both
constexpr uint32_t MERGE_SCR_BYTES = CAND_PAGES * TB + 2 * CAND_MAX * 4 + mladsa::MAX_CHUNKS * 4;
}  // namespace mlashard
