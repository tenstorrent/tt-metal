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
// ---- TP8 (the 2x4 mesh): MLA_DSA_SHARD_LINE = the chips of a mesh row (the axis-1 line); the pair: 1
// ---- TP32 (the 8x4 mesh, doc/perf/shard_8x4_design.md): MLA_DSA_SHARD_PAIR = the chips of a mesh column (the axis-0
// line, 8); unset = 2 (the pair / the 2x4 column pair: every build before the port is unchanged)
#if defined(MLA_DSA_SHARD_LINE)
constexpr uint32_t LINE = MLA_DSA_SHARD_LINE;
#else
constexpr uint32_t LINE = 1;
#endif
#if defined(MLA_DSA_SHARD_PAIR)
constexpr uint32_t PAIR = MLA_DSA_SHARD_PAIR;
#else
constexpr uint32_t PAIR = 2;
#endif
static_assert(LINE == 1 || LINE == 4, "shard: the pair (1) or the mesh rows of 4 chips (4)");
static_assert(PAIR == 2 || (PAIR == 8 && LINE == 4), "shard: the pair / the 2x4 column pair (2) or the 8x4 column (8)");
constexpr uint32_t NCHIPS = PAIR * LINE;                         // the axis-0 line x the axis-1 line
constexpr uint32_t HPC = 64 / NCHIPS;                            // heads per chip (32 / 8 / 2)
constexpr uint32_t PIECE_ROWS = NCHIPS > 2 ? 64 / NCHIPS : 32;   // q / o rows (heads) of one chip in a 32-row head tile
constexpr uint32_t PIECES = NCHIPS > 2 ? NCHIPS / 2 : 1;         // chips whose pieces form one head tile (4 / 16)
constexpr uint32_t RS1_ROWS = PAIR > 2 ? 64 / PAIR : 32;         // 8x4 round 1: the column slice rows per chip (8)
constexpr uint32_t RS2_ROWS = RS1_ROWS / LINE;                   // round 2: the row slice rows (2x4: 8; 8x4: 2)
inline uint32_t pair_pos(uint32_t rank) { return rank / LINE; }  // the mesh row (2x4 / 8x4) / the pair position (TP2)
inline uint32_t line_pos(uint32_t rank) { return rank % LINE; }  // the mesh column (2x4 / 8x4) / 0 (TP2)
inline uint32_t tile_of(uint32_t rank) { return pair_pos(rank) / (PAIR / 2); }     // the chip's 32-head tile (0 / 1)
inline uint32_t piece_of(uint32_t rank) { return rank - tile_of(rank) * PIECES; }  // its piece (rows PIECE_ROWS j ..)
// y_ag block of chip `rank`: the PAIR chips of a column are adjacent (the round-2 bundle of column c = blocks PAIR c
// .. PAIR c + PAIR - 1); the pair: block = rank
inline uint32_t ag_blk(uint32_t rank) { return line_pos(rank) * PAIR + pair_pos(rank); }

// rev 59: the global merge's messages to the merge helpers (owner chunk cores c < NCHIPS of group 0; helper q holds
// chip q's candidate list; rev 64: c < NHELP, KPH lists each): per token MERGE_MSGS 16-B slots -- 0 start, 1 + r round
// r's (prefix, k_rem, done), then the tie quotas of the NCHIPS chips' lists in chip order (2 slots: 8 words)
constexpr uint32_t QUOTA_SLOTS = NCHIPS * 4 / mladsa::MSG_BYTES;  // 16-B slots of the per-helper tie quotas (2 / 8)
static_assert(QUOTA_SLOTS * mladsa::MSG_BYTES == NCHIPS * 4, "one quota word per merge helper, whole slots");
constexpr uint32_t MERGE_MSGS = 1 + mladsa::ROUNDS + QUOTA_SLOTS;
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

// the TP8 AG block: per token ONE chunk of AG8_PPC pages ([T x 32, 320] bf16 tensors): pages [0, AG8_QPAGES) = the
// chip's q piece (q_lat rows 0..7 of tile d = 2 face-row runs of 256 B at page d / 4, offset (d % 4) 512 (+ 256)),
// then the candidate records (header + CAND_MAX records = 5 pages), then a pad page (page pairs)
constexpr uint32_t AG8_PPC = 10;
constexpr uint32_t AG8_QPAGES = 4;
constexpr uint32_t CAND_PAGES = (REC_HDR + CAND_MAX * REC_BYTES + TB - 1) / TB;
static_assert(AG8_QPAGES + CAND_PAGES <= AG8_PPC && AG8_PPC % 2 == 0, "TP8 AG chunk");
// the TP32 AG block (8x4): the 2-row q piece of every tile = 2 runs of 64 B = 128 B x 16 tiles = ONE page, then the
// candidate records (5 pages): 6 pages per token
constexpr uint32_t AG32_PPC = 6;
constexpr uint32_t AG32_QPAGES = 1;
static_assert(AG32_QPAGES + CAND_PAGES <= AG32_PPC && AG32_PPC % 2 == 0, "TP32 AG chunk");
constexpr uint32_t AG_PPC = PAIR > 2 ? AG32_PPC : (LINE > 1 ? AG8_PPC : PPC);  // pages per AG chunk (width / 32)
constexpr uint32_t AG_QPAGES = PAIR > 2 ? AG32_QPAGES : AG8_QPAGES;
constexpr uint32_t AG_CHUNKS_PER_TOKEN = LINE > 1 ? 1 : 2;
// a q piece of one bf16 tile in the compact AG formats: 2 runs of PIECE_ROWS face rows (PIECE_ROWS x 32 B each)
constexpr uint32_t PIECE_RUN = PIECE_ROWS * 32;         // bytes of one run (256 at TP8, 64 at TP32)
constexpr uint32_t PIECE_BYTES = 2 * PIECE_RUN;         // bytes of a tile's piece (512 / 128)
constexpr uint32_t PIECES_PER_PAGE = TB / PIECE_BYTES;  // 4 at TP8, 16 at TP32
// (the pair forms of ag_q_page / ag_cand_page above; these pick the format of this build)
inline uint32_t agx_q_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t d) {
    return LINE > 1 ? (blk * T + t) * AG_PPC + d / PIECES_PER_PAGE : ag_q_page(T, blk, t, d);
}
inline uint32_t agx_q_off(uint32_t d) { return LINE > 1 ? (d % PIECES_PER_PAGE) * PIECE_BYTES : 0; }
inline uint32_t agx_cand_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t byte) {
    return LINE > 1 ? (blk * T + t) * AG_PPC + AG_QPAGES + byte / TB : ag_cand_page(T, blk, t, byte);
}
// rows `rows` j .. of a 32x32 tile of tb bytes (4 faces of tb / 4; a face row = tb / 64 bytes): two runs of rows x
// tb / 64 bytes, run A at slice_off_rows(rows, j, tb) (face 0 / 2), run B = run A + tb / 4 (face 1 / 3). rows = 8
// (the TP8 pieces / the 8x4 round-1 slices): run A at (j / 2) (tb / 2) + (j % 2) (tb / 8), tb / 8 bytes
inline uint32_t slice_off_rows(uint32_t rows, uint32_t j, uint32_t tb) {
    const uint32_t r0 = rows * j;
    return (r0 / 16) * (tb / 2) + (r0 % 16) * (tb / 64);
}
inline uint32_t slice_run(uint32_t rows, uint32_t tb) { return rows * (tb / 64); }
inline uint32_t slice_off(uint32_t j, uint32_t tb) { return slice_off_rows(PIECE_ROWS, j, tb); }  // the q / o pieces
// RS round 2 (x_rs2 / y_rs2 [LINE x T x 32, 320] bf16): block q x token t = one chunk of RS2_PPC pages; x_rs2 block q
// = the slice for destination column q (the own column too: the root lands it from there), y_rs2 block s = the slice
// from source column s. A slice = rows 8 q .. of the column-combined partial: O tile d at byte d (otb / 4), the stats
// tile i at DT (otb / 4) + i (ptb / 4) (9 pages with fp32 O, 5 with bf16 O)
constexpr uint32_t RS2_PPC = 10;
inline uint32_t rs2_page(uint32_t T, uint32_t blk, uint32_t t) { return (blk * T + t) * RS2_PPC; }
// 8x4 (doc/perf/shard_8x4_design.md): round 1 = the column reduce-scatter of 8-row slices in the RS2 format above
// (x_rs1 / y_rs1 [PAIR x T x 32, 320] bf16, block q = destination / source column position); round 2 = the row scatter
// of 2-row slices (x_rs2 / y_rs2 [LINE x T x 32, 128] bf16: O tile d at d (otb / 16), stats tile i at DT (otb / 16) +
// i (ptb / 16): 16 x 256 + 2 x 256 = 4.5 KB -> 3 pages + pad)
constexpr uint32_t RS1_PPC = RS2_PPC;
constexpr uint32_t RS2B_PPC = 4;
inline uint32_t rs1_page(uint32_t T, uint32_t blk, uint32_t t) { return (blk * T + t) * RS1_PPC; }
inline uint32_t rs2b_page(uint32_t T, uint32_t blk, uint32_t t) { return (blk * T + t) * RS2B_PPC; }
// rev 65, THE FOLD COMBINE (MLA_DSA_SHARD_COMB, tt/mk_mla.py GLM53_DSA_SHARD_COMB; the 8x4 only, unset: every build of
// rev 64 unchanged). The lane root combines each round's slices in ONE pass instead of a chain of pairwise combines
// (round 1: own + 7 = 7 combines of 32 fp32 matmul tiles each on full 32-row tiles; round 2: 3 + the finalise).
// Round 1: the 8 column slices (8 rows each) land ROW-PACKED in CB_ARXO messages of COMB_MSG pages -- O tile d of
// source s = 4 p + k at rows 8 k .. of page p (4 O tiles per message); their m as 4 ROTATIONS x 2 pages (page p of
// rotation j, rows 8 k .. = source 4 p + (k + j) % 4: every element of the 8 pages' max sees all 8 sources, so the
// joint max M and c = exp(m - M) <= 1 are exact elementwise fp32) and their l packed, in CB_ARX 2-page messages (rev
// 66: the stats must unpack to DST exactly; att_c.cpp THE FORMAT RULE). F_p = FOLD1 diag(c_p) (FOLD1[8 j1 + r][8 k + r]
// = 1: the destination's rows gather the 4 row blocks), the column-combined O rows 8 j1 .. = F_0 P_0 + F_1 P_1 (2
// matmuls per O tile; every operand rounded to tf32 once, not once per chain step), L = FOLD1 (c_0 l_0) + FOLD1 (c_1
// l_1). Round 2: the 4 row slices (2 rows) in sub-blocks 8 j1 + 2 k of one page per O tile (stats: 4 rotations of m +
// the packed l in CB_ARX), G = FOLD2 diag(c) (FOLD2[2 j2 + r][8 j1 + 2 k + r] = 1), L = FOLD2 (c l), G' = diag(1 / L) G
// and o = G' P (one matmul per O tile: combine + finalise). FOLD1 / FOLD2 (fp32 0 / 1 tiles): the root NCRISC writes
// them into CB_ARX once per launch (the compute copies them to CB_APBO, a matmul-operand CB). (Both rev-65 blocks
// declare nothing without their define: the rev-64 builds keep their instructions.)
#if defined(MLA_DSA_SHARD_COMB)
constexpr uint32_t COMB_MSG = 8;  // pages per CB_ARXO message of the fold combine (2 per 16-page CB: double-buffered)
static_assert(PAIR == 8 && LINE == 4 && RS1_ROWS == 8 && RS2_ROWS == 2, "the fold combine: the 8x4 slices");
// word of element (row, col) of a 32x32 fp32 tile (4 faces of 16 x 16, row-major within a face): the FOLD tiles
inline uint32_t tile_word(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}
#endif
// rev 65, RS1 EARLY (MLA_DSA_SHARD_RS1_EARLY, GLM53_DSA_SHARD_RS1_EARLY; the 8x4 only): the root signals the RS1
// scatter lane after pass 0 too. Pass 0's slices go to the OTHER half of the column -- all of the traffic across the
// column's middle link -- so the lane streams those destinations once every lane token's pass 0 is shipped, while pass
// 1 computes, and the own half's once every pass 1 is (fs_read.cpp FS_WAIT2; tt/fabric_stream.py FabricStreamScatter
// first= / wait2=). Rev 67: the two gates count in SEPARATE FIELDS of the lane's wait semaphore -- a pass-0 signal adds
// 1 (bits [0, 16)), a pass-1 signal RS1_PASS1_INC (bits [16, 32)). Rev 65 / 66 added 1 for both and opened the first
// gate at a total of t: a token lane through both passes of its tokens opened it while a slower lane's pass-0 slices
// were still unwritten, and their stale x_rs1 pages travelled (B 16 distinct users, 2 tokens per lane: user 0 off at
// decode id 8; B 4 identical users never skewed)
#if defined(MLA_DSA_SHARD_RS1_EARLY)
static_assert(PAIR == 8 && LINE == 4, "RS1 early: the 8x4 column scatter");
constexpr uint32_t RS1_PASS1_INC = 65536;  // = 1 << 16 (fs_read.cpp FS_WAIT2's field split; tt/fabric_stream.py)
static_assert(T_MAX < 65536, "a gate's count fits its 16-bit field");
#endif
// rev 59, TP8: q_asm [2 x T x 32, 512] bf16 -- head tile g of token t assembled ONCE per chip (the index core's NCRISC,
// behind the global merge) from the 4 pieces in y_ag; the owners read 16 whole pages per push
inline uint32_t qasm_page(uint32_t T, uint32_t tile, uint32_t t, uint32_t d) { return (tile * T + t) * PPC + d; }
// rev 64: KPH chips per merge helper (MLA_DSA_SHARD_KPH, tt/mk_mla.py shard_kph / tt/shard_plan.py shard_kph; the 8x4
// only, unset = 1: every build before rev 64 unchanged). Merge helper h (lane-local chunk core h < NHELP of group 0)
// holds the lists of chips h KPH .. h KPH + KPH - 1, so a token lane needs NHELP = NCHIPS / KPH helpers among its
// C / LANES chunk cores (8x4, 64 chunk cores: KPH 2 -> 4 lanes, KPH 4 -> 8 lanes). Its round histogram is one 16-bin
// histogram over the union of its lists (the coordinator's sums, digits and T_g are the KPH = 1 integers); its tie
// report carries the KPH per-list counts of keys == T_g in words 0 .. KPH - 1 (<= 16: one histogram slot), so the
// per-chip tie quotas stay in chip order (the same quotas as KPH = 1)
#if defined(MLA_DSA_SHARD_KPH)
constexpr uint32_t KPH = MLA_DSA_SHARD_KPH;
#else
constexpr uint32_t KPH = 1;
#endif
static_assert(KPH == 1 || PAIR == 8, "shard: KPH > 1 on the 8x4 only (the pair / the 2x4 keep one chip per helper)");
static_assert(KPH >= 1 && KPH <= 16 && NCHIPS % KPH == 0, "shard: KPH divides NCHIPS; its tie counts fit one slot");
constexpr uint32_t NHELP = NCHIPS / KPH;  // merge helpers per token lane
// R8 (MLA_DSA_R8, tt/mk_mla.py GLM53_DSA_SHARD_R8; dsa_common.hpp RADIX_BITS = 8; the 8x4 only, unset: every build
// unchanged): the local select and the global merge in 4 rounds of 8-bit digits. A core's round histogram is 256 fine
// counts (uint16, digit (k >> shift) & 255) in its own mailbox past the messages (R8_FINE_OFF: the shard mailbox has
// MSGS = 5 and MERGE_MSGS = 1 + 4 + QUOTA_SLOTS slots per lane token, so the fine area fits in the bytes the 4-bit
// layout's rounds took: the host allocates MBOX_SHARD_BYTES + R8_FINE_BYTES, tt/shard_plan.py mbox_bytes); its report
// to the coordinator stays the 64-B slot of 16 coarse counts (bucket b = fine bins 16 b .. 16 b + 15). The coordinator
// picks the coarse digit from the summed reports (= the 4-bit round 2 r), reads bucket b's R8_SLICE_BYTES (16 counts)
// from every core's fine area into the core's slot (one NoC read each, one barrier) and picks the fine digit from their
// sum (= the 4-bit round 2 r + 1): the same integers, one handshake instead of two. No new L1, no new semaphores
#if defined(MLA_DSA_R8)
static_assert(PAIR == 8 && LINE == 4, "R8: the 8x4 shard");
static_assert(mladsa::RADIX_BITS == 2 * mladsa::DIGIT_BITS && mladsa::BINS == 16, "R8: two 16-bin levels per round");
constexpr uint32_t R8_FINE_OFF = MBOX_SHARD_BYTES;  // a core's fine counts in its mailbox
constexpr uint32_t R8_FINE_BYTES = 512;             // 256 uint16 counts (tt/shard_plan.py R8_FINE_BYTES)
constexpr uint32_t R8_SLICE_BYTES = 32;             // one coarse bucket's 16 fine counts (the coordinator's pull)
static_assert(R8_FINE_BYTES == 256 * 2 && R8_SLICE_BYTES == mladsa::BINS * 2, "R8: uint16 counts");
static_assert(R8_FINE_OFF % 16 == 0 && R8_SLICE_BYTES % 16 == 0, "R8: the pull's NoC read alignment");
static_assert(R8_SLICE_BYTES <= mladsa::HIST_SLOT_BYTES, "R8: a pulled bucket lands in the core's histogram slot");
static_assert(KPH * CAND_MAX <= 65535, "R8: a merge helper's counts fit uint16");
// a core's round: the fine counts of its keys src[0, n) that match (mask, prefix) at fine_addr, the 16 coarse counts in
// hist[0, 16) (the report). n <= 65535 keys: every count fits its uint16, and a bucket's 8 words (2 counts each) sum
// without a carry between the halves
inline void r8_hist(
    uint32_t fine_addr,
    volatile tt_l1_ptr uint32_t* hist,
    volatile tt_l1_ptr uint32_t* src,
    uint32_t n,
    uint32_t mask,
    uint32_t prefix,
    uint32_t shift) {
    volatile tt_l1_ptr uint32_t* f32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fine_addr);
    volatile tt_l1_ptr uint16_t* f16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(fine_addr);
    for (uint32_t w = 0; w < R8_FINE_BYTES / 4; ++w) {
        f32[w] = 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t k = src[i];
        if ((k & mask) == prefix) {
            f16[(k >> shift) & 255] += 1;
        }
    }
    for (uint32_t b = 0; b < mladsa::BINS; ++b) {
        uint32_t s = 0;
        for (uint32_t w = 0; w < R8_SLICE_BYTES / 4; ++w) {
            s += f32[b * (R8_SLICE_BYTES / 4) + w];
        }
        hist[b] = (s & 0xFFFFu) + (s >> 16);
    }
}
// the coordinator: the pulled buckets' 16 fine counts (uint16 words 0 .. 15 of each of the n slots) summed, the
// accumulators in registers (dsa_common.hpp sum_hist's form)
inline __attribute__((always_inline)) void r8_sum_fine(volatile tt_l1_ptr uint32_t* hist, uint32_t n, uint32_t* out) {
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    uint32_t a8 = 0, a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0;
    for (uint32_t c = 0; c < n; ++c) {
        volatile tt_l1_ptr uint16_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(hist + c * (mladsa::HIST_SLOT_BYTES / 4));
        a0 += p[0];
        a1 += p[1];
        a2 += p[2];
        a3 += p[3];
        a4 += p[4];
        a5 += p[5];
        a6 += p[6];
        a7 += p[7];
        a8 += p[8];
        a9 += p[9];
        a10 += p[10];
        a11 += p[11];
        a12 += p[12];
        a13 += p[13];
        a14 += p[14];
        a15 += p[15];
    }
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
    out[4] = a4;
    out[5] = a5;
    out[6] = a6;
    out[7] = a7;
    out[8] = a8;
    out[9] = a9;
    out[10] = a10;
    out[11] = a11;
    out[12] = a12;
    out[13] = a13;
    out[14] = a14;
    out[15] = a15;
}
#endif
// rev 59: the merge helpers' scratch in CB_AKEY: their chip's candidate pages, its keys, the compacted candidates, the
// per-core tie quotas (helper `rank` only). Rev 60 (review 2 item 1): it overlays the phase-A keys / candidates
// (dead once every lane token's local select is done; the merge runs after the all-gather), so CB_AKEY holds
// max(2 MLA_DSA_KEYS words, MERGE_SCR_BYTES + 64) there instead of both
constexpr uint32_t MERGE_SCR_BYTES1 = CAND_PAGES * TB + 2 * CAND_MAX * 4 + mladsa::MAX_CHUNKS * 4;
// rev 64, KPH > 1: MSTG_PAGES staging pages (the lists' record pages stream through them, double-buffered), the KPH
// lists' keys (compacted in place per list), the per-core tie quotas (helper rank / KPH only): 12.3 KB at KPH = 4
constexpr uint32_t MSTG_PAGES = 2;
constexpr uint32_t MERGE_SCR_BYTESK = MSTG_PAGES * TB + KPH * CAND_MAX * 4 + mladsa::MAX_CHUNKS * 4;
constexpr uint32_t MERGE_SCR_BYTES = KPH == 1 ? MERGE_SCR_BYTES1 : MERGE_SCR_BYTESK;
// rev 68, THE KEY-PACKED ALL-GATHER (MLA_DSA_SHARD_AGK, tt/mk_mla.py GLM53_DSA_SHARD_AGK; the 8x4 only, unset: every
// build of rev 67 unchanged). The AG moved 32 x t x 6 pages per chip (the AG32 block: q page + header + 512 16-B
// records), 5 / 6 of it records whose core id only the record's own chip reads -- at t = 16 the busiest link 0 hop
// carries 7 (AG1) + 3 x 8 (AG2) blocks = 6 MB per launch, the ~535 us AG window of the probe (doc/perf/
// shard_8x4_design.md section 9). x_ag keeps the AG32 format (the producers are unchanged); the AG1 lane's NCRISC
// (fs_read.cpp FS_KEYPACK) rewrites each staged chunk in place -- page AG32_QPAGES = the n = min(header, CAND_MAX)
// record keys in record order, zeros after -- and ships AGK_PPC = 2 pages per token: y_ag block blk, token t = the
// q-piece page (agy_q_page) + the key page (agk_key_page). The count is not shipped: chip q's list of a sparse token
// at position p holds exactly agk_count(p, q) = min(CAND_MAX, n_closed_local(p, q)) records (the local select keeps
// every key below k candidates, exactly k above). The merge helpers read one key page per list (instead of <= 5
// record pages); the helper of chip `rank` reads its per-core tie order from this chip's own x_ag records.
#if defined(MLA_DSA_SHARD_AGK)
constexpr uint32_t AGK_PPC = 2;
static_assert(PAIR == 8 && LINE == 4 && AG_PPC == AG32_PPC && AG_QPAGES == 1, "AGK: the 8x4 AG32 block");
static_assert(CAND_MAX * 4 == TB && AGK_PPC == AG_QPAGES + 1, "AGK: one key page holds a full list; page pairs");
inline uint32_t agy_q_page(uint32_t T, uint32_t blk, uint32_t t, uint32_t d) {
    return (blk * T + t) * AGK_PPC + d / PIECES_PER_PAGE;
}
inline uint32_t agk_key_page(uint32_t T, uint32_t blk, uint32_t t) { return (blk * T + t) * AGK_PPC + AG_QPAGES; }
inline uint32_t agk_count(uint32_t p, uint32_t q, uint32_t C) {
    const uint32_t n = n_closed_local(p, q, C);
    return n < CAND_MAX ? n : CAND_MAX;
}
#endif
// rev 72, LOCAL_ALL (MLA_DSA_SHARD_LOCAL_ALL, tt/mk_mla.py GLM53_DSA_SHARD_LOCAL_ALL; the 8x4 only, unset: every build
// of rev 71 unchanged). A sparse token whose chip holds n_closed_local <= TOPK closed pools keeps every one of them in
// its local top-512 -- the radix rounds and the tie round only re-derive that (T = 0 on the order keys, the quotas the
// core counts). The coordinator (ix_dm0.cpp) and every owner (att_dsa_shard_dm1.cpp phase A, all groups) skip those
// MSGS messages for that token (the SEL_SEM counts stay equal on both sides: the merge reuses the counter); each owner
// keeps all its valid keys in local order and the group-0 owners write their records at the analytic offset below:
// the same records, offsets and header count as the rounds give (tt/shard_plan.py local_select_all, test_shard_plan_8x4
// test_local_all_matches_the_rounds). Below 64 K tokens on 32 chips every chip takes it.
#if defined(MLA_DSA_SHARD_LOCAL_ALL)
static_assert(PAIR == 8 && LINE == 4, "LOCAL_ALL: the 8x4 shard");
// valid pools of chunk core c of C at n_closed (the phase-A key loop: pool tiles c, c + C, .. < ceil(n_closed / 32),
// each min(32, n_closed - its first pool) columns)
inline uint32_t local_all_pools(uint32_t n_closed, uint32_t c, uint32_t C) {
    const uint32_t npt = mladsa::pool_tiles(n_closed, c, C);
    if (npt == 0) {
        return 0;
    }
    const uint32_t tail = n_closed - (c + (npt - 1) * C) * POOL_PAGE;
    return (npt - 1) * POOL_PAGE + (tail < POOL_PAGE ? tail : POOL_PAGE);
}
// core c's record offset in its token's candidate chunk: the valid pools of the cores before it
inline uint32_t local_all_offset(uint32_t n_closed, uint32_t c, uint32_t C) {
    uint32_t off = 0;
    for (uint32_t k = 0; k < c; ++k) {
        off += local_all_pools(n_closed, k, C);
    }
    return off;
}
#endif
// rev 72, ROUND_MARKS (MLA_DSA_ROUND_MARKS, GLM53_DSA_SHARD_ROUND_MARKS; the 8x4 + R8 only, probe builds): the shard
// coordinator marks its local select's rounds for the lane's first token -- per round "histograms in" and "digits
// broadcast", then the tie round's two: 10 marks between y_in and local_coord (ix_dm0's row: 16 marks in all;
// tt/shard_plan.py decode_probe ROUND_MARKS)
#if defined(MLA_DSA_ROUND_MARKS)
static_assert(PAIR == 8 && LINE == 4 && mladsa::ROUNDS == 4, "ROUND_MARKS: the 8x4 shard with R8 (16 probe marks)");
#endif
// rev 73, SEL2 (MLA_DSA_SHARD_SEL2, tt/mk_mla.py GLM53_DSA_SHARD_SEL2; the 8x4 + R8 only, unset: every build
// unchanged): the local select's coordinator work per round and core (rev-72 round marks: ~0.4 us per core, 27 us a
// round at 64 chunk cores), cut with the same integers. (1) No per-round above[c] accumulation on the coordinator: a
// core's keys above the final T are exactly the keys its compactions drop UPWARD (a key > T leaves at the first digit
// where it differs from T, and its digit is the larger one there); every core counts them while it compacts and reports
// (eq, gt) in the tie round (slot words 0 / 1). (2) Packed sums: a core's coarse report is 8 uint32 words of 2 uint16
// counts (r8_hist_packed), as the pulled fine slice already is; the coordinator sums 8 words per core in 8 registers
// and splits the halves (sel2_sum) -- no carry while the lane's keys total <= 65535 (n_closed_local <= 65535: contexts
// below 8 M tokens on 32 chips). (3) The (quota, offset) message: owner k gets the 16-B chunk holding its own pair
// instead of the whole C x 8-B table (ix_dm0.cpp send_own). Emulated by tt/shard_plan.py local_select_sel2.
#if defined(MLA_DSA_SHARD_SEL2)
static_assert(PAIR == 8 && LINE == 4, "SEL2: the 8x4 shard");
static_assert(mladsa::RADIX_BITS == 2 * mladsa::DIGIT_BITS && mladsa::BINS == 16, "SEL2: R8's two-level rounds");
constexpr uint32_t SEL2_WORDS = mladsa::BINS / 2;  // packed words of a report / a pulled bucket
static_assert(SEL2_WORDS * 4 == R8_SLICE_BYTES, "SEL2: a pulled bucket is 8 packed words");
static_assert(mladsa::QUOTA_W == 2, "SEL2: two (quota, offset) pairs per 16-B chunk");
// a core's round: r8_hist (the fine counts in its mailbox, the 16 coarse counts in hist), then the coarse counts packed
// two per word in place: hist[w] = c[2 w] | c[2 w + 1] << 16 (word w reads 2 w and 2 w + 1 >= w before it is written)
inline void r8_hist_packed(
    uint32_t fine_addr,
    volatile tt_l1_ptr uint32_t* hist,
    volatile tt_l1_ptr uint32_t* src,
    uint32_t n,
    uint32_t mask,
    uint32_t prefix,
    uint32_t shift) {
    r8_hist(fine_addr, hist, src, n, mask, prefix, shift);
    for (uint32_t w = 0; w < SEL2_WORDS; ++w) {
        hist[w] = hist[2 * w] | (hist[2 * w + 1] << 16);
    }
}
// the coordinator: n slots' 8 packed words summed in registers, each word's halves split -> 16 counts (exact while
// every count's total <= 65535: the lane's keys)
inline __attribute__((always_inline)) void sel2_sum(volatile tt_l1_ptr uint32_t* hist, uint32_t n, uint32_t* out) {
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint32_t c = 0; c < n; ++c) {
        volatile tt_l1_ptr uint32_t* p = hist + c * (mladsa::HIST_SLOT_BYTES / 4);
        a0 += p[0];
        a1 += p[1];
        a2 += p[2];
        a3 += p[3];
        a4 += p[4];
        a5 += p[5];
        a6 += p[6];
        a7 += p[7];
    }
    const uint32_t a[SEL2_WORDS] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (uint32_t w = 0; w < SEL2_WORDS; ++w) {
        out[2 * w] = a[w] & 0xFFFFu;
        out[2 * w + 1] = a[w] >> 16;
    }
}
#endif
}  // namespace mlashard
