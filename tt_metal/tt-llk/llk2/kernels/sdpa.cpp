// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ttnn's scaled-dot-product-attention compute kernel (kernels/compute/sdpa.cpp, with its streaming protocol) on
// LLK 2.0: Tensors, the Compute API and owners for every sync. One body for all three threads; no LLK calls, no
// init / uninit / reconfig, no tile_regs_* and no per-thread code.
//
// The online softmax. A Q chunk (Sq_chunk_t tile rows of Q) meets the K chunks (Sk_chunk_t key tiles each; the last
// one has last_width) one after the other. For a group of Q tile rows (group_rows of them), each K chunk computes
//   S     = Q K^T                                  scores
//   m     = max(m_prev, row max of S)              running row max
//   alpha = exp(scale (m_prev - m))                what the earlier chunks' sums are rescaled by
//   P     = exp(scale (S - m))                     unnormalized probabilities
//   L     = alpha L_prev + row sum of P            running denominator
//   U     = alpha U_prev + P V                     running numerator
// (the first K chunk has nothing from an earlier one: m = row max of S, L = row sum of P, U = P V), and after the last
// K chunk out = U / L.
//
// A mask is added onto S before anything reads it: S += M. That is the mask step, one per group of S, between the
// group's last pack and its row max (mask_scores). M is the op's mask tensor, or it is put together from the
// few tiles the writer makes: for causal attention and for a sliding window by where the Q tile row is among the
// keys, and for a last key tile that is not whole. An op with none of these has no such step. M is added to the
// unscaled S, so it is scaled with it: the op divides its mask tensor by the scale before it reaches the reader.
//
// A Q chunk need not meet every K chunk: a causal one meets those up to its own last row's keys, one with a sliding
// window those around it, and for windowed attention or a mask block map the reader sends the range. The reader
// sends K and V for that range only. The range is a run-time value per Q chunk (KRange, read by Group); it is all
// the K chunks, at compile time, for an op with none of these.
//
// Chunked attention (Q is a piece of a longer sequence) only moves where the Q chunks are among the keys: by a
// run-time arg, or by a word the reader sends once. Paged K and V are the reader's alone.
//
// An attention sink is one more key, with a score of its own (a scalar per Q chunk, from the reader) and no value:
// L += exp(scale (sink - m)) before out = U / L, m being the last K chunk's. That m is then written for a next K
// chunk although there is none, and the term goes through a buffer of its own into the sum that precedes 1 / L.
//
// Syncs, and where each one is (compute_api/sync.h):
//   every buffer    through Tensors: wait_front / peek_front / reserve_back return the Tensor, release() pops or
//                   pushes its pages (compute_api/dataflow.h)
//   Dest            a ring of two sections (compute_api/dest.h): math writes one block while pack reads the other
//   math -> pack    inside pack, or inside the exp and the reciprocal, which run on the pack thread
//
// Flow, per Q chunk (which first reads its range of K chunks, when the reader sends one). The exp of P runs on the
// pack thread's vector unit and the matmuls on the FPU, so a pass keeps both busy: a group's S is computed one pass
// early, between the P blocks of the group before it, and its m and alpha next to that group's P V. The group before
// it may be the last group of the K chunk before: the passes go through the K chunks without a seam.
//   S (with its mask step) and m of the first group                  the first stage of a pass with no group before:
//                                                                    nothing to run them next to yet
//   one pass per group `cur` of every K chunk, `next` being the group after it:
//     P(cur) block, S(next) block, P(cur) block, S(next) block ...   the exp on pack || the matmul on the FPU
//                                                                    (a block: the tiles one Dest section holds;
//                                                                    a width's last block may be narrower)
//                                                                    after the last S(next) block, the group's
//                                                                    mask step: a mask tensor's rows are waited
//                                                                    for only there
//     with last_p_beside_m (the host's choice: for large groups) the last P(cur) block comes after the last S(next)
//     block and S(next)'s push: m(next) does not wait behind that block's exp, which runs on pack || the pooling
//     m(next)
//     L(cur), then U(cur) but for its last rows
//     alpha(next)                                                    its exp on pack || the P V of the last rows
//     U(cur), its last rows
//     last K chunk: out(cur) = U(cur) / L(cur), L(cur) with the group's sink terms when the op has a sink
//
// The row sum of P comes from the packer: each P block is packed twice, into P and, its tiles added up, into its
// row's L tile (pack .sum_tiles). An L tile is then the sum of the row's P tiles, datum by datum; alpha L_prev is
// added onto whole tiles the same way, and the sum over a tile's 32 columns is taken once, in front of 1 / L.
// So L = alpha L_prev + new is formed by the packer in L1, where U = alpha U_prev + P V is formed in Dest and packed
// once: L's new part is never in Dest as a tensor (it comes out of P's packs), and bringing it back would cost an
// unpack and a pack per group.
//
// The Dest mode is the host's (fp32_dest_acc_en, here LLK2_DEST_FP32), and the flow is the same in both. With a 32-bit
// Dest a section holds 4 tiles, not 8 (every block size here follows from dest_section_tiles), and the host keeps S,
// P, L and 1 / L in Float32.
//
// What this file takes from tt-metal beyond what the 1.0 kernel does (the LLK 2.0 branch of sdpa_program_factory.cpp):
//   - Each buffer's id is the named compile-time arg <cb> and its data format <cb>_format, the same on all three
//     TRISCs: the Tensors' specs (and through them the Resolver) need the format at compile time on every thread.
//   - S, P, m, alpha, 1 / L and the sink terms have a buffer each, holding whole groups (cb_scores two,
//     cb_probabilities one, cb_row_max one, cb_alpha two, cb_inverse one, cb_sink_term one), and the group size is
//     the named arg q_rows_per_group. Every tensor is then read at the front of a buffer and written at its back.
//     The 1.0 kernel keeps S and P in one buffer (cb_qk_im) and m and alpha in another (cb_exp_max_diff).
//   - The Dest mode and the math fidelity reach all three TRISCs as the defines LLK2_DEST_FP32 and
//     LLK2_MATH_FIDELITY (tt-metal generates MATH_FIDELITY for math and pack only).
//   - Every CB has pages of one tile (the Tensors are counted in tiles, as 1.0 counts them).
//   - The reader sends two groups of a Q chunk's rows in front of V of its first K chunk and the rest behind it
//     (SDPA_Q_ROWS_BEFORE_FIRST_V): the first pass computes S of two groups and then reads V.
//   - llk2/inc on the include path.
//   - The mask buffer's id and format are cb_mask_in and cb_mask_in_format whether or not the op has one; the
//     same for the sink's buffers, cb_attention_sink and cb_sink_term, and for the reader's pages of words,
//     cb_chunk_start_idx and cb_k_range.
// And as today: the firmware sets up each TRISC's CB pointers; the reader lays K out as K^T's tile grid and sends a
// mask tensor's tiles behind each K chunk, a Q tile row at a time, with -inf where K is padding; the writer
// pushes the reduce's scaler tile once, and the mask tiles it makes once; m, L and U go from one K chunk to the next
// through two buffers each, written by one chunk and read by the next.
//
// The tt-llk harness runs this same file (tests/sources/llk2_sdpa_compute_api_test.cpp includes it), with its own
// stand-ins for the tt-metal pieces above (tests/helpers/tt_metal_kernel.h). It builds the kernel without the
// features whose other half is tt-metal's reader or writer: causal, a sliding window, chunked, sinks, zigzag and a
// sent K range run only through tt-metal's tests (tests/ttnn/.../sdpa).
//
// What the 1.0 kernel does that this one does not: Wormhole, and the 1.0 kernel's other protocol (a whole Q chunk
// per step, which the host uses with a 32-bit Dest): here the 32-bit Dest runs on the streaming protocol.

#include <cstdint>

// LLK2_DEST_FP32 and LLK2_MATH_FIDELITY (kernel_settings.h) are the host's defines here, not this file's: see above.
// So is EXP_APPROX_MODE, which the 1.0 kernel reads too.
#include "api/compute/common.h" // get_arg_val only (tt-metal has run-time args nowhere else): no 1.0 compute API call may appear in this file
#include "api/dataflow/dataflow_buffer.h"
#include "compute_api/copy.h"
#include "compute_api/dataflow.h"
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/reduce.h"
#include "compute_api/sync.h"
#include "compute_api/vec.h"
#include "kernel_args.h"
// What the reader, the writer and this kernel must agree on, from where they take it.
#include "ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"

namespace
{

using namespace llk2;

// The op, as the host's named compile-time args.
constexpr std::uint32_t DHt        = get_named_compile_time_arg_val("DHt");            // head dimension of Q and K, in tiles
constexpr std::uint32_t vDHt       = get_named_compile_time_arg_val("vDHt");           // head dimension of V and of the output
constexpr std::uint32_t Sq_chunk_t = get_named_compile_time_arg_val("Sq_chunk_t");     // Q tile rows per Q chunk
constexpr std::uint32_t q_per_head = get_named_compile_time_arg_val("q_num_chunks");   // Q chunks of one head
constexpr std::uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");     // key tiles per K chunk
constexpr std::uint32_t k_chunks   = get_named_compile_time_arg_val("k_num_chunks");   // K chunks of the key sequence
constexpr std::uint32_t valid_Skt  = get_named_compile_time_arg_val("valid_Skt");      // key tiles of all the K chunks together
constexpr std::uint32_t scale      = get_named_compile_time_arg_val("scale");          // the softmax scale, FP32 bits
constexpr bool causal              = get_named_compile_time_arg_val("is_causal") != 0; // a Q row sees the keys up to its own position
// The op has a mask tensor, and the reader sends its tiles for every K chunk, a Q tile row at a time.
constexpr bool provided_mask = get_named_compile_time_arg_val("use_provided_mask") != 0;
constexpr bool chunked       = get_named_compile_time_arg_val("is_chunked") != 0; // Q is a piece of a longer sequence, from a Q chunk that is a run-time value
// A sliding window, in tokens: causal, a Q row sees that many keys up to its own; otherwise half of it on each side.
// 0: none.
constexpr std::uint32_t window = get_named_compile_time_arg_val("sliding_window_size");
constexpr bool sinks           = get_named_compile_time_arg_val("use_attention_sink") != 0; // an attention sink: the reader sends its tile with every Q chunk
// The columns of the last key tile that hold keys, when it is not a whole tile (0: it is). The writer then makes the
// tile that masks the others, once.
constexpr std::uint32_t partial_key_cols = get_named_compile_time_arg_val("k_partial_col");
constexpr bool zigzag                    = get_named_compile_time_arg_val("use_zigzag") != 0; // reader and writer take a head's Q chunks from both ends in turn
constexpr bool k_range_sent = get_named_compile_time_arg_val("use_k_range") != 0; // the reader sends each Q chunk's range of K chunks, two words of a page
// With a single K chunk the reader sends K and V once for this many heads of Q, and the kernel keeps them for every
// Q chunk of those heads; 0: K and V come anew with every Q chunk.
constexpr std::uint32_t heads_per_kv = get_named_compile_time_arg_val("kv_retain_heads");
constexpr std::uint32_t group_rows   = get_named_compile_time_arg_val("q_rows_per_group"); // Q tile rows that go through one op
// The host's choice of where a pass's last P block goes (the flow at the top; the host has the measurement).
constexpr bool last_p_beside_m = get_named_compile_time_arg_val("last_p_beside_m") != 0;
// The exp of alpha and of the sink term, and with a 32-bit Dest of P: the polynomial or fast one, or the exact exp.
constexpr bool exp_approx = EXP_APPROX_MODE;
// Run-time args are positional in tt-metal: the 1.0 kernel's numbers.
constexpr int chunk_start_is_tensor_arg = 1; // chunked: the op's start position comes in a page from the reader
constexpr int q_chunk_offset_arg        = 2; // chunked: otherwise, the sequence's Q chunk the op's first one is
constexpr int q_first_arg               = 4; // this core's first Q chunk
constexpr int q_chunks_arg              = 5; // this core's share of the Q chunks

constexpr std::uint32_t tile_dim = default_tile_dim; // datums along one side of a tile
// S and P go through Dest in blocks of block_tiles tiles: what one section holds, or a whole K chunk when that is less.
constexpr std::uint32_t block_tiles = dest_section_tiles < Sk_chunk_t ? dest_section_tiles : Sk_chunk_t;
constexpr std::uint32_t groups      = Sq_chunk_t / group_rows;
constexpr std::uint32_t last_width  = valid_Skt - (k_chunks - 1) * Sk_chunk_t; // key tiles of the last K chunk
constexpr bool position_masks       = causal || window != 0;                   // S is masked by where the Q tile row is among the keys
constexpr bool made_masks_used      = position_masks || partial_key_cols != 0; // from the few tiles the writer makes
constexpr bool masked               = provided_mask || made_masks_used;        // S has a mask step
// A Q chunk meets all the K chunks, or its own range of them, a run-time value (KRange).
constexpr bool k_range_per_q_chunk = position_masks || k_range_sent;
// A sliding window in key tiles, as the reader bounds its K loop: how far before and after its own position a Q tile
// row may see keys.
constexpr std::uint32_t window_before = SlidingWindowLoopGeometry<window, causal, tile_dim>::left_window_tiles;
constexpr std::uint32_t window_after  = SlidingWindowLoopGeometry<window, causal, tile_dim>::right_window_tiles;
// Where the window's edges are from a Q tile row's own key tile, as the writer cuts the edge tiles
// (generate_lightweight_mask_tiles): the leading edge window_lead_tiles tiles before it, and when the edge is not at
// a tile boundary (window_lead_cols) in the tile before that too; the trailing edge of a window that is not causal
// likewise after it.
constexpr std::uint32_t window_lead_tiles  = (causal ? window : window / 2) / tile_dim;
constexpr std::uint32_t window_lead_cols   = (causal ? window : window / 2) % tile_dim;
constexpr std::uint32_t window_trail_tiles = window / 2 / tile_dim;
constexpr std::uint32_t window_trail_cols  = window / 2 % tile_dim;
// U goes through Dest u_rows rows at a time: as many as one section holds. A group's last block has what is left.
// A row wider than a section (u_rows is then 1) goes u_cols tile columns at a time, and what is left at its end.
constexpr std::uint32_t u_cols      = vDHt < dest_section_tiles ? vDHt : dest_section_tiles;
constexpr std::uint32_t u_rows      = dest_section_tiles / u_cols < group_rows ? dest_section_tiles / u_cols : group_rows;
constexpr std::uint32_t u_last_rows = (group_rows - 1) % u_rows + 1;

// What the host does not send, and the combinations this kernel relies on it not sending.
static_assert(!chunked || causal, "chunked attention is causal");
static_assert(!(provided_mask && partial_key_cols != 0), "a mask tensor masks the last key tile's padding itself");
static_assert(!(provided_mask && position_masks), "a mask tensor holds the causal or window mask itself");
static_assert(!(causal && partial_key_cols != 0), "causal attention masks the last key tile's padding with its own mask");
static_assert(!(k_range_sent && position_masks), "the reader sends a range of K chunks only for an op it bounds no other way");
static_assert(heads_per_kv == 0 || k_chunks == 1, "K and V are kept over Q chunks only when one K chunk holds them");
static_assert(Sq_chunk_t % group_rows == 0, "the Q chunk is a whole number of groups");
static_assert(group_rows <= dest_section_tiles, "a group's statistics fit one Dest section");
static_assert(u_rows == 1 || u_cols == vDHt, "several rows of U go through Dest together only whole");
static_assert(last_width >= 1 && last_width <= Sk_chunk_t, "the last K chunk holds what the others left");

constexpr const auto& q_spec    = l1_spec<Sq_chunk_t * tile_dim, DHt * tile_dim, LLK2_CB_FORMAT("cb_q_in")>;  // [Sq, DHt] tiles
constexpr const auto& kt_spec   = l1_spec<DHt * tile_dim, Sk_chunk_t * tile_dim, LLK2_CB_FORMAT("cb_k_in")>;  // K^T: [DHt, Sk]
constexpr const auto& v_spec    = l1_spec<Sk_chunk_t * tile_dim, vDHt * tile_dim, LLK2_CB_FORMAT("cb_v_in")>; // [Sk, vDHt]
constexpr const auto& unit_spec = l1_spec<tile_dim, tile_dim, LLK2_CB_FORMAT("cb_identity_scale_in")>;        // the reduce's scaler: 1.0
// One group of Q rows in one K chunk. A statistic is one tile per row, its values in the tile's column 0.
constexpr const auto& s_spec         = l1_spec<group_rows * tile_dim, Sk_chunk_t * tile_dim, LLK2_CB_FORMAT("cb_scores")>;        // S: [group_rows, Sk]
constexpr const auto& p_spec         = l1_spec<group_rows * tile_dim, Sk_chunk_t * tile_dim, LLK2_CB_FORMAT("cb_probabilities")>; // P: [group_rows, Sk]
constexpr const auto& m_spec         = l1_spec<group_rows * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_row_max")>;                    // m: [group_rows, 1]
constexpr const auto& alpha_spec     = l1_spec<group_rows * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_alpha")>;
constexpr const auto& inverse_spec   = l1_spec<group_rows * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_inverse")>;         // 1 / L
constexpr const auto& carried_m_spec = l1_spec<group_rows * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_max_A")>;           // m, for the next K chunk
constexpr const auto& l_spec         = l1_spec<group_rows * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_sum_A")>;           // L
constexpr const auto& u_spec         = l1_spec<group_rows * tile_dim, vDHt * tile_dim, LLK2_CB_FORMAT("cb_out_im_A")>; // U: [group_rows, vDHt]
template <std::uint32_t N>
constexpr const auto& out_spec = l1_spec<tile_dim, N * tile_dim, LLK2_CB_FORMAT("cb_out")>; // N tiles of an output row
// A Q chunk's attention sink, its datum [0, 0]; and a group's sink terms, exp(scale (sink - m)), a statistic.
constexpr const auto& sink_spec      = l1_spec<tile_dim, tile_dim, LLK2_CB_FORMAT("cb_attention_sink")>;
constexpr const auto& sink_term_spec = l1_spec<group_rows * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_sink_term")>;
// The mask of one group in one K chunk, when the op has a mask tensor: a tile for every tile of S.
constexpr const auto& mask_spec = l1_spec<group_rows * tile_dim, Sk_chunk_t * tile_dim, LLK2_CB_FORMAT("cb_mask_in")>;
// Otherwise the mask buffer holds the tiles the writer made, once, a column of them: a tile of -inf; for causal
// attention the tile of a Q tile row's own keys (-inf above its diagonal); with a sliding window that tile or the
// trailing edge's, and the edges' other tiles; then, when the last key tile is not whole, its mask.
constexpr std::uint32_t minus_inf_tile        = 0;
constexpr std::uint32_t diagonal_tile         = 1; // causal: the tile of the row's own keys
constexpr std::uint32_t trailing_tile         = 1; // not causal: the window's trailing edge
constexpr std::uint32_t leading_before_tile   = 2; // the window's leading edge, when it is in two tiles: the first
constexpr std::uint32_t leading_tile          = 3; // the window's leading edge
constexpr std::uint32_t trailing_after_tile   = 4; // the window's trailing edge, when it is in two tiles: the second
constexpr std::uint32_t position_mask_tiles   = window != 0 ? 5 : (causal ? 2 : 1);
constexpr std::uint32_t partial_key_mask_tile = position_mask_tiles;
constexpr std::uint32_t made_mask_tiles       = position_mask_tiles + (partial_key_cols != 0 ? 1 : 0);
constexpr const auto& made_masks_spec         = l1_spec<made_mask_tiles * tile_dim, tile_dim, LLK2_CB_FORMAT("cb_mask_in")>;
// The reader's pages of words, for read_value. A chunked op's start position in its sequence, in tokens, sent once
// when it is a tensor: the first word of a page of 8. A Q chunk's range of K chunks: its first and its end, the first
// two words of a page of 4. Both as reader_interleaved.cpp writes them.
constexpr const auto& chunk_start_spec   = words_spec<8, LLK2_CB_FORMAT("cb_chunk_start_idx")>;
constexpr const auto& k_range_spec       = words_spec<4, LLK2_CB_FORMAT("cb_k_range")>;
constexpr std::uint32_t chunk_start_word = 0;
constexpr std::uint32_t k_first_word     = 0;
constexpr std::uint32_t k_end_word       = 1;
static_assert(
    LLK2_CB_FORMAT("cb_max_A") == LLK2_CB_FORMAT("cb_max_B") && LLK2_CB_FORMAT("cb_sum_A") == LLK2_CB_FORMAT("cb_sum_B") &&
        LLK2_CB_FORMAT("cb_out_im_A") == LLK2_CB_FORMAT("cb_out_im_B"),
    "the two buffers a statistic alternates between hold the same tensor");

/// Two Dest sections: math writes one block while pack reads the other. Each is a view of the kernel's one ring.
template <std::uint32_t R, std::uint32_t N>
using TilesRing = DestRing<dst_spec<R * tile_dim, N * tile_dim, kernel_dest_format>>; // [R, N] tiles
using BlockRing = TilesRing<1, block_tiles>;                                          // a block of S or P: up to block_tiles tiles
using StatRing  = TilesRing<group_rows, 1>;                                           // a group's m, alpha, L or 1 / L

constexpr matmul_modifiers transposed {.transpose_in1 = true}; // a tile of K^T's grid holds K's tile as it is
constexpr sub_modifiers by_column {.bcast = Broadcast::Col};
constexpr sub_modifiers by_scalar {.bcast = Broadcast::Scalar};
constexpr mul_modifiers times_column {.bcast = Broadcast::Col};
constexpr mul_modifiers plus_times_column {.bcast = Broadcast::Col, .accumulate = true}; // y = acc + a * b's column 0
constexpr reduce_modifiers row_max {.kind = ReduceKind::Max};
constexpr reduce_modifiers row_sum {.kind = ReduceKind::Sum};
// P's exp is the fast one. With a 32-bit Dest it is the host's choice (exp_approx): the fast one, or
// the exact exp. That the choice exists only there is the 1.0 kernel's rule, kept for its accuracy in both modes.
// Math moves on while pack runs it.
constexpr bool p_exp_fast = exp_approx || !kernel_settings.dest_fp32;
// The fast exp does not clamp its input: the scores have their maximum subtracted, and its pack is the relu one.
constexpr vec_modifiers p_exp_on_pack {
    .form = p_exp_fast ? VecForm::Fast : VecForm::Exact, .clamp_input = !p_exp_fast, .thread = Thread::Pack, .input_scale = scale};
// The statistics are in column 0 of their tiles: their exp and reciprocal run on the left faces only.
// alpha's exp is the host's choice in either Dest mode: the polynomial, or the exact exp.
constexpr vec_modifiers exp_on_pack {
    .form = exp_approx ? VecForm::Polynomial : VecForm::Exact, .region = VecRegion::LeftFacesEvenColumns, .thread = Thread::Pack, .input_scale = scale};
// The sink term's: exp(scale (sink - m)) as exp(-scale (m - sink)), m - sink being the form with the scalar second.
constexpr vec_modifiers negated_exp_on_pack {
    .form = exp_on_pack.form, .region = exp_on_pack.region, .thread = Thread::Pack, .input_scale = scale ^ 0x80000000u};
constexpr vec_modifiers recip_on_pack {.region = VecRegion::LeftFaces, .thread = Thread::Pack};
constexpr pack_modifiers relu {.relu = true}; // the fast exp is not clamped at 0
constexpr pack_modifiers relu_summed {.relu = true, .l1_acc = Toggle::Runtime, .sum_tiles = true};
constexpr pack_modifiers added {.l1_acc = Toggle::On};

/// The K chunks a Q chunk meets: [first_chunk, end_chunk) of the k_chunks, the last of them last_tiles key tiles
/// wide. They follow from where the Q chunk is: q_tile, its first Q tile row's place in the sequence.
template <bool PerQChunk>
struct KRangeOf
{
    std::uint32_t q_tile;
    std::uint32_t first_chunk;
    std::uint32_t end_chunk;
    std::uint32_t last_tiles;

    /// Of the Q chunk whose first tile row is the sequence's row q_tile; [sent_first, sent_end) is what the reader
    /// sent for it, when it sends a range. The reader sends K and V by the same rule (reader_interleaved.cpp):
    ///   causal      up to the chunk with the last row's own keys, and of that chunk the tiles up to them
    ///   a window    from the chunk with the first key tile the first row sees to the one with the last the last sees
    ///   sent        the chunks sent, counted as the reader sends them: for a mask block map it skips the chunks the
    ///               map masks out, so chunk c here is the c-th one sent, not the tensor's. The last one sent has
    ///               the tensor's last chunk's width only when all were sent; narrower keys than that are -inf in
    ///               the mask tensor, which such an op always has
    ///   otherwise   all of them; the tensor's last K chunk holds the keys the others left
    static constexpr KRangeOf at(std::uint32_t q_tile, std::uint32_t sent_first, std::uint32_t sent_end)
    {
        const std::uint32_t reach = q_tile + Sq_chunk_t + window_after; // one past the last key tile its rows see
        std::uint32_t first       = k_range_sent ? sent_first : 0;
        std::uint32_t end         = k_range_sent ? sent_end : k_chunks;
        if (window != 0 && q_tile > window_before)
        {
            first = (q_tile - window_before) / Sk_chunk_t;
        }
        if (position_masks && (reach + Sk_chunk_t - 1) / Sk_chunk_t < end)
        {
            end = (reach + Sk_chunk_t - 1) / Sk_chunk_t;
        }
        std::uint32_t tiles = end == k_chunks && !causal ? last_width : Sk_chunk_t;
        if (position_masks && reach - (end - 1) * Sk_chunk_t < tiles)
        {
            tiles = reach - (end - 1) * Sk_chunk_t;
        }
        return {q_tile, first, end, tiles};
    }
};

/// Every Q chunk meets all the K chunks, wherever it is: compile-time values, and nothing to hold.
template <>
struct KRangeOf<false>
{
    static constexpr std::uint32_t q_tile      = 0;
    static constexpr std::uint32_t first_chunk = 0;
    static constexpr std::uint32_t end_chunk   = k_chunks;
    static constexpr std::uint32_t last_tiles  = last_width;

    static constexpr KRangeOf at(std::uint32_t, std::uint32_t, std::uint32_t)
    {
        return {};
    }
};

using KRange = KRangeOf<k_range_per_q_chunk>;

/// The first Q tile row's place in the sequence, for the Q chunk at `flat` of the op's Q chunks (q_per_head of them
/// per head, the heads one after the other), where the op's first Q chunk is the sequence's chunk q_chunk_offset.
/// With zigzag a head's chunks come from both ends in turn, as reader and writer take them (remap_q_index).
inline std::uint32_t q_tile_at(std::uint32_t flat, std::uint32_t q_chunk_offset)
{
    const std::uint32_t in_head = remap_q_index(flat, q_per_head, zigzag) % q_per_head;
    return (q_chunk_offset + in_head) * Sq_chunk_t;
}

/// Where a group of Q rows is in the Q chunk's stream of K chunks.
struct Group : KRange // the K chunks its Q chunk meets
{
    std::uint32_t chunk; // its K chunk
    std::uint32_t index; // its place among the chunk's groups

    /// The first group of a Q chunk that meets `range`.
    static constexpr Group opening(KRange range)
    {
        return Group {range, range.first_chunk, 0};
    }

    /// False for the group after the Q chunk's last one.
    constexpr bool exists() const
    {
        return chunk < end_chunk;
    }

    /// In the Q chunk's first K chunk: it has nothing from an earlier one.
    constexpr bool first() const
    {
        return chunk == first_chunk;
    }

    /// In the Q chunk's last K chunk: it is normalized into the output.
    constexpr bool last() const
    {
        return chunk == end_chunk - 1;
    }

    /// The first group of its K chunk.
    constexpr bool opens_chunk() const
    {
        return index == 0;
    }

    /// Its first Q tile row.
    constexpr std::uint32_t q_row() const
    {
        return index * group_rows;
    }

    /// Key tiles of its K chunk: the row length of its S and P.
    constexpr std::uint32_t width() const
    {
        return last() ? last_tiles : Sk_chunk_t;
    }

    /// Blocks a row of its S or P goes through Dest in: block_tiles tiles each, the last one what is left.
    constexpr std::uint32_t blocks() const
    {
        return !exists() ? 0 : (width() + block_tiles - 1) / block_tiles;
    }

    /// Tiles of block b of such a row, for b < blocks().
    constexpr std::uint32_t block_width(std::uint32_t b) const
    {
        return width() - b * block_tiles < block_tiles ? width() - b * block_tiles : block_tiles;
    }

    /// The group after it: of the same K chunk, or the first one of the next.
    constexpr Group next() const
    {
        return index + 1 < groups ? Group {*this, chunk, index + 1} : Group {*this, chunk + 1, 0};
    }
};

/// A statistic that goes from one K chunk to the next through two buffers: a chunk writes one, the next reads it.
struct Carried
{
    DataflowBuffer a, b;

    DataflowBuffer& written_by(const Group& g)
    {
        return g.chunk % 2 == 0 ? a : b;
    }

    DataflowBuffer& read_by(const Group& g)
    {
        return g.chunk % 2 == 0 ? b : a;
    }
};

/// A row of `Total` tile columns in blocks of `Step`, and one of what is left at its end: block(c0, n) for each, c0
/// its first column and n its width as a type (n.value: a compile-time value, for the block's spec).
template <std::uint32_t Step, std::uint32_t Total, class Block>
inline __attribute__((always_inline)) void for_each_col_block(const Block& block)
{
    for (std::uint32_t c0 = 0; c0 + Step <= Total; c0 += Step)
    {
        block(c0, std::integral_constant<std::uint32_t, Step> {});
    }
    if constexpr (Total % Step != 0)
    {
        block(Total - Total % Step, std::integral_constant<std::uint32_t, Total % Step> {});
    }
}

/// One block of scores, n tile columns from column c0: S[row, block] = Q row x K^T[:, block], packed to its place in
/// the row's S. n is a run-time value, at most block_tiles.
template <class QRow, class Kt, class SRow>
inline __attribute__((always_inline)) void scores_block(const QRow& q_row, const Kt& kt, const SRow& s_row, std::uint32_t c0, std::uint32_t n)
{
    const auto s = matmul<transposed>(q_row, cols_up_to<block_tiles>(kt, c0, n), reserve_back<BlockRing>());
    pack(s, cols_up_to<block_tiles>(s_row, c0, n));
    release(s);
}

/// One block of probabilities, n tile columns from column c0: P = exp(scale (S - m)), the exp on the pack thread.
/// Packed to its place in the row's P and, its tiles added up, into the row's L tile: the row's first block writes
/// that tile, the others add to it. n as for scores_block.
template <class SRow, class M, class PRow, class L>
inline __attribute__((always_inline)) void probabilities_block(const SRow& s_row, const M& m, const PRow& p_row, const L& l, std::uint32_t c0, std::uint32_t n)
{
    const auto p = vec_exp<p_exp_on_pack>(sub<by_column>(cols_up_to<block_tiles>(s_row, c0, n), m, reserve_back<BlockRing>()));
    pack<relu>(p, cols_up_to<block_tiles>(p_row, c0, n));
    pack<relu_summed>(p, l, c0 > 0);
    release(p);
}

/// N tiles of a mask added onto as many tiles of S: S is in L1 by then, and the packer adds onto it there.
template <std::uint32_t N, class Mask, class STiles>
inline __attribute__((always_inline)) void add_mask(const Mask& mask, const STiles& s)
{
    const auto bias = copy(mask, reserve_back<TilesRing<1, N>>());
    pack<added>(bias, s);
    release(bias);
}

/// Made mask tile t onto column c of a row of S that is `width` tiles long, when the row has that column.
template <class MadeMasks, class SRow>
inline __attribute__((always_inline)) void add_made_mask(const MadeMasks& made_masks, std::uint32_t t, const SRow& s_row, std::int32_t c, std::int32_t width)
{
    if (c >= 0 && c < width)
    {
        add_mask<1>(tile(made_masks, t), cols<1>(s_row, c));
    }
}

/// -inf onto columns [c0, c1) of a row of S that is `width` tiles long, as far as the row has them: the tile is
/// copied once and packed onto each.
template <class MadeMasks, class SRow>
inline __attribute__((always_inline)) void mask_out(const MadeMasks& made_masks, const SRow& s_row, std::int32_t c0, std::int32_t c1, std::int32_t width)
{
    c0 = c0 < 0 ? 0 : c0;
    c1 = c1 < width ? c1 : width;
    if (c0 < c1)
    {
        const auto bias = copy(tile(made_masks, minus_inf_tile), reserve_back<TilesRing<1, 1>>());
        for (std::int32_t c = c0; c < c1; ++c)
        {
            pack<added>(bias, cols<1>(s_row, c));
        }
        release(bias);
    }
}

/// The column, in g's K chunk, of the key tile at g's row i's own place in the sequence: negative or past the chunk
/// when that tile is in another chunk.
constexpr std::int32_t own_key_col(const Group& g, std::uint32_t i)
{
    return static_cast<std::int32_t>(g.q_tile + g.q_row() + i) - static_cast<std::int32_t>(g.chunk * Sk_chunk_t);
}

/// The mask step for row i of the group g, whose mask is `mask` when the op has a mask tensor:
///   a mask tensor         its tile onto every tile of the row's S, a block at a time: the whole K chunk's, since
///                         a tile of S past g's width is not read
///   a sliding window      -inf onto the tiles before the window, the leading edge's tile or two; and, not
///                         causal, the trailing edge's tile or two and -inf onto the tiles after them
///   causal                the diagonal tile onto the tile of the row's own keys, -inf onto the tiles after it
///   a partial key tile    in the K chunk that holds it, the writer's mask of that tile onto its tile of S
///   none of these         nothing
template <class Mask, class MadeMasks, class SRow>
inline __attribute__((always_inline)) void mask_scores_row(const Group& g, std::uint32_t i, const Mask& mask, const MadeMasks& made_masks, const SRow& s_row)
{
    if constexpr (provided_mask)
    {
        const auto mask_row = row(mask, i);
        for_each_col_block<block_tiles, Sk_chunk_t>(
            [&](std::uint32_t c0, auto n) __attribute__((always_inline))
            { add_mask<decltype(n)::value>(cols<decltype(n)::value>(mask_row, c0), cols<decltype(n)::value>(s_row, c0)); });
    }
    else
    {
        const std::int32_t width = g.width();
        if constexpr (window != 0)
        {
            const std::int32_t lead = own_key_col(g, i) - static_cast<std::int32_t>(window_lead_tiles);
            mask_out(made_masks, s_row, 0, lead - (window_lead_cols != 0 ? 1 : 0), width);
            if constexpr (window_lead_cols != 0)
            {
                add_made_mask(made_masks, leading_before_tile, s_row, lead - 1, width);
            }
            add_made_mask(made_masks, leading_tile, s_row, lead, width);
        }
        if constexpr (causal)
        {
            add_made_mask(made_masks, diagonal_tile, s_row, own_key_col(g, i), width);
            mask_out(made_masks, s_row, own_key_col(g, i) + 1, width, width);
        }
        else if constexpr (window != 0)
        {
            const std::int32_t trail = own_key_col(g, i) + static_cast<std::int32_t>(window_trail_tiles);
            add_made_mask(made_masks, trailing_tile, s_row, trail, width);
            if constexpr (window_trail_cols != 0)
            {
                add_made_mask(made_masks, trailing_after_tile, s_row, trail + 1, width);
            }
            mask_out(made_masks, s_row, trail + (window_trail_cols != 0 ? 2 : 1), width, width);
        }
        if constexpr (partial_key_cols != 0)
        {
            if (g.chunk == k_chunks - 1) // the K chunk with the tensor's last key tile
            {
                add_made_mask(made_masks, partial_key_mask_tile, s_row, last_width - 1, width);
            }
        }
    }
}

/// The mask step of the group g, when it exists, onto its S in s_slot: after the group's S is packed, before its row
/// max. A mask tensor's rows are waited for here and not before the group's S (they can come later than K), and
/// popped here.
template <class MaskIn, class MadeMasks, class SSlot>
inline __attribute__((always_inline)) void mask_scores(const Group& g, bool exists, MaskIn& mask_in, const MadeMasks& made_masks, const SSlot& s_slot)
{
    if constexpr (masked)
    {
        const auto mask = wait_front<mask_spec>(mask_in, exists);
        if (exists)
        {
            for (std::uint32_t i = 0; i < group_rows; ++i)
            {
                mask_scores_row(g, i, mask, made_masks, row(s_slot, i));
            }
        }
        release(mask);
    }
}

/// m = max(m_prev, row max of S) for the group g, whose S is at the front of scores: into row_maxes, for g's P blocks
/// and its alpha, and into the buffer the next K chunk reads m_prev from (unless g's chunk is the last).
template <class Unit>
inline __attribute__((always_inline)) void running_row_max(const Group& g, const Unit& unit, DataflowBuffer& scores, DataflowBuffer& row_maxes, Carried& m)
{
    const auto s   = cols_up_to<Sk_chunk_t>(peek_front<s_spec>(scores), 0, g.width()); // the key tiles of g's K chunk
    const auto acc = reserve_back<StatRing>();
    if (g.first())
    {
        reduce<row_max>(s, unit, acc);
    }
    else
    {
        reduce<row_max>(s, unit, peek_front<carried_m_spec>(m.read_by(g)), acc); // the pooling starts from m_prev
    }
    release(pack(acc, reserve_back<m_spec>(row_maxes)));
    if (!g.last() || sinks) // with sinks the last K chunk's m is read by its groups' outputs
    {
        release(pack(acc, reserve_back<carried_m_spec>(m.written_by(g))));
    }
    release(acc);
}

/// An [R, N] block of a group's U, from its row r0 and column c0: P V over the width key tiles of the group's K chunk
/// (a run-time value, at most Sk_chunk_t), plus alpha U_prev when the group has an earlier K chunk (carried), in one
/// Dest tensor and one pack.
template <std::uint32_t R, std::uint32_t N, class P, class V, class UPrev, class Alpha, class USlot>
inline __attribute__((always_inline)) void numerator_block(
    std::uint32_t width, bool carried, std::uint32_t r0, std::uint32_t c0, const P& p, const V& v, const UPrev& u_prev, const Alpha& alpha, const USlot& u_slot)
{
    const auto acc = reserve_back<TilesRing<R, N>>();
    matmul(cols_up_to<Sk_chunk_t>(rows<R>(p, r0), 0, width), rows_up_to<Sk_chunk_t>(cols<N>(v, c0), 0, width), acc);
    if (carried)
    {
        mul<plus_times_column>(cols<N>(rows<R>(u_prev, r0), c0), rows<R>(alpha, r0), acc); // after P V: acc += alpha U_prev
    }
    pack(acc, cols<N>(rows<R>(u_slot, r0), c0));
    release(acc);
}

/// R rows of a group's U, from its row r0: one block, or, a row wider than a Dest section, its blocks of u_cols tile
/// columns and what is left.
template <std::uint32_t R, class P, class V, class UPrev, class Alpha, class USlot>
inline __attribute__((always_inline)) void numerator_rows(
    std::uint32_t width, bool carried, std::uint32_t r0, const P& p, const V& v, const UPrev& u_prev, const Alpha& alpha, const USlot& u_slot)
{
    for_each_col_block<u_cols, vDHt>([&](std::uint32_t c0, auto n) __attribute__((always_inline))
                                     { numerator_block<R, decltype(n)::value>(width, carried, r0, c0, p, v, u_prev, alpha, u_slot); });
}

/// N tile columns of an output row, from its column c0: U times 1 / L, pushed to the output as they are packed.
template <std::uint32_t N, class URow, class Inverse>
inline __attribute__((always_inline)) void output_block(std::uint32_t c0, const URow& u_row, const Inverse& inv, DataflowBuffer& out)
{
    const auto o = mul<times_column>(cols<N>(u_row, c0), inv, reserve_back<TilesRing<1, N>>());
    release(pack(o, reserve_back<out_spec<N>>(out)));
    release(o);
}

/// A group's sink terms, exp(scale (sink - m)), into their buffer: sink is the Q chunk's, at the front of its buffer,
/// and m the group's row max after the last K chunk, at the front of m_last and popped here.
template <class SinkIn, class SinkTerms>
inline __attribute__((always_inline)) void sink_terms_of(DataflowBuffer& m_last, SinkIn& sink_in, SinkTerms& sink_terms)
{
    const auto sink = peek_front<sink_spec>(sink_in);
    const auto m    = wait_front<carried_m_spec>(m_last);
    const auto term = vec_exp<negated_exp_on_pack>(sub<by_scalar>(m, sink, reserve_back<StatRing>()));
    release(pack(term, reserve_back<sink_term_spec>(sink_terms)));
    release(term);
    release(m);
}

/// out = U / L for a group, after the last K chunk. An L tile holds the sum of its row's P tiles: the sum over its 32
/// columns comes first (with sinks it starts from the group's sink terms, at the front of their buffer), then 1 / L
/// on the pack thread, then each row of U times it: a row wider than a Dest section in the blocks of columns U was
/// computed in.
template <class Unit, class SinkTerms>
inline __attribute__((always_inline)) void output_rows(
    const Unit& unit, DataflowBuffer& l, DataflowBuffer& u, DataflowBuffer& inverse, DataflowBuffer& out, SinkTerms& sink_terms)
{
    {
        const auto l_tiles = wait_front<l_spec>(l);
        const auto acc     = reserve_back<StatRing>();
        if constexpr (sinks)
        {
            const auto terms = wait_front<sink_term_spec>(sink_terms);
            reduce<row_sum>(l_tiles, unit, terms, acc);
            release(terms);
        }
        else
        {
            reduce<row_sum>(l_tiles, unit, acc);
        }
        const auto inv_acc = vec_recip<recip_on_pack>(acc);
        release(pack(inv_acc, reserve_back<inverse_spec>(inverse)));
        release(inv_acc);
        release(l_tiles);
    }
    const auto inv       = wait_front<inverse_spec>(inverse);
    const auto u_rows_in = wait_front<u_spec>(u);
    for (std::uint32_t i = 0; i < group_rows; ++i)
    {
        for_each_col_block<u_cols, vDHt>([&](std::uint32_t c0, auto n) __attribute__((always_inline))
                                         { output_block<decltype(n)::value>(c0, row(u_rows_in, i), tile(inv, i), out); });
    }
    release(u_rows_in);
    release(inv);
}

/// Chunked attention: the sequence's Q chunk the op's first one is. A run-time arg, or from the sequence position, in
/// tokens, that the reader sends in a page, once. 0 for an op that is not chunked.
template <class ChunkStartIn>
inline __attribute__((always_inline)) std::uint32_t first_q_chunk_of_op(ChunkStartIn& chunk_start_in)
{
    if constexpr (chunked)
    {
        if (get_arg_val<std::uint32_t>(chunk_start_is_tensor_arg) != 0)
        {
            const auto start           = wait_front<chunk_start_spec>(chunk_start_in);
            const std::uint32_t tokens = read_value(start, chunk_start_word);
            release(start);
            return tokens / (Sq_chunk_t * tile_dim);
        }
        return get_arg_val<std::uint32_t>(q_chunk_offset_arg);
    }
    else
    {
        return 0;
    }
}

/// The K chunks met by the Q chunk whose first tile row is the sequence's row q_tile: when the reader sends them, from
/// the two words of its page.
template <class KRangeIn>
inline __attribute__((always_inline)) KRange k_range_of(std::uint32_t q_tile, KRangeIn& k_range_in)
{
    if constexpr (k_range_sent)
    {
        const auto sent                = wait_front<k_range_spec>(k_range_in);
        const std::uint32_t sent_first = read_value(sent, k_first_word);
        const std::uint32_t sent_end   = read_value(sent, k_end_word);
        release(sent);
        return KRange::at(q_tile, sent_first, sent_end);
    }
    else
    {
        return KRange::at(q_tile, 0, 0);
    }
}

/// The first stage of a pass: P of the group cur and S of the group after it, next, block by block: the exp on the
/// pack thread, the matmul on the FPU. After the last S block comes S(next)'s mask step. Each P block adds into its
/// row's tile of l_slot, cur's L. S(cur) and m(cur) are at the front of cur_scores and cur_row_maxes, P(cur) goes to
/// cur_probabilities and S(next) to next_scores, behind S(cur).
/// The Q chunk's first group has its S computed with no P beside it: the stage is then called with no buffers for cur
/// (AbsentBuffer) and has no P. The last group's P has no S beside it: next does not exist and has no blocks.
/// With last_p_beside_m the pass's last P block comes after the last S block and S(next)'s push: m(next) finds S(next)
/// whole in L1 without waiting behind that block's exp on the pack thread, and the exp runs next to m(next).
template <class LSlot, class MadeMasks, class MaskIn, class CurScores, class CurRowMaxes, class CurProbabilities>
inline __attribute__((always_inline)) void probabilities_and_scores(
    const Group& cur,
    const Group& next,
    const LSlot& l_slot,
    const MadeMasks& made_masks,
    DataflowBuffer& q_in,
    DataflowBuffer& k_in,
    MaskIn& mask_in,
    CurScores& cur_scores,
    CurRowMaxes& cur_row_maxes,
    CurProbabilities& cur_probabilities,
    DataflowBuffer& next_scores)
{
    constexpr bool has_cur = is_present<CurScores>;
    const bool has_next    = next.exists();
    const auto s           = wait_front<s_spec>(cur_scores);                                    // S(cur)
    const auto m_cur       = wait_front<m_spec>(cur_row_maxes);                                 // m(cur)
    const auto kt          = peek_front<kt_spec>(k_in, has_next);                               // K^T of next's chunk
    const auto q_next      = peek_front_rows<q_spec, group_rows>(q_in, next.q_row(), has_next); // Q rows of next: Q comes a row at a time
    const auto p_slot      = reserve_back<p_spec>(cur_probabilities);
    const auto s_slot      = reserve_back<s_spec>(next_scores, has_next);
    for (std::uint32_t i = 0; i < group_rows; ++i)
    {
        const bool ends_pass = last_p_beside_m && i == group_rows - 1; // the row whose last P block is left for after the loop
        for (std::uint32_t b = 0; (has_cur && b < cur.blocks()) || b < next.blocks(); ++b)
        {
            if constexpr (has_cur)
            {
                if (b < cur.blocks() && !(ends_pass && b + 1 == cur.blocks()))
                {
                    probabilities_block(row(s, i), tile(m_cur, i), row(p_slot, i), tile(l_slot, i), b * block_tiles, cur.block_width(b));
                }
            }
            if (b < next.blocks())
            {
                scores_block(row(q_next, i), kt, row(s_slot, i), b * block_tiles, next.block_width(b));
            }
        }
    }
    mask_scores(next, has_next, mask_in, made_masks, s_slot);
    release(s_slot);
    if constexpr (has_cur && last_p_beside_m)
    {
        const std::uint32_t i = group_rows - 1;
        const std::uint32_t b = cur.blocks() - 1;
        probabilities_block(row(s, i), tile(m_cur, i), row(p_slot, i), tile(l_slot, i), b * block_tiles, cur.block_width(b));
    }
    release(s);
    release(m_cur);
    release(p_slot);
}

} // namespace

void kernel_main()
{
    // A buffer only some builds of the kernel have is a buffer_if: nothing is made for it in the others.
    DataflowBuffer q_in(get_named_compile_time_arg_val("cb_q_in"));
    DataflowBuffer k_in(get_named_compile_time_arg_val("cb_k_in"));
    DataflowBuffer v_in(get_named_compile_time_arg_val("cb_v_in"));
    DataflowBuffer unit_in(get_named_compile_time_arg_val("cb_identity_scale_in"));
    DataflowBuffer scores(get_named_compile_time_arg_val("cb_scores"));
    DataflowBuffer probabilities(get_named_compile_time_arg_val("cb_probabilities"));
    DataflowBuffer row_maxes(get_named_compile_time_arg_val("cb_row_max"));
    DataflowBuffer alphas(get_named_compile_time_arg_val("cb_alpha"));
    DataflowBuffer inverse(get_named_compile_time_arg_val("cb_inverse"));
    DataflowBuffer out(get_named_compile_time_arg_val("cb_out"));
    Carried m {DataflowBuffer(get_named_compile_time_arg_val("cb_max_A")), DataflowBuffer(get_named_compile_time_arg_val("cb_max_B"))};
    Carried l {DataflowBuffer(get_named_compile_time_arg_val("cb_sum_A")), DataflowBuffer(get_named_compile_time_arg_val("cb_sum_B"))};
    Carried u {DataflowBuffer(get_named_compile_time_arg_val("cb_out_im_A")), DataflowBuffer(get_named_compile_time_arg_val("cb_out_im_B"))};
    // The mask buffer holds a mask tensor's tiles as they come, or the tiles the writer makes.
    buffer_if<DataflowBuffer, provided_mask> mask_in(get_named_compile_time_arg_val("cb_mask_in"));
    buffer_if<DataflowBuffer, made_masks_used> made_masks_in(get_named_compile_time_arg_val("cb_mask_in"));
    buffer_if<DataflowBuffer, sinks> sink_in(get_named_compile_time_arg_val("cb_attention_sink"));
    buffer_if<DataflowBuffer, sinks> sink_terms(get_named_compile_time_arg_val("cb_sink_term"));
    buffer_if<DataflowBuffer, chunked> chunk_start_in(get_named_compile_time_arg_val("cb_chunk_start_idx"));
    buffer_if<DataflowBuffer, k_range_sent> k_range_in(get_named_compile_time_arg_val("cb_k_range"));

    const std::uint32_t q_first  = get_arg_val<std::uint32_t>(q_first_arg);
    const std::uint32_t q_chunks = get_arg_val<std::uint32_t>(q_chunks_arg);
    // Pushed once by the writer and never popped (as 1.0): the reduce's scaler, and the mask tiles it makes.
    const auto unit       = peek_front<unit_spec>(unit_in);
    const auto made_masks = peek_front<made_masks_spec>(made_masks_in);

    const std::uint32_t q_chunk_offset = first_q_chunk_of_op(chunk_start_in);

    for (std::uint32_t q_chunk = 0; q_chunk < q_chunks; ++q_chunk)
    {
        // The next Q chunk reads the same K and V (it is of the same heads_per_kv heads): they stay in their buffers.
        const std::uint32_t kv_span = q_per_head * heads_per_kv;
        const bool keep_kv          = heads_per_kv != 0 && q_chunk + 1 < q_chunks && (q_first + q_chunk) / kv_span == (q_first + q_chunk + 1) / kv_span;

        const KRange range = k_range_of(q_tile_at(q_first + q_chunk, q_chunk_offset), k_range_in);
        LLK_ASSERT(range.first_chunk < range.end_chunk, "a Q chunk meets at least one K chunk");

        // S and m of the first group.
        {
            const Group first = Group::opening(range);
            AbsentBuffer no_group(0);
            probabilities_and_scores(first, first, absent_back<l_spec>(), made_masks, q_in, k_in, mask_in, no_group, no_group, no_group, scores);
            running_row_max(first, unit, scores, row_maxes, m);
        }

        for (Group cur = Group::opening(range); cur.exists(); cur = cur.next())
        {
            const Group next    = cur.next(); // its S, m and alpha are computed in this pass
            const bool has_next = next.exists();
            const bool carried  = !cur.first();                                   // cur has m, L and U from the K chunk before
            const bool kv_done  = next.opens_chunk() && !(cur.last() && keep_kv); // cur ends its K chunk, and K and V go

            if (kv_done)
            {
                pop_front<kt_spec>(k_in); // every S of cur's K chunk is issued: the next chunk's K^T is at the front
            }

            // P(cur) and S(next), block by block.
            const auto l_slot = reserve_back<l_spec>(l.written_by(cur)); // each P block adds into its row's L tile
            probabilities_and_scores(cur, next, l_slot, made_masks, q_in, k_in, mask_in, scores, row_maxes, probabilities, scores);

            // m(next): the FPU pools S(next) while the pack thread finishes the last blocks.
            if (has_next)
            {
                running_row_max(next, unit, scores, row_maxes, m);
            }

            // L(cur) = alpha L_prev + the row sums the P blocks left in l_slot: added onto them by the packer.
            const auto alpha = peek_front<alpha_spec>(alphas, carried); // alpha(cur)
            if (carried)
            {
                const auto l_prev = wait_front<l_spec>(l.read_by(cur));
                const auto scaled = mul<times_column>(l_prev, alpha, reserve_back<StatRing>());
                pack<added>(scaled, l_slot);
                release(scaled);
                release(l_prev);
            }
            release(l_slot);

            // U(cur) = P V + alpha U_prev, as many rows at a time as fit one Dest section: all but the last rows.
            const auto p      = wait_front<p_spec>(probabilities);
            const auto v      = peek_front<v_spec>(v_in);
            const auto u_prev = peek_front<u_spec>(u.read_by(cur), carried);
            const auto u_slot = reserve_back<u_spec>(u.written_by(cur));
            for (std::uint32_t r0 = 0; r0 < group_rows - u_last_rows; r0 += u_rows)
            {
                numerator_rows<u_rows>(cur.width(), carried, r0, p, v, u_prev, alpha, u_slot);
            }

            // alpha(next) = exp(scale (m_prev - m)): its exp on the pack thread, next to the P V of the last rows.
            if (has_next && !next.first())
            {
                const auto m_prev = wait_front<carried_m_spec>(m.read_by(next));
                const auto diff   = sub(m_prev, peek_front<m_spec>(row_maxes), reserve_back<StatRing>());
                const auto a      = vec_exp<exp_on_pack>(diff);
                release(pack(a, reserve_back<alpha_spec>(alphas)));
                release(a);
                release(m_prev);
            }

            numerator_rows<u_last_rows>(cur.width(), carried, group_rows - u_last_rows, p, v, u_prev, alpha, u_slot);
            release(p);
            release(u_slot);
            if (carried)
            {
                pop_front<alpha_spec>(alphas);
                pop_front<u_spec>(u.read_by(cur));
            }
            if (kv_done)
            {
                pop_front<v_spec>(v_in); // cur was the last group of its K chunk
            }

            if (cur.last())
            {
                if constexpr (sinks)
                {
                    sink_terms_of(m.written_by(cur), sink_in, sink_terms);
                }
                output_rows(unit, l.written_by(cur), u.written_by(cur), inverse, out, sink_terms);
            }
        }
        pop_front<q_spec>(q_in);
        if constexpr (sinks)
        {
            pop_front<sink_spec>(sink_in); // the Q chunk's sink: its groups have read it
        }
    }
}
