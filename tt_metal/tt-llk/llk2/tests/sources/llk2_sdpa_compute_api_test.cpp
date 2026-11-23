// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the LLK 2.0 tt-metal kernel kernels/sdpa.cpp, the file itself, on one core in the tt-llk harness.
//
// This file gives the kernel what tt-metal's build would (tests/helpers/tt_metal_kernel.h): its positional
// compile-time args from the test's parameters, named args for the buffer ids, formats and the group size, its one
// run-time arg, and the harness's stand-in for tt-metal's DataflowBuffer (tests/helpers/test_dataflow_buffer.h).
// run_kernel does the TEST ONLY setup below and calls the kernel's kernel_main.
//
// What is emulated, and what tt-metal does instead:
//   - Q, K, V, the mask and the reduce's scaler: the host wrote them to L1 in the order the kernel pops them (the
//     reader and writer kernels' job), and preload() marks them pushed. The mask is a mask tensor's tiles as the
//     reader sends them (per Q chunk and K chunk, Sq_chunk_t rows of Sk_chunk_t tiles), or the two tiles the writer
//     makes when the last key tile is partial (-inf, and that tile's mask), or nothing. K and V hold every K chunk once per Q chunk (with
//     KEEP_KV: once, for all the Q chunks, as the reader sends a single K chunk once per head), so wait_front never
//     waits. K is laid out as the reader lays it out: each chunk a [DHT, SK] grid of K's tiles.
//   - out: holds every output row; nothing pops it (tt-metal: the writer kernel).
//   - the intermediates (scores, probabilities, row max, alpha, 1 / L, and the two buffers each of m, L and U):
//     pushed by pack and popped by unpack inside this kernel, so their counters must be 0 before either thread looks
//     at them. The pack thread zeroes them and all three threads meet at a barrier before the body starts.
//     tt-metal: the firmware zeroes every CB's stream counters before the kernel is released.
//   - buffer_configs stands in for the host's CB config, sized as sdpa_program_factory.cpp sizes it.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#include "tt_metal_kernel.h" // TEST ONLY: what tt-metal's build gives a kernel

// tt-metal's host passes the kernel's settings as these defines (sdpa_program_factory.cpp).
#define LLK2_DEST_FP32     DST_ACCUM_MODE
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#define EXP_APPROX_MODE    (!EXACT_ALPHA_EXP) // the op's exp_approx_mode: alpha's exp is the polynomial, or the exact one
#include "kernel_settings.h"
#include "tensor/tensor.h"

namespace
{

constexpr DataFormat format = static_cast<DataFormat>(UNPACK_A_IN); // the host's choice for BF16 inputs
// S, P, L and 1 / L: with a 32-bit Dest the host keeps them in Float32 (sdpa_program_factory.cpp: qk_im_df, sum_df).
constexpr DataFormat wide_format = llk2::kernel_settings.dest_fp32 ? DataFormat::Float32 : format;

// MASK, the test's mask kind.
constexpr std::uint32_t no_mask          = 0;
constexpr std::uint32_t partial_key_tile = 1; // the last key tile holds PARTIAL_KEY_COLS keys: the writer's two tiles
constexpr std::uint32_t mask_tensor      = 2; // the op has a mask tensor: the reader's tiles

// Buffer ids, in the order the host allocates them.
enum : std::uint16_t
{
    q_id,
    k_id,
    v_id,
    unit_id,
    mask_id,
    u_a_id,
    u_b_id,
    max_a_id,
    max_b_id,
    sum_a_id,
    sum_b_id,
    out_id,
    scores_id,
    probabilities_id,
    row_max_id,
    alpha_id,
    inverse_id,
    buffer_count
};

/// One buffer's two named args.
#define BUFFER_ARGS(name, id, format)                      \
    {name, id},                                            \
    {                                                      \
        name "_format", static_cast<std::uint32_t>(format) \
    }

/// The named compile-time args the kernel reads.
constexpr llk2_test::NamedArg named_args[] = {
    // The op, as the host names it: plain non-causal streaming attention. The features this test leaves off
    // (causal, chunked, a sliding window, attention sinks, zigzag, a sent K range) have tt-metal's reader and writer
    // as their other half.
    {"DHt", DHT},
    {"vDHt", VDHT},
    {"Sq_chunk_t", SQ},
    {"q_num_chunks", Q_CHUNKS}, // Q chunks of one head
    {"Sk_chunk_t", SK},
    {"k_num_chunks", K_CHUNKS},
    {"valid_Skt", (K_CHUNKS - 1) * SK + LAST_WIDTH},
    {"scale", SCALE_FP32},
    {"is_causal", 0},
    {"use_provided_mask", MASK == mask_tensor ? 1 : 0},
    {"is_chunked", 0},
    {"sliding_window_size", 0},
    {"use_attention_sink", 0},
    {"k_partial_col", MASK == partial_key_tile ? PARTIAL_KEY_COLS : 0},
    {"use_zigzag", 0},
    {"use_k_range", 0},
    {"kv_retain_heads", KEEP_KV ? 1 : 0}, // heads of Q per K / V (the test's Q is one head), or off
    {"q_rows_per_group", GROUP_ROWS},
    {"last_p_beside_m", GROUP_ROWS* SK >= 6 * (is_fp32_dest_acc_en ? 4 : 8) ? 1 : 0}, // the host's rule: 6 Dest sections of tiles
    BUFFER_ARGS("cb_q_in", q_id, format),
    BUFFER_ARGS("cb_k_in", k_id, format),
    BUFFER_ARGS("cb_v_in", v_id, format),
    BUFFER_ARGS("cb_identity_scale_in", unit_id, format),
    BUFFER_ARGS("cb_mask_in", mask_id, format),
    BUFFER_ARGS("cb_attention_sink", unit_id, format), // without a sink the kernel reads neither: placeholders, as the host's
    BUFFER_ARGS("cb_sink_term", unit_id, wide_format),
    BUFFER_ARGS("cb_chunk_start_idx", unit_id, DataFormat::Int32), // the reader's pages of words: placeholders too
    BUFFER_ARGS("cb_k_range", unit_id, DataFormat::Int32),
    BUFFER_ARGS("cb_scores", scores_id, wide_format),
    BUFFER_ARGS("cb_probabilities", probabilities_id, wide_format),
    BUFFER_ARGS("cb_row_max", row_max_id, format),
    BUFFER_ARGS("cb_alpha", alpha_id, format),
    BUFFER_ARGS("cb_inverse", inverse_id, wide_format),
    BUFFER_ARGS("cb_max_A", max_a_id, format),
    BUFFER_ARGS("cb_max_B", max_b_id, format),
    BUFFER_ARGS("cb_sum_A", sum_a_id, wide_format),
    BUFFER_ARGS("cb_sum_B", sum_b_id, wide_format),
    BUFFER_ARGS("cb_out_im_A", u_a_id, format),
    BUFFER_ARGS("cb_out_im_B", u_b_id, format),
    BUFFER_ARGS("cb_out", out_id, format),
};

} // namespace

/// tt-metal: the kernel's named compile-time arg `name`.
constexpr std::uint32_t get_named_compile_time_arg_val(const char* name)
{
    return llk2_test::find_named_arg(named_args, name);
}

/// tt-metal: the kernel's run-time arg i. The kernel reads 4, its first Q chunk, and 5, how many it has.
template <class T>
constexpr T get_arg_val(int i)
{
    return i == 5 ? Q_CHUNKS : 0;
}

#include "sdpa.cpp" // the kernel file, as tt-metal builds it

void run_kernel(RUNTIME_PARAMETERS params)
{
    using llk2_test::buffer_configs;
    constexpr std::uint32_t page    = llk2::tile_bytes(format, llk2::default_tile_datums);
    constexpr std::uint32_t wide    = llk2::tile_bytes(wide_format, llk2::default_tile_datums) / page; // pages of `format` a wide_format tile takes
    constexpr std::uint32_t chunks  = (KEEP_KV ? 1 : Q_CHUNKS) * K_CHUNKS; // K and V chunks the kernel pops
    constexpr std::uint32_t k_tiles = chunks * DHT * SK, v_tiles = chunks * SK * VDHT;
    constexpr std::uint32_t mask_tiles  = MASK == mask_tensor ? Q_CHUNKS * K_CHUNKS * SQ * SK : (MASK == partial_key_tile ? 2 : 0);
    constexpr std::uint32_t group_tiles = GROUP_ROWS * SK;

    // TEST ONLY: the host's CB config. buffer_B holds K, then V, then the scaler, then the mask; buffer_C the intermediates.
    buffer_configs[q_id]    = {params.buffer_A[0], page, Q_CHUNKS * SQ * DHT};
    buffer_configs[k_id]    = {params.buffer_B[0], page, k_tiles};
    buffer_configs[v_id]    = {params.buffer_B[k_tiles], page, v_tiles};
    buffer_configs[unit_id] = {params.buffer_B[k_tiles + v_tiles], page, 1};
    buffer_configs[mask_id]        = {params.buffer_B[k_tiles + v_tiles + 1], page, mask_tiles};
    buffer_configs[out_id]  = {params.buffer_Res[0], page, Q_CHUNKS * SQ * VDHT};
    std::uint32_t next      = 0;
    const auto scratch             = [&](std::uint16_t id, std::uint32_t pages, std::uint32_t page_pages = 1)
    {
        buffer_configs[id] = {params.buffer_C[next], page_pages * page, pages};
        next += pages * page_pages;
    };
    scratch(scores_id, 2 * group_tiles, wide);
    scratch(probabilities_id, group_tiles, wide);
    scratch(row_max_id, GROUP_ROWS);
    scratch(alpha_id, 2 * GROUP_ROWS);
    scratch(inverse_id, GROUP_ROWS, wide);
    scratch(max_a_id, SQ);
    scratch(max_b_id, SQ);
    scratch(sum_a_id, SQ, wide);
    scratch(sum_b_id, SQ, wide);
    scratch(u_a_id, SQ * VDHT);
    scratch(u_b_id, SQ * VDHT);

    // TEST ONLY: what the reader and writer kernels and the firmware do in tt-metal (see the top of the file).
    DataflowBuffer(q_id).preload(Q_CHUNKS * SQ * DHT);
    DataflowBuffer(k_id).preload(k_tiles);
    DataflowBuffer(v_id).preload(v_tiles);
    DataflowBuffer(unit_id).preload(1);
    DataflowBuffer(mask_id).preload(mask_tiles);
    for (std::uint16_t id = u_a_id; id < buffer_count; ++id)
    {
        DataflowBuffer(id).zero_counters();
    }
    llk2_test::harness_barrier(); // every counter written before any thread reads one

    kernel_main();
}
