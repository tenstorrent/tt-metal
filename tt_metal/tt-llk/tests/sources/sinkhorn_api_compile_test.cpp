// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Build-only sentinel for the Blackhole sinkhorn compute-API wrapper header.
//
// api/compute/experimental/sinkhorn.h includes llk_api/experimental/llk_sfpu/llk_math_sinkhorn.h
// into every TRISC_MATH kernel that includes it. _llk_math_sinkhorn_4x4_init_ there is a
// non-template inline function, so its body is checked as soon as the header is included,
// whether or not sinkhorn_4x4_init() is ever called. No in-tree kernel includes the header;
// of the tests, sinkhorn_lconst_neg1_regression_test.cpp does, but dispatches only a one-face,
// full-rectangle sinkhorn_4x4. This driver adds what that leaves uncompiled: the init call,
// the row-max-sub entry point (a non-template inline, so only a call makes the compiler
// generate it), the four-face forwarder the compute API defaults to, the SINGLE_SUBMAT arm,
// and both arms of the padding zeroing (whole-vector store and the per-lane masked store).
// test_sinkhorn_api_compile.py only builds it (never runs it). The other two threads are
// empty.

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_MATH

// The wrapper header is written against the tt-metal JIT build, which defines DST_ACCUM_MODE;
// the harness exposes the same value as is_fp32_dest_acc_en (same idiom as topk_test.cpp).
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "experimental/llk_sfpu/llk_math_sinkhorn.h"
#undef DST_ACCUM_MODE

using namespace ckernel;

// The EPS_BITS default of sinkhorn_4x4() in api/compute/experimental/sinkhorn.h.
static constexpr std::uint32_t DEFAULT_EPS_BITS = 0x35890000;
// Every entry point below takes a DEST tile index; one tile is enough to compile them.
static constexpr std::uint32_t INPUT_INDEX = 0;

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    _llk_math_sinkhorn_4x4_init_();

    // The comb per-row max-sub, as sinkhorn_row_max_sub() dispatches it.
    _llk_math_sinkhorn_row_max_sub_(INPUT_INDEX);

    // The forwarder with the defaults of sinkhorn_4x4(): four faces, both sub-matrices, full
    // 32x32 rectangle. Instantiates every face arm of _sinkhorn_4x4_; a full rectangle
    // compiles no padding store.
    _llk_math_sinkhorn_4x4_<4 /*NUM_FACES_USED*/, 20 /*ITERS*/, DEFAULT_EPS_BITS, false /*SINGLE_SUBMAT*/, 32 /*VALID_H*/, 32 /*VALID_W*/>(INPUT_INDEX);

    // The arms the defaults leave out: a single sub-matrix, and a 3x4 rectangle inside the
    // one 16x16 face of a one-face set. _sinkhorn_zero_rect_padding_addrs_ then takes both of
    // its arms: the masked store (row and column masks under v_if) for the two vectors at
    // rows 0..3 of face 0, which mix valid and padding lanes, and the whole-vector store
    // for every vector beyond them.
    _llk_math_sinkhorn_4x4_<1 /*NUM_FACES_USED*/, 1 /*ITERS*/, DEFAULT_EPS_BITS, true /*SINGLE_SUBMAT*/, 3 /*VALID_H*/, 4 /*VALID_W*/>(INPUT_INDEX);
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif
