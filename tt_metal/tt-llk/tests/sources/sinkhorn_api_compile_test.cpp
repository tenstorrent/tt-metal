// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Build-only sentinel for the Blackhole sinkhorn compute-API wrapper header.
//
// api/compute/experimental/sinkhorn.h includes llk_api/experimental/llk_sfpu/llk_math_sinkhorn.h
// into every TRISC_MATH kernel that includes it. _llk_math_sinkhorn_4x4_init_ there is a
// non-template inline function, so its body is checked as soon as the header is included,
// whether or not sinkhorn_4x4_init() is ever called. No in-tree kernel includes the header, so
// nothing else ever compiles it: this driver does, and test_sinkhorn_api_compile.py only builds
// it (never runs it).
//
// The math thread calls the init the way sinkhorn_4x4_init() dispatches it and instantiates
// the sinkhorn_4x4 forwarder, which no in-tree kernel does either; the other two threads are
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

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
    _llk_math_sinkhorn_4x4_init_();

    // The forwarder with the defaults of sinkhorn_4x4() in api/compute/experimental/sinkhorn.h:
    // four faces, both sub-matrices, full 32x32 rectangle. Instantiates every face arm of
    // _sinkhorn_4x4_.
    _llk_math_sinkhorn_4x4_<4 /* NUM_FACES_USED */, 20 /* ITERS */, 0x35890000 /* EPS_BITS */, false /* SINGLE_SUBMAT */, 32 /* VALID_H */, 32 /* VALID_W */>(
        0 /* input_index */);

    // The arms the defaults leave out: a single sub-matrix and a rectangle smaller than the
    // face set, which selects the padding-zeroing path.
    _llk_math_sinkhorn_4x4_<1 /* NUM_FACES_USED */, 1 /* ITERS */, 0x35890000 /* EPS_BITS */, true /* SINGLE_SUBMAT */, 16 /* VALID_H */, 16 /* VALID_W */>(
        0 /* input_index */);
}

#endif

#ifdef LLK_TRISC_PACK

void run_kernel([[maybe_unused]] RUNTIME_PARAMETERS params)
{
}

#endif
