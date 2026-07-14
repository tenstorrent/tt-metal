// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// single_tile matmul (out = in0 x in1, a 1x1x1 matmul), written in the
// static-state-tracking style. SST variant of
//   tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp
// Launched by tests/tt_metal/tt_metal/llk/test_single_core_matmul_compute.cpp
// (TensixTestSingleCoreSingleTileComputeMatmulStaticState), verified against the
// matmul golden. Exercises the SST matmul + tiled pack_tile ops.

#include <cstdint>

// Include first: brings common_globals.h (MATH/PACK/UNPACK) + get_compile_time_arg_val
// before our defs.h, so our #ifndef-guarded macros defer to the force-included ones.
#include "api/compute/compute_kernel_api.h"

#include "experiments/static-state-tracking/compute/ops.h"
#include "experiments/static-state-tracking/inc/control.h"

void kernel_main() {
    using namespace sst;
    using namespace sst::compute;
    using namespace sst::tensor;

    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);
    const uint32_t out_cb = get_compile_time_arg_val(2);

    using TileT = Tile32x32_Float16_b;

    // Matmul output is tiled, so DST stays in the natural tiled layout the default
    // packer expects -> Remap=false. hw_startup configures both input operands.
    auto s0 = hw_startup<TileT, TileT, /*Remap=*/false>();

    auto out = Tensor<TileT, Cb>::reserve_back(out_cb, 1);
    sst::compute::tile_regs_acquire();
    auto in0 = Tensor<TileT, Cb>::wait_front(in0_cb, 1);
    auto in1 = Tensor<TileT, Cb>::wait_front(in1_cb, 1);
    // out[0] = in0[0] x in1[0]  (in0 -> SrcB, in1 -> SrcA), accumulated into DST[0].
    auto s1 = matmul(s0, in0, in1, /*i0=*/0, /*i1=*/0, /*dst=*/0);
    sst::compute::tile_regs_commit();

    sst::compute::tile_regs_wait();
    // Pack DST[0] -> out[0] in natural (tiled) layout.
    auto s_out = pack_tile(s1, out, /*dst_idx=*/0);
    pop_front(in0);
    pop_front(in1);
    sst::compute::tile_regs_release();
    push_back(out);
    (void)s_out;
}
