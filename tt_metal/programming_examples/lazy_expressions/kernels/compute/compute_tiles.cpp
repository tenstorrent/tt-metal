// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "views/compute_view.h"
#include "views/transform_view.h"
#include "views/tile_views.h"

namespace NAMESPACE {
void MAIN {
    // rt args are:
    // - n_tiles
    // - any scalar params used in expression tree
    const auto n_tiles = get_arg_val<uint32_t>(0);

    using namespace tt;
    using namespace views;

    // ct args are:
    // - num_tiles_per_cycle
    // - cb_indices for input and internal nodes in postfix order
    // - cb_index for output tensor

    // 5 circular buffers
    using View = MakeComputeView<5>;

    View::init_tiles(init_sfpu<c_0, c_4>());

    // fused addcmul
    // c_0 + ((c_1 * value) * c_3) -> c_4
    View::compute_tiles(
        n_tiles,
        // c_1 * value -> c_2
        with_cb_ids</*0*/ c_1, /*1*/ c_2>(
            copy<0 /*c_1*/>(0, 0), mul_unary(0, get_arg_val<uint32_t>(1)), pack<1 /*c_2*/>(0)),
        // c_4 * c_2 -> c_4
        with_cb_ids</*0*/ c_2, /*1*/ c_3, /*2*/ c_2>(mul<0 /*c_2*/, 1 /*c_3*/>(0, 0, 0), pack<2 /*c_2*/>(0)),
        // c_0 + c_4 -> c_3
        with_cb_ids</*0*/ c_0, /*1*/ c_2, /*2*/ c_4>(add<0 /*c_0*/, 1 /*c_2*/>(0, 0, 0), pack<2 /*c_4*/>(0)));
}
}  // namespace NAMESPACE
