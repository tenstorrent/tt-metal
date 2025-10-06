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
    const auto value = get_arg_val<uint32_t>(1);

    // ct args are:
    // - num_tiles_per_cycle
    // - cb_indices for any intermediate buffered tiles in expression tree
    // - cb_indices for input tensors
    // - cb_index for output tensor

    // 4 circular buffers: cb_in0, cb_in1, cb_in2, cb_out0
    // and append 2 ct args (num_tiles_per_cycle, cb_temp)
    const auto view = views::MakeComputeView<4, 2>();

    view.init_tiles(views::init_sfpu<tt::c_0, tt::c_3>());

    // fused addcmul
    // c_0 + ((c_1 * value) * c_2) -> c_3
    const auto addcmul =
        // c_1 * value -> c_4
        views::with_cb_ids</*0*/ tt::c_1, /*1*/ tt::c_4>(
            views::copy<0 /*c_1*/>(0, 0) | views::mul_unary(0, value) | views::pack<1 /*c_4*/>(0)) |
        // c_4 * c_2 -> c_4
        views::with_cb_ids</*0*/ tt::c_4, /*1*/ tt::c_2, /*2*/ tt::c_4>(
            views::mul<0 /*c_4*/, 1 /*c_2*/>(0, 0, 0) | views::pack<2 /*c_4*/>(0)) |
        // c_0 + c_4 -> c_3
        views::with_cb_ids</*0*/ tt::c_0, /*1*/ tt::c_4, /*2*/ tt::c_3>(
            views::add<0 /*c_0*/, 1 /*c_4*/>(0, 0, 0) | views::pack<2 /*c_3*/>(0));

    view.compute_tiles(n_tiles, addcmul);
}
}  // namespace NAMESPACE
