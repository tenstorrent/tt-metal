// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE stage 5 writer, one core per column block: the T delta tiles (CB 16) -> the output pages first..first+T-1 (the
// state shift is the reader's).  With INJECT (T = 1): the delta tile's rows 0-3 become four row-0 tiles in CB 18 (zero
// elsewhere: the layer's permute of the delta into the branch-major layout), the compute adds the residual blocks,
// and the four injected tiles (CB 20) go to pages b * 20 + first of the injected residual instead of the delta.
// Compile-time args: 0 T, 1 INJECT, then TensorAccessorArgs(out), (injected).  Runtime args: 0 out addr, 1 the first
// tile of this core's column block, 2 injected addr.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t T = get_compile_time_arg_val(0);
constexpr uint32_t INJECT = get_compile_time_arg_val(1);
constexpr uint32_t BF16_TILE = 2048, BF16_FACE = 512, BF16_ROW = 32, BLOCK_TILES = 20;
constexpr uint32_t c_out = 16, c_rows = 18, c_inj = 20;

void kernel_main() {
    constexpr auto a_out = TensorAccessorArgs<2>();
    constexpr auto a_inj = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(a_out, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1);
    Noc noc;
    DataflowBuffer o(c_out);
    {
        FUSED_ZONE("fz_pl_conv_w_delta");
        o.wait_front(T);
        if constexpr (!INJECT) {
            for (uint32_t t = 0; t < T; ++t) {
                noc.async_write(
                    o, out, BF16_TILE, {.offset_bytes = t * BF16_TILE}, {.page_id = first + t, .offset_bytes = 0});
            }
            noc.async_write_barrier();
            o.pop_front(T);
            return;
        }
    }
    // INJECT: the delta tile's row b (face 0 row b, face 1 row b: 32 bytes each) -> row 0 of tile b in CB 18
    {
        FUSED_ZONE("fz_pl_conv_w_inject");
        const auto injected = TensorAccessor(a_inj, get_arg_val<uint32_t>(2));
        DataflowBuffer drows(c_rows), inj(c_inj);
        drows.reserve_back(4);
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(o.get_read_ptr());
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drows.get_write_ptr());
        for (uint32_t w = 0; w < 4 * BF16_TILE / 4; ++w) {
            dst[w] = 0;
        }
        for (uint32_t b = 0; b < 4; ++b) {
            for (uint32_t face = 0; face < 2; ++face) {
                const uint32_t s = (face * BF16_FACE + b * BF16_ROW) / 4;   // delta tile, face `face`, row b
                const uint32_t d = (b * BF16_TILE + face * BF16_FACE) / 4;  // tile b, face `face`, row 0
                for (uint32_t k = 0; k < BF16_ROW / 4; ++k) {
                    dst[d + k] = src[s + k];
                }
            }
        }
        drows.push_back(4);
        o.pop_front(T);
        inj.wait_front(4);
        for (uint32_t b = 0; b < 4; ++b) {
            noc.async_write(
                inj,
                injected,
                BF16_TILE,
                {.offset_bytes = b * BF16_TILE},
                {.page_id = b * BLOCK_TILES + first, .offset_bytes = 0});
        }
        noc.async_write_barrier();
        inj.pop_front(4);
    }
}
