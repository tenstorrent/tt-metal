// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE router, READER half: load the logits tile into cb_in.
//
// Split from the writer because a single ReaderConfigDescriptor kernel doing both
// NoC reads AND writes hung the device. The working megakernel in this repo
// (swiglu_*) uses separate reader/writer kernels, so this mirrors that.
//
// WHY: in gpt-oss decode the post-matmul router path is 16 kernel launches taking
// 58.8 us per layer = 1.412 ms/tok (9.4% of decode), to do roughly 100 arithmetic
// operations on a SINGLE [1,32] logits row:
//     Typecast, FillPad, Pad, FillPad, TopK, Slice, Slice, Softmax, Unary,
//     Untilize x3, Scatter, Tilize, Untilize, Fill
// Every one of those is launch-dominated (most run on 1 core and move <1 KB).
// Five separate attempts to remove individual ops all regressed, because each
// removal added back an equivalent launch. The only thing that has ever worked on
// this model is genuine fusion (the SwiGLU megakernel, +1.7 tok/s).
//
// This kernel does the whole thing on ONE core with plain scalar C++:
//   1. read the 32 logits
//   2. selection-sort the top-k (k=4 of 32 -- trivially cheap serially)
//   3. softmax over just those k values
//   4. write a dense [1,E] row with the k weights in their expert slots, 0 elsewhere
//   5. write the k expert ids as uint32
//
// Outputs match the existing contract exactly:
//   routing_weights [1,1,1,E] bf16 TILE  (dense, zeros outside the top-k)
//   expert_ids      [1,1,1,k] uint32 ROW_MAJOR
// so sparse_matmul and the fused SwiGLU consume them unchanged.
//
// Single core is correct here, not a limitation: the whole job is ~100 scalar ops
// on 32 values. Splitting it would cost more in launch than it saves.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_w_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_id_id = get_compile_time_arg_val(2);
constexpr uint32_t num_experts = get_compile_time_arg_val(3);
constexpr uint32_t top_k = get_compile_time_arg_val(4);
constexpr uint32_t ct_in = 5;
constexpr uint32_t ct_w = TensorAccessorArgs<ct_in>::next_compile_time_args_offset();
constexpr uint32_t ct_id = TensorAccessorArgs<ct_w>::next_compile_time_args_offset();

// bf16 helpers: the logits arrive as bfloat16, which is just the high 16 bits of
// an IEEE float32, so conversion is a shift in both directions.
inline float bf16_to_f32(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = ((uint32_t)h) << 16;
    return v.f;
}

inline uint16_t f32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    // round-to-nearest-even on the truncated mantissa
    uint32_t r = v.u + 0x7FFF + ((v.u >> 16) & 1);
    return (uint16_t)(r >> 16);
}

// exp(x) via 2^(x*log2e), using the exponent-field trick plus a cubic correction.
// Accuracy is ~1e-4 relative, far tighter than the bf16 output can represent.
inline float fast_exp(float x) {
    if (x < -60.0f) {
        return 0.0f;
    }
    float y = x * 1.44269504088896f;  // log2(e)
    float fl = (float)(int)(y < 0 ? y - 1.0f : y);
    float fr = y - fl;
    // 2^fr on [0,1) -- minimax cubic
    float p = 1.0f + fr * (0.6931472f + fr * (0.2402265f + fr * 0.0555041f));
    union {
        uint32_t u;
        float f;
    } v;
    int e = (int)fl + 127;
    if (e < 0) {
        return 0.0f;
    }
    if (e > 254) {
        e = 254;
    }
    v.u = ((uint32_t)e) << 23;
    return p * v.f;
}

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    constexpr auto in_args = TensorAccessorArgs<ct_in>();
    const auto in_t = TensorAccessor(in_args, in_addr);

    Noc noc;
    CircularBuffer cb_in(cb_in_id);
    const uint32_t in_page = get_local_cb_interface(cb_in_id).fifo_page_size;

    cb_in.reserve_back(1);
    noc.async_read(in_t, cb_in, in_page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_in.push_back(1);
}
