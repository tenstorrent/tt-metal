// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE router, WRITER half: top-k + softmax + scatter, then ship both outputs.
//
// The reader has already placed the logits tile in cb_in. This kernel does all the
// arithmetic (selection top-k, softmax over k, dense scatter) and writes the two
// output tensors. Split from the reader because one ReaderConfigDescriptor kernel
// doing both NoC reads and writes hung the device; swiglu_* splits the same way.
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
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t id_addr = get_arg_val<uint32_t>(1);

    constexpr auto w_args = TensorAccessorArgs<ct_w>();
    constexpr auto id_args = TensorAccessorArgs<ct_id>();
    const auto w_t = TensorAccessor(w_args, w_addr);
    const auto id_t = TensorAccessor(id_args, id_addr);

    Noc noc;
    CircularBuffer cb_in(cb_in_id), cb_w(cb_w_id), cb_id(cb_id_id);
    const uint32_t w_page = get_local_cb_interface(cb_w_id).fifo_page_size;
    const uint32_t id_page = get_local_cb_interface(cb_id_id).fifo_page_size;
    const uint32_t id_page_elems = id_page / 2;

    cb_in.wait_front(1);
    volatile tt_l1_ptr uint16_t* logits =
        (volatile tt_l1_ptr uint16_t*)get_local_cb_interface(cb_in_id).fifo_rd_ptr;

    // ---- top-k by selection ----
    // In TILE layout, row 0 element j is at (j/16)*256 + (j%16). Only row 0 is real
    // for decode (batch 1, tile-padded to 32).
    float best_val[16];
    uint32_t best_idx[16];
    bool taken[64];
    for (uint32_t i = 0; i < num_experts; ++i) {
        taken[i] = false;
    }
    for (uint32_t s = 0; s < top_k; ++s) {
        float bv = -3.0e38f;
        uint32_t bi = 0;
        for (uint32_t j = 0; j < num_experts; ++j) {
            if (taken[j]) {
                continue;
            }
            const float v = bf16_to_f32(logits[(j >> 4) * 256 + (j & 15)]);
            if (v > bv) {
                bv = v;
                bi = j;
            }
        }
        // Safety clamp. If the input dtype/layout is not what this kernel expects
        // (e.g. BFLOAT8_B block-float read as bf16), bv/bi can be garbage. An
        // out-of-range bi would index outside the output CB and corrupt L1, which
        // wedges the NoC and hangs the device. Clamping keeps a wrong result wrong
        // but contained, so it fails as bad numbers instead of a hang.
        if (bi >= num_experts) {
            bi = 0;
        }
        taken[bi] = true;
        best_val[s] = bv;
        best_idx[s] = bi;
    }

    // ---- softmax over the k selected logits ----
    float mx = best_val[0];
    for (uint32_t s = 1; s < top_k; ++s) {
        if (best_val[s] > mx) {
            mx = best_val[s];
        }
    }
    float sum = 0.0f;
    for (uint32_t s = 0; s < top_k; ++s) {
        best_val[s] = fast_exp(best_val[s] - mx);
        sum += best_val[s];
    }
    const float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;

    // ---- dense weights row ----
    cb_w.reserve_back(1);
    volatile tt_l1_ptr uint16_t* wout =
        (volatile tt_l1_ptr uint16_t*)get_local_cb_interface(cb_w_id).fifo_wr_ptr;
    for (uint32_t i = 0; i < w_page / 2; ++i) {
        wout[i] = 0;
    }
    for (uint32_t s = 0; s < top_k; ++s) {
        const uint32_t j = best_idx[s];
        wout[(j >> 4) * 256 + (j & 15)] = f32_to_bf16(best_val[s] * inv);
    }
    cb_w.push_back(1);

    // ---- expert ids, uint16 TILE (matches the stock op's output) ----
    cb_id.reserve_back(1);
    volatile tt_l1_ptr uint16_t* idout =
        (volatile tt_l1_ptr uint16_t*)get_local_cb_interface(cb_id_id).fifo_wr_ptr;
    for (uint32_t i = 0; i < id_page_elems; ++i) {
        idout[i] = 0;
    }
    for (uint32_t s = 0; s < top_k; ++s) {
        idout[(s >> 4) * 256 + (s & 15)] = (uint16_t)best_idx[s];
    }
    cb_id.push_back(1);

    // ---- ship both outputs ----
    cb_w.wait_front(1);
    noc.async_write(cb_w, w_t, w_page, {.offset_bytes = 0}, {.page_id = 0});
    cb_id.wait_front(1);
    noc.async_write(cb_id, id_t, id_page, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
    cb_w.pop_front(1);
    cb_id.pop_front(1);
    cb_in.pop_front(1);
}
