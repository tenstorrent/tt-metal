// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused combine kernel: waits for SEM_EXPERT_DONE, then performs weighted
// accumulation of expert output into final output.
//
// For each expert e (processed one at a time in fused mode):
//   Wait for SEM_EXPERT_DONE signal from compute leader.
//   For each assigned token t with weight w:
//     output[t, :] += w * out_buf[e][t, :]
//
// Operates at the sub-tile level: reads tiles from out_buf and output,
// locates specific rows within tile face layout, performs scalar BF16
// multiply-accumulate, writes modified output tile back to DRAM.
//
// Runs on 1 core as dm0 (RISCV_0 / reader).
// CB0: output tile (read-modify-write), CB1: expert out_buf tile.
//
// Semaphores:
//   SEM_EXPERT_DONE (id=3): Waited on before processing each expert.
//
// BF16 tile face layout (32x32 tile = 2048 bytes):
//   Face 0: rows 0-15, cols 0-15  (bytes 0-511)
//   Face 1: rows 0-15, cols 16-31 (bytes 512-1023)
//   Face 2: rows 16-31, cols 0-15 (bytes 1024-1535)
//   Face 3: rows 16-31, cols 16-31 (bytes 1536-2047)
//   Within each face: row-major uint16_t, 16 cols per row = 32 bytes/row.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t D_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_experts = get_arg_val<uint32_t>(2);
    // Per expert (packed sequentially starting at arg 3):
    //   out_buf_addr, M_e, token_row[0..M_e-1], weight_bf16[0..M_e-1]

    // Compile-time args: TensorAccessorArgs for output (reused for all same-shape tensors)
    constexpr auto tensor_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_out = 4;  // output tile (read-modify-write)
    constexpr uint32_t cb_exp = 5;  // expert out_buf tile
    constexpr uint32_t SEM_EXPERT_DONE = 3;

    const uint32_t page_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    const auto out_accessor = TensorAccessor(tensor_args, output_addr, page_bytes);

    uint32_t arg_idx = 3;

    for (uint32_t e = 0; e < num_experts; ++e) {
        // Wait for compute to finish this expert's output
        volatile tt_l1_ptr uint32_t* done_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_EXPERT_DONE));
        noc_semaphore_wait(done_sem, 1);
        noc_semaphore_set(done_sem, 0);  // Reset for next expert

        uint32_t out_buf_addr = get_arg_val<uint32_t>(arg_idx);
        uint32_t M_e = get_arg_val<uint32_t>(arg_idx + 1);

        if (M_e == 0) {
            arg_idx += 2;
            continue;
        }

        uint32_t tokens_base = arg_idx + 2;
        uint32_t weights_base = arg_idx + 2 + M_e;

        const auto exp_accessor = TensorAccessor(tensor_args, out_buf_addr, page_bytes);

        for (uint32_t d = 0; d < D_tiles; ++d) {
            // Read output tile into L1
            cb_reserve_back(cb_out, 1);
            uint32_t out_l1 = get_write_ptr(cb_out);
            noc_async_read_page(d, out_accessor, out_l1);
            noc_async_read_barrier();

            // Read expert out_buf tile into L1
            cb_reserve_back(cb_exp, 1);
            uint32_t exp_l1 = get_write_ptr(cb_exp);
            noc_async_read_page(d, exp_accessor, exp_l1);
            noc_async_read_barrier();

            // Weighted accumulate for each assigned token
            for (uint32_t i = 0; i < M_e; ++i) {
                uint32_t t = get_arg_val<uint32_t>(tokens_base + i);
                uint32_t w_bf16 = get_arg_val<uint32_t>(weights_base + i);

                // BF16 weight → float32
                union {
                    uint32_t u;
                    float f;
                } w_conv;
                w_conv.u = w_bf16 << 16;
                float w = w_conv.f;

                // Row t in tile face layout
                uint32_t face_row = t >> 4;    // t / 16
                uint32_t local_row = t & 0xF;  // t % 16

                // Process 32 columns across 2 face halves
                for (uint32_t half = 0; half < 2; ++half) {
                    uint32_t face_off = (face_row * 2 + half) * 512 + local_row * 32;
                    uint16_t* op = reinterpret_cast<uint16_t*>(out_l1 + face_off);
                    uint16_t* ep = reinterpret_cast<uint16_t*>(exp_l1 + face_off);

                    for (uint32_t c = 0; c < 16; ++c) {
                        // BF16 → float32 → accumulate → BF16
                        union {
                            uint32_t u;
                            float f;
                        } cur, val, res;
                        cur.u = static_cast<uint32_t>(op[c]) << 16;
                        val.u = static_cast<uint32_t>(ep[c]) << 16;
                        res.f = cur.f + w * val.f;
                        op[c] = static_cast<uint16_t>(res.u >> 16);
                    }
                }
            }

            // Write modified output tile back to DRAM
            noc_async_write_page(d, out_accessor, out_l1);
            noc_async_write_barrier();

            cb_push_back(cb_out, 1);
            cb_pop_front(cb_out, 1);
            cb_push_back(cb_exp, 1);
            cb_pop_front(cb_exp, 1);
        }

        arg_idx += 2 + 2 * M_e;
    }
}
