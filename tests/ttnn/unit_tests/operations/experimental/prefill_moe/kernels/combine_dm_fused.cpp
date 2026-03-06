// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused combine kernel: waits for ALL experts to finish compute, then performs
// weighted accumulation of expert output into final output.
//
// Strategy: Wait for all SEM_EXPERT_DONE signals upfront (no per-expert
// pipelining). This ensures the combine's scalar BF16 processing does not
// run concurrently with the compute pipeline, avoiding a deadlock that
// occurs when RISC-V scalar L1 access on dm0 overlaps with active matmul
// barrier signaling on 15+ compute cores.
//
// For each expert e:
//   For each assigned token t with weight w:
//     output[t, :] += w * out_buf[e][t, :]
//
// For P > 32, tiles are addressed with 2D page indexing:
//   page_idx = tile_row * D_tiles + d
// Token rows are decomposed: tile_row = t / 32, local_t = t % 32.
//
// Runs on 1 core as dm0 (RISCV_0 / reader).
// CB4: output tile (read-modify-write), CB5: expert out_buf tile.
//
// Semaphores:
//   SEM_EXPERT_DONE (id=3): Waited on (cumulative) before processing.
//
// BF16 tile face layout (32x32 tile = 2048 bytes):
//   Face 0: rows 0-15, cols 0-15  (bytes 0-511)
//   Face 1: rows 0-15, cols 16-31 (bytes 512-1023)
//   Face 2: rows 16-31, cols 0-15 (bytes 1024-1535)
//   Face 3: rows 16-31, cols 16-31 (bytes 1536-2047)
//   Within each face: row-major uint16_t, 16 cols per row = 32 bytes/row.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr auto tensor_args = TensorAccessorArgs<0>();
    constexpr uint32_t enable_fabric_reduce = get_compile_time_arg_val(TensorAccessorArgs<0>::NumArgsCT);

    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t D_tiles = get_arg_val<uint32_t>(1);
    const uint32_t P_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_experts = get_arg_val<uint32_t>(3);
    // Per expert (packed sequentially starting at arg 4):
    //   out_buf_addr, M_e, token_row[0..M_e-1], weight_bf16[0..M_e-1]
    // When enable_fabric_reduce: last 3 args are reduce_core_noc_x, reduce_core_noc_y, sem_combine_done_l1

    constexpr uint32_t cb_out = 4;  // output tile (read-modify-write)
    constexpr uint32_t cb_exp = 5;  // expert out_buf tile
    constexpr uint32_t SEM_EXPERT_DONE = 3;

    const uint32_t page_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    const auto out_accessor = TensorAccessor(tensor_args, output_addr, page_bytes);

    // Reserve CB buffers ONCE and get fixed L1 addresses.
    cb_reserve_back(cb_out, 1);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    cb_reserve_back(cb_exp, 1);
    const uint32_t exp_l1 = get_write_ptr(cb_exp);

    // Wait for ALL experts to finish compute before starting any combine work.
    // This ensures no concurrent scalar L1 access while compute barriers are active.
    if (num_experts > 0) {
        volatile tt_l1_ptr uint32_t* done_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_EXPERT_DONE));
        noc_semaphore_wait(done_sem, num_experts);
    }

    uint32_t arg_idx = 4;

    for (uint32_t e = 0; e < num_experts; ++e) {
        uint32_t out_buf_addr = get_arg_val<uint32_t>(arg_idx);
        uint32_t M_e = get_arg_val<uint32_t>(arg_idx + 1);

        uint32_t tokens_base = arg_idx + 2;
        uint32_t weights_base = arg_idx + 2 + M_e;

        if (M_e == 0) {
            arg_idx += 2;
            continue;
        }

        const auto exp_accessor = TensorAccessor(tensor_args, out_buf_addr, page_bytes);

        for (uint32_t d = 0; d < D_tiles; ++d) {
            for (uint32_t tr = 0; tr < P_tiles; ++tr) {
                uint32_t page_idx = tr * D_tiles + d;
                uint32_t row_lo = tr * 32;
                uint32_t row_hi = row_lo + 32;

                // Read output tile into L1
                noc_async_read_page(page_idx, out_accessor, out_l1);
                noc_async_read_barrier();

                // Read expert out_buf tile into L1
                noc_async_read_page(page_idx, exp_accessor, exp_l1);
                noc_async_read_barrier();

                // Weighted accumulate for tokens in this tile row
                for (uint32_t i = 0; i < M_e; ++i) {
                    uint32_t t = get_arg_val<uint32_t>(tokens_base + i);

                    // Only process tokens that fall in this tile row
                    if (t < row_lo || t >= row_hi) {
                        continue;
                    }

                    uint32_t w_bf16 = get_arg_val<uint32_t>(weights_base + i);

                    // BF16 weight → float32
                    union {
                        uint32_t u;
                        float f;
                    } w_conv;
                    w_conv.u = w_bf16 << 16;
                    float w = w_conv.f;

                    // Row within this tile (0..31)
                    uint32_t local_t = t - row_lo;
                    uint32_t face_row = local_t >> 4;    // local_t / 16
                    uint32_t local_row = local_t & 0xF;  // local_t % 16

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
                noc_async_write_page(page_idx, out_accessor, out_l1);
                noc_async_write_barrier();
            }
        }

        arg_idx += 2 + 2 * M_e;
    }

    // Signal fabric_reduce_dm that local combine is complete
    if constexpr (enable_fabric_reduce) {
        uint32_t reduce_core_noc_x = get_arg_val<uint32_t>(arg_idx);
        uint32_t reduce_core_noc_y = get_arg_val<uint32_t>(arg_idx + 1);
        uint32_t sem_combine_done_id = get_arg_val<uint32_t>(arg_idx + 2);
        // Convert semaphore ID to L1 address (same on all cores)
        uint64_t reduce_sem_addr =
            get_noc_addr(reduce_core_noc_x, reduce_core_noc_y, get_semaphore(sem_combine_done_id));
        noc_async_write_barrier();  // ensure all output writes are flushed
        noc_semaphore_inc(reduce_sem_addr, 1);
    }

    // Release CB buffers at the end
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_out, 1);
    cb_push_back(cb_exp, 1);
    cb_pop_front(cb_exp, 1);
}
