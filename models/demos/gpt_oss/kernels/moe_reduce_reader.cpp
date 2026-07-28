// SPDX-License-Identifier: Apache-2.0
//
// gpt-oss custom fused MoE expert-reduce reader kernel.
//
// Feeds the compute kernel two circular buffers:
//   cb_scores (cb_in1): E score tiles, loaded ONCE and kept resident. Each tile
//       is pre-built on the HOST in COL-broadcast format: element [token, col 0]
//       = routing weight w[token, e]; other columns are ignored by the
//       BroadcastType::COL MAC. Pre-building on host removes the all_to_all
//       score-packing machinery of the stock DeepSeek reader.
//   cb_act (cb_in0): activation (down[e]) tiles, streamed in expert-major order
//       per output tile. For output tile i (i in 0..num_output_tiles-1) we push
//       the E expert tiles act[e, i], e=0..E-1, so the compute inner loop can
//       MAC-accumulate sum_e score[e]*act[e,i].
//
// Input activation tensor layout: [E, 1, T=32, H]  -> tiles are expert-major:
//   tile(e, i) lives at DRAM page  e * num_output_tiles + i   (Tt=1).
// Score tensor layout: [E, 1, 32, 32] -> E tiles, page e == score tile e.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_act_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_scores_id = get_compile_time_arg_val(1);
constexpr uint32_t num_experts = get_compile_time_arg_val(2);       // E (reduction dim)
constexpr uint32_t num_output_tiles = get_compile_time_arg_val(3);  // Ht (Tt=1)
// TensorAccessor CT args: activation @ 4, scores @ next.
constexpr uint32_t ct_idx_act = 4;
constexpr uint32_t ct_idx_scores = TensorAccessorArgs<ct_idx_act>::next_compile_time_args_offset();

void kernel_main() {
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t scores_addr = get_arg_val<uint32_t>(1);

    constexpr auto act_args = TensorAccessorArgs<ct_idx_act>();
    constexpr auto scores_args = TensorAccessorArgs<ct_idx_scores>();
    const auto act = TensorAccessor(act_args, act_addr);
    const auto scores = TensorAccessor(scores_args, scores_addr);

    Noc noc;
    DataflowBuffer cb_scores(cb_scores_id);
    DataflowBuffer cb_act(cb_act_id);

    const uint32_t score_page = get_local_cb_interface(cb_scores_id).fifo_page_size;
    const uint32_t act_page = get_local_cb_interface(cb_act_id).fifo_page_size;

    // Prologue: load all E score tiles once; they stay resident for the whole kernel.
    cb_scores.reserve_back(num_experts);
    for (uint32_t e = 0; e < num_experts; ++e) {
        noc.async_read(scores, cb_scores, score_page, {.page_id = e}, {.offset_bytes = e * score_page});
    }
    noc.async_read_barrier();
    cb_scores.push_back(num_experts);

    // Stream activation: for each output tile i, push the E expert tiles act[e,i].
    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        for (uint32_t e = 0; e < num_experts; ++e) {
            const uint32_t page = e * num_output_tiles + i;
            cb_act.reserve_back(1);
            noc.async_read(act, cb_act, act_page, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb_act.push_back(1);
        }
    }
}
