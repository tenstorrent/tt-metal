// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused Gate + Up DRAM expert matmul — reproducer for back-to-back K-split hang.
//
// Calls MatmulExpertCompressedDRAM::Op twice on the same NCRISC/TRISC threads:
//   1) gate_proj's CTArgs (with gate's fmt sems, cb_out, DRAM addr, etc.)
//   2) up_proj's   CTArgs (with up's fmt sems,  cb_out, DRAM addr, etc.)
//
// Both share cb_in0 (activation) and cb_in1 (weight streaming buffer L1 region).
// gate has pop_in0=false, pop_index=false; up has pop_in0=true, pop_index=true.
// Both use ResetCBIn1=true with cb_in1_buf_addr so the SW write-pointer wrap
// stays anchored across the two ops.

#include "../../../unified_kernels/matmul_expert_compressed_dram.hpp"

void kernel_main() {
#if defined(COMPILE_FOR_NCRISC)
    // Sharded buffer setup: cb_in0 (activation, num_tiles_k pages),
    // cb_index (1 page) — both shared between gate and up.
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_index = get_named_compile_time_arg_val("cb_index");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");
    unified_kernels::setup_sharded_buffer(cb_in0, num_tiles_k);
    unified_kernels::setup_sharded_buffer(cb_index, 1);

    using GateArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("gate_cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("subblock_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("bank_id"),
        get_named_compile_time_arg_val("vc"),
        get_named_compile_time_arg_val("gate_expert_offsets_l1_addr"),
        get_named_compile_time_arg_val("gate_block_sizes_l1_addr"),
        get_named_compile_time_arg_val("cb_in1_size_bytes"),
        get_named_compile_time_arg_val("noc_max_page_size"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("gate_pipeline_sem_addr"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y"),
        get_named_compile_time_arg_val("cores_per_dram_bank"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("gate_cb_fmt"),
        get_named_compile_time_arg_val("gate_fmt_dram_addr"),
        get_named_compile_time_arg_val("gate_fmt_per_expert_bytes"),
        get_named_compile_time_arg_val("gate_fmt_per_core_bytes"),
        get_named_compile_time_arg_val("gate_fmt_cb_l1_addr"),
        get_named_compile_time_arg_val("gate_fmt_cb_page_size"),
        get_named_compile_time_arg_val("gate_fmt_sem_addr_0"),
        get_named_compile_time_arg_val("gate_fmt_sem_addr_1"),
        /*accum_experts=*/0,
        /*index_offset=*/0,
        get_named_compile_time_arg_val("k_parallel_per_bank"),
        get_named_compile_time_arg_val("k_slice_idx"),
        get_named_compile_time_arg_val("num_subblocks_k_local"),
        get_named_compile_time_arg_val("gate_partial_sem_addr"),
        get_named_compile_time_arg_val("primary_at_last_offset"),
        get_named_compile_time_arg_val("gate_gather_sync_sem_addr"),
        get_named_compile_time_arg_val("gate_cb_out")>;  // cb_internal_acc unused (accum=0)

    using UpArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ReaderCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("up_cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("subblock_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("bank_id"),
        get_named_compile_time_arg_val("vc"),
        get_named_compile_time_arg_val("up_expert_offsets_l1_addr"),
        get_named_compile_time_arg_val("up_block_sizes_l1_addr"),
        get_named_compile_time_arg_val("cb_in1_size_bytes"),
        get_named_compile_time_arg_val("noc_max_page_size"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("up_pipeline_sem_addr"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y"),
        get_named_compile_time_arg_val("cores_per_dram_bank"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("up_cb_fmt"),
        get_named_compile_time_arg_val("up_fmt_dram_addr"),
        get_named_compile_time_arg_val("up_fmt_per_expert_bytes"),
        get_named_compile_time_arg_val("up_fmt_per_core_bytes"),
        get_named_compile_time_arg_val("up_fmt_cb_l1_addr"),
        get_named_compile_time_arg_val("up_fmt_cb_page_size"),
        get_named_compile_time_arg_val("up_fmt_sem_addr_0"),
        get_named_compile_time_arg_val("up_fmt_sem_addr_1"),
        /*accum_experts=*/0,
        /*index_offset=*/0,
        get_named_compile_time_arg_val("k_parallel_per_bank"),
        get_named_compile_time_arg_val("k_slice_idx"),
        get_named_compile_time_arg_val("num_subblocks_k_local"),
        get_named_compile_time_arg_val("up_partial_sem_addr"),
        get_named_compile_time_arg_val("primary_at_last_offset"),
        get_named_compile_time_arg_val("up_gather_sync_sem_addr"),
        get_named_compile_time_arg_val("up_cb_out")>;  // cb_internal_acc unused

#elif defined(COMPILE_FOR_BRISC)
    using GateArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::WriterCTArgs;
    using UpArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::WriterCTArgs;

#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();

    using GateArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("gate_cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("subblock_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("gate_dram_fmt_l1_addr"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("gate_cb_fmt"),
        get_named_compile_time_arg_val("gate_dram_meta_words_per_block"),
        get_named_compile_time_arg_val("in0_page_size"),
        get_named_compile_time_arg_val("gate_fmt_cb_l1_addr"),
        get_named_compile_time_arg_val("gate_fmt_cb_page_size"),
        get_named_compile_time_arg_val("gate_fmt_sem_addr_0"),
        get_named_compile_time_arg_val("gate_fmt_sem_addr_1"),
        /*accum_experts=*/0,
        get_named_compile_time_arg_val("gate_dram_fuse_silu"),
        /*index_offset=*/0,
        get_named_compile_time_arg_val("k_parallel_per_bank"),
        get_named_compile_time_arg_val("k_slice_idx"),
        get_named_compile_time_arg_val("num_subblocks_k_local"),
        get_named_compile_time_arg_val("gate_partial_sem_addr"),
        get_named_compile_time_arg_val("gate_cb_out_silu"),
        get_named_compile_time_arg_val("gate_silu_tile_h"),
        get_named_compile_time_arg_val("cores_per_dram_bank"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y"),
        get_named_compile_time_arg_val("primary_at_last_offset"),
        get_named_compile_time_arg_val("gate_gather_sync_sem_addr"),
        get_named_compile_time_arg_val("gate_cb_out")>;

    using UpArgs = deepseek_b1_ops::MatmulExpertCompressedDRAM::ComputeCTArgs<
        get_named_compile_time_arg_val("cb_in0"),
        get_named_compile_time_arg_val("cb_in1"),
        get_named_compile_time_arg_val("up_cb_out"),
        get_named_compile_time_arg_val("cb_index"),
        get_named_compile_time_arg_val("num_tiles_k"),
        get_named_compile_time_arg_val("subblock_k"),
        get_named_compile_time_arg_val("subblock_n"),
        get_named_compile_time_arg_val("num_subblocks_k"),
        get_named_compile_time_arg_val("per_core_n"),
        get_named_compile_time_arg_val("up_dram_fmt_l1_addr"),
        get_named_compile_time_arg_val("num_active_experts"),
        get_named_compile_time_arg_val("index_l1_addr"),
        get_named_compile_time_arg_val("up_cb_fmt"),
        get_named_compile_time_arg_val("up_dram_meta_words_per_block"),
        get_named_compile_time_arg_val("in0_page_size"),
        get_named_compile_time_arg_val("up_fmt_cb_l1_addr"),
        get_named_compile_time_arg_val("up_fmt_cb_page_size"),
        get_named_compile_time_arg_val("up_fmt_sem_addr_0"),
        get_named_compile_time_arg_val("up_fmt_sem_addr_1"),
        /*accum_experts=*/0,
        /*fuse_silu=*/0,  // up has no silu
        /*index_offset=*/0,
        get_named_compile_time_arg_val("k_parallel_per_bank"),
        get_named_compile_time_arg_val("k_slice_idx"),
        get_named_compile_time_arg_val("num_subblocks_k_local"),
        get_named_compile_time_arg_val("up_partial_sem_addr"),
        /*cb_out_silu=*/0,
        /*silu_tile_h=*/0,
        get_named_compile_time_arg_val("cores_per_dram_bank"),
        get_named_compile_time_arg_val("core_in_bank_idx"),
        get_named_compile_time_arg_val("next_core_noc_x"),
        get_named_compile_time_arg_val("next_core_noc_y"),
        get_named_compile_time_arg_val("primary_at_last_offset"),
        get_named_compile_time_arg_val("up_gather_sync_sem_addr"),
        get_named_compile_time_arg_val("up_cb_out")>;
#endif

    constexpr uint32_t cb_in1_buf_addr = get_named_compile_time_arg_val("cb_in1_buf_addr");

    // gate_proj: pop_in0=false (keep activation for up), pop_index=false (up needs it).
    deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<
        GateArgs,
        /*IsActiveCore=*/true,
        /*pop_in0=*/false,
        /*pop_index=*/false,
        /*ResetCBIn1=*/true,
        cb_in1_buf_addr>
        gate_op;

    // up_proj: pop_in0=true (last consumer of activation), pop_index=true (last consumer).
    deepseek_b1_ops::MatmulExpertCompressedDRAM::Op<
        UpArgs,
        /*IsActiveCore=*/true,
        /*pop_in0=*/true,
        /*pop_index=*/true,
        /*ResetCBIn1=*/true,
        cb_in1_buf_addr>
        up_op;

    gate_op();
    up_op();

#if defined(COMPILE_FOR_NCRISC)
    noc_async_atomic_barrier();
#endif
}
