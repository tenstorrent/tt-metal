## Files read
- agent_orch/WORKER.md, campaigns/rmsnorm-prefill/campaign.yaml: the job, the metric, allowed paths
- device/kernels/compute/dit_rmsnorm_fused_compute.cpp: the PRE (DST-accumulated S, then the mm_row_stat matmul
  ones*S^T), the x*gamma prepass, and the combine (ELWADD + add_rsqrt row 0 + transpose_dest<fp32>). This node changes
  the PRE stat path only.
- device/kernels/dataflow/dit_rmsnorm_fused_worker_writer.cpp: the stick is row 0 of the fp32 stats_transposed_local_cb
  tile, two 64 B face-rows at byte offsets 0 and 1024. The layout this node must produce is unchanged.
- device/dit_fused_distributed_rmsnorm_program_factory.cpp: stats_transposed_local_cb and pre_intermediate_cb are both
  1 fp32 tile, and fp32_dest_acc_en is required.
- tt_metal/hw/inc/api/compute/compute_kernel_api.h: the `sfpu_reduce` / `sfpu_reduce_init` API. REDUCE_COL puts the
  column sums in the first row and supports Float32.
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h: `perform_reduce_col_sum_avg` does 4
  face-pair iterations (8 loads, tree add, SFPTRANSP, 3 adds, 1 store into row 0). Replay slots are [0,9). FPU
  windows start at 16, so it doesn't clash with transpose_dest.
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h: the 32-bit transpose STALLWAITs on SFPU/MATH and
  SrcA/SrcB valid, restores implied SrcA format and Fp32 ALU, and uses replay at replay_buf_offset (16+).
- llk_unpack_common.h `_llk_unpack_set_srcb_dummy_valid_`: sets dummy valid on both SrcA and SrcB (the unpack side of
  transpose_dest).
- tests/ttnn/unit_tests/kernel_lib/reduce/kernels/sfpu_reduce_col_avg.cpp: reference usage of sfpu_reduce.

## Nodes consulted
- All 27 committed nodes' reflections (history.md + git show). Most relevant:
- r01-b04-a04: the PRE tail is a fixed per-row cost, not a per-tile cost.
- r02-b02-a02: a stat straight from DST (diagonal) is ready 0.35-0.6 µs earlier, but the BRISC gather costs 0.36 µs.
- r02-b02-a03: the PRE tail lives in the S DST->L1->SrcA handoff, not in the number of FPU stages after it. Its
  suggestion (c) is to keep the stat in DST.
- r02-b02-a04, r03-b0x-a01: post-AG combine work. The siblings attack it; this node stays pre-AG to stay orthogonal.
- r02-b02-a02/tl.py: timeline script (stat ready = W_PUSH start, W_PUSH end, F_COLLECT, AG end) for the reflection.

## Docs / external references
- none beyond the in-tree LLK headers above
