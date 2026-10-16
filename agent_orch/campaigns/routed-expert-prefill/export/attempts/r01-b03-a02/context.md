## Files read
- agent_orch/WORKER.md, campaign.yaml: process, metric, accuracy gate, allowed paths.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp: CB sizing (x CBs are
  sized per_core_M_max * in0_block_w_gu; cb_in0_x single-buffered on the RM path, cb_x_rm double-buffered),
  L1 fitter (w=16, M_max=4 normally), ring-granule checks, semaphores (in0 ready/valid per M-row), UP_SPLIT /
  DOWN_SPLIT / IN1_WRITER_MCAST notes.
- kernels/adaptive_chunk.hpp: per_core_M_for_chunk (runtime per_core_M 1..M_max, identical on all RISCs).
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: gate/up kb loop. x and weight handshakes are
  lock-stepped per K-block; RM x sender reads 32 sticks x per_core_M per K-block (1 KB each) and mcasts.
- kernels/compute/fused_swiglu.cpp: matmul_phase_fused_gu. Per K-block on the RM path:
  tilize(InitAndUninit) + reconfig srca + matmul_block_init + pack reconfig + L1_ACC + packer drain.
- ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl, tt_metal/hw/inc/api/compute/tilize.h: the helper picks BH
  fast tilize for bf16 -> 32x32 tiles at any width < 256 tiles, so a wider strip is a template width change.
- tt_metal/hw/inc/api/compute/matmul.h: matmul_block's kt_dim is the in0 row stride (unpack), so
  batched x passes nb*w.
## Nodes consulted
- r01-b01-a01 (1.4376) / r01-b02-a01 (1.4017): row-group expert parallelism. Time scaled with per-core
  M-rows (dsv3 0.709x time for 0.75x M-units), so the root is compute/per-core-work bound rather than DRAM.
- r01-b03-a01 (parent, 0.9416) and r01-b04-a01 (0.9330): weight read-ahead regressed. This node reverts
  the parent's kernel changes to root and doesn't touch the weight path. Both reflections named the
  per-K-block compute overhead (tilize/matmul re-init) and handshake count as the next suspect.
## Docs / external references
- Baseline per-expert arithmetic from history.md: ~28 cycles/tile-mult + ~16 us/expert fixed.
