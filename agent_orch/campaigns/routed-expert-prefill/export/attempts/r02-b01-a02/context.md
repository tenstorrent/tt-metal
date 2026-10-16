## Files read
- agent_orch/WORKER.md, campaign.yaml: rules, gate, allowed paths.
- kernels/adaptive_chunk.hpp (parent and r01-b04-a04): the parent's fluid-DRAM picker vs the units+chunks picker. Reverted to r01-b04-a04.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp:
  - The COUNTS_BCAST prologue is one reader + L1 multicast for counts/idx.
  - Then EVERY reader read `start` from DRAM itself, behind a barrier.
  - Also read the gate/up and down sender handshakes. The down sender waits for receivers' ready before its DRAM read, so DRAM latency sits in the per-block chain. I did not change that: the read-ahead class is discouraged by r01-b03-a01 / r01-b04-a01 / r01-b03-a03.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp:
  - Every writer also reads the same `start` page from DRAM at kernel start.
  - The UP_SPLIT + IN1_WRITER_MCAST chain serialises the up read of block kb+1 behind the multicast of kb (noted for children).
- kernels/compute/fused_swiglu.cpp: per-chunk phase order. matmul_phase_fused_gu includes the SiLU gate copy on non-binary-activation shapes. binary_activation_phase / multiply_phase come next, then matmul_phase (down K-loop + copy pass).
- program_factory.cpp: CB_START_SCRATCH (c_14, writer) and CB_START_SCRATCH_READER (c_15) are separate one-page CBs on every core. COUNTS_BCAST is a rectangle multicast to all cores. The device grid is 11x10 = 110 cores, all of which are used (CORE COUNT == AVAILABLE WORKER CORE COUNT in ops_perf).
- tt_metal/tools/profiler/kernel_profiler.hpp: DeviceZoneScopedN is bounded by bufferHasRoom (zones past the room are dropped).
- tests/.../test_routed_expert_prefill.py: offsets/counts/idx are 1-row ROW_MAJOR uint32 tensors, so the start page is num_experts x 4 B (0.5-3.5 KB).
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h: reuse_a = ct >= rt; in0 -> SrcB, in1 -> SrcA.

## Nodes consulted
- All 20 nodes' reflections (history.md + git show of each reflection.md).
- r02-b01-a01 (parent): the picker regression. Its reflection says to revert to r01-b04-a04's picker.
- r02-b04-a01 / r02-b02-a01: the measured per-expert fixed cost F (2-5 per_core_M units), and U ~ 30 µs per pcM unit on gpt-oss. That U implies ~16.7 cycles per tile-mult, i.e. matmul near the LoFi rate.
- r01-b03-a03 / r01-b04-a03: total DRAM ~410-434 GB/s on the big shapes.
- r01-b01-a03: flagged the shared `start` page read at kernel start.
- r01-b03-a01, r01-b04-a01: read-ahead regressions (avoided).

## Docs / external references
- /tmp/r02b04a01/plan.py (prior worker's routing replay). I reused it in /tmp/r02b01a02/simF.py: with uniform 10/5/2 layouts, no fixed-cost value F changes any shape's pick, so an F recalibration only matters together with mixed layouts (b02/b04 branches).
