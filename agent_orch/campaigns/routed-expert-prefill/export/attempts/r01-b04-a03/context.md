## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: the job, gate, allowed paths.
- unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp (r01-b01-a02 version):
  L1 fitter `cb_footprint_bytes` / `fit_config`, the M-doubling step, CB allocation (`make_cb`),
  `check_ring` invariants, writer compile-time args/defines. The fit picks (per_core_M_max, w) at
  2-slot rings, and the extra slots are only taken from what is left over.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: UP_SPLIT / IN1_WRITER_MCAST / DOWN_SPLIT
  paths. The writer never pushes cb_in1_up/down, so it indexes slots with `(up_seq-1) % kUpNumSlots`
  and `down_blk % kDownNumSlots`, both hard-coded to 2. These are the only ring-depth assumptions in
  the kernels.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: gate/up loop (x reserve/ack/multicast, the
  in1 sender's `mcast_done >= mc_seq-2` slot-reuse wait, which is conservative and correct for any
  depth >= 2), down loop (cb_in1_down + cb_in0_down_full reserved together, ready/valid per
  block). The ready/valid handshakes are serialized per block by the receivers' valid wait, so they
  are independent of ring depth.
- kernels/compute/fused_swiglu.cpp: matmul_phase_fused_gu, plus out_subblock_h == 1 static_assert.
  In tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h the non-reused operand is unpacked once per
  tile-mult whatever the subblock shape (reuse_a = ct >= rt), so taller subblocks would not cut the
  unpack count per mult. That rules out the "taller subblock" idea suggested by r01-b02-a01 /
  r01-b03-a02.
- kernels/weight_runs.hpp: one NoC read per K-row shard run (per_core_N tiles, one bank).
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py: the weight ND-shard spec follows the
  op's exported GRID_X (shard width = per_core_N).
- tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h: L1 1536 KB.
- tech_reports/GEMM_FLOPS/GEMM_FLOPS.md: LoFi = 16 cycles/tile ideal. Big bf4 matmuls reach ~80%.
- reports/r01-b01-a02/run.log (weight_gbps, counts) and .logs/profile_log_device.csv (core
  coordinates: the op uses 11 of 11 usable columns and 8 of 10 rows; "Max Compute Cores: 120").

## Nodes consulted
- r01-b01-a02 (1.4716, best): the code this node ports verbatim as its base; its reflection's
  random 6-14% group finish spread motivated the jitter hypothesis.
- r01-b01-a01, r01-b02-a01, r01-b02-a02: row groups; DRAM GB/s figures; placement lessons
  (x reads off weight-sender RISCs).
- r01-b03-a01, r01-b04-a01: reading ahead BEFORE the current multicast hurt. This node keeps
  per-RISC order and only adds slack.
- r01-b03-a02: x batching kept the same lead time and made arrival burstier.
- r01-b04-a02 (parent): widening in0_block_w_gu fails accuracy (bf16 dst accumulation span), so
  this node keeps the fitter's width exactly. Its config change is reverted by the port.

## Docs / external references
- Derived total DRAM traffic per shape at r01-b01-a02 (weights + x read + output write) from
  the run log: 360-434 GB/s on all shapes except gpt-oss (186).
