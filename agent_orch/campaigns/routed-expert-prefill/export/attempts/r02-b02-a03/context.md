# r02-b02-a03 context

## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: the job, rules, allowed paths.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: phase 3 (gate/up) and phase 4 (down) loops.
  Phase 4 is strictly sequential per down K-block: reserve both rings, ack both senders, read + mcast
  down weights kb, then WAIT for activated block kb (compute gate/up + activation), push. So only
  down block 0 is fetched during the activation gap. That is the bottleneck this node targets.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: DOWN_SPLIT upper-half reads. The slot index
  is `down_blk % IN1_D_SLOTS`, so IN1_D_SLOTS must equal the cb_in1_down depth.
- kernels/compute/fused_swiglu.cpp: binary_activation_phase (MATH-thread SFPU, 2 dst slots per
  output tile), silu copy pass, PACKER_L1_ACC structure. I ruled out overlapping the SFPU activation
  with FPU work: on the 1-expert shapes there is no independent FPU work, and the down phase needs a
  column's full activation before its block can start. Re-blocking K would change the L1-acc rounding.
- tt_metal/hw/ckernels/blackhole/.../ckernel_sfpu_situ_glu.h, moe_gpt swiglu_sfpu.h: the activation
  cost is real SFPU math (tanh/exp/reciprocal per element). Nothing cheap to cut without touching
  approximations.
- unified_routed_expert_ffn_program_factory.cpp: cb_footprint_bytes, L1 fit, ring-depth logic
  (gu_slots / d_slots = 2 or 3), writer/reader defines.
- hybrid_routed_expert_ffn/device/kernels/hybrid_{reader,writer}.cpp: #include the unified kernels.
  The hybrid factory sets no IN1_DW_SLOTS / IN1_D_SLOTS, so it gets the defaults (2 -> look-ahead 0,
  i.e. the old loop).
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py and the model
  configs: emb/hidden dims used in /tmp/r02b02a03/l1.py, my replica of the L1 fit, to estimate
  which shapes get spare down slots.

## Nodes consulted
- All r01/r02 reflections (history.md + git show).
- r02-b01-a02, r02-b04-a02: measured per-expert ACT / GU / DN zones. ACT is 18-40 µs per expert on the
  binary-activation shapes, and DN is ~1/3 of matmul time and stream-bound on kimi-k3 / dsv4.
- r01-b04-a03: 3-slot rings were neutral, and noted the down loop is gated by activated block 0.
- r01-b03-a01, r01-b04-a01: read-ahead inside the gate/up block protocol hurt, so this node does not
  reorder within a block's protocol.
- r02-b04-a02: moving the down stream to the writer/NoC1 cost 10-22%, so the reads stay split
  NoC0/NoC1 as today and the look-ahead starts only after the group's gate/up stream ends.
- r02-b02-a02 (parent): request VCs were neutral and possibly -4% on gpt-oss. Reverted here.

## Docs / external references
- none
