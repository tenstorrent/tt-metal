## Files read
- agent_orch/WORKER.md, campaign.yaml: rules, metric, accuracy gate.
- kernels/adaptive_chunk.hpp: group-size picker (pick_group_rows), expert_cost, GroupScheduler;
  this is where the plan lives.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: per-expert / per-chunk loops, weight
  multicast (in1 sems, sender reads into its own CB write pointer and multicasts to the same
  address), row-shared x (x_blk), activated multicast (per M-row).
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: UP_SPLIT / DOWN_SPLIT / IN1_WRITER_MCAST
  slot indexing by sender-only counters (up_seq, down_blk), which would break when roles change.
- kernels/compute/fused_swiglu.cpp: UNPACK picks the schedule and mails MATH/PACK the group size
  and per-expert counts.
- device/unified_routed_expert_ffn_program_factory.cpp: semaphore allocation (13 of 16 used),
  per-group-size topology runtime args.
- hybrid_routed_expert_ffn: has its own kernel copies, so it is unaffected.
## Nodes consulted
- r01-b04-a04 (parent code): per-row end times show the tail (kimi-k2.7 groups 0-1 end at ~745,
  groups 2-4 at ~560; dsv3 heavy groups ~520 vs light ~250-313).
- r01-b03-a04: layouts are table entries; the 5-row layout works.
- r01-b01-a03: pacing / board approaches add per-block cost.
- r01-b04-a03: ring depths (IN1_GU_SLOTS) can be 3, so slot math must not assume 2.
- r01-b04-a02: in0_block_w_gu is an accuracy knob (untouched here).
- r01-b03-a03 / r01-b01-a04: the expert-boundary bubble is not on the critical path.
- r01-b01-a04 counts.py (/tmp/r01b01a04): seed-0 tile counts per shape, used to calibrate the
  model (/tmp/r02b03a01/model.py).
## Docs / external references
- tt_metal/impl/buffers/semaphore.hpp: NUM_SEMAPHORES = 16.
