## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: rules, gate, allowed paths.
- device/kernels/adaptive_chunk.hpp (r02-b03-a03 version): layout table (already has {4,3,3}, {4,2,2,2}),
  schedule_in_order / schedule_lpt / plan_groups, two-phase Plan / PhaseDealer / model_expert /
  model_phase / make_plan / plan_launch. The only file changed.
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp (phase loop ~L385-460): the phase loop takes
  any layout index per phase (layout_group, topology set index = layout, phase-A sem pair), so a mixed
  phase-B layout needs no kernel change. The writer and compute (fused_swiglu.cpp ~L1068-1095) do the same.
- device/unified_routed_expert_ffn_program_factory.cpp ~L718: the phase-A semaphore pair; topology sets are
  emitted for every layout in the table.
## Nodes consulted
- All 28 reflections (r01-*, r02-*), read in full.
- r02-b03-a03 (best, 1.5435): base code taken verbatim.
- r02-b03-a01 / r02-b03-a02: two-phase schedule. The uniform phase B lost on dsv3/glm (two rounds), and
  the one-round rule excluded them. a02's next #2 is this node's idea.
- r02-b04-a01 / r02-b02-a01: mixed-height layouts and the measured per-expert fixed cost.
- r02-b01-a01, r02-b03-a01: fewer, larger groups lose stream bandwidth (≈250 GB/s per 5-row group). Here
  phase B keeps 3 streams, the same count as today's tail.
- r02-b02-a02 / r02-b02-a03 (my lineage): request VCs and down look-ahead were neutral or negative. Both
  are reverted by taking r02-b03-a03's code.
- r02-b04-a03: kimi-k3 is DRAM-bound (~408 GB/s), so it is left alone. The model also keeps it single-phase.
## Tools
- /tmp/routed_r02b02a04/pt.cpp: test-routing replay (from /tmp/r02b03a02/pt.cpp). /tmp/routed_r02b02a04/fuzz.cpp: 300k
  random plans checking the deal invariants; the old header is in /tmp/routed_r02b02a04/orig.
