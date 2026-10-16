## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml: the job, the metric and allowed paths.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py: the shapes, routing, weight
  layout (bf4 DRAM ND-sharded, shard width per_core_N) and the per-shape byte counts used for the DRAM-floor table.
- device/kernels/adaptive_chunk.hpp: the launch planner (uniform / mixed layouts, two-phase plan, plan_launch).
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp: the counts broadcast, the start page, the plan,
  and the gate/up and down loops (look-ahead, x owner, activated stream).
- device/kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the up / down-half reads, the writer multicast,
  the output drain, and the writer's planner copy.
- device/kernels/compute/fused_swiglu.cpp: the UNPACK planner copy and the mailbox scan, multiply_phase, the phase
  zones.
- device/unified_routed_expert_ffn_program_factory.cpp: the CB indices (c_21 free), the ring-depth fit, the
  named compile-time args, and the reader / writer NoC defaults (reader NoC 1 for multicasts, NoC 0 for reads).

## Nodes consulted
- Every node r01..r04: proposal, reflection and error (dumped with git show to /tmp/r05b01/all.md). The decisive
  ones:
  - r04-b02-a01 (best, parent).
  - r04-b03-a03: the tie pass, its mm3 -4.6%, and the -Os UNPACK penalty.
  - r03-b01-a02: plan once per core and the size data.
  - r04-b01-a02 / r04-b04-a01 / r04-b04-a02: the h7168 kernel-config margin on this lineage.
  - r04-b03-a01 / r04-b03-a02 / r04-b04-a02 / r04-b02-a02: DM-handshake decoupling only redistributes on
    DRAM-bound shapes.
  - r01-b01-a03: pacing failed.
  - r02-b02-a02 next #2: bank-phase idea, judged unlikely.
- Own analysis of reports/r04-b02-a01 (scripts in /tmp/r05b01a01):
  - rows.py / rows2.py / perexp.py / cores.py / outl.py: per-row and per-core ends per call.
  - The group "lottery" is mostly whole-group, sometimes one column (mistral call 1: column x3 of y8-9 DN 107 us vs
    37).
  - Per-call mean group end is stable while the max varies.
  - kimi-k3's first one or two experts per group are the slow ones (GU 150-207 vs ~115 later).
- /tmp/w_r04b03a03c/sim/pt.cpp + n.txt: r04-b03-a03's plan replay on the test routing (counts per shape, new plans:
  only mistral / mm3 change, to 3+3+2+2).

## Docs / external references
- none beyond the code.
