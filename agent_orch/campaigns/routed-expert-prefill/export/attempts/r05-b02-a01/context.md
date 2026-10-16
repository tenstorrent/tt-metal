## Files read
- agent_orch/WORKER.md, campaigns/routed-expert-prefill/campaign.yaml — job, rules, allowed paths, accuracy gate.
- $DREAM_HOME/routed-expert-prefill/history.md — index of all 57 nodes.
- All 57 nodes' proposal.md / reflection.md (dumped to /tmp/r05b02a01/{prop,refl}.md). Rounds 3-4 were read in full.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py — shapes, routing, weights
  ND-sharded via routed_expert_weight_memory_config.
- models/demos/deepseek_v3_d_p/tt/moe/tt_routed_expert.py (routed_expert_weight_memory_config) — weight shards are
  1 K-row x per_core_N tiles, ROUND_ROBIN_1D over the 8 DRAM banks, so bank = (k*11 + c) % 8.
- tt_metal/soc_descriptors/blackhole_140_arch.yaml — DRAM channel NoC endpoints (3 per bank at x = 0 / 9).
  dram_views put worker NoC 0 on (0|9, {11,2,9,5}) and NoC 1 on (0|9, {1,10,4,7}). The SYS-1419 note is why I
  did not touch endpoints.
- tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md — states that overlapping DRAM routes on the NoC
  prevent DRAM saturation and that round-robin reads congest the NoC. It also gives NoC 0 routing as right then down.
- tests/scripts/test_moreh_microbenchmark.py — BH all-core DRAM read bound 490-512 GB/s.
- tt_metal/api/tt-metalium/kernel_types.hpp, impl/kernels/kernel_types.cpp, program_descriptors.hpp — the reader
  (NCRISC) is on NOC_0 and the writer (BRISC) on NOC_1.
- unified_routed_expert_ffn_program_factory.cpp:
  - sender placement (weight sender row = group_base + gx % rows; x owner col);
  - UP_SPLIT, DOWN_SPLIT, IN1_WRITER_MCAST;
  - CT arg layout.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp — gate/up loop (issue_gate_read, try_gu_lookahead,
  mcast_go / up_done order) and the phase-4 down loop (issue_down_read, look-ahead, down_go / down_done).
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp — UP_SPLIT up read + IN1_WRITER_MCAST multicast, DOWN_SPLIT read.
- kernels/weight_runs.hpp — coalesced shard-row reads. Any [j0, jend) range works; an empty range issues nothing.
- tt_metal/api/tt-metalium/tensor_accessor_args.hpp — get_compile_time_args, used for the gate == up layout check.
- hybrid_routed_expert_ffn — has its own copies of the dataflow kernels (shares only weight_runs.hpp), so it is unaffected.

## Nodes consulted
- r03-b03-a02 (all gate/up reads on NoC 0: -5..-11%, aggregate BW fell) and r02-b04-a02 (all down reads on NoC 1:
  -10..-22% delivery) — the two data points the NoC link model must reproduce. It does: NoC 0 at ~90, NoC 1 at ~130 GB/s
  on the hottest link.
- r02-b01-a01 / r02-b03-a01 — a lone group pulls ~250 GB/s. Modelled hottest NoC 1 link is 57-63 GB/s, the same as the
  all-groups 440 GB/s case (65 GB/s).
- r03-b04-a03, r03-b03-a03 — steady-state ~450 GB/s (mistral GU) and whole-op 380-417 GB/s per shape.
- r04-b04-a01, r04-b04-a02, r04-b03-a02, r04-b02-a02 — handshake / ack decoupling only reorders who waits; both phases
  are paced by the group's DRAM share. That points at the delivery rate itself.
- r02-b02-a02 (request VCs) — arbitration, not link load.
- r04-b02-a01 (this branch's start, best 1.6275) — the reader / writer look-ahead structure I modified.
- r04-b01-a02, r04-b04-a02, r04-b04-a01 — kernel-config-buffer budget on the h7168 programs (≤ ~0.9 KB headroom) and
  the local compile harness (/tmp/r04b01a02/build_o2.sh, which reproduces r04-b02-a01's reader size 18324 B).
- r04-b03-a03 — tie pass (not used here; noted as an untried port for later).

## Tools written for this node
- /tmp/r05b02a01/noc.py, noc2.py, noc3.py, noc4.py — BH NoC link-load model of weight-read responses (+ weight
  multicasts) for any row-group layout and NoC 0 share. noc3 also tries per-(sender, bank) NoC choice; it is no better
  than a global ~0.6 split.
- /tmp/r05b02a01/b.sh, b2.sh — local NCRISC / BRISC compile with size (DH = reader CT 30, XDEF for -DRE_UP_HEAD_ROWS,
  WSRCP / RSRCP for an alternative source placed under /tmp/r05b02a01/base/... so relative includes resolve).

## Docs / external references
- tech_reports/Saturating_DRAM_bandwidth (above).
