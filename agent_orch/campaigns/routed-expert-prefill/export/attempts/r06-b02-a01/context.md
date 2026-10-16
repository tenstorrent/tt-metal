## Files read
- agent_orch/WORKER.md, agent_orch/campaigns/routed-expert-prefill/campaign.yaml: job, rules, allowed paths, gate.
- $DREAM_HOME/routed-expert-prefill/history.md: index of all 61 nodes over rounds 1-5.
- Every node's proposal.md / reflection.md (and error tails), dumped to /tmp/r06b02/all.md with `git show` on the tags.
  Read in full for rounds 4-5 and the key round 1-3 nodes, skimmed the rest.
- tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py: shapes, seed-0 routing.
- kernels/adaptive_chunk.hpp: the launch planner (uniform / mixed layouts, two-phase plan, plan_launch). The new
  helper goes at its end.
- kernels/dataflow/unified_routed_expert_ffn_reader.cpp: prologue (counts multicast, `start` read, plan), phase / expert /
  chunk loops, up-front x reads of blocks 0/1, gate read issue, look-ahead, step order of the K-block loop.
- kernels/dataflow/unified_routed_expert_ffn_writer.cpp: the UP_SPLIT `up` read waits for the reader's up_go, so the
  writer's first DRAM read follows the reader's wait.
- device/unified_routed_expert_ffn_program_factory.cpp: NoC-balance note, d_head_tiles / up_head_rows, per-layout
  sender topology (sender row = group_base + gx % rows).
- tt_metal/soc_descriptors/blackhole_140_arch.yaml: DRAM endpoints per NoC (NoC 0: x=0 rows 11/2/9/5, x=9 rows
  11/3/8/6; NoC 1: rows 1/10/4/7 on both columns), worker grid.
- tt_metal/hw/inc/internal/tt-1xx/risc_common.h: riscv_wait (wall-clock spin). The kernel uses the low clock word only.

## Nodes consulted
- r05-b02-a01 (parent, best 1.6667): 60/40 NoC split. Its per-row table shows the lower groups gained much more than
  the top ones and the static plan now leaves 50-280 µs of tail on the top groups.
- r05-b03-a01: RE-X0 zone (x block 0 landing per core). I re-analysed its report: the pure receivers of every row reach
  their first x wait at 5-8 µs, the weight senders' first reads complete 20-30 µs later on the top rows, and x(0)
  lands per row monotonically from the bottom-most active row to the top row. Its next #1 suggested a start-up stagger.
- r05-b01-a01: plan-once + tie pass (other branch, not combined here). Its per-call analysis of the max-core lottery.
- r05-b04-a01: column-parity wrap finding (hang). Not touched here.
- r03-b03-a02 (all gate/up on NoC 0) and r04-b02-a01 vs r05-b02-a01: per-row end / GU data showing that more NoC-0
  traffic slows the top rows and speeds the bottom rows.
- r01-b01-a03: continuous group pacing via a progress board (-0.43). This node is a one-time static offset instead.
- r04-b01-a02 / r04-b04-a01 / r03-b01-a01 / r03-b02-a02: kernel-config-buffer budget on the h7168 shapes. The reader
  growth is kept to ~0.36 KB at -O2 (helper out of line, 32-bit clock).

## Analysis scripts (this node)
- /tmp/r06b02/gu.py: per-row start and duration of every RE-GU / RE-DN / RE-ACT unit (MATH zones).
- /tmp/r06b02/ends.py: per-row kernel end (works on reports without the RE-* zones).
- /tmp/r06b02/x0.py, x0grid.py: RE-X0 per row / per core (r05-b03-a01's report).
- /tmp/r06b02/noc*.py: torus link-load model. It does not reproduce the row gradient under uniform demand (every
  group touches the same hottest link), so it was not used to pick parameters.
- /tmp/r06b02/plan/drv.cpp, drv2.cpp: the kernel planner compiled natively on the test's seed-0 counts; prints each
  shape's plan and the per-row start-up delay.
- /tmp/r06b02/build.sh, build_o2.sh: local reader compile for size (copied from /tmp/r04b01a02).

## Docs / external references
- tech_reports/Saturating_DRAM_bandwidth (via r05-b02-a01's proposal): DRAM response routes congest the NoC.
