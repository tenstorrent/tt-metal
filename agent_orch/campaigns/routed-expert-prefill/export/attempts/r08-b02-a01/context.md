## Files read
- agent_orch/WORKER.md, campaign.yaml: the job, the metric and allowed paths.
- device/kernels/adaptive_chunk.hpp: the planner (uniform / mixed / two-phase plans). I replayed it natively
  (/tmp/r08b02/pt, built from /tmp/r06b01a01/pt.cpp) to check which shapes change plan: only mistral and mm3
  (5x2 with y10-11 idle -> 3+3+2+2).
- device/kernels/dataflow/unified_routed_expert_ffn_reader.cpp: the whole per-block weight path (gate/up
  look-ahead, top-sender reader multicast, down look-ahead). I read it to check that plan-once does not
  interact with r07-b02-a01's K-loop changes, and to scope a pacing design, which I dropped because
  r08-b03/b04 run pacing.
- device/kernels/weight_runs.hpp: coalesced weight reads. Not changed.
- device/unified_routed_expert_ffn_program_factory.cpp: semaphores (15 of 16 used), ring depths, the NoC
  split, CB ids (c_21 free on this lineage).
- device/kernels/compute/fused_swiglu.cpp: partials already use PACKER_L1_ACC (no spill/reload lever left).
- tt_metal/impl/program/{program.cpp,dispatch.cpp}: the kernel-config size check, and semaphore init per
  launch (only word 0 of each 16 B slot is initialized).
- tests/.../test_routed_expert_prefill.py: shapes, routing, output bf8 / x bf16, used to work out delivered
  GB/s per shape.

## Nodes consulted
- All 69 committed nodes' proposals and reflections (dump in /tmp/r08b02/all.md). Rounds 5-7 in full.
- r07-b01-a01: the diff ported here (plan once + tie pass), and its measured mistral -9.8%.
- r07-b02-a01: this round's root. Its next #2 asks for this combination.
- r05-b01-a01 / r04-b03-a03: the tie pass measured on earlier lineages.
- r07-b04-a01: an idle group on top buys nothing ("relative position"), so idle rows should be filled,
  not moved.
- r08-b03-a01 (in-flight round-8 sibling, read from its worktree): cross-group pacing. It is neutral overall
  (+0.5%): it wins on 1-2-expert phases and loses on many-expert phases. Its next #3 also suggests stacking
  r07-b01-a01. r08-b04-a01 (pacing) and r08-b01-a01 (bottom-first deal) are also in flight, so I avoided
  both directions.
- r01-b01-a03: earlier pacing failure (per-block overhead).

## Analysis done
- Fluid DRAM-share model with rank-weighted priority (/tmp/r08b02/sim/*.py), fitted per shape on the root's
  per-row phase-A ends (error 6-9 µs per group). It reproduced r07-b04-a01's kimi-k3 flip (-3.4% predicted
  and measured). It predicts only +1..+1.5% geomean from planner rebalancing, which r08-b01 is doing.
- Delivered bytes / chip time on the root: kimi-k2.7 ~470, dsv4-pro ~470, dsv3 ~450, kimi-k3 ~455,
  mm3 ~420, mistral ~390, gpt-oss ~290 GB/s, against the 490-512 GB/s BH microbenchmark
  (tests/scripts/test_moreh_microbenchmark.py). So the h7168 shapes are within ~5-10% of the DRAM limit, and
  mistral / mm3 are the DRAM-bound-ish shapes with the most headroom.
- Local -O2 profiler compile (/tmp/r08b02/cc/b.sh, a named-CT-arg version of /tmp/r07b02/sz/b.sh): reader
  20128 B vs 18928 B on the root, writer 8448 B vs 12008 B.

## Docs / external references
- tech_reports/Saturating_DRAM_bandwidth; tests/scripts/test_moreh_microbenchmark.py (BH 490-512 GB/s).
