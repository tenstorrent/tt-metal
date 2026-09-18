# PAUSED 2026-09-18 ~23:20 -- how to resume the SP scaling campaign

## Where things are
* Code: branch `imichalak/llama-sp/6-sp-fused-matmul-ccl` (local only, never pushed). Verified commits up to
  9a0f9ad5749 (fused ops, tt-train wiring, Fused default, two-queue plumbing cherry-picks). On top: two WIP commits
  (bc1d302e14c WIP(S), aec7af6b583 WIP(F)), one per agent, NOT reviewed by the coordinator: "WIP(S) two-stream backward overlap" (tt-train:
  ops/distributed/sp_overlap.{hpp,cpp}, tests/python/test_sp_overlap.py, ttnn_fixed/sp_linear_ops edits) and
  "WIP(F) fused ops round 2" (ttnn fused op dirs, sp_matmul_fusion_common, matmul 2D factory/kernels, tests).
* Notes: this directory `generated/spfuse/` (git-ignored, durable): DESIGN.md (design + all numbers), HANDOFF_S.md and
  HANDOFF_F.md (agents' own hand-offs), the harness (env.sh, build.sh, devrun.sh, bench_sp_train.sh, matrix.sh,
  final_regress.sh, bench_sp_collectives.py, kit/, mgd/), logs/ (only post-restart logs; earlier logs were lost).
* Memory: ~/.claude/projects/-home-imichalak-tenstorrent-tt-metal/memory/sp-fused-matmul-ccl.md (key numbers, lessons).
* Published report (state as of 16:00, batch 1 only): https://claude.ai/artifact/MVxm6PVrjhHxoHBz7Q4WZM

## State of the numbers (Llama-8B tp4 SP, 1x4 Blackhole galaxy, 2 links, s/step; ideal = nocomm)
| batch | mesh | ideal | composed | published fused | best now | gap | closed |
| 1 | ring | 0.533 | 0.603 | 0.599 | 0.592 two-queue backward (S, bitwise-verified) | 70 ms | 11 |
| 1 | line | 0.534 | 0.629 | 0.597 | 0.597 fused (two-queue backward 0.609; fused-fwd on the split grid loses) | 95 ms | 32 |
| 2 | ring | 0.914 | 1.053 | - | 1.019 fused | 139 ms | 34 |
| 2 | line | 0.913 | 1.107 | - | 1.029 fused | 194 ms | 78 |
| 5 memeff | ring | 2.397 | 2.907 | - | 2.814 two-queue backward (S, rows=1) | 510 ms | 93 |
| 5 memeff | line | 2.397 | 3.074 | - | 2.907 fused | 677 ms | 167 |
Backward = 60% (B=1) / 77% (B=5) of the gap. Phase split and rationale: DESIGN.md.

## Open questions when resuming (in order)
1. F's perf2 shows fused qkv 307 us vs 494 unfused on ring at B=1 (committed op: 467). Is it a real mechanism? Its
   rs_check_ring had 2 failures. Read HANDOFF_F.md; re-run check mode on both ops/topologies before trusting perf.
2. S's backward overlap (HANDOFF_S.md): bitwise vs one queue verified by S (1x2 + 1x4 ring/line, 16+16+16 tests,
   SP suite 41). Not run: line B=5 memeff, CCL rows=2 / columns=1, batch 2, recompute-forward collectives through the
   same queue (its 4 collectives per block are untouched by M1 and would drain the 4 deferred wgrads).
3. bfp8 payloads (half the collective bytes): numerics numbers from F decide whether/where to use them.
4. Then: per-site policy (fused where net positive at HiFi4, two-queue composed elsewhere), re-run matrix.sh
   (`matrix.sh <tag> "1 5" "ring line" "nocomm composed fused"` with MEMEFF=1 for batch 5), final_regress.sh, commit.
5. Later levers: S M2 forward micro-batch pipelining (B>1), F in1 reuse across same-slice sub-batches, wide-N
   subblock; weeks-class: L1-resident RS partial (strided-RS rolling window).

## How to run things
source generated/spfuse/env.sh; build: build.sh [targets]; device: devrun.sh <name> <idle_s> <hard_s> -- "<cmd>"
(needs the sandbox disabled); training rows: [BATCH=5 MEMEFF=1] bench_sp_train.sh <composed|fused|nocomm> <ring|line> 6.
Do NOT use PROFILE=1 on the 32-layer model (tracy stalls); profile a 2-4 layer variant if per-op data is needed.
Agent transcripts (resumable while the session store exists): S = two-stream scheduler, F = fused ops round 2.

## Off-machine copy
Remote backup branch (no PR): origin/imichalak/sp-fusion-campaign-wip = this branch incl. both WIP commits and this
directory. The local PR-stack branch is imichalak/llama-sp/6-sp-fused-matmul-ccl.
F's hand-off headline (HANDOFF_F.md): the perf2 jump (ring B=1 qkv 467 -> 307 us) = an all-gather signal-timing fix
(~125 us of serialisation removed) + a cheaper in1 stream (sub-batched matmul 309 -> 213 us); every check case is
still bitwise equal to the same-config unfused path; the "2 failures" are the loose PCC gate vs ttnn.linear's auto
config on the new K=7168 shape (threshold decision pending). bfp8 payloads: qkv 307 -> 226 us (bf16acc), PCC vs fp32
0.99992, max |d| 0.47 (see HANDOFF_F.md section 5 for the full numerics).
