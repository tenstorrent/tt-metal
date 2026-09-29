# GLM-5.3-Flash (layers 0-4) bring-up: supervision log

Overseer log: time, task, trigger, classification, action, resulting commit.

## 2026-09-28/29 intake
- Owner prompt: glm53_bringup_prompt.md (repo root). Study: glm53_flash_findings.md.
- Box: 4x Blackhole p300c (2 dual-chip cards), 32 GB per chip, 11x10 compute grid (not the p150b the study assumed).
  1x4 and 2x2 FABRIC_2D meshes open and all_gather exactly (probes in tests/ttnn/unit_tests/operations/glm53_box/).
- Checkpoint zai-org/GLM-5.3-Flash @ eb9eb208eb0d988989d07a6a12d0fdeb5f52574a, not gated, FP8 328 GB. glm5_next code
  (transformers 5.17) vendored in /localdev/dnijemcevic/bringup/glm53_flash_d_p/hf_modeling.
- Framework fix before the plan: mixed per-layer state (F48, commit dbd53535796; selftests 197 -> 212).
- Download died at 45/62 shards when the first intake session ended; restarted 2026-09-29 00:08 in tmux `glmdl`
  (unauthenticated, no HF token on the box).
- Owner: mesh 2x2 (not 1x4); layers 0-4 (the minimum covering all three block types; every indexer is `full`, so no
  cross-layer top-k sharing needs a second DSA layer); HF sanity on the whole model, trim after it passes; both
  dashboards. Correction to the study: the minimum is 0-4, not 0-3 (0-3 has no kda_moe).
- Owner approved the spec ("let's do 0-4"); intake approved 00:16; ledger --early 9 tasks; run1.
- Launch waits for the full download (R.1's checkpoint counts and R.2's sanity need every shard).

## 2026-09-29 run1 start
- 00:23 launched (tmux `glmorch`) after the download finished (62/62 shards, 306 GiB).
- 00:24 R.1 FAIL: shape_mismatches 45 = `layers.*.hc_attn_fn` expected `"*"` in the spec (not a wildcard for shapes;
  real shape [24, 16384]). Classified: spec error (intake). Paused, killed the fix agent before it edited anything.
  Owner approved the one-line spec fix ("yes"); intake re-approved; resume.
- 01:40 R.2 PASS (998ab90bb7a) after 1 reference attempt. Reviewed: glm_ref.py is a standalone implementation (no
  HF import; sparse indexer gathers selected pools; fp32 KDA state); HF parity L0-4 and logits PCC 1-1e-13, top1 1.0;
  whole-model (45-layer) HF sanity: smoke "Paris", text_top1_acc 0.964 (~10 min, streams FP8 experts). Accepted.
- 01:10-01:32 owner asked to investigate KDA fixes on main (mvasilijevicTT): missing ea27dd5f1ba (padded-chunk
  actual_end) + 411fb7c4fd8 (bind only if given). Owner approved: build + test in worktree
  /localdev/dnijemcevic/tt-metal-kda first; cherry-pick into the branch at the plan-approval wait.
  Baseline KDA op tests on the current build: 244 passed, 22 skipped, 1 perf-timing fail (summarize_chunk_recurrence
  production perf 319.9 us > 314.7 us limit).
- 01:44-02:12 cherry-picks in the worktree: 14 new KDA op-test failures (prepare_chunk_recurrence without actual_start:
  q bound twice, rejected by #55670). Built clean origin/main 98134127a7b in /localdev/dnijemcevic/tt-metal-main:
  the same 14 fail with the same TT_FATAL (checked: main's own build, submodules and python modules). CI never ran them
  after #57070 merged (nightly L2 stopped earlier every night; PR CI ran before #55670 was in its base); no issue filed.
  Server side: main's prefill engine changes nothing K.1 or GLM's adapter must follow (read-only comparison;
  scratchpad prefill_engine_main_vs_ours.md). Owner: port the fix by cherry-pick, not rebase; not patch the op.
- 02:59 cherry-picked #57070 + #58049 onto the branch (158a510b062, 926ca262f00), rebuilt in place (0 errors).
  KDA op tests: 236 passed, 24 skipped, 15 failed = the 14 known main failures + the summarize production-perf
  timing check (also fails on the old build). Framework selftests 212 passed; ttnn.bringup fork tests 23 passed;
  test_box on 2x2 passed. findings.yaml: kda-actual-end-padding for the implement agents.
- Plan reviewed with the owner (KDA TP2 x SP2 on 2x2 accepted as the experiment; DSA replicated + query split;
  indexer composed, not deferred; MiMo expert layout at 72/288 unverified). Owner approved the plan ("ok go ahead").
- 03:2x C.kda_dense.attn_hc PASS (fabd9566e18) pcc 0.9999986, attempt 1. Reviewed tt/mhc.py: pure ttnn (DeepSeek _project + mhc_split_sinkhorn, fp32, HiFi4), no host math; hooks' HybridDeviceModel = framework hybrid harness (CPU ref with passed device steps swapped in). Accepted.
- dashboards: standard https://claude.ai/artifact/J7Qb5GkEhxXevKJ7WVXuX9 , teletext https://claude.ai/artifact/W9yuVL9dGTr97NTVkikj8e (republish after every gate; lapsed PL.0..S.kda_dense.01, caught by owner, republished 03:3x)
- C.kda_dense.attn_collapse PASS (2cbcd201190) pcc 0.999997, attempt 1; tt/collapse.py pure ttnn (DeepSeek _streams/_cols/_mix, fp32 accumulate). Accepted.
- S.kda_dense.02 PASS (a1e3a9678b0) swap 0.999997. C.kda_dense.attn_norm PASS (3887e90de0a) pcc 0.999996, attempt 1; tt/rms_norm.py torch only for the weight at load. Accepted. Owner switched this session to Opus 5.5 high effort (agents already high via settings.json).
- C.kda_dense.attention PASS (b9bc0e0d0a6) attempt 1: pcc 0.999985, rel 0.0071, block 0.0072, state rec 0.0134 / head 0.0238, conv 0.0017. First run: worst head 0.0528 > 0.05; cause TF32 subtraction in prepare_chunk_recurrence's k_dec_t; fixed by recomputing k_dec_t on device (fp32 exp). SP order verified (row 0 first half), no bump at the split. HiFi4 everywhere. Accepted. Gap: findings.yaml is not in any brief, so the KDA actual_end note never reached the agent (module passes no actual_end). Paused to move it into knowledge/known_issues.md.
- 04:4x S.kda_dense.04 PASS (e8aeffa758b) swap 0.999984. Paused at the gate boundary; moved the KDA actual_end note into knowledge/known_issues.md (Serving contract), committed with the agents' uncommitted Proposed entries and repo_map rows; resumed.
