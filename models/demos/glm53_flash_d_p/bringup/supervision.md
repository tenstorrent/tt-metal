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
