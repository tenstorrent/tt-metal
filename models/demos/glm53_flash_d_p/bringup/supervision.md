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
- C.kda_dense.attn_residual PASS (935cb6701f0) pcc 0.999994 attempt 1; tt/residual.py pure ttnn. Accepted.
- S.kda_dense.05 PASS (d211d3ccb89) 0.999982; C.kda_dense.ffn_hc PASS (184318f240f) 0.999994 (reuses TtHcWeights); S.kda_dense.06 PASS (f59f6bda4ef); C.kda_dense.ffn_collapse PASS (fb1907ea62b).
- S.kda_dense.07 PASS (62a76fdd63f); C.kda_dense.ffn_norm PASS (4a9cb3d13bd).
- S.kda_dense.08 PASS (dcc487a552e); C.kda_dense.mlp PASS (c15c5fab05d) pcc 0.999994 attempt 1: TP=4 clamped SwiGLU from MiMo 2x2 TtDenseMLP, HiFi4 fp32 acc, fp32 all_reduce(cluster_axis=None) (bf16 AR biased +0.19%). Accepted.
- S.kda_dense.09 PASS (8096355b111); C.kda_dense.ffn_residual PASS (f6032b1e830).
- S.kda_dense.10 PASS (dc275b3c7de) 0.999977: layer 0 fully on device, 20/20 first attempt. C.dsa_moe.attn_hc PASS (476b5212a96).
- S.dsa_moe.01 PASS (a2cfb0cdbf7); C.dsa_moe.attn_collapse PASS (4eec486d4ca).
- S.dsa_moe.02 PASS (8bc76798980); C.dsa_moe.attn_norm PASS (85ec2bfedc6).
- S.dsa_moe.03 PASS (fd63888959e); C.dsa_moe.q_a PASS (615eb1131f9).
- S.dsa_moe.04 PASS (8d82d7e6639). C.dsa_moe.indexer PASS (d0e8a09e69e) attempt 1: selection overlap chunk1 mean 0.99866 / worst row 0.9922 (limits 0.9975 / 0.98), keys rel 0.0025. All device, not deferred: pooled keys by reshape/softmax, per-head ttnn.linear + fp32 addcmul scores (indexer_score_dsa's bf16 head sum gave 0.99715, failed the extra check; kept as GLM_INDEXER_SCORE=op), ttnn.experimental.topk_large_indices k=512, device tail/sentinel tables. Accepted.
- 08:5x C.dsa_moe.attention PASS (3ef3d142602) attempt 1: pcc 0.99999, rel 0.0044, block 0.0049, kv_latent rel 0.0025.
  ttnn.bringup.sparse_sdpa(high_precision=True) over the 512-wide latent (the source op's bf16 running state failed
  the per-token norm ratio at 0.993). FORK EXTENDED: ttnn/ttnn/bringup/sdpa sparse_sdpa high_precision (default off =
  source program, bit-identical checked; 118 source tests 0 regressions, 92 sparse_sdpa source tests recorded, unit 28
  passed, MiMo cases 4 passed; CHANGELOG + INDEX). Remaining bias 0.13% low inside sparse_sdpa (TF32 reads), noted.
  Classified: FRAMEWORK - the gate commit left out every fork file and the knowledge entries (stage_paths did not
  list BRINGUP_OPS / common_paths). Paused after S.dsa_moe.06 (0ffa7727e85, swap 0.99999); fix F49 a51e970a8ab
  (selftests 212 -> 214, new test fails on the old gate.py); committed the missing files 5fd4a4cfe2b (test_refusals
  moved to the repo's expect_error fixture to pass pre-commit). Resumed.
- C.dsa_moe.attn_residual PASS (381fe013e2b).
- S.dsa_moe.07 PASS (f4df333b14d); C.dsa_moe.ffn_hc PASS (525bcfbaa0b).
- S.dsa_moe.08 PASS (fa65d62a85f); C.dsa_moe.ffn_collapse PASS (9910cb051bd).
- S.dsa_moe.09 PASS (fd65fad2be0); C.dsa_moe.ffn_norm PASS (ef3af813154).
- S.dsa_moe.10 PASS (7e93b122d0e). C.dsa_moe.router PASS (79db354f597) attempt 1: selection overlap 0.99683 (CPU fp32 on bf16 input 0.99695), worst row 7/8, matched weights rel 0.0012; fp32 logits + fp32 bias + ttnn.topk on fp32 keys (from MiMo TtRouter fp32 mode). Accepted.
- S.dsa_moe.11 PASS (20276fd0050). C.dsa_moe.experts PASS (4582a981e4a) attempt 1: pcc 0.99997, rel 0.0077, coef 1.0005; clamp probe rel 0.009. Port of MiMo 2x2 TtExperts: bringup dispatch/offset_cumsum/combine (axis 0, group 2), unified_routed_expert_moe ClampedSiluGlu high_precision, bfp8 experts, HiFi4 fp32 dest; 72 local / 288 global worked with NO fork change (plan's unverified item confirmed). Accepted.
- S.dsa_moe.12 PASS (20439999cac); C.dsa_moe.shared_expert PASS (9ce52e60715).
- S.dsa_moe.13 PASS (6dda462ce0b); C.dsa_moe.moe_add PASS (35a2f14d3c2).
- S.dsa_moe.14 PASS (03df6ae15e8); C.dsa_moe.ffn_residual PASS (f99b3ae029b).
- S.dsa_moe.15 PASS (ca3a44d6854): layer 3 fully on device, all first attempt.
- C.kda_moe.attn_hc PASS (929a584b7ac).
- S.kda_moe.01 PASS (8bfcc9df126); C.kda_moe.attn_collapse PASS (acf2ae52cec).
- S.kda_moe.02 PASS (bedf05af599); C.kda_moe.attn_norm PASS (c27b749a7a1).
- S.kda_moe.03 PASS (3a1ada7086a). C.kda_moe.attention PASS (b50db8f94fb) reused TtKdaAttention: pcc 0.99998, rel 0.0073, state rec 0.0166, conv 0.0017, WORST HEAD 0.047 vs limit 0.05 (layer 0 was 0.024): thin margin, watch the KDA state metrics on the ladder (11 chunks at 56k); known cause = prepare_chunk_recurrence intra term (left for later in C.kda_dense.attention breadcrumbs).
- S.kda_moe.04 PASS (292437b531c); C.kda_moe.attn_residual PASS (c9c58454a7d).
- S.kda_moe.05 PASS (11af05d81f1).
- C.kda_moe.ffn_hc PASS (ee0312edd46).
- S.kda_moe.06 PASS (6cdb85d429b); C.kda_moe.ffn_collapse PASS (d3f1d4957a3).
- S.kda_moe.07 PASS (8aad9b4b902); C.kda_moe.ffn_norm PASS (5aa0383a1f0).
- S.kda_moe.08 PASS (22b812e28b2); C.kda_moe.router PASS (f5292b66485).
- S.kda_moe.09 PASS (8942dae2332); C.kda_moe.experts PASS (18f4a929b23).
- 19:3x S.kda_moe.10 gate PASS but its commit failed: pre-commit check-large-files refused state.json (506 KB > 500 KB);
  orchestrator exited 1. Classified: FRAMEWORK (every recorded metric copied into state.json; swap tests record
  300-400 each). Fix F50 091b5556d2f (state keeps gated metrics above 64; results/<task>.json keeps all; selftests
  214 -> 216, new test fails on the old gate.py). Migrated state.json 518 KB -> 201 KB with the same rule; replayed the
  S.kda_moe.10 gate commit (a8d385cd98d). Resumed.
- C.kda_moe.shared_expert PASS (6f465aa7343).
- S.kda_moe.11 PASS (2c4e6f1035d); C.kda_moe.moe_add PASS (302686016c0).
- S.kda_moe.12 PASS (924a3824a5b); C.kda_moe.ffn_residual PASS (3610a4cccaa).
- S.kda_moe.13 PASS (9e85460f086): layer 4 fully on device. All 5 layers / 3 block types on device, every component + swap gate first attempt.
- M.1 PASS (8509e6ecad3): layers 0.99995+, host_transfers_per_layer 0. L.s4096 PASS (7a2eb951412) layers >= 0.99995, state min 0.99916. L.s16384 PASS (043b3c6e40e) layers >= 0.99993, state min 0.99269 = kda_recurrent L01 (4k 0.99916 -> 16k 0.99269; L02 0.99966 -> 0.99624): KDA recurrent state drifts with length (limit 0.97); watch L.last / L.s56320. Likely the prepare_chunk_recurrence TF32 intra/decay terms (C.kda_dense.attention breadcrumbs).
- L.last PASS (d4c6d031d5b): layers >= 0.99994, state min 0.99836 (kda_recurrent L01, one chunk from the golden 50k prefix).
- L.s56320 PASS (56638699400): all 11 chunks on device; layers >= 0.99955 (L01 worst); state min 0.98217 = kda_recurrent L01 (L02 0.98536). KDA recurrent-state drift with length confirmed: L01 0.99916 (4k) -> 0.99269 (16k) -> 0.98217 (56k); limit 0.97 passes with 1.2 pt margin. Candidate precision follow-up: fork prepare_chunk_recurrence (TF32 subtraction in k_dec_t / intra) - the agent's 'left for later'. K.1 started (first check FAIL expected: no adapter yet).
- 21:3x K.1 failed attempts 1-2 on "hooks.contract_state_pcc is missing": the read-back was written in tt/runners/adapter.py
  but hooks.py was outside K.1's allowed paths; the agent refused a runtime monkeypatch (correct). Classified:
  FRAMEWORK. Killed attempt 3 (it would fail the same way), fix F51 79a61144519 (contract step may change and commits
  hooks.py; selftests 216 -> 217). K.1 latent / index-key read-back via the engine already PCC 0.99997 / 0.99998.
  Rerun from K.1.
- K.1 PASS (99ac5360ccc) attempt 1 after F51: contract checks 0 failed, acks_early 0, producer KV latent 0.99997 / index_key 0.99998, fixed state via contract_state_pcc: kda_recurrent 0.99920, kda_conv 0.99998 at the real end of the padded last chunk. kda_attention.py now passes actual_end (main's #57070 fix in use). Shared engine change: adapter flag acks_in_kv_slot_space (default False, other models unchanged) + runner maps slot->global layer; registry entry glm53_flash_d_p; docs updated. Accepted.
- 21:4x X.1 PASS (3f7435ee52c): warm chunk 51200->56320 device 528 ms (mHC sections 277 ms = 52%). X.2 opportunities:
  owner picked (1) the mHC residual mix (rows 1-2, attn_residual + ffn_residual 166 ms); (2) = split the 4-stream
  residual by sequence across chips, only if (1) works. Added P.1 (deps X.2; X.3 now also depends on P.1): gate =
  the 6 residual component tests + ladder `last` + profile with attn/ffn_residual < 60 ms each and total < 500 ms.
  Perf approved 21:38. Also started a read-only agent on KDA state handover prefill -> decode (the engine migrates
  only the KV table; GLM keeps KDA state in the model per slot; decode on another machine would get no KDA state).
- KDA state handover research: not supported today; tt-metal PR #56443 (open, closes #57403) adds recurrent+conv state as extra table entries + 2 adapter hooks (mocked migration only); #57184 conflicts; tt-blaze draft #3265 / branch bklockiewicz/k3-cache-migration unmerged. GLM would need a decode consumer, migratable state buffers + hooks per #56443, and acks for KDA layers (layer 44 is KDA). Report scratchpad/kda_state_handover.md.
- P.1 PASS (5b4b7ba214e) attempt 1: attn/ffn_residual 82.9 -> 31.5 ms each, chunk 528 -> 428 ms (-19%); residual PCCs 0.99999, layers >= 0.99994, state min 0.9984; per-32-token-tile matmul, HiFi4 fp32 DEST, coefficients read as tf32 (noted); old path GLM_RESIDUAL_MIX=addcmul. Owner: go with (2) if (1) works -> pausing after X.3 (already running) to add P.2 and rerun X.3 after it.
- 22:3x X.3 FAIL only timeline_ok (profiler dropped programs: 1281 vs 1319 per chip); all accuracy passed; full prefill
  0->55k 4501 ms; state min 0.98247. Fix agent 1 confirmed PROGRAM_SUPPORT_COUNT=3000 fixes it (timeline 423.7 ms,
  gaps 0.7 ms). Classified FRAMEWORK: killed fix attempt 2, F52 38574604693 (PROFILE_ENV adds
  TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000; GLM profile gates X.1/P.1/X.3 patched; selftests 217 -> 218).
- Added P.2 (owner's pick 2, after P.1 worked): mHC residual split by sequence across the 4 chips, deps P.1, X.3 now
  depends on P.2; gate = the 24 mHC component tests + ladder s4096 + last + profile total < 400 ms. Rerun from X.3.
