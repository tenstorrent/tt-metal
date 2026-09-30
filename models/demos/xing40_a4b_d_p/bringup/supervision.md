# Xing4.0-29B-A4B bring-up: supervision log

Overseer log: time, task, trigger, classification, action, resulting commit.

## 2026-09-30 intake
- Owner prompt: xing_bringup_prompt.md (repo root). Box: 8x Blackhole p150b (LoudBox), mesh 2x4, FABRIC_2D opens and
  all_gather passes on both axes (probe 02:56).
- XingChen-AGI/Xing4.0-29B-A4B @ baae3c3e813cad5f888f1f485cfff659c89076c5, not gated, apache-2.0, 62 GB bf16,
  downloaded to /localdev/dnijemcevic/bringup/xing40_a4b_d_p/hf; every spec.checkpoint.expect shape verified.
  Custom modeling code (trust_remote_code) builds on python_env's transformers 5.12.1; nothing vendored.
- Fit: all 40 layers, <= ~11 GiB per chip (experts bfp8 3.3 + rest bf16 replicated 5.1 + MLA cache 2.4).
- 03:05 owner approved the spec ("yes"); intake approved, ledger --early 8 tasks, run1.
- 03:15-05:10 R.1 PASS b72ea7fe647 (smoke "Paris<_end>", HF text_top1_acc 0.749). Overseer check of the low accuracy
  (scratchpad probe, CPU): rope_interleave=False gives 0.132 (so the interleaved layout is right), eager / fp32 give
  0.744 / 0.746 (not precision); greedy recites the book after its own markdown header. Owner's earlier small models:
  Gemma-4 26B-A4B 0.67, the big ones 0.955-0.965. Classified plausible for a 4B-active model; floor left at 0.4.
- R.2 PASS 89499441318 (attempt 1): standalone reference (reference/xing_ref.py, no HF import), parity PCC 1.0 on 40
  layers and logits, top1 match 1.0. Accepted. R.3 65a71665b03, G.s4096 / s16384 / s56320, B.1 (2x4), PL.0 PASS.
- 05:15 owner: cherry-pick F56 (dnijemcevic/f56-component-checks): reviewed the diff (checks=None unchanged, freeze
  sweep, CPU proof 73/73, device 6/6, default review on); cherry-picked 6f2d7d30e45..a7816821e06, selftests 290 passed
  8 skipped. Owner chose agents.component_review: none (spec edit, intake re-approved on their word).
- 05:25 PL.1 attempt 1 (2x4, replicated 4-stream residual + TP=8 heads / EP=8, glm53 scheme): owner REJECTED, no
  sequence parallelism. Overseer probe: the box opens as 4x2 with FABRIC_2D, all_gather / all_reduce on both axes.
  Owner: mesh 4x2, SP=4 (rows) x TP=2 (columns), the Kimi K2.7 4x4 layout. Spec: box.mesh [4, 2] + owner rule
  (agents.rules); intake re-approved on their word. Draft plan moved out of the tree (scratchpad plan_attempt1_2x4);
  its known_issues entry (mhc_split_sinkhorn differs from Xing's Sinkhorn) kept. rerun --from B.1 (B.1 for the new
  mesh, PL.1 replanned); goldens unaffected (CPU).
- 03:15-05:10 R.1 PASS b72ea7fe647 (smoke "Paris<_end>", HF text_top1_acc 0.749). Overseer check of the low accuracy
  (scratchpad probe, CPU): rope_interleave=False gives 0.132 (so the interleaved layout is right), eager / fp32 give
  0.744 / 0.746 (not precision); greedy recites the book after its own markdown header. Owner's earlier small models:
  Gemma-4 26B-A4B 0.67, the big ones 0.955-0.965. Classified plausible for a 4B-active model; floor left at 0.4.
- R.2 PASS 89499441318 (attempt 1): standalone reference (reference/xing_ref.py, no HF import), parity PCC 1.0 on 40
  layers and logits, top1 match 1.0. Accepted. R.3 65a71665b03, G.s4096 / s16384 / s56320, B.1 (2x4), PL.0 PASS.
- 05:15 owner: cherry-pick F56 (dnijemcevic/f56-component-checks): reviewed the diff (checks=None unchanged, freeze
  sweep, CPU proof 73/73, device 6/6, default review on); cherry-picked 6f2d7d30e45..a7816821e06, selftests 290 passed
  8 skipped. Owner chose agents.component_review: none (spec edit, intake re-approved on their word).
- 05:25 PL.1 attempt 1 (2x4, replicated 4-stream residual + TP=8 heads / EP=8, glm53 scheme): owner REJECTED, no
  sequence parallelism. Overseer probe: the box opens as 4x2 with FABRIC_2D, all_gather / all_reduce on both axes.
  Owner: mesh 4x2, SP=4 (rows) x TP=2 (columns), the Kimi K2.7 4x4 layout, and "tell the agent to think hard". Spec:
  box.mesh [4, 2] + owner rule in agents.rules (incl. "Plan role: think hard ..."); intake re-approved on their word.
  Draft plan moved out of the tree (scratchpad plan_attempt1_2x4); its known_issues entry (mhc_split_sinkhorn differs
  from Xing's Sinkhorn) kept. rerun --from B.1 (B.1 for the new mesh, PL.1 replanned); goldens unaffected (CPU).
- 05:50 PL.1 attempt 1 on 4x2 (SP=4 rows x TP=2 columns, Kimi layout: ttMLA chunked ring_mla over axis 0, block-cyclic
  latent cache, hy4 TtHcGates mHC, DeepSeek 2D dispatch in 4-chip columns, reduce_scatter over axis 1; 9.63 of 27.2 GiB
  per chip, 0 CPU / OPGEN steps). Overseer checked: moe README 4x2 example, mla.py:980 chunked requires
  is_balanced=False, TtHcGates, memory items. Owner approved ("ok approve"). approve plan; resumed.
- Owner delegation (05:50): run autonomously from here; reset boards when needed (no other device job running);
  investigate wrong-looking results myself, keeping in mind that swap (F49) and component (F56) tests freeze on built-in
  checks without a review agent. Still the owner's: perf picks, op-gen launches, pushes.
