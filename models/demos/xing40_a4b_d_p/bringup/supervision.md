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
- 06:00-06:25 C.dense.attn_hc: F56 sweep failed (noise1e-2 slips: [S, 24] gates output, bf16 projection, calibrated
  limit 0.0148 > the 0.007 whole-output noise), review agent ran as designed and added per-part checks (pre / post /
  comb rel, comb column sums, ranges); frozen b2e0f923971, PASS af8ce847bf0 (composed Sinkhorn on device, one
  [S/4, 32] all_reduce axis 1; no fork). Accepted. S.dense.01 PASS cb38f2865f3 (frozen without review, F49).
  C.dense.attn_collapse frozen without review (F56 sweep PASS), PASS b924b40968d. Accepted.
- 06:30 owner asked to look at the F56 branch again: new commit fb9ccf40814 (worst-column rel L2 for outputs <= 64
  columns, component_col 0.015; proof on all 42 reviewed Hy4 tests: 0 misses, 84/84 controls, 41/42 sweeps pass).
  Paused before S.dense.02, cherry-picked (e8828a10407). It applies to the already-frozen attn_hc test (checks auto):
  re-ran the gate on device: PASS, worst column vs cpu 0.0100, vs golden 0.0111 (Hy4 device iHC <= 0.0075): passes
  with ~1.35x headroom; watch ffn_hc / moe attn_hc for column-only failures. Owner noted the 64-column cutoff is a
  heuristic; proposed follow-up for the F56 branch: check the worst column on every float output with a per-column
  limit from the precision model, no NARROW cutoff (owner to decide). Resumed.
- 06:35 C.dense.attention attempt 1: ring_mla hit TT_FATAL (ring_joint_sdpa_program_factory.cpp:1352, kv_actual_isl
  needs streaming compute, which is off when fp32_dest_acc_en), the agent switched to the all_gather + chunked flash
  MLA fallback, whose SDPA CBs overflowed L1 at fp32 dest. Owner: "turn off fp32_dest_acc ... streaming compute is much
  faster" (rule 7 is HiFi4 only; fp32 accumulation was the planner's). Stopped the orchestrator and the agent between
  device runs (06:36:55), moved its WIP out of the tree (runs/run1/wip_attention_attempt1), added the owner rule
  (fp32_dest_acc_en=False for ring_mla, use ring_mla), intake re-approved on the owner's word; C.dense.attention
  restarted from its precheck.
- 06:55 C.dense.attention (attempt 1 after the restart): at fp32_dest_acc_en=False the streaming ring_mla failed the
  frozen test (worst row 0.054 > 0.045 on the x2 "big" second input: sharp softmax, 18 K tiles summed in 16-bit
  DEST). The agent extended the sdpa fork instead: latent-V ring_mla takes the streaming path at fp32 DEST (host-only,
  use_streaming_compute = !fp32_dest_acc_en || v_shares_k_buffer; bf16 DEST bit-identical to the source; 7 unit tests
  on 4x2; CHANGELOG + INDEX). Gate PASS (pcc 0.99999, rel vs cpu 0.0031), but the gate commit failed the repo's
  prefer-expect-error hook (pytest.raises in the new fork test) and the orchestrator crashed. Owner chose "keep" (the
  fork: streaming + fp32 DEST) over bf16 DEST + a test exemption. Overseer fixed the one pytest.raises -> expect_error
  (test passes) and resumed.
- 07:05 framework gap (F57 candidate): the C.dense.attention gate recorded PASS in state.json before its commit failed
  on the hook; on resume the orchestrator skipped the task (PASS, commit None) and the next gate commit (S.dense.05,
  9039a435f4e) swept in the staged attention files (tt/attention.py, sdpa fork change + tests). Content verified (the
  S.dense.05 swap test runs attention on device); bookkeeping only. Fix later: record PASS only after a successful commit.
- 07:15 owner asked why streaming excluded fp32 dest: the streaming PR (#38838) said "fp32_dest_acc_en is not functional
  with the streaming path" when the kernel hard-coded an 8-tile DEST; it is dst_size-parametrized since, and #45191
  keeps the exclusion without a reason. Owner: pause and check. Overseer check on device
  (generated/xing_fp32_stream_check, not tracked): fork ring_mla at fp32 DEST vs the source at bf16 DEST vs fp32 torch,
  every ladder rung's geometry (q32/k256; chunk 2048 / 8192 / 5120 incl. the 51200 last chunk) + q64/k256, q128/k128,
  q32/k128, q64/k128; spread and sharp scores: 22/22 pass, every case bit-identical over two runs, fork more accurate
  in every case (sharp worst row 0.14-0.16 vs 0.64-0.83 at bf16 DEST). Kept. Resumed.
- 16:05-16:50 M.1: the orchestrator's gate (s4096 ladder) sat silent 27 min after the mesh opened. py-spy: TtEmbedding
  -> ttnn.as_tensor (tt/model.py:92), 100 % CPU, no I/O. Killed the gate (16:37) and, since attempt 2 got no usable
  log, stopped the orchestrator and attempt 2's agent before any device run. Overseer probes: as_tensor of the full
  table takes 1.5 s fresh; its cache hit with an mmap-backed input never finishes (a cloned input hits in 0.6 s): the
  stale 940 MB embed tensorbin written by the first run made every later load spin. Moved the cache files to
  runs/run1/stale_cache, added a known_issues entry (fix: no cache_file_name for the embedding, or clone). Resumed.
- 16:50-17:05 L.s4096: M.1 had passed on the orchestrator's precheck (no agent ran, so nobody applied the known-issue
  fix) and wrote a fresh embed cache; the L.s4096 gate then spun in the cache hit again, and so did the fix agent's
  second run. Stopped orchestrator, fix agent and test; overseer fix in tt/model.py: TtEmbedding cache=False by default
  (fresh build 1.5 s), stale cache moved aside; resumed (L.s4096 precheck re-runs the rung).
- 17:10-17:45 L.s4096 PASS d21120da8d5 (min layer 0.9954, final 0.9977, logits 0.9988, state 0.9976, 0 host transfers).
  L.s16384 first gate failed on "No space left on device" (home 9.4 G quota full; JIT kernel builds); the fix agent
  cleared the uv download cache (3.2 G, regenerable), no model change: PASS dda19f93ad7 (min layer 0.9934, final
  0.9985, top5 1.0). L.last PASS 9885c95b1f5 (50k golden prefix + device 51200..56320: min layer 0.993, final 0.9985,
  logits 0.9991, state 0.9997, top1 0.969, top5 1.0). Paused before L.s56320 and moved the JIT kernel cache off home:
  copied ~/.cache/tt-metal-cache (3.6 G) to /localdev/dnijemcevic/tt-metal-cache; the orchestrator now runs with
  TT_METAL_CACHE=/localdev/dnijemcevic (rtoptions appends tt-metal-cache). Resumed.
