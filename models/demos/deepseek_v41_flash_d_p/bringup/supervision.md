# Supervision log: deepseek_v41_flash_d_p

Every intervention by the supervising session: time, task, trigger, classification, action, resulting commit.

## 2026-10-14 intake (owner delegated; owner away overnight)

- Owner's decisions (2026-10-14): deployment box galaxy_4x4 at 1M tokens; boxes galaxy_4x4 / galaxy_8x4 /
  loudbox_2x4 (partial only) / quietbox_2x2 (this run); partial mode, layers 0-3, 14, 20, 21, 24; engram on the host
  from the checkpoint files; vision and DSpark skipped; DeepSeek's inference code as ground truth; max precision,
  experts in a format exact for the checkpoint's fp4; the supervisor approves the intake spec, the run stops at the
  plan for the owner; framework commits stay local until the owner looks.
- 16:00 framework F65 (layers that read other layers' state: hooks state_sources) built before the run, since this
  model's attention shares KV caches, top-k indices and a candidate pool across layers; selftests 377 pass; commit
  a67c6435d58 (local). F64 layer pattern fix for bare `layers.N.` names: 5a7d562f62a (local).
- 16:15 intake approved by the supervisor on the owner's delegation. Checkpoint download (510 GB, revision
  5254448103b6) running in tmux `ds41dl`; the run starts when it completes.
- 16:50 framework F65 part 2 (token ids for steps keyed on them: ctx.extra tokens, device-model begin_chunk), for
  engram; selftests 379 pass; commit eb615f880ed (local). Cloned tt-d-gen (8a4a0042, no build; SC.1 reads it) to
  /localdev/dnijemcevic/tt-d-gen; it has no DeepSeek-V4.1 code.
- 20:10 owner delegated the plan approval (PL.1) to the supervisor as well ("no plan approval from me, you do the
  reasoning"); push still waits for the owner.
- 20:31 R.2 passed (a953697a19d) after one attempt. Reviewed: DeepSeek's model.py / engram.py run unmodified with the
  CPU port of kernel.py in place of the GPU kernels (as the owner's rule says); one shared tiled GEMM for oracle and
  reference (DS41-1: fp8 / fp4 requantization turns last-bit differences into whole steps) -- accepted, both still
  emulate the same kernels. Spec fix (supervisor's error, the reference's DS41-2): topk_idx and candidates are per
  position and grow with the sequence; removed from state.fixed; intake re-approved on the owner's delegation.
- 23:00-23:37 G.s56320, B.1, SC.1 passed (177596f4f64, 01bc89a9715, fbc3036f462). SC.1's owner questions decided by
  the supervisor (delegated): recorded in spec serving.owner_answers (bf16 row-major lossless; window KV TP-dedup;
  1 slot at 1M; prefix reuse on; d2p off; explicit geometry in our tests; runner smoke not required in a partial run).
- 00:12-07:00 PL.1 sat WAITING FOR A PERSON for ~7 h: the supervisor's own log watcher was not re-armed after the
  SC.1 event (framework/supervision gap). Fix F66 (b37744c2f2e, local): watch.py + orchestrator pid file, the skill's
  "Always be watching" section; owner asked for it explicitly.
- 07:10 framework F66 (b37744c2f2e, local; selftests 385 pass): plan placement `host` (the engram tables were the
  plan's one error: `skip` is rejected on decoder layers); K.1 requires smoke_runner_ok only for whole-model runs
  (tasks.yaml K.1 metric removed to match; the regenerated ledger equals tasks.yaml); agent rule 5 reworded to "no
  unnecessary host work" with the engram-style host input as the one exception (owner request).
- 07:14 plan.yaml: engram placement skip -> host (mechanical, F66; host 188.8 GB, never per chip). Spec: scaled.ladder
  (scaled / scaled_last at 64k, chunk 4096 = the deployment's 2048 rows per mesh row), serving.owner_answers; intake
  re-approved (delegated).
- 07:15 PLAN APPROVED by the supervisor on the owner's delegation. Reasoning:
  - Fits: run box 23.1 of 27.2 GiB at real size (8 layers), scaled 13.3; deployment galaxy_4x4 at 1M 26.9 of 27.2.
    Layout = the Kimi K2.7 4x4 / xing40 SP x TP scheme plus TtCSA's sparse attention; departures each reasoned
    (head->sequence reshard for sparse_sdpa's 32-head minimum, one sparse table refilled per source group, indexer
    heads replicated to avoid an all_reduce of 1M-key scores).
  - Routed experts as the checkpoint's packed fp4 + e8m0 bytes decoded exactly inside the expert op (fork extension
    of unified_routed_expert_ffn): accepted. The owner's rule asked for a format exact for fp4 (expected bfp8); bfp8
    is exact but needs 26.9 GiB per chip for this run's 8 layers alone and 33.6 GiB on the deployment, so it fits
    neither; bfp4 changes 16-17 % of values. Packed fp4 is exact and 0.53 B per value. The interim per-chunk decode
    into a bf16 temporary is a temporary rule-5 departure behind Setting experts_impl.
  - OPEN FOR THE OWNER (does not affect this run): the deployment fits 1M on galaxy_4x4 only with bfp8 for the fp8
    dense weights (lossy: 10 % of values one step, rel. err 3.5e-3); bf16 needs 28.4 GiB. The plan's override applies
    to galaxy_4x4 only; this run and every other box stay bf16 (exact). Lossless alternatives: an fp8 window cache
    (-1.2 GiB, server tooling change), an e4m3-decoding matmul fork, or galaxy_8x4. Measurable on this box before the
    Galaxy (same override at real size).
  - Scaled run: SG.1 must make the reference read the scaled counts from inference/config.json (scale.py copies only
    top-level files); in the scale role's scope.
- 07:30 OWNER DECISION: match the reference precision over context length: no bfp8 anywhere; galaxy_4x4 deploys at
  512K (26.5 of 27.2 GiB per chip; longest exact ~720K), 1M on galaxy_8x4 (18.5 GiB). "Keep the tests light (~50K),
  plan for the full; full tested later." Applied: spec deploy.seq 524288; plan.yaml galaxy_4x4 bfp8 override removed;
  plan.md owner-decision note on top (its 1M/bfp8 numbers superseded by the gate's results); serving answers updated.
- 07:35 SG.1 attempt 1 failed: the scaled ladder (65536, supervisor's spec) is longer than the canonical prompt (56320
  tokens). Supervisor error; attempt 2 stopped (agent + orchestrator killed, task RUNNING -> resume) before it could
  work around it. Scaled ladder set to 53248 (13 x 4096) <= the prompt and the owner's ~50K. The agent's attempt-1
  edits (reference reads the scaled counts from config.json; inference/ dir from the scale marker's source) reviewed
  and kept. Intake and plan re-approved (delegated).
- 08:00 OWNER: one serving chunk on every box, never above ~5K; no golden at a chunk this box never serves. Framework
  F67 (chunk fixed per deployment, serving.max_chunk 5120, intake asks for it, default ladder at the serving chunk).
  Spec: deploy chunk 5120, context 522,240 (102 chunks, ~510K) on galaxy_4x4 (1,044,480 on galaxy_8x4); ladder
  s10240 (2 x 5120, full dumps) / last / s56320; scaled ladder 56320 / 5120. tasks.yaml regenerated (old copy
  runs/run1/tasks_before_F67.yaml): G.s4096, G.s16384, L.s4096, L.s16384 -> G.s10240, L.s10240; G.s56320 reused.
  plan.md note (2); plan check passes on every box at 5120. SG.1's scaled golden (53248 / 4096) stopped mid-way.
  Intake and plan re-approved (delegated). Consequence noted for perf: ~1.6x more chunks per prefill, so the
  per-chunk sparse-table refill is paid more often.
- 08:20 OWNER (correcting 08:00): serving chunk 4096 on the deployment (SP 4 -> 1024 tokens per row), never larger on a
  bigger box; this 2x2 proxy runs 2048-token chunks (1024 per row) scaled and at real size; experts 384 / 16 = 24 per
  chip -> 96 on the scaled 2x2 (as scaling.yaml). "Generate no goldens other than those this box runs; delete the
  ones for other boxes." F67 reworked (chunk = deploy.chunk x min(1, sequence factor)); spec deploy 524,288 / 4096,
  target 55,296 / 2048, ladder s4096 / last / s55296, scaled ladder 55,296 / 2048. Deleted goldens s16384_c8192,
  s56320_c5120 and the partial scaled ones (s53248_c4096, s65536_c4096); kept s4096_c2048. tasks.yaml regenerated
  (G.s56320 / L.s56320 -> G.s55296 / L.s55296); plan check passes on every box. Intake and plan re-approved.
- 08:20 after the resume, G.s4096 failed ("golden exists; goldens are generated once") and a fix agent started: the
  supervisor had dropped G.s4096's PASS record when regenerating the ledger. Supervisor error: agent and orchestrator
  stopped, G.s4096's PASS record (2026-10-14, 34e5c9ccb0a) and results restored, the fix attempt's report removed.
  DS41-12 (prompt shorter than the 65536 scaled rung) marked resolved (scaled.ladder 55296 / 2048).
- 09:55 TT_METAL_HOME / PYTHON_ENV_DIR (from ~/.bashrc) point at /localdev/dnijemcevic/tt-metal; harmless for this build (runtime root compiled in as tt-metal2, profiler dir set by run_safe_pytest) but a trap: pause after G.s55296 and relaunch with both pointing at tt-metal2 before any device step.
- 10:05 shared kernel cache: ~/.bashrc TT_METAL_CACHE=/localdev/dnijemcevic/tt-metal/built is used by both trees; the JIT build key hashes flags, not the source root, so the two checkouts can overwrite each other's firmware / kernels. Relaunch after G.s55296 with TT_METAL_CACHE=tt-metal2/built, TT_METAL_HOME=tt-metal2, PYTHON_ENV_DIR=tt-metal2/python_env.
- 10:10 correction: the JIT cache validates each target against its include paths and a dependency hash, so cross-tree entries are recompiled, not reused; the separate cache only avoids churn and concurrent writes. Owner may run device code in tt-metal now (this run is CPU-only until the relaunch).
- 11:05 G.s55296 PASS (02bfcedd31d). Relaunched from tt-metal2 with F68 (env pinned; child processes verified: TT_METAL_HOME / PYTHONPATH / TT_METAL_CACHE / PYTHON_ENV_DIR all tt-metal2), --until SG.1 so a full-box check runs before the first device step (owner may have run single-card tests). Owner's ~/.bashrc / ~/.zshrc now derive the four from the starting checkout (tt_env; .bak kept).
- 12:35 SG.1 PASS (b3255d56c0d) after the agent's attempt-1 work: the reference reads the scaled counts from config.json (inference/ from the scale marker's source). Checked: on the real checkpoint inference_config() equals inference/config.json exactly; scaled: 32 heads, 4 groups, 96 experts, top-6. Accepted. --until SG.1 did not stop the run (C.* tasks do not depend on SG.1); the run went on to C.swa.attn_hc; a broken fabric would stop it as an infra stop.
- 13:15 C.swa.kv test attempt 1 failed only on the supervisor's DS41-12 edit (614 > 600 chars; the test role treats the problem as a failure). Supervisor error; left to attempt 2 (in its brief); if it stops, shorten the entry and rerun.
- 13:45 KNOWLEDGE CONTRADICTED DS41-15 (S.swa.05 implement 1): the kv fp8 flips are not GEMM sum order; the FPU rounds each in-tile partial dot to ~TF32 at every fidelity (probe). Agent's fix tt/exact_mm.py (32 masked copies, one batched matmul, SFPU adds; ~25x the matmul cost, behind ATTN_IN_MM). Relayed to the owner.
- 14:32 KNOWLEDGE CONTRADICTED K-31 (C.swa.attention implement 1): the agent extended the sdpa fork (sparse_sdpa high_precision + attention_sink; CHANGELOG entry, unit test 4 passed). Verify at the pause: the fork's unit suite + every model's cases (bringup-fork-op step 5).
- 14:36 pause: C.swa.attention PASS (94ca2380717). sdpa fork change (sparse_sdpa high_precision + attention_sink,
  under SPARSE_SDPA_HIGH_PRECISION; off = source program) verified by the supervisor: unit suite 32 passed / 10
  skipped, model cases (fork_tests) 5 passed / 1 skipped (Xing 8-device), as the agent recorded. Accepted. Shared
  K-31 is out of date (the agent's evidence); fix at the run-end curation.
- 14:40 framework F69 (overseer notes and aborts, overseer files ignored by the path check, dashboards refreshed
  per gate, watch --quiet-passes) and F70 (timing breakdown, timeline, device timing log, dashboard section);
  dcf8da27d5a (local); selftests 397 pass. From now on corrections go to the agents as overseer notes.
- 15:40 pause: S.swa.06 PASS (35beb8bf9ce): exact window attention (tt/window_exact.py; load-time masks only, device-only forward; WINDOW_SDPA / ATTN_IN_MM default exact, fast paths one setting away). Accepted. Precision bar question (1% tail after fp8 requantization vs exact kernels' cost) raised with the owner. F71 (dashboard links + DASHBOARD STALE wake, overseer CLI fix) applied; selftests 399 pass.
- 15:58 OWNER DECISION (precision bar, option 2): exact-emulation kernels (32-way split matmuls ~25x; piecewise window
  attention 64 ms per layer) are "completely unacceptable"; thresholds must come from the noise floor of the exact
  dtypes with the CPU op, not fixed numbers. Framework F72 (db4539a6d62): the swap tail limit = 2 x the bf16
  precision model's block-out error (no fixed floor or cap). Spec rule "Accuracy is judged against the noise floor"
  (intake re-approved). Note OV-3: fast paths back as defaults, exact paths as settings, re-run S.swa.05 / S.swa.06.
  OV-1 closed (both agents answered done; DS41-12 is 352 characters).
- 16:10 OV-3 widened to every implement task (S.swa.07 / C.swa.ffn_hc passed without an agent, so no one read it). Owner: no work-order feature now, but the fast path must not wait until the end: check at each progress check; work orders (overseer task) are a pending framework item.
- 16:20 OV-3 done by C.swa.ffn_collapse (fast paths default; S.swa.05 PASS on fpu: 1.1 % <= 4.6 %), answered disagree with evidence: S.swa.06 fails only the fixed 2 % swap step cap (step 2.1 %, its noise floor 2.2 %). S.swa.09 had started; orchestrator stopped (SIGTERM, agent ended with it, exit recorded). Framework F73: swap step limit = 2 x max(component error, bf16 resolution), no cap or floor; selftests 400 pass. OV-3 closed; OV-4 asks to update DS41-20.
- 18:40 C.swa.experts PASS (48ecee42266): model-side loop (tt/experts.py), packed fp4 decoded to bf16 every chunk (EXPERTS_IMPL=loop, ~4 ms per matrix: the plan's interim, rule-5 departure until the fork decodes in-op); fp32 routing weight carried through dispatch as 3 bf16 pieces. vs CPU rel 0.0022 (limit 0.015). Accepted as the plan's interim; the in-op decode must come before any perf number counts.
- 19:27 layer 0 (swa) fully on device: S.swa.15 PASS (ee442a47dd7). C.swa_engram.engram_rows PASS (1f9ad1c0bec): host hash + lookup in begin_chunk, tables np.memmap (never loaded), only the chunk's rows sent; follows rule 5's host-input exception. Accepted.
- 20:13 the orchestrator crashed: the supervisor's own git commit (links.yaml) held index.lock while the gate committed C.swa_engram.q_a (supervisor error). F74: framework git retries on the lock; publish time moved to the run dir; the overseer does not touch git during a run. Resumed.
- 21:55 OWNER: 'the deployment box is ONLY for computing the scale factor; once computed EVERYTHING goes scaled: goldens, model (fewer experts, fewer heads), smaller chunk.' No real-size track on a smaller box. The supervisor had built F64 with both (real-size tests + a scaled run) and listed it in the intake questions without calling it out: supervisor's error. Pausing to make scaled-only the framework behaviour (F75) and re-point this run.
- 22:25 F75 applied (c06b93b7766): this run is now the scaled model only. Spec: no separate scaled section; target
  and ladder (s4096 / last / s55296 at 2048) are this box's units. tasks.yaml regenerated (old copy
  runs/run1/tasks_before_F75.yaml, old state state_before_F75.json): SG.1 -> G.s4096 / G.s55296 (scaled) -> every
  device step; R.* / PL.1 read the real checkpoint. Kept: R.*, B.1, SC.1, PL.0, PL.1, SG.1. Reset: G.*, and all 60
  C / S tasks of layers 0 and 1 (re-run and re-frozen on the scaled model; their real-size passes stay in git
  history only). Real-size goldens deleted; the scaled s55296 golden (from SG.1) is reused. Intake and plan
  re-approved (delegated).
- 23:10 C.swa.attention (scaled) attempt 1: the agent found the frozen contract tests read hf/inference/config.json, absent / unscaled in the scaled checkpoint (KNOWLEDGE CONTRADICTED DS41-11). Framework gap: orchestrator stopped; F75(2): scale.py copies weight-less subfolders and rewrites scaling.yaml config_files with the cut counts; hard links broken first (the scaled inference/ files were hard links to the real checkpoint: rewriting in place would have changed the real config). scaling.yaml: config_files for inference/config.json (n_heads, o_groups, n_routed_experts). Re-cut reused the weights; real config verified unchanged. Resumed.

## 2026-10-16 06:40: review of the overnight gates (78/287)
- swa and swa_engram blocks complete (C.* and S.* all pass). S.swa.06 failed once (all-zero swapped outputs) and passed on attempt 1 (516f4185ac6).
- S.swa.11 (router vs golden 0.9887 < 0.99) took 3 attempts; passed (2c7fefed7c1) with a sdpa fork option fp32_scores (Float32 scores/probabilities, TF32 round-to-nearest of FPU operands, fork unit test added) and q rounded to TF32 on device (ttnn bit ops), plus hc_fn weight rounded to TF32 at load. Reviewed: device-only, load-time host work only, behind settings (WINDOW_SDPA_SCORES, FP32_WEIGHT_ROUND) with a bf16 perf-pick fallback; +2.1 ms per layer. Accepted; listed for the end-of-run report as a precision step above the reference.
- Dashboards republished (version 27). OV-4 / OV-5 still open (no answers yet).
- OV-5 (experts as bfp8 at load) never reached an agent: posted 22:27, after the only experts implement agent (18:25); since then every experts step passes on the shared code with no implement agent (C.swa.experts and C.swa_engram.experts ran the test role only). The 6 remaining C.*.experts will do the same. This is the work-order gap (a note only reaches an agent that runs). Plan: one hand-added task for it after the component steps, before the ladder (owner: work orders "not now, but not at the very end").

## 2026-10-16 06:45: FIX.1 injected (owner: the weights fix now, not at the end)
- Added task FIX.1 (step implement, role implement, precheck false: the agent always runs) before S.enc_source.03: routed experts as bfp8 at load, no per-chunk fp4 decode. Gate: the frozen tests C.swa.experts, C.swa_engram.experts, S.swa.15, S.swa_engram.18. S.enc_source.03 depends on it. OV-5 closed into it. Plan re-approved (delegated).
- Next: F76 (every per-step limit from the noise floor; the owner: no fixed bars), F77 (`overseer fix`: this injection as a command), then FIX.2 (attention back to the reference's precision: q bf16, scores fp32, probabilities bf16).
