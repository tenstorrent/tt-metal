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
