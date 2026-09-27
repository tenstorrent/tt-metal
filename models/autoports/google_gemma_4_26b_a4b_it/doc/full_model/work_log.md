# Full model work log

Stage 06, google/gemma-4-26B-A4B-it, checkpoint revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Starting commit `6b1776a0bc`.
Status: implementation and validation in progress; no acceptance claim.

- `timeout 60 tt-smi -ls --local`: four Blackhole P300c ASICs visible.
- `timeout 90 python` TP4 `open_mesh_device(MeshShape(1,4), trace_region_size=100000000)` / close: passed.
- Read accepted Stage05 README, residual and precision contracts, memory plan,
  current multichip decoder, HF text model, readiness contract, both common samplers.
- Fresh AIME24 chat reference, 100 continuation tokens, top-100: generated.
  Exact command/revision/hash in `reference_metadata.json`.
- Initial HF runner failed on Transformers 5.12.1 chat-template `BatchEncoding`.
  AutoFix isolated worktree proved the return-type mismatch; explicit
  `return_dict=False` repairs it. Ten host tests passed. Reports and patch:
  `bringup/artifacts/reference-fix/{AUTODEBUG.md,AUTOFIX.md,reference_tokens.patch}`.
  Full original reference command passes after integrating that patch.
- Reduced real-layer probe: `OMP_NUM_THREADS=8 timeout 300 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.probe_full_model --output
  models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/probe.json`.
  Initial setup failure: nonexistent `BlackholeComputeKernelConfig`; corrected
  to the checkout's `WormholeComputeKernelConfig`, also used on Blackhole.
  Initial log `probe.log`, rerun `probe_retry1.log`.

- Fresh source-only AutoFix diagnosis: `AUTODEBUG_prefill_l1.md`. The 261-token
  standard readiness prefill collides at SDPA CB end 1,315,840 vs occupied L1
  1,297,664. Reduced layers0/5 + sampler + 30 semaphore sets pass; reserving the
  missing 28×4 router buffers reproduces the same CB end against frontier
  1,301,504 (`probe_l1_routers_retry.log`). Router buffer storage is 8 KiB per
  layer. The parent applied a serial full-model router pool, preserving the
  selected decoder operations/precision; full readiness and trace validation
  are pending. The diagnosis agent used source/arithmetic only and changed no
  implementation files. The initial clone-based reservation failed before
  prefill and is not the reproducing experiment.

No vLLM work performed. Full-model metrics, qualitative verdict, capacity
validation and independent review are pending. No stage checkpoint committed.

## Focused trace and L1 repairs

- Reduced layers0/5 plus full terminal, 261-token prefill and 28 additional CCL
  semaphore sets passes under `TT_METAL_TRACE_ALLOC_TRACKING=1` and
  `TT_METAL_TRACE_ALLOC_TRACEBACKS=1`; `probe_l1_retry.log`.
- Adding28 sets of four matching 2KiB router allocations reproduces SDPA L1
  collision, CB end1315840 versus frontier1301504. Original full-stack frontier
  was1297664. `probe_l1_routers_retry.log`; source diagnosis
  `AUTODEBUG_prefill_l1.md`. The earlier clone reservation was an invalid probe
  because sharded clone rejects its non-origin worker; empty exact-spec buffers
  are the successful discriminating control.
- Model now owns one serial pool of the four identical/scratch router buffers.
  Gate results are copied to interleaved buffers before downstream use. This
  saves29*8192=237568B of persistent L1 address space without changing any
  decoder op, dtype/fidelity, shard placement, residual or CCL policy.
- `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1
  OMP_NUM_THREADS=8 timeout180 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace --output
  .../doc/full_model/trace_contract.json` passes. Exact command uses `timeout 180`.
  `trace_contract_retry.log`: sampled token equals next replay's captured token
  snapshot, position increments, unchanged page table copies0, changed table
  copies1 into same address, reset retains buffers/traces and zeroes owned KV,
  repeated generation agrees. Reduced correctness evidence, not full performance.
- Both common greedy strategies pass a standalone TP4 known-winner oracle for
  tokens65540 and196633 on all ranks. `sampler_comparison.json`. Its timings
  include trace-allocation tracking (GC on replay) and are diagnostic only.
- Fresh HF controls completed for all six shared prompts in
  `models/common/readiness_check/vllm_prompts.txt`, each rendered with the exact
  HF chat template, maximum128 tokens and greedy sampling. `qualitative_hf.json`
  includes rendered text, token IDs and raw completions. HF dtype is BF16.

## All-layer quality and headline token-out measurement

- `OMP_NUM_THREADS=8 timeout 600 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.run_full_readiness
  --skip-readiness --qualitative --performance --output
  models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/generation_run.json`
  passed (`generation_run.log`). All six chat-template prompts ran on full30
  layers. Actual completions were inspected; see `qualitative_verdict.md`.
- Exact4096 input/128 generated/B1/C1 autoregressive measurement:
  TTFT2118.337ms, token-out49.318t/s/user. `performance.json`.
  127model and127sampler replays; no steady-state input/position/page-table
  refresh or full-logit readback. Allocation tracking disabled for measurement.
- Shared AIME check passed after router scratch pooling: `readiness.json`,
  `readiness_retry.log`, top1/5/100 prefill=.96/1/1; decode=.94/1/1 over100
  positions. Teacher forcing uses traced token-out with explicit callback
  injection; tracking-enabled timing is not an inference performance result.
- Same-prefix HF divergence controls for six prompts: `qualitative_divergence.json`.
  TT differing-token ranks2,3,7,2,2,2. Lexical differences remain coherent.
- `check_full_trace --batch 3` exposed sampler logical3 rows versus32 offsets,
  `trace_mixed_slots.log`. Source-only AutoDebug:
  `AUTODEBUG_sampler_batch.md`; focused raw/padded sampler probe prepared.
  This is required repair, not an accepted batch restriction.
- `OMP_NUM_THREADS=8 timeout 1200 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.run_full_capacity` is the full
  30-layer262143/262144 prompt and maximum cache capacity command; `capacity.log`.

## Maximum context and sampler contracts

- Full30-layer maximum-context run passes262143 (158.24s) and262144 (156.21s)
  logical prompts, with maximum B1 cache allocation and final-position decode
  at262143. `capacity.json`; durations are host-wall functional runs, not
  headline performance or device time. Context remains262144.
- Standalone rawB3 sampler probe reproduces Invalid subtile broadcast; actual
  `ttnn.pad` to32 logical rows fixes normal greedy and sampled B1/B3/B32.
  `sampler_raw_batch3.json`, `sampler_padded.json`,
  `sampler_selected_batches.json`. Every live row and all4ranks checked against
  known winners over changed-input replays. `--pad-batch --batch-sizes 3 32
  --force-mode split --modes greedy sampled --iterations 100` is the passing
  selected-path command to `tests.probe_full_sampler`.
- Untracked isolated B1 semantically greedy sampling: normal508.87us,
  force-argmax3302.49us. Force also has a B3 changed-input global-index mismatch;
  source diagnosis records uncertainty. Normal common split sampler selected.
- Original mixed-slot test after padding catches a new unsafe public-output
  allocation with trace tracking (`trace_mixed_slots_retry.log`). Persistent
  output storage is being verified; this is not waived.
- Initial independent review `stage_review_initial.md` returned more-work-needed.
  Added fresh entropy for omitted seeds, explicit rejection of unsupported
  sampled host-compatibility requests, and independent-cache batch controls.
  Runtime validation and rereview remain required.

## Native Watcher failure and recovery

- First `TT_METAL_WATCHER=5` attempt failed before model execution because
  inlined Ethernet firmware28464B exceeds26624B kernel buffer. Rerun adds
  `TT_METAL_WATCHER_NOINLINE=1`, matching accepted Stage05 settings.
- That rerun trips BRISC assert279 in native multicast all-gather writer on
  device0, worker0,0 (`trace_watcher_retry.log`). AutoTriage attempted after
  Watcher aborted the host; Inspector data no longer existed. Explicit
  `--dev=all --run=check_eth_status --run=check_arc` records healthy Ethernet
  and ARC. `watcher_failure/AUTODEBUG.md` traces the malformed scatter header.
- `timeout 180 tt-smi -r`, `timeout 60 tt-smi -ls --local`, TP4 open/close smoke
  pass after evidence preservation. No process needed killing. Artifacts:
  `watcher_failure/{reset.log,device_list_after.log,mesh_smoke.log}`.
- Focused component control reproduces: BF16 tiles2048B pass; uint32 tiles4096B
  fail the same assert279. Fabric payload4352/page4096 yields1chunk, below
  scatter minimum2, although that specialization sends via unicast writes.
  Native shared helper initializes the unused scatter state unconditionally.
  AutoFix proposes compile-time guards around both scatter-state initializers,
  preserving primary/alternate unicast initialization and all payload logic.
  Before evidence is in `bringup/artifacts/reference-fix/watcher_gather_before.*`.

## Fixed native assertion and final validation

- Both scatter initializers are now guarded by `if constexpr (use_scatter_write)`.
  Focused before/after controls and linear/ring traced checks are copied into
  `watcher_failure/`; the original reduced model Watcher regression passes:
  `TT_METAL_WATCHER=5 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_TRACE_ALLOC_TRACKING=1
  OMP_NUM_THREADS=8 timeout 240 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_trace --output
  models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/trace_watcher.json`
  (`trace_watcher_fixed.log`). This supersedes the failed Watcher attempts.
- `.github/scripts/copilot-build.sh --build-ttnn-tests` could not execute because
  Docker is unavailable (`native_build.log`). Full native build remains
  unverified; affected kernels JIT-compiled and passed real-device controls.
- Persistent public token output fixes the tracked allocation failure. Reduced
  mixed B3 and B32 independent-cache controls pass, followed by full30-layer B32
  endpoint-slot controls (`trace_full_batch32.json`).
- Trace-reuse checks pass repeated requests and changed prompt contents against
  fresh capture. Final metrics include request trace setup and expose reuse.
- Sampled-mode replay has device-owned seed advancement and token feedback;
  fresh entropy and explicit-seed repeatability pass. Explicit host sampling
  rejects unsupported sampled policies. `sampling_contract_retry2.log` records
  these checks. The extended logprob check exposed the common TP4 support gate;
  `sampling_contract_final.log` preserves the failure for follow-up.
- Reduced profile command (Watcher disabled): `OMP_NUM_THREADS=8 timeout 600
  python -m tracy -r -p -v --op-support-count 100000 --no-op-info-cache
  --disable-device-data-dump-to-files --disable-device-data-push-to-tracy
  -o models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/profile_terminal
  -n terminal_two_layers -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.profile_full_terminal`.
  `profile_terminal.log`; layers0/5 with real4096-token shapes and terminal.

## Final full-stack measurements

- `OMP_NUM_THREADS=8 timeout 600 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.run_full_readiness
  --qualitative --performance --output
  models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/readiness_final.json`
  passed (`readiness_final.log`). Final prefill top1/5/100=.96/1/1;
  traced decode=.94/1/1,100 positions each. Teacher-forcing decode49.84t/s/u
  is recorded separately on its161-input/100-position workload.
- Final exact4096-input/128-output/B1/C1 autoregressive measurement:
  TTFT2115.067ms, token-out49.2247t/s/u, trace_setup_ms0, reused_request_trace=true.
  127model and127sampling replays; zero steady token/position/page-table uploads,
  explicit synchronizations or full-logit readbacks. Output token readback127.
  `performance.json` supersedes the earlier provisional measurement.
- All six full-stack qualitative outputs were read again after the final run;
  coherent with the same lexical differences, no mechanical repetition or
  wrong-language drift. `qualitative_tt.json` is the current artifact.
- Reduced profiler succeeds; `profile_terminal/README.md`, `summary.json` and
  `decode_perf_report.csv` record complete per-device windows plus op breakdown.
  Whole reduced window3078.76us; sampling trace512.37us, not dominant. This is
  never substituted for an all-layer device-time or roofline measurement.
- `python -m pytest -q models/common/readiness_check/test_generate.py`:
  10passed (`reference_host_tests_final.log`). Stage source/docs pre-commit
  hooks pass (`precommit_final.log`). Native build limitation remains disclosed.
- Standard comparison command: `OMP_NUM_THREADS=8 timeout 600 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.run_full_autoregressive`;
  `autoregressive.log`, artifacts under `autoregressive/`.
- AutoDebug established that both common logprob paths explicitly exclude TP4.
  Optional logprob requests now raise before request mutation. Penalty testing
  remains independent. See `AUTODEBUG_logprobs.md`.
- Explicit host sampling now also uses host argmax for the prefill token. The
  reduced softcap-tie oracle checks both initial choices are maxima instead of
  assuming identical tie-breaking (`sampling_contract_validated.log` preserves
  the obsolete exact-equality failure; subsequent validation is recorded below).

- Standard `run_autoregressive` completed successfully. HF BF16 and TT each
  emit128 tokens from the same28-token chat prompt. Actual text was read;
  coherent English introductions to light/color, no repetition or wrong-language
  drift. First lexical difference at generated index3, both truncate at the
  imposed token cap. `qualitative_verdict.md` records the assessment.
- `python models/common/readiness_check/check_degenerate_output.py --hf-model
  google/gemma-4-26B-A4B-it --missing-artifacts critical --scope autoregressive
  --json models/autoports/google_gemma_4_26b_a4b_it/doc/full_model/degeneracy.json`
  exits0 with no findings (`degeneracy.log`).

- Final `TT_METAL_TRACE_ALLOC_TRACKING=1 OMP_NUM_THREADS=8 timeout 240 python -m
  models.autoports.google_gemma_4_26b_a4b_it.tests.check_full_sampling` exits0
  (`sampling_contract_pass.log`). The refreshed `sampling_contract.json` proves
  penalties, seeded sampling, no-host fallback, greedy prefill/decode maxima,
  host compatibility, no-mutation rejection of optional TP4 logprob requests,
  and the single-token callback. This supersedes both extended test failures.
- All runtime gates are now passing. Only independent review/checkpoint remains.

- Final pre-commit over all stage-owned source and compact artifacts passes.
  Hooks normalized artifact trailing whitespace/newlines only; generated token
  IDs and measurements are unchanged. `precommit_final.log`.

## Completion and checkpoint

Independent fresh xhigh review returns **clean-pass** with no required work:
`stage_review_final.md`. The previous more-work-needed findings are fixed and
rereviewed. Full-model Stage06 is complete. No vLLM integration was started.

The full native build remains unverified because Docker is unavailable; this
is disclosed alongside affected-kernel JIT/Watcher validation. Optional TP4
logprob requests are rejected, never silently handled on the host. Full-model
device-time/roofline metrics remain unknown; no reduced-profile substitution.

Stage implementation/evidence checkpoint: `3d5a5654b5a514ce35d2ee01013ff02700402ac5` in repo
`/workspace/tt-metal`, branch `gemma-4-26b-a4b-it`. All commit-time hooks pass.
Only tt-metal was touched. No push was performed. This provenance update is a
separate local documentation commit; its SHA is recorded in the final local
telemetry packet alongside the implementation checkpoint.
