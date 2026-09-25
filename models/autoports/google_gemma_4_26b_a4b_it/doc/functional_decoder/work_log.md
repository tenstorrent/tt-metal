# Functional decoder work log

Stage 01 only; model `google/gemma-4-26B-A4B-it`.
Initial checkout `a3a9fb4229a045ad9361b4e39ad854b491346ea9`; initially clean.
HF revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.

## Startup

- `timeout 60 tt-smi -ls --local`: exit 0; four Blackhole p300c devices.
- `timeout 60 python` calling `ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0)` and close: `MESH_SMOKE_OK`.
- Installed plugin environment.py validated selected tt-autodebug 0.1.6 package.
- `AutoConfig.from_pretrained('google/gemma-4-26B-A4B-it')` succeeds.
- Config: hidden 2816, 30 layers, sliding window 1024, context 262144,
  128 experts, top 8, shared MLP 2112, expert intermediate 704.
  Sliding: 16 Q heads, 8 KV heads, dimension 256. Full: 16 Q heads,
  2 KV heads, dimension 512, tied K/V projection, partial RoPE .25.
- HF source inspected: installed transformers/models/gemma4/modeling_gemma4.py,
  attention, router, experts and decoder (lines 1177–1458).
- Snapshot download requested only safetensors at the pinned revision. The two
  monolithic shards contain both selected layers; no full causal LM is loaded.

## Initial checks

Run prefix: `python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder`.

- `--output .../synthetic_sliding_32.json`: real config, provisional synthetic
  weights, paged sliding prefill, length 32, PCC 0.9713391913952353; FAIL.
- `--decode --output .../synthetic_trace_32.json`: decode warmup rejected a
  cache-position vector of length 32 for logical batch 1. Fixed harness to
  supply one int32 cache position; RoPE lookup remains padded independently.
- `--decode --output .../synthetic_trace_32_retry.json`: prefill still fails;
  traced decode PCC 0.997149425797613, identical repeated replay. This is a
  diagnostic synthetic result, not stage completion or real-weight evidence.
- Fresh AutoDebug subagent investigating prefill divergence before AutoFix.

No performance measurement, full-context validation, watcher acceptance,
independent clean-pass, or checkpoint commit yet.

## Real weights and AutoFix localization

- Snapshot download completed (two safetensors shards, 51,611,872,412 bytes of
  tensor data per index). Standard HF shard download was used; only layer 0/5
  tensors are materialized by the runner. This downloaded more data than an
  HTTP range-based layer extractor would require; it is not a claim that a
  full-model download was necessary.
- `--real --decode --output .../real_sliding_32.json`: PASS; prefill
  0.9988936448679125, traced decode 0.9994597686704707, repeat identical.
- `--real --decode --layer 5 --length 33 --output .../real_full_33.json`:
  prefill PASS 0.9990409416886573; traced decode FAIL 0.9832279444343548.
- Same command plus `--diagnostic`, output `real_full_33_diagnostic.json`:
  attention PCC .9995312, shared post-norm .9999387, routed post-norm .9005982.
- AutoFix has exclusive serialized device access for route/expert isolation.
  Oracle original-HF routing restores final PCC .9998725; expert HiFi4 did not
  fix the failure and is not a selected change.
- CPU-only `tests.router_diagnostic` reproduced .972244 output PCC by rounding
  only synthetic full-softmax probabilities to BF16 before top-k. 23/32 expert
  sets changed. Real-weight statistics are in `weight_stats.json` (43 tensors).
  The primary harness now generates synthetic weights from these statistics;
  the diagnostic script preserves the original provisional initialization.
- `python -m pip install tt-perf-report` failed because pip is absent.
  `uv pip install --python /workspace/tt-metal/python_env/bin/python tt-perf-report`
  succeeded, version 1.3.0. `tt-perf-report --help` checked.
- `python -m black --target-version py310` on the two primary runners and
  `python -m compileall -q models/autoports/google_gemma_4_26b_a4b_it/tests` pass.

## Wider contract checks

- `tests.request_reuse --layer 0 --output .../reuse_sliding.json` (same module
  prefix as above): one loaded instance, requests 31/32/33/1023/1024/1025/2049/33,
  random new physical-page ownership each time, one reused decode trace.
  All prefill PCCs pass; decode at 31, 1025 and 2049 fails. This remains work.
- `tests.batched --batch 2 --layer 0 --output .../batch2_sliding.json`:
  both prefill slots pass; slot 0 decode fails .9885266, slot 1 passes .9991798.
  Both repeated trace output equality and runtime fallback guard pass.
- `AUTODEBUG_boundaries.md` records competing hypotheses for those failures.
  AutoFix continues; prior single-case repair is not universal acceptance.
- `tests.long_context --layer {0,5} --length 262144 --reference-only --output
  .../reference_{sliding,full}_262144.pt` completed CPU reference preparation.
  All K/V positions are present, with 291 exact HF query rows sampled for
  comparison. Binary references stay local/ignored; these are not TT results.
- Added per-slot fixed-batch decode orchestration and a device-only
  prefix-continuation path using tensor positions allocated during setup.
  Continuation correctness has not yet been run.

- `tests.long_context --layer 0 --length 262144 --reference-file
  .../reference_sliding_262144.pt --output .../long_sliding_262144.json`
  passed full TT prefill, sampled-query PCC .99832883 (291 rows, all K/V),
  and traced decode at positions 262143/262142/262143 (minimum .99926937).
  This precedes the later QKV HiFi4 and exact SDPA policy; rerun required.
- CPU reference preparation for both kinds at 262143 also completed.
- Source audit found imported short sliding-tail helpers create host zeros.
  Fresh single chunks now omit unnecessary chunk-tail bookkeeping; short
  final sliding chunks pad physically to one window and use tensor offsets.
  Runtime guards now reject native creation entrypoints as well as Torch and
  tensor conversion calls. These orchestration changes await hardware checks.

- `RUNTIME_AUDIT.md`: independent source-only audit found a missed 993–1023
  row sliding tail branch, corrected via valid-row rather than padded-row
  predicate. Setup now validates 128-token chunk alignment and chunk divisibility
  of context. Cache and RoPE physical-capacity requirements are documented.
- Broader numerical validation is still failing. Sliding final-policy batch32
  fails five decode slots; full batch32 fails four. Every prefill slot passes,
  runtime guards and repeated trace outputs pass. See batch32 result JSONs.
  HiFi2 versus HiFi4 full-attention QKV exchanges which inputs fail; neither
  is a universal repair. AutoFix is testing FP32 norm/head/storage boundaries.
- Added offline pytest regression entrypoint with pinned `tests/config.json`
  and recorded statistics; it has not yet been executed on devices.

- Reproducible CPU controls: `tests.hf_precision_controls --control all`
  (see `HF_PRECISION_CONTROLS.md` for full command). HF BF16 itself changes
  some discrete routes, but fails different slots from most TT failures.
  Rounding only cached K/V to BF16 passes all 64 decode slots with unchanged
  routes (minimum .99999275 sliding, .99999956 full), refuting cache format
  alone as the cause. These are CPU controls, not TT acceptance evidence.
- Combined FP32 diagnostic computation is under test. A closure bug in an
  intermediate `probe_policy` residual experiment rebound input norm to the
  post-attention norm; its catastrophic artifact is invalid and superseded
  by a corrected retry. No production fix or gate waiver used that artifact.

- Fresh xhigh source investigation `AUTODEBUG_fp32.md` identified native RoPE
  BF16 intermediates/config handling, and TF32 unpack in Float32 matmul and
  fused RMSNorm. Controlled diagnostics tested SFPU compositions and split
  operands rather than assuming a dtype label proved arithmetic precision.
- AutoFix found that Float32 input to paged cache update did not match an
  explicit BF16 cast of the same K/V. The explicit on-device BF16 cast before
  update makes the combined diagnostic pass all 32 sliding slots (minimum
  decode PCC .99968503), with no oracle tensors. Full-kind validation and
  minimal local integration remain pending; this is not a stage pass.
- Statistical synthetic weights now reproduce the recorded checkpoint dtype
  as well as mean/std: sampled values round to BF16 before FP32 HF compute.
  All recorded source weights are BF16; real-weight loading is unchanged.

- Direct `tests.run_decoder --layer 0 --length 4096 --real --decode --steps 128`
  (`headline_sliding_accuracy.json/.log`) passes prefill .99852634 but fails
  minimum decode .99494137. Subsequent AutoFix diagnostic localizes position4110.
- The same headline command with `--layer 5` passes prefill .99944062 and all
  128 traced decode steps, minimum .99977066 (`headline_full_accuracy.json/.log`).
  Per-position PCC records are now preserved by the runner.
- Root source fixes: public forwards reject non-BF16/non-32-token caches before
  mutation; physical chunks cap at16384 (logical context is unchanged).
  Cache embedding row arithmetic is now UInt32 to avoid Float32 aliasing.
- `python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.cache_indices
  --output .../cache_indices.json` passed exact row addresses through physical
  page262144 for both KV-head shapes; guard clean. No enormous cache allocation
  was required for this address-arithmetic regression.
- Hardware lease returned to AutoFix after these commands; no root device
  process remains active while the position4110 investigation runs.

- Final AutoFix integration uses decode-only SFPU QKV row dots in groups of 256;
  both 128-step headline, both batch-32, and both nine-request reuse tests pass.
  Artifacts are `*_exact_qkv_integrated.json/.log`; full control details are
  in `AUTOFIX_decode.md`. BF16 tables/cache and PCC>=0.995 are unchanged.
- `python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.prefix_continuation
  --layer {0,5} --output .../continuation_{sliding,full}.json` passes PCC
  .99914565/.99967083, partial-page prefix preservation and other-slot isolation.
- `python_env/bin/python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py
  --basetemp models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/pytest_tmp`
  passes 2/2; `synthetic_pytest.log` records the run, temporary resultJSONs retained.
- Root acquired hardware lease after AutoFix finished; final full-context reruns
  use `long_context --layer {0,5} --length {262144,262143}` with corresponding
  saved `reference_{sliding,full}_{length}.pt` and `long_*_final.json/.log`.

- Final sliding 262144 passes prefill .99889163 and decode .99979693 minimum;
  final full 262144 passes prefill .99627766 but fails decode .101604/.114831.
  Exact `long_*_262144_final.json/.log` retained. AutoFix resumed for full-context
  decode; no capability reduction or acceptance waiver selected.
- Sliding profile command: `python_env/bin/python -m tracy -r -p -v
  --op-support-count 150000 -o models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/tracy/sliding/raw
  -n headline -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder
  --layer 0 --length 4096 --real --decode --steps 128 --profile
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/profile_sliding.json`.
  Increased profiler storage retains complete 128 replays; watcher remains disabled.

- First sliding profile executes all 128 steps and passes first/last PCC, then
  aborts during close/collection. No deviceCSV is produced; `profile_sliding.json`
  is correctness-only and supplies no latency. `profile_sliding.log` preserves
  abort and missing-CSV postprocessing error. AutoFix source investigation started.
- After aborted profiling, `timeout 60 tt-smi -ls --local` exits 0 and bounded
  `open_mesh_device(MeshShape(1,1),trace_region_size=0); close_mesh_device` exits 0.
  `post_profile_device_list.log` and `post_profile_mesh_smoke.log` retained;
  no reset or lock deletion was needed.

- Final sliding 262143 passes sampled 291-row prefill PCC .99889113 and traced
  decode PCC .99985036 minimum. This completes sliding maximum/near-maximum
  context coverage without reduction (`long_sliding_262143_final.json`).
- Profiler source investigation `AUTOFIX_profiler.md` finds UInt32 host-buffer
  size overflow for 150000 supported programs; retry 100000 remains pending.
  Saved host trace has 560 programs per trace and 130 executions (2 correctness,
  128 measured), 78391 total programs; this is capacity evidence, not device latency.

- Optional combined full262144 watcher run was cancelled with SIGINT after
  watcher samples proved steady forward progress (~70 program IDs/s). It
  closed cleanly at19:57:29; no reset or SIGTERM was needed. This partial run
  is not passing context/watcher evidence (`long_full_262144_fixed_watcher.log`).
  Full262144 and262143 correctness remain required and are rerun without
  watcher; separate watcher10 tests use the full headline4096/128 workload.
  No context capacity reduction or gate waiver follows from watcher cost.
- `profile_sliding_retry.log` exits0 with safe profiler capacity100000.
  The device CSV has78391 records:560 programs ×130 trace replays plus5591
  nontrace programs. The measured window contains exactly128 replays.
- `tests.summarize_perf .../tracy/sliding/ops.csv --layer-type sliding_attention
  --output .../tracy/sliding/whole_layer.json` validates complete firmware
  start/end windows, yielding4996258.54us prefill and9811.40us mean decode.
  All four tt-perf-report calls (prefill/decode × CSV/human text) exit0;
  files use corresponding PERF_PREFILL/PERF_DECODE start/end signposts,
  `--active-experts8 --no-advice --no-color`, with `--no-summary` for text.
- Profiler cache flags are stale serializer metadata; see AUTOFIX_profiler.md
  and profiler_cache_metadata_probe.json. Optional `--verify-program-cache`
  now forbids misses during warmed prefill/capture; runtime control pending.

- Full262144 correction rerun exits0: `long_full_262144_fixed.json`, sampled
  prefill PCC .9962776617, maximum-context traced decode minimum .9998845259.
  Full advertised capability is retained for both layer kinds.
- Fresh profiler collection disables op-info caching, retaining invocation
  shapes rather than program-hash-cached shapes. Safe capacity remains100000.
  `--disable-device-data-dump-to-files --disable-device-data-push-to-tracy`
  suppress optional bulk zone dumps, while C++ device analysis remains enabled.
  Fresh sliding runtime passed `--verify-program-cache`: warmed prefill and
  capture reject cache misses, with195 entries after initial prefill.
- Earlier sliding traffic estimate is withdrawn: cached op shapes overstated
  decode operands, and the initial estimator counted the whole pool for cache
  updates. `whole_layer_stale_metadata.json` explicitly marks it invalid.
  Final estimator uses uncached shapes, actual sparse nnz, one-page cache RMW,
  and retains full-pool traffic for actual layout conversions before embedding.

- Fresh sliding collection/export and all five report commands exit0.
  `tracy/sliding/provenance.json` identifies the raw CSV/trace; report commands
  are in `report_commands.json`. Complete-layer times are4998536.8156us
  prefill and9811.0668us mean traced decode (128 windows); useful FLOPs
  882489950208, estimated decode DRAM1211068608bytes. Rooflines are
  .1064270% useful FLOPs and24.1091863% DRAM, using one theoretical ASIC.
  The independent reviewer re-derived record counts, cache hits, shapes,
  replay-window times and corrected traffic from the fresh source CSV.

- Full262143 correction rerun exits0 (`long_full_262143_fixed.json/.log`):
  sampled prefill .9962239651, traced decode .9998608009 minimum. All four
  maximum/near-maximum context tests now pass; context_contract.json retains
 262144 with no capability reduction.
- Fresh full profiler device execution passes PCC, repeated equality and
  program-cache miss guard (192 entries after initial prefill), then closes
  devices at20:20:01. CPU export overlaps only the subsequent watcher run;
  hardware execution remains serialized. Exact final profiler/watcher recipes
  are in `COMMANDS.md`; console/result stems are `profile_{sliding,full}_final`
  and `watcher_{sliding,full}_final`.

- Full fresh profile/export and all five report commands exit0. Final whole-layer
  times are5009442.5444us prefill and10894.3714us mean decode (128 windows,
 502 native operations each). Useful FLOPs1215408635904, estimated decode
  DRAM1508803108bytes; one-ASIC rooflines .1462574% /27.0495741%.
  Independent review reproduced these from the CSV, including complete cache
  hits, invocation shapes, replay counts and traffic. CSV SHA256 values are in
  each `tracy/<kind>/provenance.json`.
- Sliding watcher10 headline exits0 at20:22:37, no disabled features or watcher
  errors. Prefill PCC .99852634 and minimum of128 traced decode PCC .99904322;
  repeated outputs identical, runtime guards and cache-miss guard pass.

- Full watcher10 headline exits0 at20:24:49, no disabled features or watcher
  errors. Prefill PCC .99944062 and minimum of128 distinct traced positions
  .99977388; repeated outputs identical, both runtime guards and cache-miss
  guard pass. `watcher_summary.json` records13/10 completed watcher dumps and
  minimum free stack1328/1252bytes (sliding/full). Each JSON contains129 PCC
  comparisons because the initial position is checked once again; there are
  exactly128 distinct successive positions4096 through4223.
- Final statistical-weight regression after the page-table fix:
  `python_env/bin/python -m pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py
  --basetemp models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/pytest_final_tmp`
  exits0; `synthetic_pytest_final.log` and both resultJSONs retained. All device
  jobs are finished and devices closed. No subsequent hardware work is pending.

- Final pre-commit invocation on all stage-owned files exits0;
  `precommit_final.log`. Hooks normalized trailing spaces in rendered perf
  tables without changing data. Python/docs-only change: no C++ build needed.
  Final synthetic result is2 passed in12.13s (three dependency deprecation warnings: two SWIG types and Pydantic class config).

## Final acceptance and checkpoint

Fresh independent xhigh `$stage-review` returned **clean-pass**, with no required
work, in `STAGE_REVIEW.md`. The reviewer inspected final source, raw watcher
logs, fresh profiler CSVs, all required correctness evidence and the supplied
telemetry packet, and independently reproduced whole-layer times and traffic.
The stage retains advertised context262144 and PCC>=0.995 without a waiver.
Only the functional-decoder stage was implemented. No later stage or push ran.

The mandatory local packet is
`/workspace/tt-metal/bringup/artifacts/multigoal-runs/20260925T171711Z/telemetry/packets/c7de8060-b30e-40b0-ab7e-a6ae3c7cc666.json`,
created from the supplied template with exact identifiers. It contains compact
measurements/references only. Raw captures, weights and full tables stay local;
compact evidence and replay/stacked CSV summaries are checkpointed.

Local stage checkpoint (after independent clean-pass):

| Repository | Branch | Commit | Validation |
| --- | --- | --- | --- |
| `/workspace/tt-metal` | `gemma-4-26b-a4b-it` | `de9abb0d8c3c3dcc0c8ad04a5e66896b190cbb71` | Reviewed source and compact evidence; pre-commit passed before and during commit |

The subsequent documentation-only commit records this checkpoint SHA. Its own
SHA is recorded in the external telemetry packet so the committed work log does
not require a self-referential hash. Both commits are local; nothing was pushed.
The working tree was clean immediately after the stage checkpoint.
