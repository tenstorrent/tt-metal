# Functional decoder work log

Stage 1 only, ornith-ai/Ornith-1.5-9B, revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
Base tt-metal `9e8a4511`; worktree initially clean. This is a chronological record: early pending/failing entries are superseded by the later validation and closure sections. Current scope and metrics are in README.md. No remote writes or pushes.

## Setup

Enabled installed plugins verified from `/home/ttuser/ornith-9b/setup-evidence/plugins.json` and active Codex config: model-bringup 0.1.23, autodebug 0.1.8.
Environment loaded by lane launcher; package environment.py verified. Existing multigoal run and persistent logs retained. No remote writes or push.

AutoConfig loaded target text config: width 4096, MLP 12288, 32 layers, types linear/linear/linear/full repeated; full attention 16 Q / 4 KV heads, head width 256, partial RoPE width 64, theta 10000000, output sigmoid gate; linear attention 16 key heads / 32 value heads of width 128, convolution width 4, FP32 recurrent state. HF advertises 262144 positions. No capability reduction authorized or claimed.

Inspected installed `transformers/models/qwen3_5/modeling_qwen3_5.py`: offset RMS norms (except gated recurrent norm), projection layout, convolution/cache continuation, recurrent/chunk reference, partial RoPE, attention gate and residual order. Read TT gated-delta, paged cache and chunked SDPA sources and existing TT-transformers attention.

`fuser /dev/tenstorrent/*` found no holders. `timeout 60 tt-smi -ls --local` found four p300c Blackhole chips. Single chip selected.
`TT_VISIBLE_DEVICES=0 timeout 120 python ...` failed before device open: isolated half-P300 classified CUSTOM and required mesh graph descriptor (`mesh_smoke.log`).
Direct `open_mesh_device(MeshShape(1,1), physical_device_ids=[0], trace_region_size=0)` with topology discovery retained passed ones/add/readback and close (`mesh_smoke_physical_id.log`). No reset or kill used; this is a topology selection issue, not model failure.

## Implementation and validation in progress

`tt/functional_decoder.py`: model-specific weight-loading boundary, both layer kinds, paged full-attention cache, linear recurrent and convolution state, chunked prefill, device-only decode.
`tests/run_decoder.py`: real target config and real layer shards, CPU HF layer reference, permuted page table, prefill and follow-on decode. Weight stats recorded per layer. CPU/host conversions occur at setup and comparison boundaries.
`pre-commit run --files .../tt/functional_decoder.py .../tests/run_decoder.py`: passed (`hooks.log`).
`timeout 600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder --layer 0 --length 32`: running/initial smoke; see `linear_smoke.log`.

Remaining work: repair initial smoke as necessary, arbitrary continuation starts/full-attention tails, batched inputs, all coverage/long context, traces and allocation safety, determinism, watcher, warmed profiler evidence, runtime audit, final reports, independent review, local commits.

The setup cache contained the downloaded 19 GB Xet blobs but broken legacy snapshot links after setup cleanup. Inspected the pinned tree metadata, verified each source size, and repaired only dangling snapshot symlinks to its exact Xet blob. No weights were downloaded or modified. Initial `linear_smoke.log` records the missing legacy link; r2/r3 record API name and weight-selection setup corrections. On this checkout the compute-config class is `WormholeComputeKernelConfig` also for Blackhole. Projection selection now covers all 2D layer matrices including `in_proj_z/a/b/qkv`.

## Initial passing measurements

- `linear_smoke_r5.log`: real layer 0, S=32, prefill PCC 0.99994266, eager decode 0.99994802.
- `full_smoke_r3.log`: real layer 3, S=32, prefill PCC 0.99983531, eager decode 0.99980688. Embedding indices require UINT32 while paged cache update requires INT32; conversion is now a device-side typecast.
- `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 600 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder --layer {0,3} --length 33 --trace` (serialized): `linear_trace_33.log` and `full_trace_33.log`, each 3 deterministic traced replays. Linear decode PCC 0.99990463; full decode PCC 0.99977303.
- `timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.validate --layer 0 --lengths 31,32,33,1023,1024,1025,2051,17 --resume --batch 32 --output .../linear_validation.json` passed. Full layer 3 repeated with lengths `1,31,32,33,1023,1024,1025,2051,17` (`full_validation.json`). These initial batch results are eager; the final harness additionally traces batch decode and that affected gate will be rerun.
- Runtime guard uses TorchDispatchMode and patches TTNN host input/readback/refresh entrypoints inside measured forwards. Both complete batteries passed the guard. Python chunk/shape planning stays host-side; all tensor computation is on device.
- Resume comparisons subtract the residual and test each row at PCC >=0.995, starts 32,33,1024,1025,2050. Empty-history negative controls fail that same gate for both kinds.
- Trace lifetime: all tested prefill op shapes compile before first capture. One decode trace is retained while requests change lengths, token activations, page maps and positions. Three repeats restore the same request state; outputs are bit-equal. The only acknowledged allocation is that trace's output, read before its next replay; it has no other live trace consumer. No allocation scopes or skipped program-cache tracking.
- Installed `pip` and `tt-perf-report` in the existing active tt-metal environment with `uv pip install --python .../python_env/bin/python pip tt-perf-report` (`perf_install.log`).
- Long probe command: `timeout 7200 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.long_context --layer 0 --output .../linear_long.json`. Full advertised context retained; linear HF reference processes every token. Full-attention long reference projects K/V for all tokens and uses the actual HF decoder forward on selected query windows against that complete history; sampling is reported explicitly, never presented as whole-output PCC.

## Final correctness and context gates

The implementation is now functional; initial entries above describe historical work, not pending failures. Final reports supersede the earlier smoke and eager-batch probes.

All commands below run from `/home/ttuser/ornith-9b/tt-metal` in `python_env`, one hardware job at a time on physical chip 0. Shell variable abbreviations for reproduction:

```bash
root=models/autoports/ornith_ai_ornith_1_5_9b
art="$root/doc/functional_decoder"
module=models.autoports.ornith_ai_ornith_1_5_9b.tests
```

```bash
# Serialized for layer 0 and then layer 3; output names linear_final/full_final.
timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m "$module.validate" --layer 0 \
  --lengths 1,31,32,33,1023,1024,1025,2051,17 --resume --batch 32 \
  --output "$art/linear_final.json" > "$art/linear_final.log" 2>&1
# Long context: repeat for layer 3 with full_long_fixed as the output basename.
timeout 7200 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m "$module.long_context" --layer 0 \
  --output "$art/linear_long.json" > "$art/linear_long.log" 2>&1
# Offline synthetic-weight CI and allocation-tracker positive control.
timeout 1800 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m pytest "$root/tests/test_functional_decoder.py" -q -s --timeout 1500 \
  > "$art/synthetic_pytest.log" 2>&1
timeout 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python -m "$module.validate" --layer 0 --lengths 32 --positive-control \
  --output "$art/trace_alloc_tracking/positive_control.json" \
  > "$art/trace_alloc_tracking/positive_control.log" 2>&1
python "$TT_MODEL_BRINGUP_ROOT/scripts/check_context_contract.py" \
  --model-dir "$root" --hf-model ornith-ai/Ornith-1.5-9B --require-contract \
  > "$art/context_gate.log" 2>&1
```

Both final batteries passed with real weights, exact HF configuration, batch 32 traced decode, physical paged-KV reconstruction, per-row continuation contribution comparisons and failing dropped-history controls. Minimum positive output PCC: linear 0.99966466, full 0.99966115. Minimum positive continuation row contribution PCC: linear 0.99897540, full 0.99961668. Each single-row trace has one capture, zero request releases and 27 replays; each batch trace has three identical repeated outputs. Runtime Torch/host-boundary guards passed. `synthetic_pytest.log` reports both parameterized tests passing. The positive control raised the expected one-live-buffer error; it is a successful detection test.

Long tests passed at 262139,262143,262144 for both kinds, with three deterministic traced decodes at final position 262143. No capability reduction. Long sampled prefill minima: linear 0.99951565, full 0.99820769; decode PCC: 0.99949205, 0.99977314. Full TTNN outputs were computed; numerical reference compares sampled windows against complete history, as specified in README and context_contract.json. The context gate passed.

## Full-attention compilation investigation

The original full-context run was silent while compiling. Preserved `triage/full_long.txt`, `triage/full_long_summary.txt`, and `triage/full_long_host_stack.log` before any intervention. All triage checks passed; host stack waited in program compilation. The original job finished naturally with a passing result. No process termination, reset, or lock cleanup was used.

Applied AutoTriage, AutoDebug and AutoFix. AutoDebug's bubblewrap preflight failed because unprivileged namespaces are disabled; the read-only child completed with the supported `AUTODEBUG_SKIP_CHILD_SANDBOX=1` override in this unrestricted, user-authorized environment. Exact launch output is in `triage/autodebug_launch_retry.log`; independent report is `AUTODEBUG.md`. A separate source-only slice-cache hypothesis audit confirmed which offsets affect binary geometry.

The correction bounds each fill-table slice to its actual chunk pages and uses the existing device-valued SDPA start API. A first focused probe revealed flexible SDPA's 32-byte row requirement; padding only the attention table to eight INT32 entries resolves it without changing real cache capacity. `full_fix_probe_r2.json`, `full_long_fixed.json`, and `full_final.json` pass on the corrected implementation. See AUTOFIX.md for causal evidence. No measured speedup factor is claimed. Linear long evidence retains the old source hash because these changes touch only full attention; the final linear battery also passes on the corrected source.

## Watcher and profiler commands

`tests/run_evidence.sh` reproduces the final short/medium battery, positive control, offline pytest, watcher and profiler sequence. Original queue `evidence_queue.log` stopped at the watcher compile failure, after earlier gates passed. Resume script/log `resume_evidence.sh` and `resume_evidence.log` retain the exact affected-only restart.

The first watcher command used `TT_METAL_WATCHER=10`. GDN prep compilation exceeded the 70688-byte kernel-config region by 1168 bytes (`linear_watcher_inline_failed.log`). It did not execute an invalid kernel; device shutdown completed. Retried with `TT_METAL_WATCHER_NOINLINE=1`, which changes code inlining while retaining checks. No watcher features were disabled.

```bash
# Repeat serially for layer 3 / kind full.
kind=linear
timeout 1800 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 \
  TT_METAL_LOGS_PATH="$PWD/$art/watcher/$kind" \
  python -m "$module.validate" --layer 0 \
  --lengths 1,31,32,33,1023,1024,1025,2051,17 --resume --batch 32 \
  --output "$art/${kind}_watcher.json" > "$art/${kind}_watcher.log" 2>&1
timeout 1800 env TRACY_NO_WEB_SERVER=1 \
  python -m tracy -r -p -v -o "$art/tracy/$kind" -m "$module.validate" \
  --layer 0 --lengths 1024 --profile --output "$art/${kind}_profile.json" \
  > "$art/${kind}_profile.log" 2>&1
python "$root/tests/perf_reports.py"
```

Profiler windows synchronize after warmup, signpost start, execute the warmed prefill or captured decode replay, synchronize, then signpost end. Prefill state and decode snapshots are restored outside the window. Profiler and watcher run separately. Performance reports use filtered `Device Time` in microseconds; human tables, CSV and commands are retained per layer kind under `tracy/`.

## Anomaly classification

- Broken HF snapshot links: setup-only; exact pinned Xet blob/size mapping repaired, real weights pass.
- P300 CUSTOM topology under TT_VISIBLE_DEVICES: setup-only; retain topology and select physical device 0, mesh smoke and all tests pass.
- Initial projection/config API and INT32/UINT32 setup errors: corrected; real-weight smoke, final tests and trace replay pass.
- HF warning about missing fused causal-convolution/GDN packages: CPU reference uses its official Torch implementation. TTNN runs its device kernels; runtime guards pass.
- Unknown B850M-C motherboard and subset-of-MMIO warnings: UMD uses bus_id for tray_id and notes potential remote-access speed; local chip 0 opens/closes and computes correctly. This stage uses one chip, no remote tensor work.
- Long-context silence: controlled host compilation wait; original run passed, caller specialization repaired and revalidated; no hardware hang claim remains.
- Flexible SDPA table alignment failure: fixed with on-device attention-only table padding; real 65-page table and full 8192-page context revalidated.
- Watcher inline kernel-size failure: retry with supported noinline instrumentation; result recorded below.
- Intentional negative-control and trace-allocation errors: expected detection results, not failing model gates.

Independent stage review, final watcher/performance results and local checkpoint SHAs are appended after collection. No push is permitted or performed.

## GDN watcher repair and affected-gate reruns

The noinline-only watcher retry also exceeded kernel-text capacity (1152 bytes). AutoFix source inspection plus an independent hypothesis agent traced it to GDN's64-token Horner prep kernel, documented in `chunk_gdn_math.hpp` as close to the config-buffer limit. Switched the model's internal GDN prefill chunk to32 and its eye/tril/ones constants to32x32. On Blackhole this selects the supported FP32 triangular solve. Public prefill chunks stay1024; the full-attention path and decode math are unchanged. Both ordinary and watcher execution use the same configuration. No checks were weakened and no shared code changed.

`gdn32_watcher_probe.json/log` passed real-weight lengths1,33,1025, continuation, negative controls and traced batch2 with watcher enabled. `revalidate_gdn32.sh/log` records the serial affected-gate queue. The final linear battery and full262144 context have passed again; original results are preserved as `linear_final_gdn64.*` and `linear_long_gdn64.*`. The revised final linear output PCC minimum is0.99968177; positive resume row contribution minimum is0.99890006. These supersede the earlier64-token numbers. Current numbers and source hashes are in README and JSON. Positive-control detection also passed again. The full-attention-only results remain applicable because this repair does not change that path; full synthetic, watcher and profile runs additionally exercise the latest source.

## Final watcher and performance evidence

Both watcher batteries passed at lengths 1,31,32,33,1023,1024,1025,2051,17 with prefix continuation and batch 32 traced decode. `watcher_audit.json` records source hashes and clean log scans; logs include normal attach/dump/detach and no sanitize/overflow/fault/assert failures. The supported noinline flag remains enabled, with no disabled watcher checks. GDN32 full-context revalidation passed: minimum sampled prefill PCC 0.99866915 and final-position traced decode PCC 0.99865657; full-attention minima remain 0.99820769 and 0.99977314. The acceptance bar remains 0.995.

Original profiler correctness passed but export detected a 16-bit source-ID collision between GDN `rd_kick` and `pp_kdec`. This was diagnosed from the source strings and the exact hash function, not suppressed. `tests/run_profiles.sh` builds a model-local relative symlink overlay for the unchanged reader and four sibling headers, verifies byte equality, and runs from the model directory with explicit `TT_METAL_KERNEL_PATH`, `TT_METAL_HOME`, `PYTHONPATH` and a fresh profiler directory. `profiler_collision_resolution.json` records symlink targets, source hashes and collision-free emitted metadata (46 zones for linear, 31 for full). This only changes instrumentation source identity.

A reader-only overlay attempt exposed missing quoted sibling headers. Tracy returned zero and wrote partial CSVs despite that target compile failure. The final runner therefore also requires target JSON, VALIDATION_PASS and no exception/fatal markers. The failed attempt is preserved under `tracy/linear_overlay_missing_headers/`; it contributes no reported timing. The original collision capture is under `tracy/linear_hash_collision/`, also excluded. `profile_queue.log` is historical wrapper-only status; `profile_retry_queue.log`, successful target logs and nonempty signpost windows establish the accepted captures.

Final capture commands: `bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_profiles.sh` (full) and `bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_profiles.sh 0` (affected-only linear retry). Each child is bounded by timeout1800 and serializes hardware use. `python models/autoports/ornith_ai_ornith_1_5_9b/tests/perf_reports.py` renders tables/CSV/provenance. The final `tests/run_evidence.sh` invokes the corrected profile runner for reproduction.

Warmed device-kernel sums, from filtered `Device Time [μs]` divided by1000: linear prefill13.239976 ms, traced decode1.654715 ms; full prefill8.194264 ms, traced decode1.840723 ms. Summed op gaps are0.103700,0.084252,0.053003,0.067933 ms respectively. Traced decode host walls are1.819328 and2.011775 ms. All measurements are serial chip0, batch1, prefill1024, decode position1024. Every selected decode row carries a trace ID; exact rows/counts/hashes are in `performance_summary.json`. `functional_decoder.md` states the single-warmed-sample limitation and distinguishes device sums from host wall latency.

All required hardware/evidence runs are now finished. `hooks_pre_review.log` and later final hook logs record formatting/lint checks. Independent stage review follows; no later bringup stage has begun.


## Independent review and checkpoint preparation

`stage_review_round1.md` returns **clean-pass** from a fresh xhigh independent reviewer. No required work or hardware rerun remains. The reviewer independently checked HF config equality, source-version differences, PCC minima, raw/filtered performance sums and trace IDs, watcher logs, runtime helper paths, trace allocation lifetimes and every archived hash.

Closed packaging notes: the four accepted `tracy/{linear,full}/{prefill,decode}_perf_report.csv` files are explicitly force-added because the repository ignores CSVs; the latest successful hook log is retained as `hooks_final.log.gz`. `archive_manifest.json` records lossless log/raw-ops archives and local-only intermediate marker streams. Small filtered CSVs and human-readable tables remain directly readable. All stage-owned paths are inside this autoport root. No shared TTNN/Metal source was changed and no push is permitted or performed.

Later-stage validation risks: long-context accuracy compares sampled windows; deterministic replay restores the prefix snapshot, so sustained autoregressive trajectories remain for generator validation. A future generator must prepare and test its actual resume/batch interleaving while traces stay live. This goal implements and certifies only the decoder layer stage; no later stage was started.

Local checkpoint SHA and branch are recorded below after creating the source/evidence commit. The subsequent documentation commit records that immutable checkpoint identity.

## Local checkpoint

Repository: `/home/ttuser/ornith-9b/tt-metal`; branch: `main`.
Source, tests, evidence and independent clean-pass review committed locally as `7fbc0040fbf5f75a912751f9a2a9f085d32e2ca4`. No push performed.

Commit hooks passed. The final four filtered CSVs retain tt-perf-report’s native CRLF bytes and their reviewed hashes; a model-local `.gitattributes` entry declares those line endings for whitespace checking without changing data. The follow-up documentation commit records this checkpoint SHA and that artifact-format declaration.
