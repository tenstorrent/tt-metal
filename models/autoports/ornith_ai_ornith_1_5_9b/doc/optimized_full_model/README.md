# Optimized full model: Ornith-1.5-9B

| Warmed complete-model measurement | Before | After |
|---|---:|---:|
| B1 TTFT, 128 input / 128 output | 90.017 ms | 89.839 ms |
| B1 resident traced token-out, no per-token transfers | 112.477 t/s/user | 114.842 t/s/user |
| B1 traced token-out including caller token readback | 112.029 t/s/user | 114.599 t/s/user |
| B1 traced teacher forcing, same 128/128 sequence | 111.582 t/s/user | 113.956 t/s/user |
| B32, 31 active fixed slots, resident decode | 55.502 t/s/user | 87.159 t/s/user |
| B32, all active, resident decode | 93.102 t/s/user | 93.101 t/s/user |

TP4 on four P300c Blackhole chips, IDs 0–3, mesh 1x4, FABRIC_1D. Complete
32-layer model; HF revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
B1 primary uses context2048, three warmed repetitions and median latency.
Resident timing covers126 decode steps after the first two output tokens;
readback/teacher-forcing timing covers127 steps. B32 uses context512, diverse
logical prompt lengths33..64 and128 timed steps. All ranks execute tensor-parallel
shards; the selected replicated-across-ranks residual is locally L1-sharded.

The largest gain is masked decode: **18.017→11.473 ms/token, 36.3% lower latency**.
B1 resident decode improves2.1%. Final post-review warmed TTFT is89.839ms
versus90.017ms before; no material TTFT gain is established. The earlier final
run measured92.049ms (+2.26%) with overlapping samples; it remains in
[optimized_final.json](optimized_final.json). A capacity-matched interleaved-head
control at context65536 measures92.544→91.325ms at128 tokens. These distinct
runs and capacities are retained rather than attributing noise to the head.

Before artifacts are [baseline.json](baseline.json),
[baseline_resident.json](baseline_resident.json), [masked_control_final.json](masked_control_final.json)
and [batch32_all_active.json](batch32_all_active.json). The resident/teacher-forcing
control adds the measurement API to the original interleaved head; it does not
change decoder math. Final B1 after guard repair: [optimized_guard_final.json](optimized_guard_final.json),
[batch32_guard_final.json](batch32_guard_final.json), [all_active_final.json](all_active_final.json).
The all-active control predates the host-only admission repair; the full B32
masked path and B1 path were remeasured after that repair.
Teacher forcing includes traced sampling, explicit token uploads and caller
readback; it is not the fixed-input model-only timing or resident token-out.

## Selected changes

- The final norm stays in the inherited residual layout. Decode uses
  `tt_transformers.LMHead1D` with BF8 vocabulary-sharded DRAM weights, HiFi2/FP32
  accumulation, BF16 output, three readers/bank, K block2 and output target64.
  Head time on the same real activation falls0.767→0.570ms. Selected logits
  have PCC0.99999994, max absolute difference0.03125 and the same argmax; full
  model accuracy is independently checked. [Geometry search](head_geometry.csv)
  retains74 rows including repeats and illegal L1 configurations, one-reader
  split adaptations, and legal multi-reader controls.
- `decode_resident` queues the canonical nonblocking model trace and sampling
  trace with persistent `tt_out_tok`, positions/RoPE and page tables. There is
  no per-token host copy, readback or synchronization. It returns the borrowed
  final device token tensor, not an accumulated sequence. Existing streaming,
  explicit cache/page-table/position/prompt-length and async-read APIs remain.
- Masked decode saves/restores only inactive recurrent rows with an exact FP32
  NoC copy. The former full-pool copy/merge/copy cost is removed. Convolution
  protection remains. Scheduler permutations still snapshot all rows.
  Bitwise four-rank checks cover B1/B3/B32 and30 changing masks; full-stack B32
  output, positions, cache hashes and trace reuse pass.

The decoder stack, policy and rejection ledger are preserved:
BF8 projections/KV/CCL, attention HiFi2, MLP LoFi, FP32 recurrence, BF16
inter-layer residual and the selected phase-dependent prefill policy.
[Source preservation](preserved_source.json), [mesh plan](mesh_plan.md),
[checklist](checklist.md) and [runtime audit](runtime_audit.md) give exact scope.
No datatype frontier search or vLLM integration is part of this stage.

Correct split greedy is **0.571ms**, versus2.609ms for force-argmax on the same
real logits ([sampler_choices.json](sampler_choices.json)). The selected common
sampler gathers only32 candidates/rank, not the vocabulary. Top-k/top-p,
penalties, seeds and inactive-slot modes remain traced and pass
[sampling transitions](sampling_transitions_final.json).

## Full-path accounting and prefill

The reduced real-layer profile gives a packed stack estimate of7.947ms
(24 linear +8 full-attention layers), plus1.232ms for embedding/terminal work.
Measured resident decode is8.708ms, within that budget. The independently
measured stage5 private-layer budget is9.176ms before terminal work, with its
entry/exit and S1024 differences explicitly retained. [Profile tables and
advice](profile/README.md), [accounting](profile/layer_stack_accounting.json)
and [perf_summary.json](perf_summary.json) separate extrapolation, same-run
firmware span, host time and bandwidth estimates. The main masked gap is closed;
no force-argmax, full-vocabulary gather or generic TopK dominates decode.

| Logical prefill length, B1/context65536 | Interleaved-head control TTFT | Selected path TTFT |
|---|---:|---:|
| 128 | 92.544 ms | 91.325 ms |
| 513 | 102.014 ms | 103.696 ms |
| 16385 | 1916.067 ms | 1913.323 ms |
| 32769 | 2834.808 ms | 2832.233 ms |

Two warmed repetitions per length; this is a capacity-matched head control,
not a replacement for the primary context2048 baseline. The513-row regression
is retained. [Prefill summary](prefill_summary.json) records physical buckets;
[per-chunk kernel table](profile/prefill_chunk_kernels.csv) covers short and
chunk-crossing full-model paths, including raw profiler metadata caveats.

## Correctness and capacity

| AIME24 original chat-template, 100 reference tokens | Before top1 / top5 / top100 | After |
|---|---:|---:|
| Prefill | 99% / 100% / 100% | 99% / 100% / 100% |
| Teacher-forcing decode | 98% / 100% / 100% | 98% / 100% / 100% |

[Accuracy summary](accuracy_summary.json) links final logs and reference.
Reference-driven decode is100.88t/s/user; its6744.66ms setup-inclusive TTFT is
not the warmed primary TTFT. The previous reference-driven decode was99.15t/s/user.
Raw-prompt and AIME24 chat-template free-running HF/TT artifacts are in
`../../readiness_autoregressive_optimized` and
`../../readiness_autoregressive_aime_optimized`. The six shared qualitative
prompts reproduce the prior stage's tokens exactly; [qualitative verdict](qualitative_verdict.json)
records manual reading, previous matched HF controls and token-cap limitations.

[Context contract](../context_contract.json) retains **262144 tokens**.
Final full32-layer tracked execution passes262143 non-aligned prefill,
terminal decode at262143 and262144 prefill. Program count stays822 with two
model/four sampler traces live. Added head storage is270,745,600bytes/device;
measured free memory after all persistent preparation is20,214,922,240bytes/device.
This proves capacity/finite output, not max-context HF accuracy.
B32/context512 and mixed fixed slots pass; B32 times maximum context is not
advertised and exceeds physical KV capacity. [Capacity evidence](capacity_final.json).

`run_prefill_resume_check` passes logical length513 at aligned and non-aligned
starts, PCC minimum0.9999667 and top1 agreement100%; missing-history controls
fail as expected. Public logical lengths do not require page/chunk/tile alignment.
[Tracked B32](selective_batch32_tracked.json), [Watcher](watcher_selective_final.json),
[resident tracking](resident_tracked.json), [runtime fallback tripwires](runtime_fallback_final.json)
and49 host contract tests pass. The stage's autoregressive mechanical gate passes.

## Reproduction and evidence

Source `python_env/bin/activate` and `/home/ttuser/ornith-9b/lane-env.sh`.
The new direct dependency is pinned in [dependencies.json](dependencies.json):
`python -m pip install --no-deps --no-build-isolation -e /home/ttuser/ornith-9b/dependencies/tt-transformers`.
Its checkout is25ebcdb3311bb86d1cb4ceff77bb8bf481f4cc73; no dependency source changed.

[commands.jsonl.xz](commands.jsonl.xz) records exact argv, environment, timestamps,
source hashes and exit codes. [final_validation_jobs.json](final_validation_jobs.json)
is the serial device queue; [work_log.md](work_log.md) records decisions,
failures, repairs and local commit SHAs. Failed candidates are evidence, not
passing gate results. Quiet-host timings are separate from profiler and Watcher.
Profiler source-ID collision repair and the rejected broad B32 profiling fixture
are documented in [AutoFix](profiler_debug/AUTOFIX.md). No device reset was used.

Logs, generated text and raw profiler windows use lossless gzip archives;
the command journal uses XZ compression to stay below the repository file limit. [archive_manifest.json](archive_manifest.json) maps each original path
to its archive and uncompressed SHA256; original local files remain available.
[Final audit](final_audit.json) distinguishes pre-review compute evidence from
current-source resident admission, scheduler and performance reruns. Final tt-smi listing is healthy and fuser reports no owners.

## Resident page admission repair

Round1 review found that resident replay could cross an unmapped causal page.
Both bulk `decode_resident` and ordinary one-step resident `decode_forward` now
validate the complete requested window using the authoritative CPU position
and page-table metadata before mutation or trace submission. Inactive rows
remain exempt. No device transfer or per-token synchronization was added.

[Tracked boundary evidence](resident_page_guard.json) compares mapped replay
and changed-table continuation with explicit token feedback, rejects sparse
windows without changing replay counters, and retains all412 programs and
trace IDs. The existing explicit `reload_page_table=True` command uploads all
fixed-slot table rows; steady resident calls do not upload page tables.
[Round1 review](stage_review_round1.md) and the work log retain the finding and
repair evidence. Decoder math and recorded profiler kernels are unchanged.

Independent [round2 stage review](stage_review_round2.md): **clean-pass**, no
required work. Local checkpoint SHAs are recorded in the work log.
