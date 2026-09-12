# Stage Review

Verdict: clean-pass

Stage 1 functional decoder for `Qwen/Qwen3.8-27B`. Fresh independent review
of the live worktree on branch `mvasiljevic/qwen38-full-bringup`, base HEAD
`a3a9fb4229a045ad9361b4e39ad854b491346ea9`, on 2026-09-11. The 108 staged
stage-owned files were inspected before checkpoint creation. Only this report
was written by the reviewer; no hardware commands or model runs were issued.

## Required Work

None. No concrete implementation defect or unmet Stage 1 evidence requirement
was found. The three pending gates in `stage_review_1.md` are closed by the
completed context runs, synthetic pytest results, and final capability/command
documentation. The local checkpoint and SHA bookkeeping follow this review
under the stage-review workflow; their pre-review absence is not a defect.

## Other Concerns

- Prefill batches have equal logical lengths and prefix positions. This is an
  explicit decoder API boundary, not an advertised ragged-batch capability.
  Full attention requires caller-owned disjoint pages and matching device
  positions/RoPE. Linear recurrence requires state for the intended prefix;
  changing its unused position tensor does not rewind recurrence.
- The largest context was tested at batch 1; batch 32 was tested at prefill
  257 and decode context 258. `context_contract.json` and the README disclose
  these separate coverage points. The review does not infer coverage of every
  batch/context combination.
- Decode replay variants restore the same prefix before replay. A continuous
  multi-step generation test belongs to subsequent model/generator integration.
  It would strengthen accumulated-state evidence, but the current persistent
  state updates, trace-buffer refresh checks, and continuation controls satisfy
  this decoder stage without inventing another acceptance gate.
- The measured timings are one warmed batch-1 execution per mode at prefill
  128/decode context 129. They establish device kernel time and recorded gaps,
  not a latency distribution, full-context performance, or a speedup claim.

## Hard-Check Gaps

- This inspection independently checked source, saved results, hashes, logs,
  and profiler rows. It did not independently rerun numerical computations on
  TT hardware, import TTNN, reset/open devices, or collect new profiler data.
- PCC is aggregate over all logical output elements. The stated contract does
  not require per-token PCC, and the reviewer found no contradictory output
  artifact justifying an additional threshold or metric.
- The stage changes Python and evidence files only. Existing final pre-commit
  output reports passing applicable checks. A native build is not required by
  the supplied AGENTS.md change table; no build is claimed here.

## Anomaly Ledger

- Observed anomaly: batch-32 linear prefill initially failed with
  `num_heads 1536 exceeds compute cores 110`.
  Evidence: `linear_b32_watcher.log`; native
  `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp:137`;
  `tt/functional_decoder.py:253`.
  Affected path: linear prefill scan.
  Control or comparison: `linear_b32_watcher_retry.json/log` passes prefill,
  continuation, traced decode and changed-input replay; `linear_b3.json/log`
  also passes the uneven 2+1 grouping. Final watcher logs are clean.
  Likely subsystem: native scan allocation requires batch times value heads
  to fit the worker grid.
  Investigation performed: checked the native limit and the repair's split
  along the independent batch axis, preserving output/state request order.
  Resolution: fixed by the tested model-level grouping; no capability reduction.

- Observed anomaly: initial mesh initialization failed on a frozen Ethernet
  firmware heartbeat, with teardown repeating the failure.
  Evidence: `AUTOTRIAGE.md`, `mesh_smoke.log`, `reset_1.log`,
  `device_list_after_reset_1.log`, `mesh_smoke_after_reset_1.log`.
  Affected path: infrastructure before decoder execution.
  Control or comparison: one reset restored mesh open/close on unchanged
  firmware 19.8.0; later decoder, watcher and profiler runs completed.
  Likely subsystem: stale active-Ethernet execution state; the original writer
  or workload is not established.
  Investigation performed: read the independent diagnosis and recovery logs;
  retained its warning that the first-sample triage heartbeat boolean cannot
  prove advancement. No device action was performed during this review.
  Resolution: controlled. Firmware 19.8.0 versus documented 19.8.1 and possible
  recurrence remain disclosed infrastructure risks.

- Observed anomaly: the saved HF config says `output_gate_type: "swish"`, but
  full attention applies sigmoid.
  Evidence: `hf_config.json`; installed Transformers 5.12.1
  `models/qwen3_5/modeling_qwen3_5.py:714`; decoder line 151.
  Affected path: full-attention output gating.
  Control or comparison: the pinned canonical HF implementation explicitly
  executes `torch.sigmoid(gate)` and does not consume that config field;
  both real-weight attention kinds pass the unchanged HF oracle.
  Likely subsystem: config metadata versus the installed model implementation.
  Investigation performed: directly compared HF and TTNN gating, zero-centered
  norms, residual/MLP order and linear SiLU gate.
  Resolution: controlled; the decoder follows the specified HF execution path,
  and the distinction is documented rather than silently changing the oracle.

- Observed anomaly: full-attention long-context PCC is lower than short-case
  PCC, and first-use long prefill compiles many per-offset SDPA variants.
  Evidence: `full_context.json/log`, `full_context_progress.log`; prefill
  PCC 0.9967552062 at 262143 and 0.9968009230 at 262144.
  Affected path: full-attention long prefill and subsequent decode.
  Control or comparison: all logical outputs pass unchanged PCC >=0.995;
  final-context replay PCC is 0.9977424741, repeated restored replay is bitwise
  equal, changed-input replay passes, and subsequent short reuse passes.
  Likely subsystem: long-context numerical accumulation and offset-specific
  program specialization; no component-level numerical attribution is proven.
  Investigation performed: checked HF absolute causal masks, complete cache
  retention, bounded reference temporary chunks, full-length TTNN inputs,
  result/log agreement and separation from warmed short-shape profiling.
  Resolution: controlled within the stated accuracy bar. No cap or threshold
  relaxation was introduced; no full-context latency claim is made.

- Observed anomaly: successful runs emit compiler/topology warnings; profiler
  logs report a failed optional viewer trace copy, pandas mixed-column warnings,
  and full-profile AICLK settling at 1337 MHz versus requested 1350 MHz.
  Evidence: both final watcher console logs and `linear_profile.log` /
  `full_profile.log`, including full-profile lines 20 and 2411.
  Affected path: host topology/JIT diagnostics, optional viewer export and
  measurement environment.
  Control or comparison: watcher reports disabled features `None`; final
  watcher files have no suspicious messages. Both original host Tracy captures
  exist in the requested per-kind report directories and match their manifests.
  Copied ops CSVs match those originals; all measured rows have device timing,
  and raw/filtered kernel sums agree exactly after ns-to-us conversion.
  Likely subsystem: host tooling and clock settling, not evidence of corrupt
  model outputs or missing measured operations.
  Investigation performed: separated optional viewer-copy failure from actual
  retained capture, checked report windows, trace IDs, totals and single-device
  row attribution. The clock warning itself places settling within 5%.
  Resolution: controlled for these recorded samples; they are not comparative
  speedup evidence or a guarantee of identical clocks on a future run.

## Scope Inspected

- Goal/skill paths: supplied Stage 1 contract and AGENTS.md instructions;
  repository `.agents/skills/{stage-review,functional-decoder,tt-device-usage,tt-enable-tracing}/SKILL.md`;
  installed `tt-model-bringup/0.1.4/skills/functional-decoder/SKILL.md`, including
  its additional chunked-prefill and feasible unchunked-control requirements.
- Artifact paths: `doc/context_contract.json`, `doc/functional_decoder.md`;
  `doc/functional_decoder/README.md`, `work_log.md`, `environment.json`,
  `hf_config.json`, `weight_stats.json`, both manifests, interim review and
  recovery report/logs; real-weight smoke/boundary/audited/continuation/batch-32/
  batch-3/context/control JSON and logs; synthetic pytest log/XML and both
  synthetic result JSONs; profile JSON/logs, all four original and copied ops
  CSVs, filtered CSVs and rendered tables; final watcher logs and pre-commit log.
- Code paths: complete `tt/functional_decoder.py` and all five stage test/helper
  files; imported rotary helper in
  `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py`;
  native delta-rule adapter and scan limit; installed HF Qwen3_5 decoder,
  attention, normalization, convolution, recurrence and MLP implementation.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `head`, `tail`, `wc`,
  `git status`, `git diff`, `git branch --show-current`, `git rev-parse HEAD`;
  standard-library `python3` JSON/CSV, SHA256, AST and log analyses. The sole
  mutation was writing this report.
- Reviewed decoder SHA256:
  `c790a57d53a3d2bbf9461400088dbd6281aba72ce98285513badc6bb8ad9623c`.
  Runner SHA256:
  `c51c6861b29be120d4720989e6687cd268377844d97a46e08e6c21001c5e48aa`.

Re-derived acceptance evidence:

| Requirement | Independent inspection result |
|---|---|
| Real model and both kinds | Config has 48 linear/16 full layers at hidden 5120, MLP 17408, full 24 query/4 KV heads of 256, linear 16 key/48 value heads of 128. Statistics cover all 14 linear and 11 full layer tensors; config hash matches. Layer-only HF loading is strict and the same weights feed TTNN. |
| Correct block semantics | Residual order, SwiGLU, zero-centered norms, full-attention GQA/gate/partial RoPE and linear causal convolution/delta recurrence match the inspected HF structure. Native scan padding adds neutral zero q/k/v/beta/g rows and slices logical outputs. Constants are caller-supplied, avoiding the native host-upload fallback. |
| Full capability | Both context runs contain 4097,262143,262144,31 on one loaded instance and exit 0. Exact-limit prefill PCC is linear 0.9990556382/full 0.9968009230. Final-context traced-decode PCC is linear 0.9997068644/full 0.9977424741. All JSON rows match their console records. |
| Boundaries, cache and continuation | Real and synthetic sweeps cover 1,31,32,33,127,128,129,257 and short reuse; long nondivisible contexts pass. Prefix-33 continuation, batch 32, permuted physical pages, unowned pages and refreshed page-table ownership pass. Unaligned continuation uses per-token paged update until page alignment, preserving live prefix rows. |
| Chunked control | Both real-weight batch-2 257-token controls compare 128-token outer chunks with one outer chunk and unchanged HF, then traced decode from each filled state. Every reported comparison passes >=0.995. |
| Synthetic CI | JUnit records 2 tests, zero failures/errors/skips, 42.513 seconds; all 18 copied case rows appear in the common pytest log. Minimum prefill/decode PCC is linear 0.9972204566/0.9965006113 and full 0.9983780888/0.9977924824. |
| Device-only traced decode | Forward guards prohibit Torch dispatch and TTNN host conversion. Imported/native helpers inspected for fallback. State snapshots/restoration occur outside measurement; captured inputs retain their device tensors and are refreshed before replay. All replay PCCs pass, with bitwise repeated restored replay. |
| Warmed performance | Raw signpost windows contain only device ops: linear prefill/decode 88/100, full 52/57. Every decode row has Metal trace ID 0. Raw ns totals and filtered us totals agree: linear 4623.045/3009.414 us; full 3407.822/2378.159 us. Rendered tables are actual tables, with zero host ops. |
| Watcher | Both final batch-32 runs complete with watcher enabled, attach/poll/detach evidence, and no fatal/exception/invalid/overflow/out-of-bounds/error/corrupt/sanitize messages in the final watcher files. |
| Artifact integrity | All 102 compact manifest entries and all 6 original raw-capture entries exist and match size/SHA256. All stage Python sources parse. Current staged scope is limited to this autoport; no unstaged implementation differences were present. |

## Residual Risk

The review establishes this functional decoder stage against the pinned HF
implementation and recorded one-device environment. It does not establish
full-model accumulated accuracy, generation quality, multi-device behavior,
serving integration, every context/batch combination, or long-run firmware
stability. Those limitations are disclosed and do not leave required Stage 1
work outstanding. The parent must preserve this reviewed source/evidence in
the local stage checkpoint and record its SHA; no push is authorized here.
