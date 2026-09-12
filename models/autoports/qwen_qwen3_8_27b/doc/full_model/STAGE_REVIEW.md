# Stage Review

Verdict: clean-pass

Independent final review of Stage 6 full-model for
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, completed on
2026-09-12. Reviewed the live worktree on
`mvasiljevic/qwen38-full-bringup`, starting from
`7550299ba237fa0d578938de1424398046500968`. This verdict supersedes the
intermediate more-work-needed reviews in this file.

The delivered implementation and recorded validation satisfy the supplied
full-model goal. The review found no unresolved required work. The stage owner
should now stage the final reviewed artifacts, create the authorized local
checkpoint, and record its SHA under the stage-review follow-up procedure.
No push or vLLM work is included in this verdict.

## Required Work

None.

## Evidence Supporting the Verdict

| Contract | Independently inspected evidence and result |
| --- | --- |
| Complete text model and standard generator | `tt/model.py` loads embeddings, all 48 linear and 16 full-attention layers, final gamma-plus-one norm, and untied TP4 vocabulary head. `tt/generator.py` implements the standard builder and explicit traced generation and low-level cache/table/position APIs. |
| Inherited implementation and precision | The wrapper calls the unchanged selected `MultichipDecoder` without conversions between layers. Final profiler rows on all four devices confirm BFP4/LoFi decoder projections, BF16 activation/residual/collectives, and BF16 decode updates into BFP8 paged K/V. FP32 recurrence and the BF8-prefill/BF16-decode cache split are preserved. |
| Readiness | Fresh pinned AIME24 chat reference has 100 HF greedy tokens and top-100 sets. `readiness_confirmed.json` records prefill top1/top5/top100 = 99/100/100% and teacher-forced decode = 98/100/100%; its process/launcher exits 0. Actual HF/TT autoregressive texts and token IDs were read. |
| Public context and shapes | The final selected-head run passes 262143-token prefill plus last-position decode and 262144-token prefill, both ending at position 262144. The reduced shape matrix covers 1/31/32/33/4095/4096/4097 and repeated shapes. The all-layer memory plan and context contract retain the full advertised context. |
| Full batch and external state | `contract_full_b32.json`, exit 0, uses all 64 layers and 32 fixed slots. It covers mixed 31/33-token prompts, inactive linear state, exact logits after physical page permutation, unchanged-table copy elision, batch/repeat logit determinism, host compatibility, and 31+2 continuation versus 33 with PCC 0.99921447. |
| Trace and sampling | Source and runtime checks show separate model/common-sampling traces, stable `tt_out_tok` feedback into the next embedding, device position/RoPE increments, and no steady token/position/RoPE/page-table host refreshes. Greedy and sampled parameter modes alternate. Common force-argmax and semantically greedy split sampling were compared; the faster split path is delivered. |
| Integrity and native repair | Local CMake build and runtime installation pass after the prescribed Docker wrapper was unavailable. The focused wide-BFP8 watcher regression passes reader counts 1/2/3. `watcher_fixed` repeats the original generator trigger with tracing and stronger state checks, exit 0. No watcher/barrier/precision check was disabled. |
| Qualitative quality | All six selected TT answers reach EOS and are coherent: original 256-token p1/p4 plus matching-cap 1024-token extensions for p0/p2/p3/p5. Both original and extended shared degeneration checks have no findings. Direct reading agrees with `qualitative_review.md`; its prefix/capacity caveat is retained. |
| Final performance | Post-repair, unprofiled full B1 S128/G128: 97.244 ms TTFT and 39.055 tokens/s/user. AIME S203/G100: 99.557 ms and 38.760 tokens/s/user, explicitly including sampled predictions and teacher-forced feedback uploads. |
| Reduced profiling and accounting | All 48 phase-table hashes/counts/timing sums across baseline/selected/final captures match. All 16 final advice-enabled phase reports exist. Exact layer boundaries reproduce the 22.890 ms representative kernel-only stack sum and the separately labeled instrumented estimates. The inherited Stage 5 weighted comparison is 25.156384 ms with isolated replay/synchronization overhead explicitly qualified. |
| Final checks and provenance | `stage_check.log` reports no AIME degeneration and full 262144-token context, exit 0. Final source hashes match every listed file. Final formatting checks pass; original normalized evidence bytes remain archived with hashes. |

## Other Concerns

The initial audit found three issues, all resolved before this verdict:

- External-cache binding omitted convolution-state validation. It now checks
  presence, shape, dtype, layout, and DRAM placement of all relevant state
  tensors before replacing the binding. Independent AST-only host probes
  reject missing/malformed convolution state without changing the old binding,
  while accepting the valid BF16 row-major control. Subsequent full device
  contract coverage passes.
- The full-stack profiling prohibition originally ran after model execution.
  It now rejects `--full --profile` immediately after argument parsing and
  before mesh creation.
- The extended TT suite initially selected the old 128-token HF control.
  It now explicitly selects matching controls and preserves the original
  six-prompt 256-token evidence in a separate artifact.

The native regression documentation now correctly distinguishes three eager
address-rebinding iterations from the separate generator watcher trace coverage.

## Hard-Check Gaps

These limit the scope of evidence without hiding an observed defect:

- Sampled-mode testing covers legal outputs, parameter updates, and greedy/sample
  alternation; it is not a statistical distribution or seed-repeatability study.
- The inactive-state test reads device 0. Source inspection shows identical
  preservation operations on all ranks, but the test itself does not read each
  rank's inactive state.
- Maximum context at B1 and short contexts at B32 are separate physical coverage
  points. B32 at the maximum context is not claimed to fit.
- Kernel-only layer extrapolation is representative reduced-model accounting,
  not a measured all-layer device profile or a universal mathematical bound.
- The final checkpoint occurs after this review, as required by the skill.
  Normalized artifacts were restaged, and the independent cached whitespace
  check passes with `core.whitespace=cr-at-eol`, preserving valid raw CSV CRLF
  endings. Working-tree formatting, preserved-original hashes, and current
  profiler CSV hashes pass. Final qualitative-report formatting also passes.

## Anomaly Ledger

- Observed anomaly: a 17408-byte BF8 head row was issued through a helper
  promising at most a 16384-byte packet.
  Evidence: `AUTOTRIAGE_dram_head.md`, original watcher capture, and native source.
  Affected path / likely subsystem: split-bank DRAM matmul reader, NoC packet
  accounting.
  Control or comparison: the repaired constexpr transfer bound selects the
  existing any-length helper while retaining transaction tags.
  Investigation performed: inspected `Noc::async_read` and `noc_async_read`,
  repaired source hash, build/install logs, focused watcher regression, and
  original/full-model confirmations.
  Resolution: fixed.

- Observed anomaly: new eager prefill signatures could leave persistent
  allocations conflicting with captured scratch.
  Evidence: `AUTOFIX_trace_prefill.md`, generator lifecycle code, tracked shape
  and all-layer contract runs.
  Affected path / likely subsystem: prefill program buffers and trace allocation
  lifetime.
  Control or comparison: release traces before an unseen signature; reuse them
  for known signatures.
  Investigation performed: source inspection and recorded allocation-tracked
  shape/full-B32 passes.
  Resolution: fixed; the generic untracked warning is classified, not suppressed.

- Observed anomaly: malformed external convolution state was accepted.
  Evidence: independent AST-only reproduction and corrected `bind_cache`.
  Affected path / likely subsystem: external cache validation.
  Control or comparison: valid BF16 row-major convolution state.
  Investigation performed: pre/post-fix guard probes and subsequent full
  contract evidence.
  Resolution: fixed.

- Observed anomaly: 256 tokens did not complete several shared TT answers.
  Evidence: directly read original and extended HF/TT outputs, EOS/think
  boundaries, `qualitative_prefix_comparison.json`.
  Affected path / likely subsystem: fixed generation budget and model reasoning.
  Control or comparison: matching prompt/template/revision and 1024-token caps.
  Investigation performed: all six final selected answers read independently.
  TT p0/p2/p3/p5 terminate at 418/531/433/337 tokens; p1/p4 already terminated
  at 148/145. The haiku scans 5/7/5, the story has a coherent ending, the laws
  are appropriately stated, and Fibonacci has the required loop and return.
  HF p2 remains coherent but capped at 1024; no HF story completion is claimed.
  Resolution: controlled; the recorded six-prompt suite passes.

- Observed anomaly: extended TT p0 differs from its old 256-token prefix at
  generated token 32.
  Evidence: exact saved-token comparison; HF prefixes and TT p2/p3/p5 prefixes
  are preserved.
  Affected path / likely subsystem: allocation-dependent numerical execution.
  Control or comparison: prompt/rendered token IDs and sampler settings match;
  changing the requested generation cap changes cache capacity and crosses an
  inherited short/general SDPA configuration branch.
  Investigation performed: inspected source and counters plus both actual
  outputs. The SDPA branch is a plausible explanation, not experimentally
  isolated causality.
  Resolution: controlled and limited in the verdict. The extended haiku is
  correct and complete; termination of the old branch and capacity-invariant
  token streams are not claimed.

- Observed anomaly: selected profiled prefill was about 17.16 ms versus 6.38 ms
  baseline, mostly additional gaps.
  Evidence: original and final phase tables, unprofiled candidate timings,
  `performance.md`, and recorded CPU-control pause.
  Affected path / likely subsystem: collection/host contention and eager dispatch.
  Control or comparison: final capture is 8.917 ms, comprising 3.099 ms device
  work and 5.819 ms gaps; it still exceeds the baseline profile.
  Investigation performed: independently reproduced table sums/hashes and read
  the final disposition. The selected terminal adds eager prefill dispatches;
  no profiled prefill speedup is claimed. Headline TTFT uses the final unprofiled
  97.244 ms result.
  Resolution: controlled and accurately reported.

## Scope Inspected

- Goal: supplied Stage 6 all-layer text autoregressive, standard generator,
  inherited TP4/precision/layout, cache/context/batch, tracing/sampling,
  correctness/quality, reduced profiling and local-checkpoint contract.
- Skills: `.agents/skills/{stage-review,full-model,tt-enable-tracing,qualitative-check,tt-device-usage}/SKILL.md`.
- Code: model/generator; inherited decoder cache/prefill/decode paths; both
  common sampling contracts and delivered `TTSampling`; readiness/reference,
  qualitative, state/shape, memory and profile helpers; shared prompt-whitespace
  fix; native DRAM reader and regression; relevant NoC helper definitions.
- Artifacts: stage/top README, work log, context/memory plans, diagnosis/repair
  reports, readiness reference metadata/results, actual autoregressive and
  qualitative text/token outputs, state/shape/watcher/check logs, final source
  hashes, formatting-original manifest, raw compact profiler inputs, per-device
  reports, performance accounting and archive manifest.
- Commands: read-only source/git queries, stdlib JSON/CSV/hash/AST analysis,
  fake-tensor cache guard probes, artifact identity comparisons, and whitespace
  inspection. No TTNN import, device access, server, build or model execution
  was performed by this reviewer. Only this report was written.
- Final runtime SHA256:
  - `tt/model.py`: `25726bd020846dcffca14fcb2fb46ab0784f0fc1d59bc68d5650c10aa222b82b`
  - `tt/generator.py`: `7509ddc079fbadac267bd067660a6a22eb2cea90fb795b41e9c64c7ed59cc411`
  - Repaired native reader: `95584c41a4fd87c86b8b0e6737d08a43811e5d8883f47ae0320ecfd8b4963e36`
  - Complete source and installed-library provenance: `final_source.sha256`,
    `native_fixed.sha256`, and successful run manifests.

## Residual Risk

This is a full-model stage pass for the recorded model, four-device hardware
and workload coverage. It does not establish universal reasoning reliability,
identical HF/TT free-running text, bitwise identity across different cache
capacities, arbitrary joint batch/context allocations, or later serving behavior.
No vLLM adapter or serving release is claimed. Earlier review findings are fixed
or controlled by the specific evidence above; no unresolved issue prevents this
stage's local checkpoint.
