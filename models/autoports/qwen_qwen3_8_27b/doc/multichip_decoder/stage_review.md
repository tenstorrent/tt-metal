# Stage Review

Verdict: clean-pass

Independent review of stage 4, multichip-decoder, for Qwen/Qwen3.8-27B.
Reviewed the live tree on branch `mvasiljevic/qwen38-full-bringup`, starting
at `1477a0eecf4f6654cebb4010aa89b852d2ca75fb`. Final runtime SHA256:
`e8898b5a80cd8f182c82a9fd94d194d6636522378b003cca37552c66770f02ce`.
Final runner SHA256:
`b5752898c589e72955f42e750fc12506a6921c0a9e786dde0a69fcb6720bca87`.

## Required Work

None. The findings raised during review have been repaired or resolved with
source and runtime controls. Local checkpointing and recording its SHA follow
this review; this verdict does not claim that checkpointing has already occurred.

## Evidence Supporting the Verdict

- The frozen `optimized_decoder.py` is unchanged. Changes are confined to the
  multichip implementation, stage tests, stage documentation, and the context
  contract. This Python/docs change requires no C++ or CMake build.
- TP4 packing assigns six full-attention query heads and one KV head to each
  rank, or four linear key heads and twelve value heads. Inspection confirms
  complete local head groups, independently partitioned convolution channels,
  padded B/A gates, local recurrence, replicated positions/page tables, and
  row-parallel output/down reductions. The dense model has no expert path.
- Both layer kinds preserve replicated BF16 logical input/output shapes.
  The repaired B>1 public DRAM boundary avoids physical TILE-row expansion
  exhausting L1 in a stack. Compact internal norms, projections, residuals,
  and direct all-reduce remain in L1. One shared `TT_CCL` owns the scratch
  workspace across the documented ordered CQ0 stack.
- Independently checked all **31 regression JSONs**, matching frozen baseline
  hashes, and `regressions.xml`: 31 passed, no failures/errors/skips,
  326.881 seconds. Coverage includes both layer kinds, non-aligned lengths
  through 4097, continuation at 31, B1/B3/B8/B32 single layers, and
  B1/B2/B3/B8/B16 linear/full stacks. Minimum regression PCC is
  **0.9998691678**; minimum per-user output PCC is **0.9999901056**.
- The final B32 stack passes **100 evolving eager calls and 100 queued trace
  replays without inter-call host barriers**, using the same workspace object.
  Outputs and raw state match bitwise. Stressed output PCC is 0.9999812245,
  minimum per-user stressed output PCC is 0.9999963641, and recurrent-state
  PCC is **0.9998265382**, the overall validation minimum. The changed-input
  replay tests also refresh RoPE, page tables, and differing user positions.
- All **five final capacity artifacts** match the final source and retain
  262144-token prefill for both layer kinds, final valid decode position
  262143, and a retained-input two-layer stack. Each reserves
  **12,759,483,008 bytes/device** in addition to its actual decoder/input/state
  allocations. Complete outputs and valid state rows pass the baseline PCC
  threshold; final-position and replay checks pass where decode is legal.
  Re-derived the **29,939,351,552-byte/device** full-model planning total,
  including both weight layouts, all KV/recurrent states, terminal weights,
  constants, and the 18 GiB activation/trace allowance. The context contract
  is validated with no capability or public alignment reduction.
- All **four final watcher runs** match the final source. Their console logs
  show all four devices checked, and the raw watcher logs contain no detected
  corruption/assert/error. ETH-only exclusion is supported by the recorded
  instrumented-fabric size failure and the explicit optimize-skill allowance.
  Worker/NoC instrumentation remains active. Final device health lists all
  four p300c devices.
- The runner prohibits Torch operations and host tensor conversions inside
  every forward. Code inspection found no model runtime host-compute path.
  Setup, reference comparisons, state restoration, and refresh transfers are
  outside the measured decode interval.

Final medians independently recomputed from the benchmark artifacts, using
20 warmed prefill samples and 50 warmed traced decode samples:

| Layer kind | Phase | Single-chip ms | TP4 ms | Speedup | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear attention | Prefill | 2.041607 | 1.297140 | 1.57393x | 39.35% |
| Linear attention | Decode | 0.822928 | 0.465395 | 1.76824x | 44.21% |
| Full attention | Prefill | 1.715780 | 1.228964 | 1.39612x | 34.90% |
| Full attention | Decode | 0.661916 | 0.350631 | 1.88778x | 47.19% |

Final profiler rows on every rank show four dominant BF16-activation,
BFP4-weight, LoFi projections and two BF16 direct all-reduces per decode.
Device 0 has 425.601/315.055 us kernel totals for linear/full decode,
49.185/40.875 us gaps, and 31.583/31.641 us of direct all-reduce.
Verified all 20 phase/rank summary rows against their CSV tables and source
paths, including retained `.csv.gz` sources after compaction.

The optimization comparison is substantive: packed versus separate MLP,
precision-matched projection blocks/core ownership, Linear/Ring and link
counts, persistent buffers, replicated versus hidden-sharded residuals,
adapted AGMM/MMRS through distributed norm consumers, and coherent BF16/BFP8
collective families. The direct-AR BFP8 candidate passes correctness but loses
both paired 50-replay comparisons. The final default reproduces the selected
performance. The remaining `SLOW` output projection has measured 4/6/12 K-block
alternatives and a source-proven wider-reader blocker; missing CSV subblock
fields are explained by the native factory's automatic 1x4 selection.

## Other Concerns

- The shared workspace is qualified for one ordered CQ0 model stream. Keep
  the context, buffers, and captured inputs alive; independent concurrent
  models or traces must not share it. The README states this caller contract.
- Historical partial policy overrides cannot reproduce older runs against
  arbitrary newer defaults. Saved effective policies/source snapshots and
  pinned historical sweep controls preserve the reviewed comparisons.

## Hard-Check Gaps

None blocking this decoder stage. Full-model generation, full-model trace
allocation, serving behavior, and the complete batch/context capacity frontier
belong to later stages. The present full-context budget is B1; B32 and full
context are separate demonstrated coverage points.

## Anomaly Ledger

| Observed anomaly / affected path | Evidence and control | Subsystem and investigation | Resolution |
| --- | --- | --- | --- |
| B32 stack L1 collision before capture | `stack_batch32_l1_failure.log`; single-layer controls passed; final default and neighboring stacks now pass | Public TILE row expansion and retained caller inputs; native CB arithmetic in `AUTOFIX_stacked_batch_layout.md` matches the collision | **Fixed:** batched public DRAM placement; final B32 queued stress and expanded regressions pass |
| Baseline eager stress collided with captured L1 allocations | `stress_baseline_trace_lifetime_failure.log`; corrected frozen-baseline stress passes | Fixture lifetime; eager oracle was moved before capture and outputs released before the next invocation | **Fixed:** final baseline and TP4 stress complete |
| Whole-cache PCC failed for one-token prefill despite matching outputs | `cache_padding_failure.log`, `AUTOFIX_cache_padding.md`; baseline also has nonzero future padding | Native paged fill writes physical tile padding; compare initialized logical rows while retaining exact raw replay/ownership checks | **Controlled:** all final logical-state, untouched-row, and unowned-page checks pass |
| Advertised-context repeat OOM | `capacity_overreservation_failure.log`, `AUTOFIX_capacity_fixture.md`; first prefill had completed | Fixture retained its previous full output and duplicated the activation allowance; source review also established the unaligned concat peak | **Fixed:** lifetime/reservation correction, conservative six-stream budget, all five final capacity probes pass |
| Per-layer direct-AR workspaces would consume 1 MiB/core across 64 layers | `AUTOFIX_shared_ccl.md`; native outputs do not alias scratch; final stack asserts shared identity | Workspace ownership, native cache rebinding and scheduling audit; common context preserves one allocation | **Fixed within the stated stream contract:** final 100-call eager/queued control passes |
| Multi-reader DRAM projection rejects parent mesh | `AUTODEBUG_dram_mesh.md`, `AUTOFIX_dram_mesh.md`, both-kind one-reader controls | Native coordinate-free worker assignment requires a unit mesh; model-local alternatives were measured | **Controlled exact op limitation:** one-reader default, no unscoped native edit |
| Fused AGMM BFP8 path produced nonfinite output | `AUTOFIX_fused_ccl_dtype.md`, repaired AGMM and actual BFP8 MMRS payload reports | Gather/input/output raw-byte dtype mismatch, not a synthetic precision veto | **Fixed experimental family:** coherent dtype controls pass and are slower |
| Watcher fails before model execution | `watcher_fabric_size_failure.log`; program 29072 bytes versus 26624-byte config buffer | Instrumented ACTIVE_ETH size; optimize skill explicitly permits ETH exclusion for this case | **Controlled instrumentation limitation:** four final worker/NoC watcher runs pass |
| Spurious merged profiler gap and later marker overflow | Initial per-device clock comparison, `profile_marker_overflow.log`; final four profiles and tables complete | Unsynchronized device clocks and excessive marker lifetime; per-rank tables, bounded profiling, flushes outside signposts | **Fixed evidence collection:** final raw rows have valid durations and match the selected runtime |
| Initial interface errors and final metadata/viewer diagnostics | `commands.log`, work log, `profiler_interpretation.md`, final successful runs and archived captures | Corrected API overload/shape/quoting choices; source-classified motherboard mapping, optional viewer copy and metadata warnings | **Fixed or controlled:** no unresolved model fallback, stale-input, corruption, or failed-enrichment warning |

## Scope Inspected

- Goal contract supplied to this reviewer; `.agents/skills/stage-review/SKILL.md`,
  `multichip/SKILL.md`, `tt-device-usage/SKILL.md`, relevant optimize requirements,
  and `tech_reports/LLMs/llms.md` section 3.3.
- `tt/multichip_decoder.py`, frozen `tt/optimized_decoder.py`, stage runners,
  regression/capacity/stress/watcher/profiler/summary scripts, common `TT_CCL`,
  and relevant native all-reduce and DRAM-sharded matmul source.
- Final README, mesh plan, memory/context contracts, work log, candidate and
  validation/performance/profiler summaries, AutoFix/AutoDebug reports, source
  snapshots, raw logs/JSON/XML, human profiler tables and operation CSVs.
- Read-only `git status`, `git diff`, `git diff --check`, `rg`, `cat`, `sed`,
  and Python standard-library AST/hash/JSON/XML/CSV/gzip analyses. An initial
  host Black check identified a summary-script formatting issue; inspected
  final `precommit.log` and `host_checks.log` confirm its repair and passing
  stage checks. Verified all 97 archive entries exist and every retained gzip
  payload matches its recorded SHA256. No TTNN import, hardware job, reset,
  server, implementation edit, or additional agent was launched by this reviewer.

## Residual Risk

Capacity evidence reserves planned full-model persistent storage while running
representative real layers; it is not a measurement of a complete 64-layer
generator's peak or trace size. Long-context inputs repeat recorded activations,
so this is TTNN parity/capacity evidence rather than a new long-context HF
accuracy oracle. Long-context cache checks sample first/end/unowned pages while
comparing complete outputs. ETH instrumentation and arbitrary concurrent shared
workspace use are outside the qualified contract. These limits are disclosed
and do not leave required work for this decoder-stage gate.

## Checkpoint Formatting Addendum

Verdict remains **clean-pass**. Independently verified the hook's 76 generated
artifact changes: 22 `.log`, 38 `.txt`, 15 `.yaml`, and `regressions.xml`.
Each archived original matches both its recorded SHA256/byte count and the
pre-normalization Git index bytes exactly. Each working file equals the
original after trailing-space and terminal-newline normalization, preserving
leading indentation and all non-whitespace content. The regression XML also
has identical parsed structure.

The archive manifest is an append-only extension from 97 to 173 entries; all
archive paths exist. Source snapshots still match their filename hashes.
Runtime and runner hashes remain the values reviewed above, with no runtime
or test changes. Only the manifest and explanatory work-log update accompany
the normalized artifacts. This formatting/provenance change does not alter
the measured evidence or require a hardware rerun.
