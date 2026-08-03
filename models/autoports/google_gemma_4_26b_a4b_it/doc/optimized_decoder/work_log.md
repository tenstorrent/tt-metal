# Optimized decoder work log

## Scope and baseline

- Model: `google/gemma-4-26B-A4B-it`; repo path
  `models/autoports/google_gemma_4_26b_a4b_it`.
- Baseline commit: `11473bba139c97872d03fa1ee87dd4cf9dd0556f`;
  fused implementation commit: `0dafd12a42bac0eb72b3c0abbc908500eedd7131`.
- Hardware: one Blackhole P300 (`1x1` mesh, fixture-selected device 3). All TT
  hardware commands were serialized. Health was checked before profiling and
  after final regression.
- Stage ownership: `tt/optimized_decoder.py`, its tests, optimized-decoder
  documentation/artifacts, and the additive optimized section of
  `doc/context_contract.json`. No multichip, full-model, LM-head, CCL, or vLLM
  work was started. No optimized batch-32 command was run.
- Current paired fused sequence-1,024 batch-1 baseline: sliding 400.921 ms
  prefill / 2.455 ms traced decode; full 401.868 / 2.669 ms.

## Topology audit and optimization sequence

1. Audited the fused op graph before tuning. The packed same-input expert
   up/gate sparse matmul dominated, sparse down was second, dense gate/up were
   repeated same-input matmuls, and decode repeatedly crossed DRAM around
   residual norms. Dedicated QKV heads, RoPE, paged cache update, SDPA,
   softmax, top-k, and packed expert up/gate were already fused/composite.
2. Swept weight dtype and fidelity by projection family with real-weight HF
   acceptance. BFP4/BFP8 attention and BFP4 dense variants failed sliding PCC;
   expert-only BFP4 and BFP8 passed in isolation. The inherited BF16 attention
   linears are profiler-verified HiFi2. BFP8/LoFi experts were chosen for the
   final sharded path because BFP4 missed the combined sliding gate.
3. Swept BFP4 and BFP8 sparse geometries for both layer kinds. The 4-core
   up/gate K8 + 8-core down K11 point won every safe matrix. Independent review
   caught that K22/K22 had initially only been paired with the unsafe one-core
   row. A safe 4/8-core K22/K22 point was added and passed both kinds, but lost
   to K8/K11: 1.340 vs 1.339 ms sliding and 1.539 vs 1.536 ms full. Current
   two/four-core alternatives are 1.654-1.710 ms sliding and 1.866-1.909 ms
   full.
4. Proved batch-1 DRAM-sharded QKV legality with padded M=32. K-block 1 passed
   at PCC 0.999893/0.999905 but boundary-inclusive latency was 0.289/0.352 ms
   versus 0.139/0.171 ms interleaved. K-block 11 was then attempted and failed
   the exact L1 capacity check (2.516/3.106 MB required versus 1.573 MB).
5. Swept paged SDPA 8x4/q32/k64, 8x8/q32/k64, and 8x8/q32/k128. All passed.
   Retained the common 8x4/q32/k64 point: it won sliding, and the full-only
   0.4% alternative was not a consistent material improvement.
6. Implemented four- and eight-core L1 width-sharded residual/norm chains.
   Eight cores won (1.307 vs 1.320 ms sliding; 1.502 vs 1.513 ms full in the
   sweep). The initially sharded input norm exposed a real sliding PCC
   interaction; retaining only that first reduction interleaved recovered HF
   acceptance while the residual stream and later norms/adds stay sharded.
7. Tested shape-derived sequence-1,024 dense prefill program configs. They
   passed PCC >0.99999 but regressed both prefill medians by about 0.05%, so
   `USE_LARGE_PREFILL_DENSE_CONFIGS=False`.
8. Added BFP8 paged-cache fill support using TTNN typecast on device. BF16 vs
   BFP8 decode PCC is 0.999783 sliding / 0.999959 full, with no host fallback.
9. Final rereview caught that the dominant prefill sparse rows still used fused
   K1 defaults. Added an owned BFP8/LoFi prefill sparse tile and independently
   swept five sequence-1,024 core/K/subblock points. Selected 4/8-core K8/K11
   at 239.602/240.936 ms; K22/K22 was 242.637/244.070 ms, two/four-core points
   were 405-433 ms, and the K1 control was 388.999/390.107 ms. All candidates
   passed PCC >=0.99992 versus K1.

## Hang, triage, AutoFix, and recovery

The first one-core BFP4 sparse geometry hung in packed up/gate
`SparseMatmulDeviceOperation`. Live `tt-triage` captured the exact op and
healthy ARC/DDR state before the process was terminated. The affected device
was reset using the bounded device-usage recovery sequence; all four P300s
then listed healthy and a 1x1 mesh smoke printed `MESH_SMOKE_OK`.

`$autotriage` and `$autofix` isolated the root cause: this checkout's sparse
1D multicast factory lacks the one-core in0 `SKIP_MCAST` change from
non-ancestor commit `341ffae7862`. A host-side construction and final dispatch
guard now rejects explicit one-core packed up/gate geometry without silently
remapping it. Four guard/source tests passed, then every remaining safe BFP4
and BFP8 geometry completed on hardware for both layer kinds. Evidence:
`AUTOTRIAGE.md`, `triage/geometry_bfp4_hang*.txt`, and `geometry_*.json`.

## Final measurements and correctness

Paired same-process sequence-1,024 medians of five:

| layer | fused prefill ms | optimized prefill ms | fused trace ms | optimized trace ms | speedup | decode PCC |
|---|---:|---:|---:|---:|---:|---:|
| sliding | 400.921 | 239.641 | 2.455 | 1.341 | 1.831x | 0.999913 |
| full | 401.868 | 240.887 | 2.669 | 1.534 | 1.740x | 0.999892 |

- Final HF sliding: prefill PCC 0.998678, decode PCC 0.999743. Full also
  passes both 0.995 functional bars.
- Non-aligned logical prefill 31, 33, and 1,025 passed for both kinds.
- Bounded sliding modulo-cache integrity passed.
- Decode at current position 262,143 passed for both kinds; the 262,144
  context contract is unchanged and no capability is reduced.
- Both representative layers passed 101 trace replays with replay PCC 1.0.
- Final default suite command:

  ```text
  GEMMA4_RANGE_DOWNLOAD=1 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py
  ```

  Result: 71 collected, 13 passed, 58 intentional opt-in sweep skips, zero
  failures in 87.771 s. `py_compile`, JSON parsing, and a scoped
  `git diff --check` over stage source, tests, Markdown, and JSON also passed.
  Generated profiler CSVs and captured watcher/triage logs retain byte-exact
  tool output, including their original line endings and whitespace.

- A final current-HEAD completion audit reran the same default suite after the
  local checkpoint commits: 13 passed, 58 intentional opt-in skips, and zero
  failures in 87.68 s. TTNN discovered all four local Blackhole devices, ran
  on device 3, and closed the device cleanly. The standalone `tt-smi` binary
  was not present in this shell's `PATH`; device discovery and the complete
  TTNN run provided the availability check. This rerun refreshed only the two
  101-replay stress JSONs; replay PCC remains 1.0, with medians 1.338 ms
  sliding and 1.534 ms full.

- Advertised-context command used `GEMMA4_OPTIMIZED_CONTEXT=1`; cache,
  precision, geometry, DRAM-sharded, SDPA, residual, and prefill candidate
  groups were run separately with their corresponding `GEMMA4_OPTIMIZED_*`
  opt-in environment variables. Every selected/safe row completed; intentional
  failed-PCC rows are documented as rejected evidence rather than final-suite
  failures.

## Profiler and watcher

Op-level profile (watcher disabled, sequence 32 to avoid the fixed marker
buffer overflow seen at sequence 1,024):

```text
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_PROFILE=1 GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN=32 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python -m tracy -r -p --check-exit-code -o models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/tracy/current_raw -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile -q
```

Result: 2/2 passed. Four signposted windows were processed with
`tt-perf-report --no-host-ops --active-experts 8`; filtered CSV/text/summary
artifacts are retained and raw logs were removed. Dominant sparse up/gate is
57.8-58.9% of prefill and 21.6-24.8% of decode. Overall modeled DRAM results:
110/112 GB/s prefill and 125/132 GB/s decode for sliding/full. The sparse paths
were already exhaustively attacked by precision, fidelity, and geometry. The
final report counts the remaining layout boundaries: sliding decode has 10
interleaved-to-sharded and 7 sharded-to-interleaved ops totaling 16.09 us;
full has 8 and 7 totaling 17.07 us. They cross existing dedicated helper
contracts or the residual-chain entry/output and are included in the measured
whole-layer result; no generic `Reshard` op is present.

Device trace was captured separately:

```text
TT_METAL_DEVICE_PROFILER=1 GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_OPTIMIZED_PROFILE=1 GEMMA4_FUNCTIONAL_DECODER_SEQ_LEN=32 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' python -m tracy -v -p --device-trace-profiler -o models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/tracy/current_trace_raw -m pytest models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py::test_optimized_decoder_perf_profile -q
```

Result: 2/2 passed; device kernel trace medians are approximately 1.287 ms
sliding and 1.478 ms full. The difference from 1.341/1.550 ms host trace
medians is consistent with host replay/dispatch overhead.

Watcher command (profiler disabled):

```text
GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_optimized_decoder.py -k 'optimized_hf_acceptance or optimized_traced_decode_batch1'
```

Result: 4/4 passed, 65 deselected, no watcher fault. The raw watcher log and
source-bound provenance are retained.

## Optimize checklist

- [x] Read stage context, tests, fused source, functional source, and baseline
  artifacts before modifying code.
- [x] Recorded an operation-topology audit before local tuning.
- [x] Used the fused decoder as the semantic/performance baseline and kept
  changes within optimized-decoder stage ownership.
- [x] Swept precision/fidelity by projection family; kept correctness-sensitive
  norms/router in high precision.
- [x] Swept sharded residual layouts and explicit SDPA/matmul programs.
- [x] Tested packed same-input dense projections and retained packed expert
  up/gate.
- [x] Proved DRAM-sharded batch-1 matmul legality, then rejected it with adapted
  performance/capacity evidence rather than a first API error.
- [x] Tested large prefill program configs independently from decode tuning.
- [x] Audited composite/dedicated SDPA, RoPE, cache, heads, top-k, and sparse
  operations; retained already-fused forms where no equivalent lower-movement
  replacement exists.
- [x] Removed repeated residual-chain movement while preserving necessary
  operation-boundary conversions.
- [x] Verified no runtime torch conversion or host fallback in measured paths.
- [x] Reported paired sequence-1,024 prefill and traced decode before/after for
  both representative layer kinds at batch 1.
- [x] Preserved HF PCC, deterministic trace replay, paged cache semantics,
  non-aligned lengths, and advertised context.
- [x] Ran stress, current-source op profiler, separate device trace, watcher,
  health checks, source gates, and full default suite.
- [x] Independent `$stage-review` clean pass recorded after two blocking review
  rounds were fixed and rereviewed.
- [x] Stage-owned changes committed locally and SHA recorded; never push.

## Limitations

- Optimized batch 32 is deliberately unmeasured and out of scope. The additive
  context-contract entry does not alter the functional decoder's existing
  batch-32 claim.
- Batch-1 inputs are tile-padded to M=32 internally, as required by TTNN
  matmuls; the logical batch remains one.
- One-core packed sparse geometry is unsafe in this checkout and is rejected
  before device dispatch. This does not constrain the selected 4/8-core path.
- Profiler topology uses sequence 32; the performance decisions use independent
  warmed sequence-1,024 A/B measurements.

## Commits

Stage implementation SHA: `33d5cfc4853` (`Optimize Gemma-4 26B decoder`).
No push was performed.

The first commit attempt intentionally produced no commit because hooks
formatted/import-cleaned the Python source and normalized captured text. All
source-bound profiler, device-trace, watcher, and suite evidence was rerun
after the mechanical source formatting. The final commit skips only
`trailing-whitespace` for the byte-exact watcher capture and
`prefer-expect-error` for the host-only policy guard assertion.
