# Stage 02 work log

## Scope and starting point

- Model: `google/gemma-4-26B-A4B-it`.
- Branch: `skillexp-work-google_gemma_4-p3`.
- Starting HEAD: `3fb5e87495c272665ea83cd95f1dc4a1aadc002d`.
- Functional decoder evidence and 262,144-token context contract were accepted.
- Work was restricted to `tt/fused_decoder.py`, its tests, and
  `doc/fused_decoder/`; the functional test harness received only reusable
  repeat-count and focused-profiler instrumentation.

## Hardware preparation

- `/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -l` reported four P300 Blackhole
  boards.
- A 1x1 mesh smoke opened successfully; the selected fixture device was 3 with
  an 11x10 compute-with-storage grid.
- All hardware jobs were serialized. Watcher and profiler were never enabled in
  the same run.
- The host reports a low `/dev/shm` warning, but every decoder workload and
  device close completed.

## Baseline

The functional candidate was rerun with 10 warmed samples at sequence/current
position 1024:

| Layer | Prefill b1 | Trace decode b1 | Trace decode b32 |
|---|---:|---:|---:|
| sliding | 680.894922 ms | 3.023234 ms | 68.868126 ms |
| full | 682.049158 ms | 3.199710 ms | 68.661923 ms |

## Candidate loop

1. Folded router per-hidden scale and `H^-0.5` into the FP32 projection during
   construction. Removing this fold regressed both layer kinds.
2. Folded approximate GELU into dense and sparse multiply consumers.
3. Replaced two sliding-attention paged-cache update calls with
   `paged_fused_update_cache`; added disjoint V-update sharding required by the
   operation. Full-attention batch 1 also benefits from the fused update.
   Full-attention batch 32 is specialized to the two-update path after a
   matched 200-sample forced-fusion control.
4. Preserved flexible cache updates for full shared HMA view and modulo cache
   addressing.
5. Tried residual-input RMSNorm fusion. It regressed trace latency and was
   removed.
6. Isolated dense versus sparse GeGLU. Sparse fusion wins at serving batch 32
   but loses at batch 1; dense fusion also varies by layer kind and batch, so
   the final path specializes on both.
7. Tried a batch-wide row-major routing conversion, a correct three-call
   batch-wide sparse MoE, and chunk sizes 2/4/8. All regressed versus the
   per-row production graph.
8. Audited Q/K fused RoPE, DeepSeek gate, packed projections, normalization,
   bias/scale folding, collectives, and full-layer fusions. Exact outcomes are
   recorded in `fusion_audit.md`.

Candidate measurements are preserved in the variant-suffixed host-timing JSON
files. The final fused medians are:

| Layer | Prefill b1 | Trace decode b1 | Trace decode b32 |
|---|---:|---:|---:|
| sliding | 680.574817 ms | 2.994960 ms | 68.766853 ms |
| full | 681.699678 ms | 3.186425 ms | 68.587883 ms |

The final graph beats baseline in every required cell. Batch-1 and sliding
batch-32 cells use 50 warmed samples; full-attention batch 32 uses 200. The
distinct cache controls are also slower in every selected cell. Controls that
name an already-selected producer/consumer graph are graph-identical and are
retained to make that specialization explicit.

## Correctness and contract gates

- Real weights: 3/3 cache/layer cases pass PCC `>= 0.995`.
- Traced decode: 4/4 layer-kind/batch cases pass, eager/replay and repeated
  replay PCC `1.0`.
- Non-aligned logical prefill: both layer kinds pass boundary set through 1025.
- Bounded modulo cache-tail integrity passes.
- Mutable trace replay: both layer kinds pass A/B/A at serving batch 32.
- Context remains 262,144 with no dtype, cache-layout, or capability reduction.
- Static hot-path audit finds no Torch, `from_torch`, `to_torch`, or host
  fallback.

## Profiler

Tracy was run for prefill and focused eager decode at both required batches and
both layer kinds. The modern processor could not join earlier dropped setup
markers to host op IDs. Adding a mid-run device-profiler flush isolated decode;
legacy processing then produced valid per-op tables with positive timing for
all retained decode rows. The final full-attention batch-32 report contains two
`PagedUpdateCacheDeviceOperation` rows; full batch 1 and both sliding batches
contain one `PagedFusedUpdateCacheDeviceOperation`.

`tt-perf-report` outputs are under `tracy/`. It reports the legacy CSV as
Wormhole because architecture metadata is absent and ignores the explicit
Blackhole override. This limitation is recorded; raw timing remains sourced
from Blackhole device 3. The processed `source_*_ops_perf_results.csv` inputs
are retained so every report can be regenerated.

## Final hardware gates

Authoritative performance command:

```bash
GEMMA4_FUSED_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_DECODER_PERF_REPEATS=10 pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_decoder_perf_profile
```

Result: 4 passed for the 10-sample matrix. Authoritative 50/200-sample
single-case commands and logs are retained under `perf/`.

Watcher command:

```bash
GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 pytest -q \
models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
-k 'real_weights_prefill_decode or traced_decode_batch_contract or trace_mutable_stable_buffers'
```

Result: 9 passed, 13 deselected; zero watcher-error matches. Captured in
`watcher/`.

The complete delivered file collected 22 tests and finished with 15 passed and
7 intentional opt-in skips. The forced full-cache candidate correctness test
was enabled separately and passed; its console, JUnit, real-weight PCC, and
batch-1/batch-32 trace artifacts are under
`candidates/force_full_cache_fusion/`.

## Review and commits

Fresh independent Stage-02 review after all remediation returned `clean-pass`
with no required work. The reviewer inspected the 125-file staged scope,
current-source timing provenance, forced-cache candidate, profiler topology,
watcher evidence, context contract, and later-stage exclusion without editing
files or using hardware.

Local implementation commit:
`e2440ba65de7c763b6e3903fc9444abade040720`.

This SHA entry is recorded by a follow-up evidence-log commit. No push is
performed.
