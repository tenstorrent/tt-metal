# Gemma-4 26B-A4B-IT fused decoder

Stage 02 is complete for the repo-local single-device TTNN autoport path.
`tt/fused_decoder.py` preserves the accepted functional decoder semantics while
removing runtime router scaling dispatches, folding GeGLU activation into its
consumer where it wins, and using a fused K/V paged-cache update for the
native geometries where it wins.

Later optimized-decoder, multichip, full-model, and vLLM work is intentionally
not included.

## Delivered graph

The production class is `FusedDecoder`, not a functional fallback. Tests route
the accepted HF oracle through that class and count native
`paged_fused_update_cache` calls. The shared full-attention HMA view is also
tested and deliberately uses the flexible two-update path because the fused
operation cannot express its physical-view metadata.

Kept fusions:

- immutable router hidden scales folded into the FP32 router projection;
- approximate GELU folded into the dense FFN multiply for the winning
  layer-kind/batch shapes;
- approximate GELU folded into sparse FFN multiply at serving decode batch 32;
- sliding-attention paged K/V decode updates fused into one device operation;
- full-attention batch-1 paged K/V decode updates fused into one device
  operation.

Batch-1 sparse GeGLU remains on the accepted functional implementation because
that graph is faster there; batch 32 uses the fused consumer. Dense GeGLU uses
the consumer fusion for prefill and full-attention batch 32, and the producer
GELU for both sliding decode batches and full-attention batch 1.
Full-attention batch 32 keeps the faster two-update cache graph; full-attention
batch 1 uses the fused update. Matched 50/200-sample controls and the opt-in
forced-fusion correctness test cover both choices.
See [fusion_audit.md](fusion_audit.md) for the complete operation sequence and
every assessed graph-fusing pattern.

## Correctness

Acceptance is PCC `>= 0.995`. All meaningful layer kinds pass:

| Layer | Cache view | Prefill PCC | Decode PCC |
|---|---|---:|---:|
| 0 sliding | shared/native | 0.998618 | 0.999656 |
| 5 full | natural/native | 0.998197 | 0.999860 |
| 5 full | shared HMA view | 0.998197 | 0.999860 |

Traced decode agrees with eager decode at PCC `1.0`; repeated replay PCC is
`1.0` at batch 1 and 32 for both layer kinds. HF-vs-trace PCC ranges from
`0.999379` to `0.999860`.

Non-aligned logical prefill lengths `1, 31, 33, 63, 65, 1023, 1025` are covered
for both layer kinds. Internal tile padding is removed before returning the
logical result; no public chunk-alignment restriction was added. Bounded modulo
cache-tail integrity also passes.

The mutable trace A/B/A test overwrites hidden state, RoPE tensors, current
positions, page tables, and both cache tensors. Both layer kinds produce
`A-repeat PCC = 1.0` and materially distinct B output.

The final watcher run used `TT_METAL_WATCHER=10`: 9 passed, 0 failed, 0 skipped,
and the captured console contains no watcher, NoC, kernel, assertion, or timeout
error. Evidence is in `watcher/console.log` and `watcher/junit.xml`.

## Context contract

`doc/context_contract.json` remains unchanged:

- advertised and supported context: 262,144;
- largest non-aligned functional context: 262,143;
- decode batches: 1 and serving batch 32;
- traced mutable-buffer replay at batch 32.

Stage 02 does not change cache dtype, physical layout, or capacity. Native
geometry selects the measured cache-update graph. Sliding attention and
full-attention batch 1 use the fused update. Full-attention batch 32, shared
physical views, and modulo addressing retain the flexible update without
reducing capability.

## Performance

Numbers are medians of warmed host measurements on one P300 Blackhole, device
3. Batch-1 and sliding batch-32 cells use 50 samples; full-attention batch 32
uses 200 because its cache-update alternatives are particularly close. Decode
is trace replay with blocking device synchronization. Sequence length/current
position is 1024. Lower is better.

| Layer | Path | Batch | Functional ms | Fused ms | Improvement |
|---|---|---:|---:|---:|---:|
| sliding | prefill | 1 | 680.894922 | 680.574817 | 0.0470% |
| sliding | traced decode | 1 | 3.023234 | 2.994960 | 0.9352% |
| sliding | traced decode | 32 | 68.868126 | 68.766853 | 0.1470% |
| full | prefill | 1 | 682.049158 | 681.699678 | 0.0512% |
| full | traced decode | 1 | 3.199710 | 3.186425 | 0.4152% |
| full | traced decode | 32 | 68.661923 | 68.587883 | 0.1078% |

The final candidate beats the best correct functional baseline in all six
required comparisons. Raw samples are retained in the corresponding
`*_host_timings_{functional,fused}.json` files and summarized in
`perf_summary.csv`.

Focused eager-decode `tt-perf-report` totals corroborate the traced host scale:

| Layer | Batch | Device-timed op rows | Sum device time |
|---|---:|---:|---:|
| sliding | 1 | 72/72 | 2.971 ms |
| full | 1 | 74/74 | 3.167 ms |
| sliding | 32 | 756/756 | 67.455 ms |
| full | 32 | 757/757 | 67.289 ms |

Prefill reports contain 554 rows for each layer kind, with device totals
555.287 ms sliding and 547.559 ms full. The remaining end-to-end time includes
host dispatch/gaps.

The current profiler postprocessor asserts when earlier setup/prefill device
markers are dropped. Captures therefore flush setup data with
`TT_METAL_PROFILER_MID_RUN_DUMP=1`, use focused eager-decode signposts, and run
the repository legacy device-log processor. Every retained decode row has
positive device time. `tt-perf-report` cannot infer Blackhole metadata from the
legacy CSV and labels it Wormhole; device provenance and the hardware run are
Blackhole. This affects roofline advice, not measured device duration or op
topology. The four retained `tracy/source_*_ops_perf_results.csv` inputs make
every compact report reproducible.

## Commands

Correctness and trace:

```bash
GEMMA4_RANGE_DOWNLOAD=1 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k 'real_weights_prefill_decode or traced_decode_batch_contract or non_aligned_logical_lengths or bounded_modulo_tail_integrity or trace_mutable_stable_buffers'
```

Authoritative warmed A/B performance:

```bash
GEMMA4_FUSED_DECODER_PERF=1 GEMMA4_RANGE_DOWNLOAD=1 \
GEMMA4_DECODER_PERF_REPEATS=10 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_decoder_perf_profile
```

Set `GEMMA4_FUSED_DECODER_VARIANT=functional` for the baseline. Candidate names
are enumerated in the test. The authoritative artifacts use 50 repeats for
batch 1 and sliding batch 32, and 200 for full-attention batch 32; exact
commands are embedded in artifact provenance and retained under `perf/`.

Focused profiler example:

```bash
TT_METAL_PROFILER_MID_RUN_DUMP=1 GEMMA4_FUSED_DECODER_PERF=1 \
GEMMA4_RANGE_DOWNLOAD=1 GEMMA4_DECODER_EAGER_PROFILE=1 \
python -m tracy -r -p -o /tmp/gemma4-fused-tracy -m pytest \
'models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py::test_fused_decoder_perf_profile[blackhole-batch1-sliding_attention_1024-device_params0-mesh_device0]'

python -m tracy.process_ops_logs -o /tmp/gemma4-fused-tracy \
  --force-legacy-device-logs
```

Watcher:

```bash
GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 pytest -q \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  -k 'real_weights_prefill_decode or traced_decode_batch_contract or trace_mutable_stable_buffers'
```

## Artifacts

- `pcc_*.json`: real-weight prefill/decode PCC and provenance.
- `trace_{sliding,full}_attention_batch{1,32}.json`: traced correctness and repeat determinism.
- `trace_mutable_buffers_*.json`: A/B/A stable-buffer replay.
- `prefill_boundaries_*.json`: aligned and non-aligned logical lengths.
- `layer*_host_timings_*.json`: functional, candidate, and final raw latency samples.
- `perf_summary.csv`: authoritative before/after summary.
- `tracy/{prefill,decode}_*.csv`: per-op `tt-perf-report` tables.
- `tracy/source_*_ops_perf_results.csv`: retained processed profiler inputs.
- `tracy/*_summary.csv` and `.png`: grouped operation summaries.
- `tracy/*.txt`: exact report command, warnings, and conclusions.
- `watcher/`: final watcher-clean console and JUnit XML.
- `candidates/force_full_cache_fusion/`: isolated correctness evidence for the
  full-attention forced-fusion candidate.

There is no known remaining compatible, correctness-preserving decoder fusion
that improves both required decode batches on the current TTNN/P300 stack.
