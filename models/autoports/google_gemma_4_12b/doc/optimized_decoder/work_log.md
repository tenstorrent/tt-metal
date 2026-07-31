# Optimized decoder work log

Repo revision: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`
Date: 2026-06-08
Target: `google/gemma-4-12B`

## Baseline

Functional correctness evidence from
`models/autoports/google/gemma-4-12B/doc/functional_decoder/pcc_results.jsonl`:

| Layer | Seq | Prefill PCC | Decode PCC | Bars |
| --- | ---: | ---: | ---: | --- |
| sliding_attention | 128 | 0.9973848698 | 0.9933475834 | 0.995 / 0.993 |
| full_attention | 128 | 0.9958167831 | 0.9968326543 | 0.995 / 0.995 |
| sliding_attention | 1024 | 0.9937211762 | 0.9942062761 | 0.992 / 0.992 |
| full_attention | 1024 | 0.9924927439 | 0.9939581613 | 0.992 / 0.992 |

Functional warmed latency from
`models/autoports/google/gemma-4-12B/doc/functional_decoder/tracy/perf_summary.json`:

| Layer | Mode | Device time us | Op gap us |
| --- | --- | ---: | ---: |
| sliding | prefill | 3296.464 | 36.423 |
| sliding | decode | 2881.813 | 58.091 |
| full | prefill | 3655.096 | 54.138 |
| full | decode | 3198.814 | 58.716 |

## Implementation summary

- Added `tt/optimized_decoder.py` with an `OptimizedDecoder` class, no import or
  reference to the functional fallback.
- Added `tests/test_optimized_decoder.py`, which dynamically reuses the
  functional test utilities but instantiates `OptimizedDecoder`.
- Added optimized tests for source fallback audit, paged prefill/decode PCC,
  long-context PCC, traced decode replay and determinism, repeated stress,
  warmed perf signposts, and real checkpoint layer 0 when local weights exist.
- Used local LLM guidance from `tech_reports/LLMs/llms.md`: DRAM/interleaved
  prefill, L1 width-sharded decode activations, DRAM-sharded decode matmuls,
  optimized SDPA/flash decode contracts, and tensor current-position inputs for
  traceable decode.

Final code choices:

- BF16 activations and norms.
- BF16 KV cache.
- MLP prefill BF16 DRAM-interleaved weights with fused GELU*up.
- MLP decode BFP8 DRAM-sharded weights and L1 width-sharded residual outputs.
- Sliding attention prefill BF16 DRAM-interleaved default TTNN path.
- Sliding attention decode QKV BFP8, O BF16, selective long-position QKV
  HiFi3/fp32 accumulation, paged SDPA decode.
- Full attention prefill BF16 DRAM-sharded explicit prefill configs.
- Full attention decode BF16 QKV/O with DRAM-sharded decode matmuls and paged
  SDPA decode.

## Trials and decisions

| Trial | Evidence | Decision |
| --- | --- | --- |
| Initial BFP8 attention and BFP8 MLP decode | Sliding short passed at 0.9971605697 / 0.9949398587; full short failed decode at 0.9941040762 below 0.995. | Keep BFP8 where accepted; make full attention BF16. |
| Global BF16 attention | Full short recovered, but sliding short decode fell to 0.9899617797. | Reject globally; use per-layer attention precision. |
| BFP8 sliding attention with long context | Long sliding decode 0.9916129407 below 0.992. | Add selective long-position compute precision. |
| BF16 MLP for long sliding | L1 OOM during long prefill gate/up matmul: `CB grow to 1906080 B > 1499136 B`. | Reject. |
| MLP down BF16 only | Long sliding decode stayed around 0.9916086. | Reject. |
| BF16 attention or HiFi4/fp32 attention globally | Worsened PCC or hit L1 limits. | Reject. |
| Sliding O projection BF16 | Improved sliding decode and is in final path. | Keep. |
| Sliding long-position QKV HiFi3/fp32 | Long sliding decode passed at 0.9967197443. | Keep selectively for `decode_position >= sliding_window`. |
| BFP8 KV cache | Long sliding decode 0.6177840. | Reject; keep BF16 KV. |
| `GEMMA4_12B_OPT_ATTENTION_DTYPE=bfp4` | Sliding decode 0.9538122188; full prefill 0.9157667111. | Reject. |
| `GEMMA4_12B_OPT_MLP_DTYPE=bfp4` | Sliding decode 0.9891755304; full decode 0.9884254788. | Reject. |
| Full default/interleaved attention prefill | Full short decode 0.9868869619 with TT cache; 0.9942429944 with HF cache; decode MLP BF16 variant 0.9942266101. | Reject; keep slower BF16 DRAM-sharded full prefill path for PCC. |
| MLP prefill BF16 DRAM-interleaved default path | Preserved PCC and kept sliding prefill neutral/slightly faster. | Keep. |
| Explicit large prefill configs for sliding MLP/default attention alternatives | No end-to-end win over final default prefill path. | Reject for sliding. |
| 4D decode RoPE | No accepted PCC/perf improvement over 2D path. | Reject. |
| Interleaved decode O projection | No accepted improvement. | Reject. |
| Decode-only MLP BF16 | Did not recover failed full default prefill decode PCC and slowed the path. | Reject. |
| Unfused MLP GELU | No benefit versus fused GELU*up. | Reject. |
| Interleaved decode norms | No benefit; adds movement. | Reject. |
| MoE active-expert path | Model is dense; no routed experts exist. | Not applicable. |
| Fused matmul-CCL | Single-chip TP=1 decoder stage. | Not applicable. |

## Final correctness commands

Full optimized correctness, trace, stress, and real-weight coverage:

```bash
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_runtime_fallback_audit_source_clean \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_paged_prefill_then_decode_pcc \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_long_context_paged_prefill_decode \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_decode_trace_replay_pcc_and_determinism \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_repeated_prefill_decode_stress \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_real_weight_layer0_prefill_decode \
  --tb=short --timeout=420
```

Result: `10 passed, 3 warnings in 88.05s`.

Watcher run:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_WATCHER_DISABLE_ETH=1 \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_LOGS_PATH=models/autoports/google/gemma-4-12B/doc/optimized_decoder/watcher/default_disable_eth \
pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_paged_prefill_then_decode_pcc \
  --tb=short --timeout=300
```

Result: `2 passed`. Watcher log:
`models/autoports/google/gemma-4-12B/doc/optimized_decoder/watcher/default_disable_eth/generated/watcher/watcher.log`.

Source fallback audit:

```bash
grep -nE 'import torch|ttnn\.from_torch|ttnn\.to_torch|FunctionalDecoder' \
  models/autoports/google/gemma-4-12B/tt/optimized_decoder.py || true
```

Result: no matches.

## Final performance commands

Sliding Tracy collection:

```bash
python -m tracy -r -p -v \
  -o models/autoports/google/gemma-4-12B/doc/optimized_decoder/tracy/sliding/raw \
  -m pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_perf_warmed_prefill_and_traced_decode \
  --tb=short -k sliding --timeout=240
```

Full Tracy collection:

```bash
python -m tracy -r -p -v \
  -o models/autoports/google/gemma-4-12B/doc/optimized_decoder/tracy/full/raw \
  -m pytest -q \
  models/autoports/google/gemma-4-12B/tests/test_optimized_decoder.py::test_optimized_perf_warmed_prefill_and_traced_decode \
  --tb=short -k full --timeout=240
```

Advice-enabled report commands were run for each `layer in sliding full` and
each `mode in prefill decode`:

```bash
tt-perf-report "$ops_csv" \
  --start-signpost "$start_signpost" \
  --end-signpost "$end_signpost" \
  --no-summary > "$report_txt"

tt-perf-report "$ops_csv" \
  --start-signpost "$start_signpost" \
  --end-signpost "$end_signpost" \
  --csv "$report_csv" \
  --summary-file "$stacked_prefix" > "$console_log"
```

Final latency:

| Layer | Mode | Functional us | Optimized us | Delta us | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| sliding | prefill | 3296.464 | 3294.000 | -2.464 | 1.0007x |
| sliding | traced decode | 2881.813 | 1374.000 | -1507.813 | 2.0974x |
| full | prefill | 3655.096 | 4404.000 | +748.904 | 0.8299x |
| full | traced decode | 3198.814 | 1740.000 | -1458.814 | 1.8384x |

All optimized signposted windows report `0 host ops`.

## tt-perf-report advice handling

Sliding decode:

- Report marks QKV, O, and MLP decode matmuls optimized.
- High op-to-op gap advice estimates only 3 us, 0.2% of total; measured decode
  is already trace replay.

Full decode:

- MLP matmuls are marked optimized.
- Attention QKV/O are DRAM-bound BF16 matmuls. The remaining advice is to use
  HiFi4 for full accuracy if not FLOP-bound; BF16 HiFi2 already meets the 0.995
  decode PCC bar and keeps traced decode at 1.8384x faster than functional.
- High op-to-op gap advice estimates only 4 us, 0.2% of total; measured decode
  is already trace replay.

Sliding prefill:

- Advice suggests DRAM-sharded program configs for attention and O matmuls and
  L1 input placement for the MLP gate/up matmuls. The final default
  BF16/interleaved prefill path is latency-neutral versus functional and lower
  op-gap than functional; explicit alternatives did not produce an accepted
  end-to-end win.

Full prefill:

- Advice calls out small `in0_block_w=1` and L1 input placement for full
  attention prefill. Default/interleaved full attention prefill was rejected on
  PCC, so the accepted path preserves explicit BF16 DRAM-sharded full prefill
  despite the latency regression.

## Optimize skill checklist

- [x] Functional checks still pass against the optimized path: final full
  optimized pytest command passed.
- [x] Prefill/decode PCC remain at the functional acceptance bar: see
  `pcc_results.jsonl` and README correctness table.
- [x] Material delta explained: full-attention short decode and full prefill
  latency tradeoff documented above.
- [x] Paged KV-cache behavior covered: paged prefill/decode tests use permuted
  page tables and rank-2 cache positions for both layer kinds.
- [x] Warmed trace replay covered: traced decode replay PCC and repeated replay
  determinism pass for both layer kinds.
- [x] Runtime fallback audit clean: source audit found no fallback tokens and
  Tracy windows report 0 host ops.
- [x] Stress/repeated-run coverage exists: 3 repeated runs for both layer kinds.
- [x] Warmed prefill and traced warmed decode latency before/after recorded:
  `tracy/perf_summary.json`.
- [x] Advice-enabled `tt-perf-report` output exists: `tracy/*/*_perf_report.txt`
  and `tracy/*/*_perf_report.csv`.
- [x] Watcher clean: `TT_METAL_WATCHER=10` correctness run passed.
- [x] Decoder path traced with no host fallback in measured decode: traced
  replay perf test, 0 host ops.
- [x] Decode activations width-sharded in L1 across norm, attention, residual,
  MLP, and output projection boundaries where TTNN op contracts allow it.
- [x] Prefill activations DRAM/interleaved where accepted; explicit large
  prefill program configs retained for full attention where needed for PCC.
- [x] SDPA/composite ops used: prefill SDPA and paged SDPA decode.
- [x] Important `memory_config`, `program_config`, and
  `compute_kernel_config` values set explicitly for decode matmuls, full
  prefill attention, SDPA, norms, and sensitive precision paths.
- [x] Shard specs/core grids divide tensor dimensions cleanly into tile-aligned
  widths; DRAM shard specs are generated from hidden/QKV/MLP dimensions.
- [x] DRAM-sharded decode matmuls used for attention and MLP.
- [x] Fused matmul-CCL ruled out: single-chip TP=1.
- [x] MoE active-expert path ruled out: dense model.
- [x] Reduced precision/fidelity experiments documented, including attention
  and MLP BFP4 failures, BFP8 KV failure, and selective precision choices.

## Limitations

- Full-attention prefill is 748.904 us slower than the functional baseline.
  Faster prefill variants were rejected because they broke decode PCC.
- Final correctness stress covers representative 128-token and 1024-token
  decoder-layer workloads, plus real checkpoint layer 0 at 32 tokens. It does
  not claim full-model or serving-path accuracy.
- Multi-chip, full-model, and vLLM serving optimization are intentionally out
  of scope for this decoder-only optimized stage.

