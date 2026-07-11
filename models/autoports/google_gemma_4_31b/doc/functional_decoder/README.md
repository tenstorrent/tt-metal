# google/gemma-4-31B functional decoder

Status: complete; independent stage rereview returned `clean-pass`. Scope is the
single-device functional decoder only.

## Contract

`FunctionalDecoder.from_state_dict(state_dict, *, hf_config, layer_idx,
mesh_device, ...)` consumes canonical HF weights and performs all host weight
conversion before runtime.

`prefill_forward(hidden_states, *, rope_mats, page_table, kv_cache,
batch_size=1, user_id=0, valid_seq_len=None)` runs causal paged prefill. It
accepts arbitrary logical lengths through 262144, pads internally, masks by
causality, slices the logical output, and populates the paged cache.

`decode_forward(hidden_states, *, rope_mats, page_table, kv_cache,
current_position, current_position_cache=None, token_index=0, batch_size=1)`
runs one paged decode step. Decode uses 2-D row-major RoPE tables and device
position tensors, so the complete pass is trace-capturable and replayable.

The two meaningful layer kinds are:

| Kind | Representative | Q/KV geometry | Attention contract |
|---|---:|---|---|
| sliding | layer 0 | 32 Q heads, 16 KV heads, head dim 256 | causal, 1024-token window, bounded circular paged cache |
| full | layer 5 | 32 Q heads, 4 KV heads, head dim 512 | causal full context, tied K/V projection |

Full advertised-context attention streams 1024-token Q/K/V chunks. This avoids
the exact 2^32-element Q tensor at sequence 262144 and bounds individual SDPA
kernel duration. The MLP streams 4096-token chunks at long sequence lengths.
These are functional memory/correctness mechanisms, not optimized-decoder work.

## Correctness

The acceptance threshold is PCC >= 0.995. All recorded real-weight results use
the cached `google/gemma-4-31B` checkpoint and real target shapes.

| Path | Kind | Coverage | PCC |
|---|---|---|---:|
| paged prefill | sliding | seq 32 | 0.999603 |
| paged prefill | full | seq 32 | 0.999592 |
| non-aligned paged prefill | sliding | seq 33 | 0.999282 |
| non-aligned paged prefill | full | seq 33 | 0.999309 |
| traced paged decode replay | sliding | position 32 | 0.999655 |
| traced paged decode replay | full | position 32 | 0.999620 |
| boundary prefill | both | 63/64/65, 1023/1024/1025 | min 0.998827 |
| batch-32 non-aligned prefill | sliding/full | 32 x 33 | 0.999309 / 0.999349 |
| batch-32 paged decode | sliding/full | position 32 | 0.999416 / 0.999480 |
| advertised prefill oracle | sliding | seq 262144, final 1024-token HF window | 0.998786 |
| advertised prefill oracle | full | seq 262144, HF prefix control through 2049 | 0.999089 |
| distinct-token traced decode vs HF | sliding/full | 262143 history, position 262143 | 0.999406 / 0.998875 |
| changed-input traced decode vs HF | sliding/full | random and 1023->1024 wrap | min 0.998948 |
| sliding padded-cache ownership | sliding | seq 1025/1057 then distinct decode | min decode 0.997702; min K/V 0.999885 |

The real-weight traced decode tests replay twice and require bitwise-identical
outputs. Page tables are a provably non-identity one-block rotation, including
separate physical ranges for each batch user. Advertised-context decode prefills
exactly 262143 history tokens, then changes a stable captured allocation to a
distinct final token and replays at absolute position 262143. A reduced
one-query oracle reproduces the official HF layer math and was first validated
against stock HF at near-1.0 PCC for both layer kinds. Random changed positions
and the 1023->1024 transition prove traced position buffers are consumed. Full
and sliding prefill also pass at 262144 and the near-context non-divisible length
262113.

Main evidence:

- `logs/final_autofix_standard_suite.log`: 25 passed, 8 explicitly gated
  long/performance tests skipped.
- `logs/prefill_boundaries_real.log`: real-weight tile/page/window boundaries.
- `logs/batch32_prefill_decode_real.log`: real-weight batch-32 prefill/decode.
- `logs/autofix_exact_context_decode_262144_{sliding,full}.log`: genuine
  262143-token history, distinct final token, direct HF-vs-TTNN traced decode.
- `logs/autofix_focused_suite.log`: reduced-oracle validation, sliding cache
  ownership after non-aligned prefill, and changed-input trace controls.
- `logs/prefill_near_context_262113_{sliding,full}.log`.
- `logs/decode_context_262144_{sliding,full}.log`.
- `logs/autofix_watcher_trace_mutation.log` and
  `watcher_autofix_trace_mutation/generated/watcher/watcher.log`.

## Performance

The measured prefill shape is batch 1, sequence 128. Decode is a warmed replay
of a complete captured layer at position 32. The totals below sum the filtered
`Device Time` column from `tt-perf-report` CSV; the column is in microseconds.

| Kind | Warmed prefill | Traced warmed decode |
|---|---:|---:|
| sliding | 3.521 ms | 2.577 ms |
| full | 4.254 ms | 2.911 ms |

Each `tracy/<kind>/<mode>/` directory contains the canonical raw ops CSV,
filtered report CSV, human-readable report table, and command console log.
Signposts are `PERF_PREFILL` / `PERF_PREFILL_END` and `PERF_DECODE` /
`PERF_DECODE_END`.

## Runtime fallback and safety

`tt/functional_decoder.py` contains no `torch`, `ttnn.from_torch`, or
`ttnn.to_torch` reference. The source audit covers every functional runtime
helper. The signposted Tracy windows contain TTNN device operations only;
conversion and PCC checks are outside the measured pass.

The final separate `TT_METAL_WATCHER=10` run passed random and wrap-boundary
changed-input traced decode for both real-weight layer kinds. Its watcher log contains normal
attach/check/detach records and no fatal exception, assert, invalid NOC,
overflow, or sanitizer finding.

## Reproduction

All commands require the checkout runtime library:

```bash
export LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/mpl
pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -s
GEMMA4_LONG_PREFILL=262144 pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -k long_nonaligned -s
GEMMA4_LONG_DECODE=262144 pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -k exact_context_distinct_traced_decode -s
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/google_gemma_4_31b/doc/functional_decoder/watcher_autofix_trace_mutation pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -k changed_trace_buffers_random_and_boundaries -s
```

Profiler commands use `GEMMA4_PERF=1 python -m tracy -r -p -v
--output-folder <artifact-dir> -m pytest <single test node> -s`, followed by
`tt-perf-report` with the matching signposts. Exact console provenance is in
`logs/perf_*_autofix.log` and
`tracy/*/*/*_perf_report.console.log`.

## Limitations

This stage intentionally contains no optimized decoder, multichip decoder,
full model, generator, or vLLM integration. Batch 2 and batch 32 are covered for
non-aligned prefill; batch 32 is also covered for paged decode.
