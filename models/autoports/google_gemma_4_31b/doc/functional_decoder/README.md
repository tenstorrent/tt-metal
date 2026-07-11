# google/gemma-4-31B functional decoder

Status: complete pending independent stage review. Scope is the single-device
functional decoder only.

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
| populated-cache traced decode parity | sliding/full | position 262143 | 0.998715 / 0.996611 |

The real-weight traced decode tests replay twice and require bitwise-identical
outputs. Page tables are a provably non-identity one-block rotation, including
separate physical ranges for each batch user. Advertised-context tests prefill
the cache, then trace and replay decode at absolute position 262143. Full and
sliding prefill both pass at 262144 and the near-context non-divisible length
262113.

Main evidence:

- `logs/final_revision_standard_suite.log`: 17 passed, 6 explicitly gated
  long/performance tests skipped.
- `logs/prefill_boundaries_real.log`: real-weight tile/page/window boundaries.
- `logs/batch32_prefill_decode_real.log`: real-weight batch-32 prefill/decode.
- `logs/final_long_pcc_populated_decode_262144_{sliding,full}.log`: exact-limit
  PCC plus populated-cache traced replay.
- `logs/prefill_near_context_262113_{sliding,full}.log`.
- `logs/decode_context_262144_{sliding,full}.log`.
- `logs/final_watcher_batch32_stable_trace.log` and
  `watcher_batch32_stable_final/generated/watcher/watcher.log`.

## Performance

The measured prefill shape is batch 1, sequence 128. Decode is a warmed replay
of a complete captured layer at position 32. The totals below sum the filtered
`Device Time` column from `tt-perf-report` CSV; the column is in microseconds.

| Kind | Warmed prefill | Traced warmed decode |
|---|---:|---:|
| sliding | 3.533 ms | 2.578 ms |
| full | 4.248 ms | 2.913 ms |

Each `tracy/<kind>/<mode>/` directory contains the canonical raw ops CSV,
filtered report CSV, human-readable report table, and command console log.
Signposts are `PERF_PREFILL` / `PERF_PREFILL_END` and `PERF_DECODE` /
`PERF_DECODE_END`.

## Runtime fallback and safety

`tt/functional_decoder.py` contains no `torch`, `ttnn.from_torch`, or
`ttnn.to_torch` reference. The source audit covers every functional runtime
helper. The signposted Tracy windows contain TTNN device operations only;
conversion and PCC checks are outside the measured pass.

The final separate `TT_METAL_WATCHER=10` run passed traced batch-32 decode for
both real-weight layer kinds. Its watcher log contains normal
attach/check/detach records and no fatal exception, assert, invalid NOC,
overflow, or sanitizer finding.

## Reproduction

All commands require the checkout runtime library:

```bash
export LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/mpl
pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -s
GEMMA4_LONG_PREFILL=262144 pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -k long_nonaligned -s
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/google_gemma_4_31b/doc/functional_decoder/watcher_batch32_stable_final pytest -q models/autoports/google_gemma_4_31b/tests/test_functional_decoder.py -k batch_32_paged_decode_pcc -s
```

Profiler commands use `GEMMA4_PERF=1 python -m tracy -r -p -v
--output-folder <artifact-dir> -m pytest <single test node> -s`, followed by
`tt-perf-report` with the matching signposts. Exact console provenance is in
`logs/perf_*.log` (the oversized sliding-prefill log is committed as
`logs/perf_prefill_sliding.log.gz`) and
`tracy/*/*/*_perf_report.console.log`.

## Limitations

This stage intentionally contains no optimized decoder, multichip decoder,
full model, generator, or vLLM integration. Batch 2 and batch 32 are covered for
non-aligned prefill; batch 32 is also covered for paged decode.
