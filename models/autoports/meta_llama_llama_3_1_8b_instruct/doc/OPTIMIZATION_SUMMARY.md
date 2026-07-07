# Llama-3.1-8B-Instruct — Agentic Bringup & Optimization Summary

Target: `meta-llama/Llama-3.1-8B-Instruct`
Autoport: `models/autoports/meta_llama_llama_3_1_8b_instruct`
Forge source: `ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0`

This work was produced autonomously by Codex driving the `.agents` pipeline:
`$forge-functional-decoder` (prepare the TTNN decoder from the tt-forge emit),
then `$optimize` (single-chip decoder-layer performance tuning). Scope is one
dense decoder layer, single-chip TTNN — no multichip, full-model, or vLLM path.

## Headline: before vs after

| Path | Prefill (seq 16) | Traced decode (prefix 16) |
| --- | ---: | ---: |
| Functional baseline (BF16 act / BFP8 weights) | 5.590 ms | n/a (functional decode was a stub) |
| First correct traced optimized candidate (BFP8/BFP8) | — | 1.845 ms |
| **Final optimized decoder** | **1.439 ms** | **1.289 ms** |
| Speedup | **~3.9x** vs functional prefill | **~1.43x** vs first correct decode |

Host timings are the fastest final uninstrumented rows from
`doc/optimized_decoder/perf_host_timings.csv`. Under the profiler: prefill device
time 1,570 us, decode device time 1,229 us. Modeled decode DRAM roofline ≈ 91 GB/s
(31.6%); the remaining gap is matmul program efficiency / op scheduling, not host
fallback (0 host ops in both windows).

## Correctness

| Check | PCC |
| --- | ---: |
| Real-weight prefill, seq 16 | 0.9999939 |
| Real-weight paged decode, prefix 17 | 0.9999949 |
| Eager-vs-traced decode replay | 1.0 |
| Synthetic prefill (seq 16/17/64) | ~0.961–0.962 |
| Synthetic paged decode (prefix 16/17) | ~0.957–0.966 |

Real weights validated after `HF_TOKEN` access was provided. The lower synthetic
PCC is the expected effect of the BFP4/LoFi policy on random weights and was not
used to veto faster real-weight-correct policies (per the optimize skill).

Full optimized suite: 13 passed, 1 skipped; watcher-clean (no ERROR/ASSERT/hang/
fault/timeout).

## Final precision policy

- BF16 activations, residuals, and norms.
- BFP4 attention weights, BFP4 MLP weights, LoFi matmul fidelity.
- BF8_B paged KV cache (`[max_num_blocks, num_kv_heads, block_size, head_dim]`,
  default `max_num_blocks=4`, `block_size=32`).

## Key optimization decisions (evidence in `doc/optimized_decoder/*.csv`)

| Area | Decision | Evidence |
| --- | --- | --- |
| Prefill Q/K/V | Kept **separate** projections (1.289 ms vs packed 1.428 ms) | `qkv_projection_trials.csv` |
| Decode Q/K/V | Kept **packed** projection (1.294 ms vs separate 1.413 ms) | `qkv_projection_trials.csv` |
| Gate/up MLP | Kept **separate** gate/up (1.289 ms vs packed 1.333 ms) | `gate_up_projection_trials.csv` |
| Decode down proj | **DRAM-sharded**, `in0_block_w=14` (56 hit L1 limit, 28 slower) | `down_geometry_trials.csv` |
| Prefill K/V geometry | 16-core `1x2` for ≤32-token prefill (1.197 ms vs 1.406); auto-config fallback for seq 64 to preserve context | `prefill_kv_geometry_trials.csv` |
| KV cache dtype | **BF8_B** LoFi cache is fastest correct (1.276 ms); rejected HiFi2 BF8_B (1.310 ms) | `fidelity_cache_trials.csv` |
| Precision | BFP4/BFP4 attention+MLP weights (1.287 ms), real PCC 0.99999 | `precision_trials.csv` |
| Attention | TTNN `scaled_dot_product_attention` (prefill) + `paged_scaled_dot_product_attention_decode` (decode, 12 us) | `tt_perf_report_decode.txt` |
| L1 input movement | **Rejected** — slower (prefill 1.765 ms vs 1.412 ms) | `l1_movement_trials.csv` |

## Deliverables

- `tt/functional_decoder.py` — prefill functional decoder (from the forge emit).
- `tt/optimized_decoder.py` — optimized prefill + paged traced decode.
- `tests/test_functional_decoder.py`, `tests/test_optimized_decoder.py`.
- `doc/functional_decoder/` and `doc/optimized_decoder/` — full work logs, perf
  reports (`tt_perf_report_*.txt/csv`), Tracy ops, and per-decision trial CSVs.

## Known limitations

- Prefill + paged decode for a single decoder layer only; no multichip / full
  model / vLLM in this stage.
- Public context contract stays at the stage-validated 64 tokens (default paged
  cache can hold 128 logical tokens).
- `decode_forward` in the *functional* decoder remains a documented stub; the
  optimized decoder implements the real paged decode path.
