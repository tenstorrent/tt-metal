# Optimized Decoder Work Log

## Scope

Skills used: `$optimize`, `$tt-device-usage`, `$graph-rewrite`.

Scope stayed within:

- `tt/optimized_decoder.py`
- `tests/test_optimized_decoder.py`
- `doc/optimized_decoder/*`

No multichip, full-model, or vLLM work was started.

## Hardware Checks

```bash
timeout 60 tt-smi -ls --local
```

Result: `tt-smi` was not on PATH.

```bash
timeout 120 python - <<'PY'
import ttnn
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
ttnn.close_mesh_device(mesh)
print("MESH_SMOKE_OK")
PY
```

Result: `MESH_SMOKE_OK`.

## Implementation Notes

- Added `OptimizedDecoder`, independent of `FunctionalDecoder`.
- Packed QKV as `[Q,K,V]` to match TTNN decode helper contracts.
- Prefill uses packed QKV matmul, TTNN slice/reshape/permute heads, RoPE, SDPA, concat heads, output projection, residual/norm/MLP.
- Decode uses packed QKV matmul with WIDTH_SHARDED output, `nlp_create_qkv_heads_decode`, `rotary_embedding_llama`, paged cache update, `paged_scaled_dot_product_attention_decode`, `nlp_concat_heads_decode`, output projection, residual/norm/MLP.
- MLP gate matmul fuses SiLU.
- Final MLP weights use BFP4 and LoFi compute; attention weights use BFP8 and HiFi2.
- Decode MLP uses explicit 1D program configs with `in0_block_w=32`, promoted from the geometry and traced replay sweeps.
- Runtime forwards contain no `torch`, `ttnn.from_torch`, or `ttnn.to_torch`.

Bringup fixes:

- `nlp_create_qkv_heads_decode` required WIDTH_SHARDED QKV output.
- Decode RoPE required `rotary_embedding_llama` rather than plain RoPE.
- Decode cos/sin required HEIGHT_SHARDED memory.
- SDPA output pre-concat memory config uses physical head width 128, not hidden width 4096.
- `nlp_create_qkv_heads` for prefill rejected the current rank/shape contract; manual TTNN slice/reshape/permute stayed in the measured path.

## Correctness And Stress

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
timeout 1200 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  --tb=short -s
```

Result: 9 passed.

PCC evidence:

- `OPT_REAL_WEIGHT_PREFILL_SEQ_5_PCC=0.999991`
- `OPT_REAL_WEIGHT_DECODE_PREFIX_5_PCC=0.999991`
- `OPT_REAL_WEIGHT_REPEATED_DECODE_POS_31_PCC=0.999992`
- `OPT_REAL_WEIGHT_REPEATED_DECODE_POS_32_PCC=0.999992`
- `OPT_REAL_WEIGHT_REPEATED_DECODE_POS_33_PCC=0.999991`

Trace replay:

- `test_optimized_decode_trace_replay_correctness` captures one decode call with `ttnn.begin_trace_capture`, replays it three times, and checks PCC `>= 0.9999`.
- `test_perf_decode_trace_signposted` captures decode, warms trace replay, and reports `OPT_DECODE_TRACE_PREFIX_31_WARMED_MS=1.755` in the latest full-suite run and `1.750` in the final standalone log.

Watcher:

```bash
timeout 1200 env TT_METAL_WATCHER=10 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k 'not perf' --tb=short -s
```

Result: 6 passed, 3 deselected; watcher stopped and detached cleanly.

## Perf Commands

Isolated final signpost run:

```bash
timeout 1200 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_perf_prefill_signposted \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_perf_decode_signposted \
  --tb=short -s
```

Final isolated values:

- `OPT_PREFILL_SEQ_5_WARMED_MS=48.659` from final Tracy prefill run.
- `OPT_DECODE_PREFIX_31_WARMED_MS=1.941` from the latest full-suite final run.

Profiler capture:

```bash
timeout 1200 python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/final_prefill \
  -n final_prefill \
  -m pytest \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_perf_prefill_signposted \
  --tb=short -s

timeout 1200 python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy/final_decode \
  -n final_decode \
  -m pytest \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_perf_decode_signposted \
  --tb=short -s
```

`tt-perf-report`:

```bash
tt-perf-report \
  doc/optimized_decoder/tracy/final_prefill/reports/final_prefill/2026_07_09_09_29_00/ops_perf_results_final_prefill_2026_07_09_09_29_00.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --csv doc/optimized_decoder/tracy/final_prefill/prefill_perf_report.csv \
  --summary-file doc/optimized_decoder/tracy/final_prefill/prefill_perf_summary.csv

tt-perf-report \
  doc/optimized_decoder/tracy/final_prefill/reports/final_prefill/2026_07_09_09_29_00/ops_perf_results_final_prefill_2026_07_09_09_29_00.csv \
  --start-signpost PERF_PREFILL --end-signpost PERF_PREFILL_END \
  --no-color \
  > doc/optimized_decoder/tracy/final_prefill/prefill_perf_report.txt

tt-perf-report \
  doc/optimized_decoder/tracy/final_decode/reports/final_decode/2026_07_09_09_54_13/ops_perf_results_final_decode_2026_07_09_09_54_13.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --csv doc/optimized_decoder/tracy/final_decode/decode_perf_report.csv \
  --summary-file doc/optimized_decoder/tracy/final_decode/decode_perf_summary.csv

tt-perf-report \
  doc/optimized_decoder/tracy/final_decode/reports/final_decode/2026_07_09_09_54_13/ops_perf_results_final_decode_2026_07_09_09_54_13.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --no-color \
  > doc/optimized_decoder/tracy/final_decode/decode_perf_report.txt
```

## Candidate Results

| Candidate | PCC | Warmed decode | Decision |
| --- | ---: | ---: | --- |
| Baseline BFP8/HiFi2 MLP | 0.999998 | 2.385 ms | Replaced by BFP4/LoFi. |
| BFP4/LoFi MLP auto config | 0.999992 | 2.316 ms sweep, 2.220 ms earlier full-suite run | Replaced by explicit program configs. |
| BFP4/LoFi MLP explicit `in0_block_w=8` | 0.999992 | 1.843 ms eager, 1.766 ms full-suite traced, same-process trace best 1.475 ms | Rejected after larger legal sweep. |
| BFP4/LoFi MLP explicit `in0_block_w=32` | 0.999992 | 1.941 ms eager, 1.755 ms full-suite traced, same-process trace best 1.285 ms | Kept; best same-process traced replay. |
| All matmuls HiFi4 | 0.999999 | 2.581 ms | Rejected, slower. |
| MLP-only DRAM-sharded weights/configs | 0.999991 | 2.842 ms | Rejected, slower after required reshard/output/block fixes. |

Traced comparison:

- `TRACE_CANDIDATE baseline_bfp8_hifi2 WARMED_MS=1.921`
- `TRACE_CANDIDATE final_bfp4_lofi WARMED_MS=1.629`
- `OPT_DECODE_TRACE_PREFIX_31_WARMED_MS=1.750` after promoting explicit `in0_block_w=32`.
- `trace_block_sweep.json`: `w=8` best/median 1.475/1.492 ms; `w=32` best/median 1.285/1.492 ms.

Geometry sweep artifact:

- `decode_geometry_sweep.md`
- `decode_geometry_sweep.json`

The sweep locks MLP precision at BFP4/LoFi and uses real layer-0 weights. It
measured the final DRAM-interleaved auto MLP, L1 WIDTH_SHARDED input variants,
explicit `in0_block_w=1/2/4/8/16/32/64`, and DRAM-sharded program variants.
The `w=32` candidate was promoted. L1 input auto variants hit the TTNN
requirement to move SiLU into the program config; the explicit legal L1
versions were then timed and were slower. DRAM-sharded program candidate `w=2`
hit an input-B WIDTH_SHARDED requirement; `w=4/8` hit shard divisibility
blockers.

DRAM-sharded QKV candidate:

- First direct QKV DRAM-sharded candidate hit `bad optional access`.
- Follow-up MLP-only candidate was pursued through multiple actionable API errors:
  - SILU had to move from `activation=` to program-config fused activation.
  - Input A had to be sharded.
  - Output memory had to be sharded.
  - `in0_block_w` had to satisfy the shard tile divisibility constraints.
- The corrected MLP-only candidate ran and was slower, so it was not kept.

## Final Notes

- The optimized path preserves valid non-aligned logical sequence lengths; tests use seq/prefix length 5 and repeated decode positions 31, 32, 33 across a page boundary.
- The model is dense Llama, so MoE active-expert execution is not applicable.
- `doc/context_contract.json` now records the optimized decoder's BFP8 paged-KV decode scope, page size, cache limits, and validation status.
- Remaining `tt-perf-report` advice was either applied where it produced a win (BFP4/LoFi MLP and explicit MLP program configs) or rejected with measured evidence (DRAM-sharded decode MLP and L1-input blockers).
