# AutoFix Report: Decode Gate/Up Geometry

## Starting Evidence

- Source: optimized-decoder stage review P1. Final decode report showed MLP gate/up matmuls still dominant and `SLOW` with `in0_block_w=2`.
- Original restored-stage decode report: gate/up rows were 258 us and 270 us with output subblock `1x7`.
- Existing evidence only covered packed-vs-separate gate/up projection topology, not program geometry.

## Hypothesis Experiments

Hypothesis: the decode post-attention RMSNorm output is L1 width-sharded with 128 elements per shard, so the largest legal gate/up K block is 4 tiles. Setting gate/up decode matmuls to 64 cores, `in0_block_w=4`, output subblock `1x7` should reduce gate/up time without changing numerics.

Experiment: monkeypatch only one `OptimizedDecoder` instance's `_mlp` in a serialized paged traced-decode harness, keep final BFP4/LoFi policy, and sweep 64/32/16/8-core 1D matmul configs plus output subblock variants. Each candidate ran synthetic HF decode PCC and eager-vs-trace PCC.

Result: `gate_up_geometry_trials.csv` selected `8x8/in0_block_w=4/out_subblock_w=7`. Repeated traced host rows were 1.058298, 1.059280, and 1.059390 ms. Correctness stayed at HF decode PCC 0.965033525382 and trace replay PCC 1.0.

Blockers:

- `in0_block_w=8` and `16` fail TTNN validation: `shard_shape[1] (128) / in0_tile.get_width() (32) must be divisible by in0_block_w`.
- `4x8/in0_block_w=4/out_subblock_w=7` runs but is numerically wrong, HF decode PCC 0.360581298179.
- 32-core correct candidates were slower than the selected 64-core candidate.

Fix: added `_decode_gate_up_program_config()` in `tt/optimized_decoder.py` and pass it only on the decode MLP path. Prefill MLP remains on TTNN auto geometry.

## Final Status

Fixed.

Verification:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_paged_decode_matches_full_hf_layer models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_batched_decode_uses_disjoint_page_rows models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_trace_replay_is_deterministic models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_runtime_has_no_host_fallback --tb=short
LLAMA31_8B_OPT_RUN_PERF=1 LLAMA31_8B_OPT_PERF_OUT=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/perf_host_timings_gate_up_geometry.csv pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
LLAMA31_8B_OPT_RUN_PERF=1 LLAMA31_8B_OPT_PERF_OUT=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/perf_host_timings_gate_up_geometry_tracy.csv python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy_gate_up_geometry -m pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
tt-perf-report models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy_gate_up_geometry/optimized_ops.csv --start-signpost PERF_TRACE_DECODE --end-signpost PERF_TRACE_DECODE_END --no-summary --no-color > models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tt_perf_report_decode_gate_up_geometry.txt
```

Results:

- Compile passed.
- Focused decode correctness/static guard: `5 passed, 1 warning`.
- Direct perf signpost: traced decode 1.066837 ms.
- Tracy perf signpost: traced decode 1.099665 ms.
- Final decode report: 38 device ops, 0 host ops, 1,020 us device time.
- Gate/up rows are now 202 us and 201 us with `in0_block_w=4`, output subblock `1x7`.
