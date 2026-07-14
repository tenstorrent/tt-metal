# Optimized Decoder: meta-llama/Llama-3.1-8B-Instruct

## Status

This stage adds `tt/optimized_decoder.py` for the repo-local autoport and keeps the functional decoder's emitted contract: single-token decode for batch 32 with a paged KV cache, cache length 128, and no TTNN prefill graph.  Prefill remains the same explicit `NotImplementedError` because the source forge emit did not contain a prefill graph.

The final decode policy is:

- Linear weights: `ttnn.bfloat8_b`
- Activations and KV cache: `ttnn.bfloat16`
- Attention/MLP matmul compute: `HiFi2`
- Norm compute: `HiFi4`
- Dominant projection weights: DRAM width-sharded
- MLP gate/up: packed into one same-input projection, then split on device before SiLU and multiply
- Norm outputs and MLP multiply input: L1 width-sharded where they feed DRAM-sharded matmuls
- SDPA decode: explicit `SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=0, k_chunk_size=0)`

`doc/context_contract.json` was not changed.  The optimized decoder does not reduce the advertised contract; `.agents/scripts/check_context_contract.py --model-dir models/autoports/meta_llama_llama_3_1_8b_instruct` reports `target=131072, supported=128 (DRAM-limited)`.

## Correctness And Performance

Final correctness artifacts:

| Check | Result |
| --- | ---: |
| Synthetic layer 0 PCC | 0.9990221858024597 |
| Synthetic layer 31 PCC | 0.9990462064743042 |
| Real weights layer 0 PCC | 0.9998587369918823 |
| Trace replay PCC | 0.9990221858024597 |
| Trace replay key-cache PCC | 0.9998999238014221 |
| Trace replay value-cache PCC | 0.9999070763587952 |
| Repeated stress PCCs | 0.9990221858024597, 0.9990221858024597, 0.9990221858024597 |

Latency evidence:

| Path | Latency |
| --- | ---: |
| Functional eager decode, synthetic | 3.8332337513566017 ms |
| Optimized eager decode, synthetic | 2.6933498680591583 ms |
| Optimized traced decode, synthetic | 2.688268944621086 ms |

The optimized traced decode is 1.4259x faster than the functional eager decode in the same evidence runner.  Prefill latency is not reported because both functional and optimized stages preserve the source emit's missing-prefill contract.

## Artifacts

- `benchmark_decode_latency.json`: functional-vs-optimized latency comparison.
- `candidate_precision_trials.json`: precision/layout/head-creation candidate evidence.
- `test_results_decode_synthetic_layer0.json`, `test_results_decode_synthetic_layer31.json`, `test_results_decode_real_layer0.json`: PCC results.
- `test_results_trace_latency.json`, `test_results_repeated_stress.json`: trace and repeated-run evidence.
- `tracy/decode/decode_ops.csv`: final Tracy ops CSV for the signposted decode window.
- `tracy/decode/decode_perf_report.txt` and `.csv`: final `tt-perf-report` outputs.
- `shard_advise/report.json` and `shard_advise/final_ir.mlir`: required shard-advisor artifacts.
- `watcher/watcher.log`: watcher-clean optimized correctness run log.

## Perf Report Conclusions

The final `tt-perf-report` filtered window contains 5 traced decode replays.  It verifies the selected dtype/fidelity policy in measured rows:

- QVK matmul `32 x 4096 x 6144`: 115-116 us, 12 cores, `HiFi2 BF16 x BFP8 => BF16`, DRAM-bound.
- O projection `32 x 4096 x 4096`: 87-88 us, 12 cores, `HiFi2 BF16 x BFP8 => BF16`, DRAM-bound.
- Packed gate/up projection `32 x 4096 x 28672`: 495-497 us, 12 cores, `HiFi2 BF16 x BFP8 => BF16`, DRAM-bound.
- Packed output split rows: about 10 us each, followed by a 14 us SiLU row and a 12 us multiply row.
- Down projection `32 x 14336 x 4096`: 247 us, 12 cores, `HiFi2 BF16 x BFP8 => BF16`, DRAM-bound.
- SDPA decode: 126-127 us with BF16 cache tensors.

The remaining visible costs are decode head creation plus RoPE/slice/reshape around the emitted attention layout, single-core RMSNorm rows, the packed gate/up projection, and the post-packed split/SiLU/multiply rows.  The attempted decode-native head creation candidate failed on layer 31 because its output layout expects a cosine cache sequence length of 8 while the emitted single-token RoPE cache has length 1.

## Limitations

- Prefill remains unimplemented because the source emit was decode-only.
- Public sequence-length semantics are unchanged; this stage does not introduce a new public chunk-alignment restriction.
- The final path still uses layout conversions needed to feed DRAM-sharded matmuls and decode cache/head ops.  They are visible in the profiler but are not host fallbacks.
- BFP4 candidates are faster on real layer-0 weights but fail real layer-31 PCC coverage: MLP-only BFP4 reaches 0.9893350005149841 on real layer 31, MLP BFP4 LoFi reaches the same PCC, and all-linear BFP4 reaches 0.9733318090438843.  BFP8 remains the final default.
- Adapted L1 core-count candidates for QVK/O/gate/up/down were correct, but did not beat the default BFP8 DRAM-sharded path.  TTNN also canonicalized several requested output memory configs back to the computed 64-core width-sharded layout, so these are kept as rejected evidence rather than final defaults.
