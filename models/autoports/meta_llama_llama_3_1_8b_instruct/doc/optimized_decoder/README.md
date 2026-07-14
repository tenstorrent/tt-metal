# Llama 3.1 8B Instruct Optimized Decoder

This stage adds the repo-local optimized TTNN decoder for `meta-llama/Llama-3.1-8B-Instruct`.
It is scoped to `tt/optimized_decoder.py`, `tests/test_optimized_decoder.py`, and this `doc/optimized_decoder`
evidence directory. It does not start multichip, full-model, or vLLM work.

## Runtime Path

`OptimizedDecoder` subclasses the completed functional decoder only for construction and shared helpers; the measured
prefill/decode methods are overridden and tested directly.

- Weights: projection and MLP weights default to `ttnn.bfloat4_b`; activations, norm weights, and KV cache remain BF16.
- Fidelity: decode and prefill dense matmuls use LoFi math.
- Prefill: non-aligned logical `seq_len=17` is accepted; internal tile padding is private.
- Decode: QKV, output, gate, up, and down matmuls use DRAM-sharded width layout around the matmul, then return to the
  existing interleaved contract.
- Decode residual/norm: the viable advisor recommendation is applied for hidden-size residual adds and post-attention
  RMSNorm by keeping that chain in L1 width-sharded layout.
- Attention: paged KV-cache update and `scaled_dot_product_attention_decode` remain TTNN composite ops with explicit
  decode program config.
- Host fallback guard: tests inspect optimized runtime methods for `torch`, `from_torch`, `to_torch`, and functional
  forward fallback references.

The context contract is unchanged. The optimized decoder does not reduce the advertised capability in
`doc/context_contract.json`.

## Operation Topology Audit

| Area | Starting topology | Action | Evidence |
| --- | --- | --- | --- |
| QKV projection | One packed QKV matmul inherited from functional stage; layer 0-30 use Q,V,K physical order, layer 31 uses Q,K,V. | Kept packed projection and preserved layer-kind slicing. | Synthetic prefill/decode tests cover layers 0 and 31. |
| Decode dense matmuls | Interleaved matmuls dominated traced decode latency. | Added DRAM-sharded decode matmul path for QKV/O/gate/up/down. | `tracy/decode/decode_perf_report.txt` shows BFP4/LoFi DRAM-sharded matmuls at 62, 49, 139, 140, and 134 us. |
| Decode residual/norm layout | Advisor recommended L1 width-sharded residual/norm/eltwise layout. | Applied hidden-size residual/norm/final-add subset; rejected full intermediate activation chain after L1 OOM. | `test_logs/l1_chain_candidate_final.log` shows full-chain OOM and residual candidate 5.298 ms; final code times at 5.252 ms. |
| Prefill dense matmuls | Functional path used default matmul configs. | Added one-tile prefill program configs for logical M <= 32 and BFP4/LoFi weights. | Clean wall timing improved prefill from 108.032 ms to 26.645 ms. |
| SDPA and cache | Functional path already used TTNN paged cache update and SDPA decode. | Kept composite TTNN ops; added explicit decode SDPA program config. | Decode report includes `PagedUpdateCacheDeviceOperation` and `SdpaDecodeDeviceOperation`; real cache slot PCC >= 0.991. |
| MLP activation | Gate and up are separate same-input matmuls, followed by `silu` and multiply. | Kept separate gate/up matmuls; there is no local packed MLP projection helper in this stage. | Decode gate/up DRAM-sharded matmuls are 139/140 us and total optimized traced decode is 5.252 ms. |
| Reshards/layout movement | Decode requires interleaved-to-sharded before matmuls and sharded-to-interleaved after. | Kept required boundaries for the existing single-chip decoder tensor contract. | Source guard prevents host fallback; profiler still shows device-side layout movement only. |

## Shard Advisor

Required advisor artifacts:

- `shard_advise/report.json`
- `shard_advise/final_ir.mlir`
- `shard_advise/report.txt`

Run shape: final-policy BFP4 dense attention+MLP block around SDPA/cache for decode. The advisor tracer could not trace
`scaled_dot_product_attention_decode` with `TracedTensor` arguments, so the capture target intentionally excludes SDPA
and paged-cache update while preserving the rewritten dense block that feeds and follows attention.

Advisor summary from `report.json`: 13 total ops, 12 final choices, spill pass ran with 0 spills.
The raw advisor `pipeline.log` was 5.6 MB and is not retained because it exceeds the repository large-file hook; the
required `report.json` and `final_ir.mlir` artifacts plus `report.txt` are retained.

Recommendations:

| Recommendation | Decision | Reason |
| --- | --- | --- |
| RMSNorm and residual chain in L1 width-sharded layout. | Applied for hidden-size decode residuals, post-attention RMSNorm, and final residual add. | Probe candidate preserved PCC and ran at 5.298 ms; final code improved to 5.252 ms traced decode. |
| Full intermediate MLP `silu`/multiply chain in L1 width-sharded layout. | Rejected with hard device evidence. | Adapted physical-height candidate reached `silu` then OOMed: 29,360,128 B allocation, 458,752 B per bank, only 59,616 B free. |
| Dense matmuls in DRAM/interleaved layout. | Rejected for decode, partially kept for prefill. | Measured decode path is faster with DRAM-sharded matmul inputs: optimized traced decode 5.252 ms vs functional traced decode 106.336 ms. |
| No spill after selected layouts. | Applied as a constraint. | Advisor reports 0 spills; final decode keeps L1 width-sharded tensors only where capacity was proven. |

## Correctness

Command:

```bash
pytest -q models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short -s
```

Result: 13 passed in 148.05s.

Representative PCC:

- Synthetic prefill: layer 0 seq17 0.997604, layer 31 seq17 0.997655, layer 0 seq64 0.995995.
- Synthetic decode output: 0.998525 for layers 0 and 31.
- Synthetic cache update: key/value around 0.999818/0.999821.
- Real prefill: 0.999994.
- Real traced decode output: 0.999995.
- Real decode cache update: key/value 0.991727/0.993251.
- Determinism: 3 traced replay rounds were bitwise equal.

Watcher-clean command:

```bash
TT_METAL_WATCHER=10 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_real_weight_single_layer_prefill_matches_hf_if_weights_available \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_real_weight_decode_trace_replay_if_weights_available \
  --tb=short -s
```

Result: 2 passed in 21.67s, watcher attached and detached without a fault.

## Performance

Clean warmed timing, same real-weight layer 0 inputs:

| Path | Functional best | Optimized best |
| --- | ---: | ---: |
| Prefill seq17 | 108.032 ms | 26.645 ms |
| Traced decode | 106.336 ms | 5.252 ms |

Tracy/`tt-perf-report` artifacts:

- Prefill: `tracy/prefill/prefill_ops.csv`, `tracy/prefill/prefill_perf_report.txt`.
- Decode: `tracy/decode/decode_ops.csv`, `tracy/decode/decode_perf_report.txt`.

The profiler did not retain the `PERF_PREFILL`/`PERF_DECODE` signposts in the CSV, so both `tt-perf-report` runs analyze
the whole captured test region. The decode report includes one warmup/capture region plus replay rows with 0 us device
time; the first nonzero region is the useful op-topology evidence. The refreshed decode report shows the final L1
width-sharded residual/norm path through 64-core `InterleavedToShardedDeviceOperation`, `LayerNormDeviceOperation`,
and `ShardedToInterleavedDeviceOperation` rows.

## Limitations

- `tt-smi -ls --local` is unavailable in this environment because `tt_smi` is not installed, but TTNN device open,
  correctness, watcher, trace, and profiler runs succeeded.
- The advisor run needed `PYTHONPATH=/localdev/mvasiljevic/tt-metal/ttnn:/localdev/mvasiljevic/tt-metal:$PYTHONPATH`
  and a temporary untracked `third_party/tt-metal` workaround for tt-mlir/UMD path resolution; the workaround was removed
  and only advisor output artifacts are retained.
- Signposts were emitted by the tests but not retained in the profiler CSVs.
