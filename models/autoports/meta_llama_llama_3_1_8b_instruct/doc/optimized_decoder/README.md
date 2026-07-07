# Llama 3.1 8B Instruct Optimized Decoder

## Scope

This stage adds `tt/optimized_decoder.py` for `meta-llama/Llama-3.1-8B-Instruct`. It stays within the repo-local optimized-decoder goal: one decoder layer, single-chip TTNN, no multichip decoder, no full model, and no vLLM path.

The optimized path is not a functional fallback. The tests import `OptimizedDecoder` directly and statically check that the measured runtime does not reference `FunctionalDecoder`, `PENDING_DECODE_MESSAGE`, `NotImplementedError`, or `torch`/`from_torch`/`to_torch` inside `prefill_forward` and `decode_forward`.

## Final Path

- Precision policy: BF16 activations/residuals/norms, BF8_B paged KV cache, BFP4 attention weights, BFP4 MLP weights, LoFi matmul fidelity.
- Prefill topology: RMSNorm, separate Q/K/V projections with tuned short-prefill K/V program config, RoPE, TTNN SDPA, concat heads, output projection, residual, RMSNorm, gate/up/silu/multiply/down MLP, residual.
- Decode topology: RMSNorm, packed QKV projection, RoPE, L1-sharded decode heads, `paged_update_cache`, `paged_scaled_dot_product_attention_decode`, decode concat heads, output projection, residual, RMSNorm, separate gate/up/silu/multiply, DRAM-sharded down projection, residual.
- Paged KV cache: `[max_num_blocks, num_kv_heads, block_size, head_dim]`, default `PagedKVConfig(max_num_blocks=4, block_size=32, cache_dtype=ttnn.bfloat8_b)`, default logical capacity 128 tokens.
- Public context contract: preserved at the functional stage's 64-token validated context. The cache dtype changed to BF8_B after timing evidence, but logical capacity was not reduced.

## Final Correctness And Perf

Final uninstrumented host timing is recorded in `perf_host_timings.csv`. The CSV retains two final direct rows to show repeat variance; the fastest final direct row is the headline timing.

| mode | seq/prefix | host timing |
| --- | ---: | ---: |
| warmed prefill | seq 16 | 1.439334 ms |
| traced decode | prefix 16 | 1.289340 ms |

Tracy-instrumented host timing: warmed prefill 1.528488 ms, traced decode 1.317992 ms.

Before/after comparison:

| path | prefill seq16 | traced decode prefix16 |
| --- | ---: | ---: |
| functional baseline | 5.590055 ms | not available; functional decode is a stub |
| first correct BFP8/BFP8 traced decode candidate | n/a | 1.844985 ms |
| final optimized decoder | 1.439334 ms | 1.289340 ms |

Final correctness:

| check | PCC |
| --- | ---: |
| synthetic prefill seq16 | 0.9614727139688691 |
| synthetic prefill seq17 | 0.9615703904005161 |
| synthetic prefill seq64 | 0.9620206558188167 |
| layer31 prefill seq16 | 0.9614727139688691 |
| real-weight prefill seq16 | 0.9999939188931128 |
| synthetic paged decode prefix16 | 0.9655116653551553 |
| synthetic paged decode prefix17 | 0.9566400793894835 |
| real-weight paged decode prefix17 | 0.9999948761421247 |
| batch-2 disjoint page rows | 0.9616335836175772 |
| eager-vs-traced decode replay | 1.0 |

The lower synthetic PCC versus the functional BF16/BFP8 path comes from the selected BFP4/LoFi policy. Real-weight prefill and decode both stay above 0.99999, and synthetic/random PCC was not used to veto faster real-weight policy wins.

## Topology Audit

Measured final topology:

- Prefill: 26 device ops, 0 host ops, 0 host fallback.
- Decode trace window: 37 device ops, 0 host ops, 0 host fallback.
- Dominant decode ops: packed QKV matmul, O projection, gate/up MLP matmuls, DRAM-sharded down projection.

| area | measured topology issue | candidate | action and evidence |
| --- | --- | --- | --- |
| Prefill Q/K/V | Same post-norm input feeds projections. Packed QKV was inherited from functional, but separate Q/K/V was legal. | Packed vs separate projections on real weights. | Kept separate prefill: `qkv_projection_trials.csv` measured separate prefill 1.288909 ms vs packed 1.428290 ms with identical real PCC. |
| Prefill K/V geometry | Final `tt-perf-report` advised increasing K/V output subblock from `1x1`. | K/V output-subblock/core geometry sweep under final BFP4/LoFi policy. | Kept 16-core K/V config for tile-padded <=32-token prefill: `prefill_kv_geometry_trials.csv` measured seq16 default 1.406425 ms vs 16-core `1x2` `mcast_in0=True` 1.197349 ms, real PCC 0.999993565943. The same config is invalid for seq64 with `Number of blocks exceeds number of cores: 32 blocks > 16 cores`, so larger prefill uses TTNN auto config to preserve the 64-token context contract. |
| Decode Q/K/V | Same one-token input feeds projections; decode target is traced latency. | Packed vs separate projections on real weights. | Kept packed decode: `qkv_projection_trials.csv` measured packed traced decode 1.293605 ms vs separate 1.413042 ms. |
| Attention | Hand-built attention would add softmax/matmul movement. | TTNN SDPA and paged FlashDecode-style op. | Kept `scaled_dot_product_attention` for prefill and `paged_scaled_dot_product_attention_decode` for decode. Decode trace-window report shows `SdpaDecodeDeviceOperation` at 12 us. |
| KV cache | Functional had no decode cache. BF16 cache was correct but not the fastest cache policy. | TTNN paged fill/update with BF16 and BF8_B cache trials. | Implemented paged fill/update and selected BF8_B cache. `fidelity_cache_trials.csv` measured LoFi BF8_B cache at 1.275702 ms, the fastest same-harness traced decode cache/fidelity row, PCC 0.999995791152. |
| Decode MLP down | Down projection is dominant in decode. | DRAM-sharded down projection and legal `in0_block_w` sweep. | Kept DRAM-sharded decode down. `down_geometry_trials.csv` selected `in0_block_w=14`; `56` failed L1 circular-buffer validation, `28` was slower. |
| Gate/up MLP | Gate and up matmuls dominate decode and use the same input. | Packed gate-up vs separate gate/up on real weights. | Kept separate gate/up: `gate_up_projection_trials.csv` measured separate at 1.288511 ms traced decode vs packed at 1.333333 ms with matching PCC. |
| Precision/fidelity | Decode dominated by MLP and attention matmuls. | BFP8/BFP4 attention and MLP weight sweep with LoFi and HiFi2 cache/fidelity follow-up. | Kept BFP4 attention and MLP weights with LoFi. `precision_trials.csv` shows BFP4/BFP4 at 1.286569 ms, PCC 0.999994672646. `fidelity_cache_trials.csv` kept BF8_B cache and rejected HiFi2 BF8_B at 1.310287 ms. |
| L1 movement advice | `tt-perf-report` suggested placing some matmul input0 tensors in L1. | Move prefill attention input and prefill down input to L1. | Rejected: `l1_movement_trials.csv` measured the L1 candidate at prefill 1.765099 ms and traced decode 1.307348 ms vs its paired baseline row 1.411823/1.283127 ms. |
| Host fallback | Final measured methods must avoid host fallback. | Static audit and profiler. | Static test passes; `tt-perf-report` windows show 0 host ops for both prefill and decode. |
| Multi-device/MoE/LM head | Not present in this stage. | n/a | Not applicable: this is a single dense decoder layer only. |

## tt-perf-report Conclusions

Artifacts:

- `tracy/optimized_ops_final.csv`
- `tt_perf_report_prefill.txt`
- `tt_perf_report_prefill.csv`
- `tt_perf_report_decode.txt`
- `tt_perf_report_decode.csv`
- `tt_perf_report_decode_tracing_mode.txt`

Prefill report:

- Signpost window: `PERF_PREFILL_WARMED` to `PERF_PREFILL_WARMED_END`.
- 26 device ops, 0 host ops.
- Device time: 1,570 us under profiler.
- Uninstrumented host timing: 1.439334 ms.
- Tracy host timing: 1.528488 ms.
- Main rows: Q projection 97 us, tuned K/V projections 61 us each on 16 cores with output subblock `1x2`, O projection 96 us, gate/up 267/262 us, down 323 us.
- K/V output-subblock advice was implemented for tile-padded <=32-token prefill after a geometry sweep; the same 16-core config is rejected for seq64 and the runtime falls back to TTNN auto config for larger prefill to preserve non-restricted logical sequence support.
- Advice about L1 input movement was tried and rejected as slower.

Decode report:

- Signpost window: `PERF_TRACE_DECODE` to `PERF_TRACE_DECODE_END`.
- 37 device ops, 0 host ops.
- Device time: 1,229 us under profiler.
- Uninstrumented traced host timing: 1.289340 ms.
- Tracy traced host timing: 1.317992 ms.
- Main rows: packed QKV 102 us, SDPA decode 12 us, O projection 58 us, gate/up 258/270 us, DRAM-sharded down 134 us.
- DRAM-sharded down row is marked optimized by `tt-perf-report`.
- Report-modeled DRAM roofline: 91 GB/s at 31.6% for decode. The same modeled path at 100% roofline would be about 0.39 ms; the measured device time is 1.229 ms, so matmul program efficiency and op scheduling, not host fallback, dominate the remaining gap.

## Tests

Final commands:

```bash
python -m py_compile models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short
LLAMA31_8B_OPT_RUN_PERF=1 LLAMA31_8B_OPT_PERF_OUT=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/perf_host_timings.csv pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
LLAMA31_8B_OPT_RUN_PERF=1 LLAMA31_8B_OPT_PERF_OUT=models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/perf_host_timings_tracy.csv python -m tracy -r -p -v -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/optimized_decoder/tracy -m pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_perf_signposts --tb=short
TT_METAL_WATCHER=10 pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py --tb=short
grep -nEi 'ERROR|ASSERT|hang|fault|timeout' generated/watcher/watcher.log
pytest -q -s models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_runtime_has_no_host_fallback models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_default_context_matches_default_paged_cache models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py::test_optimized_rejects_context_beyond_paged_cache --tb=short
```

Results:

- Full optimized suite: 13 passed, 1 skipped, 1 warning.
- Final BF8-cache watcher-clean optimized suite: 13 passed, 1 skipped, 1 warning.
- Perf signpost run: 1 passed, 1 warning.
- Watcher log grep found no `ERROR`, `ASSERT`, `hang`, `fault`, or `timeout` lines.

## Optimize Checklist

- Functional checks against optimized path: complete.
- Prefill/decode PCC: complete; material BFP4 synthetic delta explained and real-weight PCC recorded.
- Paged KV cache and traced replay: complete.
- Runtime fallback audit: complete.
- Stress/repeated coverage: complete via full suite, batch-2 cache rows, trace replay, repeated direct perf rows, and watcher run.
- Warmed before/after latency: complete.
- `tt-perf-report` with advice: complete; advice tried or rejected with evidence.
- Watcher clean: complete.
- Operation topology audit: complete.
- Best-candidate comparison: complete; delivered BF8_B cache LoFi path is the fastest correct traced-decode cache/fidelity candidate and beats the first correct traced optimized baseline.
- Dtype/fidelity policy verified in profiler rows: complete; rows show LoFi BF16 x BFP4 matmuls and BFP8 paged cache update/SDPA decode.
- SDPA/composite ops: complete.
- Projection packing/separation: complete; prefill separate wins, decode packed wins, gate/up separate wins.
- Prefill K/V output-subblock advice: complete; 16-core `1x2` K/V config is implemented for tile-padded <=32-token prefill and seq64 fallback evidence is recorded.
- Material decode matmul configs: complete for decode down via DRAM-sharded geometry sweep; other matmul report advice is accepted where already used or rejected with before/after evidence.
- DRAM-sharded decode matmul: complete.
- Batch capability preserved: complete for batch 2 page isolation; batch 1 remains primary latency target.
- Multi-device collectives, MoE active experts, LM head, sampling, and vLLM serving items: not applicable to this decoder-layer stage.
- No remaining decoder-layer optimization work is knowingly deferred inside this goal.

## Limitations

- Functional decoder has no decode implementation, so there is no functional traced-decode baseline. The before/after decode comparison uses the first correct traced optimized candidate as the baseline.
- The context contract remains at the stage-validated 64 tokens even though the default paged KV cache can hold 128 logical tokens.
- `tt-smi` is not installed in this environment (`timeout 60 tt-smi -ls --local` returned `tt-smi: No such file or directory`); TTNN mesh open/close, perf runs, Tracy, and watcher runs were used for device evidence.
