# Gemma 4 26B-A4B fused decoder

Stage 02 adds `tt/fused_decoder.py`, a distinct `FusedDecoder` runtime that
inherits the completed single-device functional contract and overrides the
dense, prefill-MoE, and decode-MoE GeGLU graphs. In all three places the separate
`gelu_pytorch_tanh(gate)` dispatch is folded into the consuming `ttnn.mul`
through `input_tensor_a_activations`. There is no functional forward fallback.

The public contract is unchanged: prefill accepts every positive logical
length (including non-tile-aligned lengths), decode remains trace-safe at
batches 1 and 32, both sliding and full-attention layer kinds retain their
paged-cache views, and the advertised 262144-token context is unchanged.

## Correctness

Real checkpoint weights, sequence 32, threshold PCC >= 0.995:

| Layer/cache | Prefill PCC | Decode PCC |
| --- | ---: | ---: |
| sliding/shared | 0.9986195 | 0.9996651 |
| full/natural | 0.9984834 | 0.9998653 |
| full/shared view | 0.9984834 | 0.9998653 |

The fused suite also passes batch-2 prefill, logical lengths
`1,31,32,33,63/127,64/128,65/129,1023,1024,1025`, bounded-modulo tail cache
integrity, eager/trace replay equivalence, and deterministic repeated trace
replay at batches 1 and 32. The final watcher run passed 9/9 selected
real-weight and trace cases; `watcher.log` has no error/fatal/assert/hang match
and SHA-256
`4e6596677b4c925d812d3e71c4bb03661d9ef69b7128ae96684bb9f81c3f4d2b`.

## Performance

Sequence/current position 1024, P300, warmed median of 7 prefill executions and
101 traced-decode replays. Batch 32 is the serving decode batch.

| Layer | Mode/batch | Functional ms | Fused ms |
| --- | --- | ---: | ---: |
| sliding | prefill/b1 | 679.399 | 678.469 |
| full | prefill/b1 | 680.690 | 679.587 |
| sliding | traced decode/b1 | 3.02073 | 3.02005 |
| full | traced decode/b1 | 3.20694 | 3.20680 |
| sliding | traced decode/b32 | 68.9046 | 68.8736 |
| full | traced decode/b32 | 68.6930 | 68.6255 |

This final-source table supersedes every older timing row. It uses 7 warmed
prefill samples and 101 warmed traced replays per case on the same checkout,
extension, and device. The final fused source wins every required row. Raw
samples and exact source/test/build provenance are in the functional and fused
`layer*_host_timings.json` files.

Final profiler evidence is in `profiler_summary.md`. Both layer kinds have
Blackhole/110-worker, nonzero prefill `tt-perf-report` CSVs. All four decode
cases have Blackhole device-trace firmware/kernel durations. Retained-trace
replay has no new per-op TTNN rows; the modern join limitation and its accepted
device-profiler equivalent are recorded explicitly rather than substituting
capture times.

AutoFix added two further rewrites. The immutable router tensor
scale and `HIDDEN_SIZE**-0.5` scalar are folded once during decoder setup, so
prefill and decode each execute one fewer broadcast multiply. Prefill MoE now
preserves canonical 32-token sparse chunking but folds each standalone GELU
into the consuming multiply. A TILE-layout normalization preserves the sparse
down-matmul contract without adding a layout conversion when the tensor is
already tiled.

Final-source correctness, boundaries, advertised context, trace determinism,
stress/repeated execution, and watcher evidence are enumerated and hashed in
`final_manifest.md` and `final_manifest.sha256`.

## Graph-fusing audit

The full functional op sequence and candidate disposition are in
`work_log.md`. Applicable dedicated kernels were already present for QKV head
creation, RoPE, SDPA/paged SDPA, paged cache update/fill, head concatenation,
RMSNorm, TopK, and sparse expert matmuls. Shared-LHS QKV was already packed.
The dedicated router family was also assessed explicitly. The only
semantically relevant member, `generalized_moe_gate`, cannot run on the
current P300 checkout: a direct batch-32 Blackhole probe fails at JIT compile
because the Blackhole LLK header
`experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h` is
absent. Even where supported, it accepts a 256-slot height-sharded block and
emits compact top-k values/indices, so Gemma's 128-expert router would still
need sentinel padding, token chunking, layout conversion, dense scatter, and
the nonuniform per-expert post-scale. `deepseek_moe_gate` uses grouped
linear-renormalized DeepSeek routing; `moe_gate_mm` is a fixed 12-core
Wormhole-only pipeline. `moe_compute` can spell the expert GeGLU math only
behind a new selected-index, BF4-packed, compute-only adapter;
`TTMoEDecode` is decode/CCL-oriented and has no non-aligned prefill contract.
Exact source references and the failed capability-probe command are in
`work_log.md`.
Dense gate/up packing was rejected because splitting the packed result adds
movement and changes the weight/setup contract. Residual-plus-RMSNorm cannot
replace the materialized residual because the residual is also consumed by the
router and later add. Decode RoPE/layout conversions are required by the
different full/sliding op contracts. Conv, pooling, batchnorm, collective,
LM-head, sampler, and multichip patterns are absent or out of this stage.

## Commands

```bash
GEMMA4_DECODER_IMPL=fused GEMMA4_RANGE_DOWNLOAD=1 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_fused_decoder.py \
  models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py

GEMMA4_DECODER_IMPL=fused GEMMA4_RANGE_DOWNLOAD=1 TT_METAL_WATCHER=10 \
TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}' \
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_functional_decoder.py \
  -k 'real_weights_prefill_decode or traced_decode_batch_contract'
```
