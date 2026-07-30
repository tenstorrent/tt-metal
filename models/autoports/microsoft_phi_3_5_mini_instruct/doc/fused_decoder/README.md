# Phi-3.5 Mini fused decoder

Stage 02 adds `tt/fused_decoder.py`. `FusedDecoder.from_state_dict` constructs
the fused class directly; inherited functional methods dispatch to its `_mlp`
override, so the measured prefill/decode path uses the fused SiLU-multiply
kernel and never calls a functional fallback. The public prefill/decode shapes,
32-token paged KV cache, BF16 cache dtype, page tables, LongRoPE selection, and
131072-token context contract are unchanged. Consequently
`doc/context_contract.json` requires no change.

## Selected rewrite and correctness

The packed gate/up matmul remains shared. The old
`silu(gate) -> multiply(gate, up)` pair is now one `multiply` with
`input_tensor_a_activations=[SILU]`. This removes one device dispatch without
changing weights, layout, sharding, or cache state.

Fused-vs-functional PCC is 1.0 for prefill 31/33/65 and decode batch 1/32.
Fused-vs-Torch prefill PCC is 0.9999979 or better at all three non-aligned
boundary lengths, above the functional acceptance bar of 0.995. The only target
layer kind is Phi-3.5's dense decoder layer. The authoritative run is
`fused_decoder_tests_final.log` (9 passed); `fused_decoder_tests.log` is a
retained rejected run whose noisy component-latency assertion failed before
end-to-end traced decode was made the selection gate. The independent watcher
run is under `watcher/`.

## Before/after latency

One Blackhole p300c, sequence/context 128, warmed host measurements:

| Path | Batch | Functional mean/min (ms) | Fused mean/min (ms) |
| --- | ---: | ---: | ---: |
| Prefill | 1 | 1.620700 / 1.551029 | 1.565702 / 1.544115 |
| Prefill | 32 | 37.688525 / 37.618348 | 37.275651 / 37.214218 |
| Traced decode | 1 | 1.050437 / 1.045684 | 1.048476 / 1.041767 |
| Traced decode | 32 | 1.214373 / 1.207763 | 1.211165 / 1.204788 |

The final path beats the best correct functional traced-decode baseline at both
required batches. Exact initial output is in `perf_before_after.log`.
`repeated_ab/` contains the robustness gate: 10 independent processes x 1000
paired/interleaved trace replays for each batch and candidate order (40,000
pairs total). Every hierarchical-bootstrap mean and median 95% CI is below
zero:

| Batch | Pair order | Mean fused-functional (ms), 95% CI | Median, 95% CI |
| ---: | --- | ---: | ---: |
| 1 | functional,fused | -0.002442 [-0.002854,-0.001666] | -0.002555 [-0.002606,-0.002503] |
| 1 | fused,functional | -0.002451 [-0.002623,-0.002348] | -0.002405 [-0.002465,-0.002353] |
| 32 | functional,fused | -0.002818 [-0.003498,-0.002355] | -0.002686 [-0.002806,-0.002575] |
| 32 | fused,functional | -0.002786 [-0.003167,-0.002430] | -0.002674 [-0.002834,-0.002534] |

See `repeated_ab/analysis.txt`, all 40 JSON sample files, and their process
logs. The final correctness test additionally performs five bitwise-identical
trace replays at batch 1 and 32.

Final fused-only profiler evidence is `tracy/ops.csv`. Human-readable and CSV
`tt-perf-report` outputs exist for prefill and decode at both batches:
`{prefill,decode}_b{1,32}.{txt,csv}` plus summary CSV/PNG files. The initial
over-wide profiler attempt overflowed marker buffers and is retained in
`tracy/profile_console.log`; the bounded successful rerun is
`tracy/profile_console_final.log`.

## Topology audit

| Sequence | Movement | Assessment |
| --- | --- | --- |
| RMSNorm -> packed QKV linear -> split heads | DRAM; decode split emits required L1 height shards | Dedicated RMSNorm and QKV head ops already selected |
| explicit Phi rotate-half -> paged cache update -> paged SDPA | decode temporarily returns to DRAM, then required cache shards | Dedicated generic RoPE rejects head width 96; fused Llama Q/K RoPE is tile-local interleaved rotation and is not Phi's half-width rotation |
| concatenate heads -> output linear -> residual add | decode uses the required rectangular one-core/user concat layout | Dedicated concat ops already selected; no redundant reshape/permute pair |
| RMSNorm -> packed gate/up linear -> split -> SiLU -> multiply -> down linear -> residual add | DRAM | SiLU folded into multiply; selected |

Every graph-fusing family was assessed:

- Dedicated activation, RMSNorm, SDPA, QKV split/create, head concat, and paged
  cache operations are used. Softmax, TopK, convolution, MoE, collective, and
  distributed-norm patterns do not exist in this single-device layer.
- A dedicated RoPE retry is not expressible: `rotary_embedding` requires a
  width divisible by 64, while Phi head width is 96 with a 48-wide rotate-half;
  `rotary_embedding_llama_fused_qk` requires tile-local Llama rotation.
- Shared-LHS QKV and gate/up matmuls were already packed by the functional
  stage. There is no RepVGG, spatial mean, or identity
  permute-reshape-permute subgraph.
- Residual-add+RMSNorm cannot remove the residual add here: the unfused
  post-attention value is also the MLP residual output, while `rms_norm`
  returns only normalized data. Adding the required explicit residual add
  leaves two ops and duplicates input reads.
- There are no bias, convolution, batchnorm, pad-pooling, explicit
  transpose-matmul, post-matmul narrowing, max-subtract softmax, or
  sum-scale/reduction-reshape candidates.

`tt-perf-report` confirms no Torch conversion, collective, or host fallback in
the measured ranges. It does show layout operations. `layout_autofix.md` maps
them exactly: each recorded range contains four forward calls; per call,
prefill has 4 tilize-pad, 4 tilize, 2 untilize, 4 untilize-unpad, and 6 permute
ops, while decode has the same explicit Q/K RoPE family plus two B1-only
embedding-output padding conversions. Ordered adjacency places every one
before cache update/fill and in the two `_apply_rope` groups. Norm, head
split/concat, cache, and the long-prefill mask were refuted as sources.

The family cannot be removed with the current op set. Generic
`experimental.rotary_embedding` accepts padded width 32 or a multiple of 64;
Phi is width 96 and rotates at midpoint 48.
`rotary_embedding_llama(_fused_qk)` applies a one-tile transformation
independently to each 32-wide tile and therefore cannot move values across the
48-wide Phi half boundary. Decode embeddings require ROW_MAJOR weights, so
their TILE-output conversions cannot be removed independently either. A new
Phi-specific fused kernel would remove this family, but no such TTNN op exists
and adding a core op is outside this stage's model-local file scope.

## Commands

```bash
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_fused_decoder.py
pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_perf.py
FUSED_PROFILE_ONLY=1 FUSED_PROFILE_ITERATIONS=3 python -m tracy -r -p -v -m pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/fused_decoder_perf.py
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/microsoft_phi_3_5_mini_instruct/doc/fused_decoder/watcher pytest -q -s models/autoports/microsoft_phi_3_5_mini_instruct/tests/test_fused_decoder.py -k 'runtime_dispatch_contract or fused_matches_functional'
```

The final Tracy process reports nanobind teardown leaks also present in the
functional logs; they occur after all tests pass and devices close. Its
histogram overflow buckets are duration-distribution buckets, not dropped
profiler markers. `tt-perf-report` labels several dedicated paged/decode ops as
unclassified, so roofline category totals for those ops are incomplete; their
operation rows and timings remain present.
