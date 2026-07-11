# google/gemma-4-31B fused decoder

Status: Stage 02 complete; final independent review verdict `clean-pass`.

## Contract

`FusedDecoder` preserves `FunctionalDecoder`'s single-device public API,
real-weight semantics, non-aligned logical-length support, paged KV-cache
behavior, bounded sliding cache, trace replay, and 262144-token context
contract. No dtype, cache allocation, page size, public shape, or advertised
capacity changed, so `doc/context_contract.json` is unchanged.

Non-aligned sliding prefill writes only logical K/V rows. Padded tail rows are
never allowed to wrap modulo 1024 and overwrite still-live cache slots. The
wrapped-window regression covers lengths 1025 and 1057, and the capacity run
covers 262113.

The delivered tests instantiate `FusedDecoder` exactly and assert that the MLP
is `_FusedSharedMLP`; runtime does not dispatch to
`FunctionalDecoder._forward_device` or a host fallback.

## Selected graph

| Step | Operation and tensor flow | Movement / fusion conclusion |
|---:|---|---|
| 1 | learned RMSNorm on residual | decode uses the existing width-sharded norm and required I2S/S2I |
| 2 | packed QKV linear | decode output is written directly to interleaved L1 |
| 3 | dedicated QKV-head creation | prefill and decode dedicated ops retained |
| 4 | Q/K/V per-head RMSNorm | dedicated RMSNorm; required sharded-to-interleaved movement around decode norms |
| 5 | Q/K rotate-half RoPE | dedicated rotary op; Q/K-fused Llama op is semantically incompatible |
| 6 | paged cache write | full decode uses one fused K/V update on disjoint grids; sliding uses two updates because modulo/override arguments are mandatory |
| 7 | causal/sliding SDPA | dedicated SDPA/chunked SDPA retained |
| 8 | head concatenate | sliding prefill up to seq 128 uses bounded-L1 dedicated chunks; full/long prefill uses multi-core permute+reshape; decode uses dedicated sharded concat |
| 9 | output projection + logical-batch crop | linear, no bias; repeated device A/B selected the post-projection crop |
| 10 | post-attention RMSNorm + residual add | distinct semantics; cannot use pre-add RMSNorm |
| 11 | pre-MLP RMSNorm | dedicated RMSNorm |
| 12 | gate linear + approximate GELU; up linear | GELU is in the explicit matmul program config, not a composite unary |
| 13 | gate/up multiply + down linear | no compatible adjacent fold |
| 14 | post-MLP RMSNorm | dedicated RMSNorm |
| 15 | residual add + layer scalar | scalar is a fused binary post-activation |

## Correctness and capacity

The acceptance bar remains PCC >= 0.995 on real cached checkpoint weights.

| Path | Sliding | Full |
|---|---:|---:|
| prefill, seq 32 | 0.999597 | 0.999586 |
| non-aligned prefill, seq 33 | 0.999301 | 0.999273 |
| prefill, seq 128 | 0.999265 | 0.999233 |
| traced decode, position 32 | 0.999629 | 0.999624 |
| batch-32 non-aligned prefill | 0.999335 | 0.999349 |
| batch-32 traced decode | 0.999410 | 0.999479 |
| advertised prefill control, seq 262144 | 0.998772 | 0.999089 |
| populated-cache decode parity, position 262143 | 0.998679 | 0.996631 |
| non-aligned prefill, seq 262113 | 0.998834 | 0.999089 |
| non-aligned populated-cache decode parity, position 262112 | 0.998888 | 0.998192 |
| distinct-token traced decode vs HF, position 262143 | 0.999380 | 0.998937 |

Each batch-1 traced decode test executes the same captured graph eight times
and requires bitwise-identical output. Page tables remain non-identity block
rotations. The hash-bound `standard_suite_final.log` records 23 passed and 9
explicitly gated candidate/long/performance tests skipped, including mutable
token/position trace replay and direct wrapped-window K/V ownership checks at
PCC 0.99989 or better. `long_context_262144_final.log` records both exact
context cases passing in 220.15 seconds total;
`long_nonaligned_262113_final.log` records both largest non-aligned cases
passing in 222.62 seconds total.

`exact_context_distinct_262144_final.log` closes the advertised-context decode
composition gate. It prefills 262143 periodic nonconstant tokens, captures
decode with a sentinel, replaces the stable traced input with a distinct final
token, and compares against the HF one-query oracle. Both layer kinds replayed
bitwise deterministically; correct-position RMSE beat the wrong-position
negative control (0.02148 vs 0.08729 sliding, 0.02017 vs 0.02247 full).

## Performance

The comparison uses the functional stage's exact workloads and profiler
method: batch 1, sequence 128 warmed prefill; complete traced warmed decode at
position 32. Values sum `Device Time` in each filtered `tt-perf-report` CSV.

| Kind | Mode | Functional | Fused | Change | Ops |
|---|---|---:|---:|---:|---:|
| sliding | prefill | 3.521 ms | 3.427 ms | -2.68% | 25 -> 26 |
| full | prefill | 4.254 ms | 4.192 ms | -1.46% | 25 -> 23 |
| sliding | traced decode | 2.577 ms | 2.560 ms | -0.65% | 43 -> 40 |
| full | traced decode | 2.911 ms | 2.881 ms | -1.05% | 43 -> 39 |

The sliding prefill op count increases by one because two bounded single-core
head-concat kernels plus slices/concat replace a much slower transpose and
reshape; latency, not op count, selected the result. Final decode reports have
no `CopyDeviceOperation` or standalone `UnaryDeviceOperation`. Full decode has
one `PagedFusedUpdateCacheDeviceOperation`; sliding retains two cache updates
to preserve circular-cache addressing.

Canonical artifacts are under `tracy/<kind>/<mode>/`: raw ops CSV,
`*_perf_report.csv`, and the report text. The signposts are
`FUSED_PERF_PREFILL[_END]` and `FUSED_PERF_DECODE[_END]`.

## Exhaustive pattern assessment

| Graph-fusing pattern | Result |
|---|---|
| dedicated activation | selected: matmul+GELU |
| softmax / spelled-out SDPA | already dedicated SDPA; no primitive subgraph remains |
| RMSNorm / distributed RMSNorm | already dedicated; distributed form is inapplicable on 1x1 |
| split/create/concatenate heads | selected where valid; full minimum tile rejected by exact P150 L1 limit |
| RoPE | already dedicated; Llama fused-QK op uses incompatible interleaved semantics |
| fused K/V cache update | selected for full; sliding op lacks required modulo, block-size, and head-count overrides |
| TopK | absent from a decoder layer |
| shared-LHS packing | QKV already packed; measured gate/up packing was slower |
| permute-reshape-permute | full/long head concat already reduced to the proven permute+reshape structural path |
| conv/BN/pad/pool/spatial mean | absent |
| bias/activation/scale folding | no biases; GELU and final scalar selected |
| transpose + matmul | weights are transposed once at load time; no runtime transpose operand remains |
| slice + matmul | repeated paired device A/B selected moving the logical-batch crop after projection |
| stable softmax / reduction rewrites | encapsulated by SDPA/RMSNorm; no exposed sequence exists |
| pre-add RMSNorm | rejected: Gemma computes `residual + norm(branch)`, not `norm(residual + branch)` |

Measured rejected candidates live under `candidates/`:

- merely passing `activation=` did not populate the program config and left a
  unary op: 3.466 ms sliding prefill;
- scalar/GELU folding before movement fusion: 2.564 ms sliding decode;
- packed gate/up: 2.593 ms and 42 ops versus final 2.560 ms / 40 ops;
- slice placement: the original one-sample ordering was refuted. A same-process
  12-sample-per-arm device A/B selected post-projection slicing by 2.512 us
  median, with all six ABBA-cycle means favoring it; PCC was 1.0 and both arms
  were deterministic;
- full head concat at the minimum 32-token tile requested 2208512 B single-core
  L1 versus 1572864 B available;
- direct sharded SDPA output is rejected by the exact TTNN GQA contract.
- rounded non-aligned sliding prefill tail: refuted by the wrapped-window
  regression (decode PCC 0.994710 and a live all-zero K slot); exact logical
  tail writes are selected and pass direct cache ownership PCC >= 0.999885.
- long-prefill fused GELU: all F4/F2/F1 real-M4096 configs were correct but
  66.5%-551.4% slower than the 11.196 ms gate+unary baseline. Adapted
  C2048/C1024/C128 families normalized to 18.725/18.993/20.618 ms per 4096
  rows, so the measured B0 path is retained.

## Runtime safety

The source audit covers every fused runtime helper and forbids `torch`,
`ttnn.from_torch`, `ttnn.to_torch`, and functional-forward fallback. Final
profiler windows contain no host conversion, tilize/untilize, or generic
reshard operations. Decode's I2S/S2I rows are required by the faster sharded
RMSNorm, GQA SDPA output contract, cache writers, and decode concat input.

The final hash-bound `TT_METAL_WATCHER=10` run passed all four mutable-input
trace cases (both layer kinds, random non-block-aligned and 1023-to-1024
boundary positions). `watcher_final/generated/watcher/watcher.log` contains
normal attach/check/detach records and no fatal, assert, invalid NOC, overflow,
sanitizer, or exception finding.

## Reproduction

```bash
export LD_LIBRARY_PATH=$PWD/build/lib:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/mpl
pytest -q models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py -s
GEMMA4_LONG_PREFILL=262144 pytest -q models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py -k fused_long_nonaligned_prefill_capacity -s
GEMMA4_LONG_PREFILL=262113 pytest -q models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py -k fused_long_nonaligned_prefill_capacity -s
GEMMA4_LONG_DECODE=262144 pytest -q models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py -k fused_exact_context_distinct_traced_decode -s
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=$PWD/models/autoports/google_gemma_4_31b/doc/fused_decoder/watcher_final pytest -q models/autoports/google_gemma_4_31b/tests/test_fused_decoder.py::test_fused_changed_trace_buffers_random_and_boundaries -s
```

Profiler runs use `GEMMA4_FUSED_PERF=1 python -m tracy -r -p -v
--output-folder <artifact-dir> -m pytest <single performance node> -s`, then
`tt-perf-report` filtered by the matching signposts.

## Limitations

This is only Stage 02 fused-decoder work. It does not begin optimized-decoder,
multichip, full-model, generator, serving, or vLLM work. Long prefill keeps the
multi-core head-layout path and measured B0 MLP GELU composition; every
source-admissible explicit 1D/block-height and adapted chunk family in the
AutoFix matrix regressed normalized gate latency. Repository search found the
same-family `models/demos/gemma4` implementation but no same-model,
same-stage, single-P150 optimized reference with comparable profiler evidence.
