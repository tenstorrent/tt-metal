# mHC hyper-connections without the data-movement composites (`tt/mhc.py::TtMHCFast`, 2026-10-20)

Design of record section 7 ("mHC composite"): the two hyper-connection wraps of a decoder layer cost ~220 ops
and 1.9-2.2 ms per layer in traced decode (`doc/mesh/tracy/README.md`), replicated on every chip. The composite
`TtMHCWrap` (`models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py`) extracts every per-stream coefficient as a
`[T,1]` column slice of the kernel's `[T,k]` output (untilize -> slice -> tilize, single core), mixes the
streams with one `addcmul` per (stream, coefficient) pair, and runs the `[T,16384] x [16384,24]` projection and
its sum of squares on one core. `TtMHCFast` keeps the arithmetic, the fp32 tensors and the fused
`mhc_split_sinkhorn` kernel and rebuilds every other stage as compute ops (matmul / broadcast elementwise /
reduce) on zero-copy views; the sublayer's input RMSNorm and the LM head's stream collapse are folded in.

## Results (chip 1 of the Blackhole Galaxy, layer 4 KDA+MoE, position 1024, bfp8 experts; logs `_logs/optmhc_*`)

Per-block device time of one traced decode step (Tracy, one replay, every op attributed to its module by the
labelled eager pass: `tests/perf_layer_report.py`, tables under `layer4_B{4,32}_{wrap,fast}/`). "wrap" =
the composite `TtMHCWrap` + the decoder's typecast + single-core `rms_norm`; "fast" = `TtMHCFast` with the
norm inside the pre-block.

| B | block | composite ops | composite us | fast ops | fast us |
|---|---|---|---|---|---|
| 4 | attention wrap (pre-block incl. input norm, post-block) | 110 (+1 norm, 46 us) | 935 (+46) | 22 | 228 |
| 4 | FFN wrap | 110 (+1 norm, 46 us) | 936 (+46) | 22 | 227 |
| 4 | both wraps + both norms | 222 | 1964 | 44 | 455 |
| 4 | whole layer step | 466 | 10630 | 288 | 9046 |
| 32 | attention wrap | 110 (+1 norm) | ~1112 (+46) [1] | 21 | 212 |
| 32 | FFN wrap | 110 (+1 norm) | ~1112 (+46) [1] | 21 | 213 |
| 32 | both wraps + both norms | 222 | 2317 [1] | 42 | 426 |
| 32 | whole layer step | 479 [1] | 22320 (wall) | 251 | 20550 |

[1] The composite path at B=32 cannot be post-processed by Tracy on this build (`process_ops_logs`:
"Device data missing: Op ... SliceDeviceOperation", a kernel-less column slice at T=32, with and without the
eager pass; `_logs/optmhc_layer4_tracy_wrap_b32*.log`). The composite B=32 numbers are the project's own
measurement of the same code at 32 users per chip (`doc/mesh/tracy/README.md`, layer 4, chip (0,0): mHC 2224 us
over 220 ops, RMSNorm x2 93 us, 479 ops/step), cross-checked here by the single-chip wall delta:
2317 - 426 = 1.89 ms predicted vs 1.91 ms measured (22.32 -> 20.41 ms/step).

Fast-path ops of one wrap at B=32 (device us, `layer4_B32_fast/summary.md`): K-split projection matmul
`b=32 x 32x512x32` 13 (32 cores), `x*x` 24, `sum` over W 10 (32 cores), 2 x `fast_reduce_nc` 4+4, rsqrt-add 5,
scale 7, `mhc_split_sinkhorn` 52 (1 core, one tile), `pre @ S_pre` 3, `* I_pre` 3, `P @ x` 7 (32 cores,
bf16 width-sharded out), sharded `rms_norm` 7 (32 cores), `sharded_to_interleaved` 2, `typecast` y 3,
coefficient masks + sums 4+2+3+2, `I * comb_c` 7, `A @ x` `128x128x4096` 19 (64 cores), `addcmul` 30 = 212 us;
the composite's single-core `32x16384x32` matmul (131), `Reduce` (115), the 19 x 3 slice/untilize/tilize
triplets, the 4 + 16 ternaries (16 us each) and the concat are gone. The Sinkhorn kernel is now half of the
wrap; the rest is bandwidth-bound multicore ops on the 2 MB stream tensor.

Traced decode step, wall (20 replays, no profiler; `GLM53_PERF_B`):

| users B | composite ms/step | fast ms/step | delta | fast tok/s per user / aggregate (single layer) |
|---|---|---|---|---|
| 1 | 8.04 | 6.57 | -1.47 ms (-18 %) | 152 / 152 |
| 4 | 10.47 | 8.97 | -1.50 ms (-14 %) | 111 / 446 |
| 32 | 22.32 | 20.41 | -1.91 ms (-9 %) | 49 / 1568 |

One wrap alone, traced (`tests/test_mhc_perf.py`, bf16 identity sublayer, no norm, 20 replays): T=32 1.120 ->
0.216 ms (5.2x, output PCC 0.9999998 vs the composite), T=1024 (prefill shape) 5.611 -> 5.155 ms.
Layer-4 eager prefill S=1024 (`GLM53_PERF_PHASE=prefill`, warmed wall, two back-to-back reruns each): composite 288.1 / 285.2 ms, fast 282.7 / 282.7 ms (a first fast run at 306.7 ms was host noise: the eager prefill wall is dispatch-bound).

Regression (`tests/test_decoder_layer.py` 5 passed, `tests/test_decode_trace.py` 5 passed, fast path, logs
`_logs/optmhc_regress_*.log`): layer 0 / 3 / 4 prefill S=256 vs HF 0.999844 / 0.999920 / 0.999966, traced decode
steps 0.99989-0.99993 / 0.99995-0.99998 / 0.99999 (identical digits to the composite runs in
`doc/functional_decoder/README.md`).


## The layout fact everything rests on

A packed `[1,1,T,n*C]` TILE tensor whose `T <= 32` rows fit one tile row stores its 512 tiles in column order,
which is also the tile order of the stream-major `[1,n,T,C]` tensor (stream `i` = tiles `[i*C/32, (i+1)*C/32)`),
of its `[(i,t), c]` matrix form `[n*T, C]` and of the K-split `[1,KS,T,n*C/KS]` tensor. All three are
`ttnn.experimental.view` reinterpretations of the same buffer (no op). The logical row count is widened to the
32-row tile with `ttnn.reshape(t, logical, padded)` (its explicit-padded-shape branch is a view) so every stage
runs whole tiles, and narrowed back the same way.

## Decode step (`T <= 32`), per wrap

| block | ops | what |
|---|---|---|
| pre (incl. the input norm) | 13 | `xk @ fn_k` K-split batched matmul (32 cores, `MatmulMultiCoreReuseProgramConfig`), `fast_reduce_nc` of the partials, `xk*xk`, `sum` over W (32 cores), `fast_reduce_nc`, `add`+rsqrt activation (`sqrt(nC)` folded into `fn`), scale, `mhc_split_sinkhorn`, `pre_tile @ S_pre`, `* I_pre` (the block-diagonal `P[t,(i,t')] = delta(t,t') pre[t,i]`), `P @ x[(i,t),c]` written bf16 width-sharded over 8x4 cores, `rms_norm` (`LayerNormShardedMultiCoreProgramConfig`), `sharded_to_interleaved` |
| post | 8 (+1 `where` at `T < 32`) | `typecast` y, `comb` tile `* mask` + `sum` over W (exact `[n,n,T,1]` columns), the same for `post`, `I_tile * comb_c` (the block-diagonal `A[(j,t),(i,t')] = delta(t,t') comb[t,i,j]`), `A @ x[(i,t),c]`, `addcmul(mix, y, post_c)` on the `[1,n,T,C]` view, which IS the packed output |
| post, collapse mode (last layer) | 8 | the mean masks fold `1/n sum_j` into the coefficient columns: `mean_j new_j = sum_i cbar_i x_i + pbar y` is the same matmul + `addcmul` with a `[T, n*T]` matrix; the LM head's 8-op collapse (4 slices + 3 adds + scale) disappears (`TtModelHead.collapse` passes a `[1,1,B,hidden]` through) |

No slice / untilize / tilize / concat / repeat / fill_pad in the decode path. The one data-movement op left is the
`sharded_to_interleaved` after the sharded norm (the sublayers take interleaved `[1,1,B,4096]` bf16; a sharded-in0
variant of their first matmuls would remove it).

Prefill (`T > 32`): the stream-major form costs one tile-block transpose in and one out per wrap
(`[T/32, n, 32, C] <-> [n, T/32, 32, C]`, `ttnn.transpose(0, 1)`), the projection splits over the `n` streams, `h`
is a broadcast multiply + `fast_reduce_nc`, the residual update is the exact fp32 broadcast `addcmul` chain over
the packed column slices (the block-diagonal matrices grow with T), the norm is the interleaved multicore one
(`T/32` tile rows).

## Numerics (vs float64, `_logs/optmhc_probe{2,3,4,5}.log`)

| op | rel rms error | note |
|---|---|---|
| fp32 `multiply` / `add` / `addcmul` (incl. broadcasts), `ttnn.sum` | 3e-8 | exact fp32 (SFPU) |
| HiFi4 fp32 matmul, operands fp32 | 8e-4 .. 1.2e-3 | rounds both operands to bf16 (exact, 2e-8, when both are bf16-exact; LoFi 2.8e-2, HiFi2 4e-3); the composite path's projection is this op |
| `fast_reduce_nc` fp32 | 4e-4 (fp32 config) / 1.3e-2 (default) | rounds the inputs to bf16 |
| `ttnn.sum(dim=1)` on the `[1,32,T,32]` partials | 6e-8 | exact but 15 us (and 218 us on `[1,4,T,C]`) |
| selector matmul coefficient extraction | 3.9e-4 | lossy -> the mask multiply + `sum` (exact) is used instead |

Consequences: `mixes` keeps the composite path's precision class (K-split + bf16-rounded partial sum: 1.5e-4
rel rms vs the composite); `h` carries 4e-4..3e-3 (the attention wrap's streams nearly cancel, so the relative
error grows there) and is bf16 for the sublayer anyway; the residual update's `A @ x` rounds `comb` and `x` to
bf16 (~1e-3 rel rms per wrap on the fp32 residual, where the composite chain is exact; the bf16 HF reference rounds
the same products). All stages are PCC >= 0.9999 vs the composite on real streams (`tests/test_mhc.py`). A bf16
hi/lo split of the operands makes the mixing matmul exact (3.9e-7, `_logs/optmhc_probe3.log`) at the price of
splitting the 2 MB residual (3 ops) per wrap; not taken.

Pad-row contract (`T < 32`): the block-diagonal contractions run over the pad rows too (`0 * NaN = NaN`), so the
input streams' pad rows must be finite (host uploads zero-pad, the embedding gathers pad-token rows), `y`'s pad
rows are zeroed by the `where` guard, and the output's pad rows are finite garbage (the contract already says
"unspecified"). `tests/test_mhc.py::test_mhc_fast_pad_rows`: finite garbage in the pad rows leaves the real rows
bit-identical.

## Exact residual update: `GLM53_MHC_EXACT` (2026-10-20, after the merge into `arg/glm53_mesh`)

The default residual update (`A @ x`, one HiFi4 matmul) rounds `comb` and the fp32 residual to bf16 (~1e-3 rel
rms per wrap); on the layer-3 mesh test it moved the known near-tie user from PCC 0.995003 to 0.994980 (median
users unchanged at 0.99994). `GLM53_MHC_EXACT=3` (or `1`; `TtMHCFast(exact=3)`) recovers the composite path's
fp32 precision: the HiFi4 matmul is exact for bf16-exact operands, so `A` (tiny) and the residual are each split
into 3 bf16 parts (hi/mid/lo: typecast, widen, subtract, twice = 7 ops on the 2 MB residual + 7 tiny) and the 6
products whose part indices sum to < 3 are stacked along K into ONE bf16 matmul `[128, 768] @ [768, 4096]` with
fp32 accumulation (two concats build the stacked operands). `GLM53_MHC_EXACT=2` is the 2-way split (3 products,
4 + 4 split ops). Measured on fp32 streams that are not bf16-exact (dump rows + 1e-3 relative noise), vs the
composite (`tests/test_mhc.py::test_mhc_exact_residual`, `_logs/optmhc_test_mhc_exact.log`):

| variant | residual update rel rms vs composite (T=4 / 32) | collapse mode | extra ops per wrap | device us per wrap (B=4 / B=32) |
|---|---|---|---|---|
| exact=0 (default) | 8.2e-4 / 8.2e-4 | 9.5e-4 / 9.4e-4 | 0 | 228 / 212 |
| exact=2 | 1.9e-6 / 2.0e-6 | 2.1e-6 / 2.5e-6 | +10 (32 per wrap) | 321 (+93) / not measured (chip handed to the 8x4 e2e) |
| exact=3 | 1.1e-7 / 1.1e-7 | 1.7e-7 / 1.7e-7 | +16 (38 / 37 per wrap) | 394 / 379 (+166 / +167) |

exact=3 per wrap at B=32 (`layer4_B32_exact3/summary.md`): the residual's 3-way split = 7 ops, 109 us (typecast
fp32->bf16 13, bf16->fp32 11, subtract 24, twice, + the last typecast 13), the `A` split 7 tiny ops ~15 us, the two
operand concats 2 + 32 us, the stacked `128 x 768 x 4096` bf16 matmul 27 us (vs 19 us for the default `128 x 128`),
i.e. +167 us and +16 ops per wrap, +0.33 ms per layer (both wraps; the default fast path is 0.43 ms per layer, the
composite 2.3 ms). Wall per step not measured for the exact variants (chip handed to the 8x4 e2e); the device-time
delta is the expectation (+0.33 ms per layer).

Error budget (`_logs/optmhc_probe6.log`, device == host emulation): with 2-way splits the bf16 rounding of the lo
parts dominates (`A` 2.4e-6, `x` 1.6e-6, the dropped lo*lo term ~1e-6), hence the 3-way split for the 1e-6 bar.
The h matmul (`P @ x`) stays inexact in every mode (the hidden is bf16 for the sublayer).

## Tests

| test | what |
|---|---|
| `tests/test_mhc.py::test_mhc_fast_vs_wrap[attn,ffn]` | project / hc_pre / hc_post / forward / forward+fused norm / collapse vs `TtMHCWrap` and vs float64 at T = 1, 2, 32, 256, 1024 on real layer-4 streams and parameters |
| `tests/test_mhc.py::test_mhc_fast_pad_rows` | the pad-row contract |
| `tests/test_mhc.py::test_decoder_collapse` | `FunctionalDecoder.decode_forward(collapse=True)` vs the head's collapse of the streams (layer 4, B=2) |
| `tests/test_mhc.py::test_mhc_exact_residual` | residual update / collapse of exact = 0, 2, 3 vs the composite on non-bf16-exact fp32 streams (bars 5e-3 / 1e-5 / 1e-6) |
| `tests/test_mhc_perf.py` | one wrap, traced, T=32 and T=1024, both implementations |
| `tests/test_perf_layer.py` (`GLM53_PERF_B`, `GLM53_PERF_EAGER=1`) + `tests/perf_layer_report.py` | layer-4 traced decode step, wall and per-module Tracy attribution |

Knobs: `GLM53_MHC_IMPL=wrap` restores the composite path (`build_mhc(impl=)`); `GLM53_MHC_EXACT=0|2|3` (above).
