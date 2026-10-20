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

### Plan: the cheapest exact-enough residual update (host analysis while chip 1 is reserved, 2026-10-20)

Error model on real layer-4 streams (dump rows + 1e-3 relative noise = an fp32 residual) and real Sinkhorn
coefficients, torch emulation of the HiFi4 matmul (bf16-rounded operands, fp32 accumulate; the emulation matched
the device bit-for-bit in probe 6), `_logs/optmhc_host_model.log`, rel rms vs the exact chain (attn / ffn wrap):

| candidate | products | rel rms |
|---|---|---|
| S0 default: A 8-bit, x 8-bit | 1 | 1.55e-3 / 1.59e-3 |
| S1: x 16-bit (hi/lo), A 8-bit (`in0 = [A|A]`, `in1 = [x_hi; x_lo]`) | 2 | 1.20e-3 / 1.26e-3 |
| S2: A 16-bit, x 8-bit | 2 | 9.9e-4 / 9.8e-4 |
| S3: both 16-bit, drop lo*lo | 3 | 2.8e-6 / 2.8e-6 |
| S3b: both 16-bit, 4 products | 4 | 2.6e-6 / 2.5e-6 |
| S4: A 24-bit, x 16-bit | 5 | 1.6e-6 / 1.6e-6 |
| S5: both 24-bit (`exact=3` as measured) | 6 | 2.8e-9 / 2.7e-9 |

The Sinkhorn coefficients are not bf16-exact (max relative deviation 3.8e-3 / 3.7e-3), so (a) "split only x" is
S1 and stays at 1e-3: for <= 1e-5 BOTH operands need their low 8 bits as a separate bf16-exact tensor (S3). Each
operand therefore costs at least: typecast to bf16 (hi), subtract (lo; one op if binary_ng takes fp32 - bf16,
else typecast-widen + subtract), and one stacking op for the K-stacked `[x_hi; x_hi; x_lo]` (`ttnn.where` with a
`[3,1,1,1]` condition broadcasting two `[1,1,128,C]` tensors, one compute op instead of a concat; for `A` the
`[hi32, hi32, lo]` source stack feeds the existing mask extraction with a `[n,3,n,1,32]` mask, so the
block-diagonal `[128, 384]` `in0` costs no extra op). Lean S3 op list per wrap (replacing the default's 1 matmul):

| side | ops | expected device us (from the measured per-op times at B=32) |
|---|---|---|
| x: typecast -> bf16, subtract (mixed dtype) or widen + subtract, `where`-stack `[hi; hi; lo]` (6 MB fp32) | 3-4 | 13 + (11) + 24 + ~40 |
| A: typecast, (widen), subtract, `where`-stack on the `[32,32]` comb tile; mask extraction unchanged | 3-4 tiny | ~8 |
| matmul `128 x 384 x 4096` (in1 fp32 6 MB; or bf16 3 MB with one more typecast of x_lo) | 0 (+0) | 32 (vs 19) |
| total | **+6 (if the mixed-dtype subtract and the batch-broadcast `where` exist) else +8** | **~+100 us per wrap** (212 -> ~310 at B=32, i.e. +0.2 ms per layer) |

Measured so far: `exact=2` as committed (concat-based stacking) = +10 ops / +93 us (2.0e-6), `exact=3` = +16 /
+167 us (1.1e-7). The two `where`/mixed-dtype assumptions and the lean S3 numbers are to be verified when chip 1
returns. Conclusion: with the existing ttnn primitives, <= 1e-5 costs >= +6 ops / ~+90-100 us per wrap, because
the HiFi4 matmul rounds every fp32 operand to bf16 and no exact (SFPU) reduction over the stream dim exists
(`fast_reduce_nc` rounds, `ttnn.sum(dim=1)` transposes with 8x padding). A <= +3-op exact residual update needs
a dedicated fp32 kernel: one eltwise-style program computing `out[j, t, c] = sum_i comb[t,i,j] x[i,t,c] +
post[t,j] y[t,c]` per output tile from 4 x-tiles, 1 y-tile and the kernel's `[32,32]` coefficient tiles with
SFPU fp32 FMAs (the column-broadcast multiply binary_ng already implements), bandwidth-bound (~2.5 MB in, 2 MB
out, ~10-15 us): it replaces the coefficient extraction (4 ops), the `A` build, the matmul and the `addcmul`, so
the post-block becomes 2 ops (typecast y + the kernel) at exact fp32, ~70 us less than today's default; a
`collapse` flag covers the last layer. Estimated 2-3 days of kernel work.

## Fused exact fp32 post-block kernel: `ttnn.experimental.deepseek_prefill.mhc_post_mix` (2026-10-20)

`ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/mhc_post_mix/` (next to the Sinkhorn op; built in this
worktree: `./build_metal.sh --build-dir $WT/build_Release --cpm-source-cache .../wt_optkda/.cpmcache --release`
with `tt_metal/third_party/{tracy,umd,tt-cluster-descriptors}` symlinked to the main tree and `build/lib ->
build_Release/lib`; run with `PYTHONPATH=$WT/ttnn:$WT:$W TT_METAL_HOME=$WT TT_METAL_RUNTIME_ROOT=$WT`).

    out[t, j*C + c] = sum_i comb[t, i*n + j] * x[t, i*C + c] + post[t, j] * y[t, c]      (collapse: mean over j)

One program on the packed streams (any T multiple of 32; the work unit is one (token-tile r, column-tile c)
position = the n stream tiles + the y tile in, n output tiles out, split over the full grid):
- reader: per token-tile r, fetches the kernel's `comb` and `post` tiles and writes the n*n + n coefficient
  COLUMN tiles (coefficient k in column 0 of every row, zeros elsewhere: 32 word copies per coefficient; the
  set is zeroed once per core with `noc.async_write_zeros`); per position, the n x-tiles and the y tile.
- compute: phase A per token-tile: `unary_bcast<COL>` of each column tile into a resident broadcast CB (the
  32-bit column-broadcast datacopy unpacks straight into the fp32 dest: exact); phase B per position and
  output stream j: `copy_tile` (fp32, unpack-to-dest) of the broadcast coefficient and the stream tile,
  `mul_binary_tile` / `add_binary_tile` in the fp32 dest (SFPU, exact), the post * y term, one pack per
  output tile. Dest slots: 0 acc, 1 coefficient/product, 2 operand, 3 collapse total (4 fp32 slots per acquire).
  `collapse` sums the n stream results in dest and scales by 1/n before the single pack.
- `unpack_to_dest_mode = UnpackToDestFp32` for the four fp32 input CBs in the program factory is what makes
  both the copies and the broadcast datacopy exact (SrcA/SrcB are 19 bits): without it the op lands at
  4e-4 (`_logs/optmhc_k_unit.log` first run vs `_logs/optmhc_probe7.log`).
- rows are independent: pad rows of y / x only reach pad rows of the output (no `where` guard needed).

Precision (`tests/test_mhc_post_mix.py`, 6 passed): random fp32 inputs 4.2e-8 rel rms vs float64 (T=32/256,
both modes); real layer-4 streams (+1e-3 noise) with real Sinkhorn coefficients 5.2e-8 / 6.7e-8 vs float64 and
3.0e-8 / 3.4e-8 vs the composite `TtMHCWrap.hc_post` chain (streams / collapse).

Wiring: `GLM53_MHC_POST_KERNEL=1` (`TtMHCFast(post_kernel=True)`): `_post` becomes `typecast(y)` + the kernel
for every T (prefill included: no stream-major transpose, no column slices), collapse through the same op.

Measurements with the kernel (`GLM53_MHC_POST_KERNEL=1`, chip 1, worktree build, `GLM53_KDA_FUSED=0`; logs
`_logs/optmhc_k_*.log`):

| what | default fast path (matmul post-block) | fast path + kernel |
|---|---|---|
| post-block ops per wrap (decode) | 8 (+1 `where` at T<32) | 2 (typecast y + kernel) |
| post-block device us per wrap, B=4 / B=32 | ~75 / ~67 (coef 11, A 7, matmul 19, addcmul 30, typecast 3, +where at T<32) | 50 / 51 (typecast 3 + kernel 47 / 48 on 120 cores) |
| whole wrap ops / device us, B=4 | 22 / 228 | 15 / 192 |
| whole wrap ops / device us, B=32 | 21 / 212 | 15 / 193 |
| one wrap traced, T=32 (`test_mhc_perf.py`) | 0.216 ms | 0.199 ms |
| one wrap traced, T=1024 (prefill shape) | 5.155 ms | 2.789 ms (no stream-major transposes, no addcmul chain) |
| layer-4 step wall, B=4 (20 replays) | 7.85 ms | 7.78 ms |
| layer-4 step wall, B=32 | 20.03 ms | 20.01 ms |
| residual update vs the composite chain | ~1e-3 rel rms (bf16-rounded matmul operands) | 3e-8 (exact fp32) |

Per-block tables: `layer4_B{4,32}_kernel/summary.md`. The kernel at 47-48 us is above the ~15 us bandwidth
estimate: per position it issues 4 x (5 coefficient + 5 operand `copy_tile`, 5 SFPU multiplies, 4-5 adds) plus
the 20 broadcast datacopies + packs of phase A per token-tile, i.e. ~100 tile ops per position on one core;
obvious follow-ups are `dst_full_sync_en` (8 fp32 dest slots: keep the 4 stream tiles and y resident across the
4 output streams, halving the copies), reading y as bf16 (drops the typecast op) and splitting phase A across
cores with a multicast. The whole layer step at these shapes moved 288 -> 143 ops between the two Tracy runs
because of the merged tree's other optimizations (router/experts), not this change.

Tests with the kernel: `tests/test_mhc.py` 5 passed (hc_post vs composite exact, pad-row contract, decoder
collapse), `tests/test_decoder_layer.py` 5 passed, `tests/test_decode_trace.py` 5 passed (unchanged PCCs).


## Re-homed as a model-local generic op: `custom_ops/mhc_post_mix` (2026-10-20, user rule)

Per the "custom kernels: model-local generic ops ONLY" rule the C++ op is gone from this branch: the same three
kernels live in `models/demos/glm53_flash/custom_ops/mhc_post_mix/kernels/` and `custom_ops/mhc_post_mix/op.py`
(`MhcPostMix`) builds the `ttnn.ProgramDescriptor` (CBs, the three `KernelDescriptor`s with the same compile-time
args, per-core `RuntimeArgs` with `buffer_address()`es, `TensorAccessorArgs`, `ComputeConfigDescriptor` with
`fp32_dest_acc_en` and `unpack_to_dest_mode[cb] = UnpackToDestFp32` for the fp32 operand CBs -- the descriptor API
exposes it, sized to the build's 64 CB slots) and launches `ttnn.generic_op([x, y, post, comb, out], program)`
with the output pre-allocated by the caller (`MhcPostMix.allocate_output`). The program-descriptor hash excludes
runtime-arg values, so the cached program is reused across addresses (trace-safe; `test_post_mix_trace`). Runs
against the main-tree build (no rebuild): `TtMHCFast(post_kernel=True)` / `GLM53_MHC_POST_KERNEL=1`.
Contract and CB plan: `custom_ops/mhc_post_mix/README.md`. Validation (`tests/custom_ops/test_mhc_post_mix.py`,
chip 1): random T=32/256 4.2e-8, decode B=1/4/32 (padded 32-row tile) 4.2e-8..4.9e-8, trace replay 4.1e-8, real
layer-4 streams 5.2e-8 / 6.7e-8 vs float64 and 3.0e-8 / 3.4e-8 vs the composite chain -- the C++ op's numbers;
`tests/test_mhc.py` with the kernel path: hc_post fast vs wrap 1.5e-8..3.3e-8 (`_logs/optmhc_cop_*.log`).


## v2 post-block kernel: resident operands, lane-layout coefficients, SFPU FMAs (2026-10-20)

`custom_ops/mhc_post_mix/kernels/{reader,writer,compute}_mhc_post_mix_v2.cpp` + `kernels/mhc_sfpu_x8.h`
(`MhcPostMix(n, version=2)`, `GLM53_MHC_POST_KERNEL=2`). What changed against v1:
- `dst_full_sync_en`: the 8-slot fp32 dest holds x_0..x_3, y, the current stream's coefficient tile, the post
  tile and the output; after each output stream's pack the dest is released WITHOUT clearing
  (`_llk_pack_dest_section_done_<DST_SYNC_MODE, true, false>`), so the operands are copied once per position.
- y is read as the bf16 sublayer output (widened on unpack, exact): the typecast op is gone, the post-block is
  the single generic op.
- the per-row coefficient broadcast is done once per token-tile by hand-written SFPU code (mask lane k, three
  subvector rotate-adds, two chains interleaved) into "lane-layout" tiles -- column group m = face m/2, column
  parity m%2, the 32 datums one SFPU slot load covers -- so the per-position mix is plain slot loads + FMAs
  (`mhc_sfpu::stream_mix`: per band 5 coefficient loads, 20 data loads, 20 SFPMADs, 4 stores; accumulators in
  LREGs). The measured Blackhole dest geometry (a slot = 4 rows x the 8 same-parity columns of a face) differs
  from the "columns 0-7 / 8-15" wording of tt-llk's `ckernel_sfpu_binary_bcast.h`; `_logs/optmhc_probe_v2c.log`
  is the lane map that settled it, `tests/custom_ops/test_mhc_post_mix.py::expand_x8` the host reference.
  The lane layout is also the natural output of the fused pre-block (the Sinkhorn coefficients land in it
  directly, no expansion in the post kernel).
- phase A (20 FPU column-broadcast datacopies per core) and the reader's column-tile building are gone.

Precision (`tests/custom_ops/test_mhc_post_mix.py`, 32 passed): streams 3.2e-8, collapse 6.2e-8 vs float64;
real layer-4 streams 3.0e-8 / 1.1e-7 vs the composite chain (streams / collapse); identical with raw
coefficients (expanded on device), pre-expanded lane-layout tiles, fp32 or bf16 y, and under trace replay.

| what (chip 1, main-tree build, `GLM53_KDA_FUSED=0`) | v1 custom op | v2 custom op |
|---|---|---|
| post-block ops per wrap | 2 (typecast + kernel) | 1 |
| post-block device us per wrap, B=4 | 50 (3 + 47) | 32 |
| whole wrap ops / device us, B=4 | 15 / 192 | 14 / 173 |
| one wrap traced, T=32 / T=1024 (`test_mhc_perf.py`) | 0.199 / 2.789 ms | 0.178 / 2.549 ms |
| layer-4 step wall, B=4 / B=32 | 7.78 / 20.00 ms | 7.75 / 19.95 ms |

(B=32 Tracy: the post-processor dropped an op in this run; the B=4 per-op table is `layer4_B4_cop2/summary.md`.)
The remaining 32 us at B=4 are ~1 position per core: dispatch, the 4 + 1 + 1 tile reads, the per-token-tile
expansion on every core (~2 us) and the 4 x (copy + 360-instruction mix + pack); the fused pre-block will hand
over lane-layout coefficients and the mix is then the floor.


## Accuracy gate: the prefill path is now the composite's, op for op (2026-10-20)

The 22-layer gate on the 2x4 block (test_mesh_model -k chain, 128-token prompt) put the composite at chain PCC
0.965 vs HF with the right first token and the fast path at 0.903 with a wrong one; its decode chain is in the
chaos regime for every variant, so the prefill (T > 32) path is what mattered. `tests/test_mhc_stages.py` isolates
it per stage on real layer-4 streams (float64 rel rms; `_logs/optmhc_stages_{old,exact}.md` have T=4 and T=128 for
both wraps). The old T > 32 path (stream-major n-split projection + `fast_reduce_nc`, broadcast multiply +
`fast_reduce_nc` for h) vs the composite, hc_attn T=128:

### hc_attn, T=128: rel rms (float64)
| stage | fast vs wrap | wrap vs f64 | fast vs f64 |
|---|---|---|---|
| mixes (Sinkhorn input) | 3.20e-04 | 4.33e-05 | 2.99e-04 |
| pre | 7.26e-04 | 3.59e-03 | 3.92e-03 |
| post | 1.00e-03 | 2.97e-03 | 3.49e-03 |
| comb | 1.14e-03 | 2.48e-03 | 2.48e-03 |
| h = sum_i pre_i x_i | 1.87e-03 | 6.98e-03 | 7.71e-03 |
| h vs f64 on own pre | nan | 4.34e-08 | 3.94e-04 |
| post-block (same coefficients) | 2.16e-08 | 4.97e-08 | 4.98e-08 |
| normed hidden (bf16 sublayer input) | 2.82e-03 | 3.57e-03 | 3.68e-03 |
| wrap output (streams), same sublayer | 2.99e-03 | 4.77e-08 | nan |

With `GLM53_MHC_PREFILL_EXACT=1` (default) `_project` calls the composite's `_project` (same `fn_T`, same
matmul + sum of squares + rsqrt scale) and `_collapse` the composite's `_mix` (multiply + addcmul over stream
slices) for T > 32, then the same typecast + rms_norm; the post-block stays the exact custom op:

### hc_attn, T=128: rel rms (float64)
| stage | fast vs wrap | wrap vs f64 | fast vs f64 |
|---|---|---|---|
| mixes (Sinkhorn input) | 0.00e+00 | 4.33e-05 | 4.33e-05 |
| pre | 0.00e+00 | 3.59e-03 | 3.59e-03 |
| post | 0.00e+00 | 2.97e-03 | 2.97e-03 |
| comb | 0.00e+00 | 2.48e-03 | 2.48e-03 |
| h = sum_i pre_i x_i | 0.00e+00 | 6.98e-03 | 6.98e-03 |
| h vs f64 on own pre | nan | 4.34e-08 | 4.34e-08 |
| post-block (same coefficients) | 2.16e-08 | 4.97e-08 | 4.98e-08 |
| normed hidden (bf16 sublayer input) | 0.00e+00 | 3.57e-03 | 3.57e-03 |
| wrap output (streams), same sublayer | 5.03e-08 | 4.77e-08 | nan |

Two facts behind the gate bar: the composite's fp32 projection is itself 4-6e-5 off float64 (operand rounding
in the HiFi4 matmul) and the shared Sinkhorn kernel is 5e-4..3.6e-3 off the float64 HF math (pre / post / comb,
both implementations); an exact custom pre-block lands ~1e-7 from float64 and therefore ~4e-5 from the
composite, so exact kernels are measured against float64 (<= 1e-6) with the 22-layer chain as the arbiter.
The decode path (T <= 32) is made exact the same way behind `GLM53_MHC_DECODE_EXACT=1` (default): the K-split
partials are summed with `ttnn.sum` (SFPU, exact) instead of `fast_reduce_nc`, and h = sum_i pre_i x_i runs
through the post-mix kernel's pre-mix mode (`MhcPostMix.h_mix`, HMODE: the raw pre tile expanded like a post
tile, `mhc_sfpu::pre_mix`) followed by the composite's typecast + rms_norm. Cost of the exact decode path (chip 1, B=4, `layer4_B4_exact/`):
the wrap goes 14 ops / 173 us -> 17 ops / 237 us (`ttnn.sum` over the batch dim expands into transpose + reduce
+ fill_pad triplets, the interleaved rms_norm is 47 us on one core vs the 8 us sharded one, the pre-mix kernel
18 us vs the 8 + 3 us matmul + typecast); walls 7.88 / 20.12 ms (B=4 / B=32), one wrap traced 0.198 ms at T=32
and 2.580 ms at T=1024 (the composite-identical prefill projection and mix cost nothing measurable against the
old 2.549). All of the decode pre-block's 16 ops are what the fused pre-block custom op (`custom_ops/mhc_pre_block`)
replaces with one program.


## Fused pre-block custom op: `custom_ops/mhc_pre_block` (2026-10-20)

One `ttnn.generic_op` program on the whole 12x10 grid for a decode wrap's pre-block (T <= 32, with the fused
norm; `GLM53_MHC_PRE_BLOCK=1`, needs `GLM53_MHC_POST_KERNEL=2`): projection, Sinkhorn, pre-mix and the sublayer
RMSNorm, with `post` / `comb` coming out in the post kernel's lane layout (no expansion in the post kernel). Core
k owns column tile(s) of the C-wide streams (128 tiles over 120 cores: 8 cores take two); per core: the 4 (8)
stream tiles are split into three bf16 parts (x = hi + lo + lo2 exactly), 12 (24) FPU matmul tile-pairs
accumulate the partial projection in fp32 and an SFPU routine adds the rows' sum of squares into column 24 of the
same tile; the partials go up a two-level tree of exact SFPU adds (column leader (x, 0) -> coordinator (0, 0)) by
unicast NoC writes into per-sender slots plus a counting semaphore; the coordinator scales the rows by
rsqrt(col 24 + eps*nC) (a condition-code lane extract, since the other lanes of the statistics copy hold the
rsqrt of negative numbers), runs the Sinkhorn kernel's tile sequence on one core, expands pre / post / comb into
lane-layout tiles (`mhc_sfpu_x8.h`), writes post / comb to DRAM and fans the pre tile out to the leaders, whose
readers fan it out to their columns; every core mixes h = sum_i pre_i x_i with SFPU FMAs, sends its rows' sums
of h^2 up the same tree, receives rsqrt(total/C + eps) the same way and writes h * rs * w as bf16. Semaphores
are reset by their waiters (trace replay); the dataflow kernels own the sync, the compute kernel the math.
Contract, CB plan and the measured precision facts: `custom_ops/mhc_pre_block/README.md`.

Two facts found on the way: (1) the packer's output format follows the CB it last packed into -- a kernel that
packs into both bf16 and fp32 CBs must `pack_reconfig_data_format` per target (the first fp32 partials came out
as bf16 bit patterns); (2) the FPU matmul is not exact even on bf16 x bf16 inputs (3.2e-4 rel rms vs float64 at
K = 32, `ttnn.matmul` HiFi4 fp32-acc), so the 3-way split does not make the projection exact: the fused mixes
land 5.9e-5 from float64, the composite's own projection 4.3e-5 (same mechanism); an exact projection is SFPU
FMAs (~25 us per core at B=32), the planned exact variant.

Precision (`tests/custom_ops/test_mhc_pre_block.py`, real layer-4 parameters, B = 1/4/32 and random streams,
trace replay): post / comb lane tiles 3.2e-4 / 3.2e-4 vs the standalone Sinkhorn kernel on the float64 mixes
(the FPU projection's rounding through the Sinkhorn; 0 at B=1) and 1-4e-3 vs the float64 HF math (the Sinkhorn
kernel's own precision); hn bit-identical (B=1/4; 1.9e-7 at B=32, 4e-5 random) to the bf16 rounding of the
float64 norm of h built from the kernel's own pre -- the mix and the norm are exact.

Model path (chip 1, `GLM53_MHC_PRE_BLOCK=1 GLM53_MHC_POST_KERNEL=2`, `layer4_B4_preblock/`): a wrap is 2 ops
(pre-block 101 us + post-block 33 us = 134 us at B=4; the exact decode path was 17 ops / 237 us, the inexact
one 14 / 173, the composite 110 / 935 + 46), the traced layer-4 decode window 117 ops / 7.80 ms device;
`tests/test_mhc.py` 5 passed, the T=4 stage table: the fused op's sublayer input differs from the composite's
by bf16-ulp flips (3.0e-3 rel rms, the composite is 3.5e-3 from float64 itself) driven by the FPU-class mixes.
The 101 us are dominated by the single-core Sinkhorn sequence (52 us standalone) and the two sync rounds; the
projection itself is ~1 us (FPU) -- next: the Sinkhorn on dest-resident tiles, overlapping phase A with the
coordinator's work, and the exact SFPU projection variant.


## Tests

| test | what |
|---|---|
| `tests/test_mhc.py::test_mhc_fast_vs_wrap[attn,ffn]` | project / hc_pre / hc_post / forward / forward+fused norm / collapse vs `TtMHCWrap` and vs float64 at T = 1, 2, 32, 256, 1024 on real layer-4 streams and parameters |
| `tests/test_mhc.py::test_mhc_fast_pad_rows` | the pad-row contract |
| `tests/test_mhc.py::test_decoder_collapse` | `FunctionalDecoder.decode_forward(collapse=True)` vs the head's collapse of the streams (layer 4, B=2) |
| `tests/custom_ops/test_mhc_post_mix.py` | the fused post-block custom op vs float64 and vs the composite chain (random T=32/256, decode B=1/4/32, trace replay, real streams; both modes; mesh cases behind `GLM53_MESH_TESTS=1`) |
| `tests/custom_ops/test_mhc_pre_block.py` | the fused pre-block custom op vs float64 and vs the standalone Sinkhorn kernel (real parameters, B=1/4/32, random, trace replay) |
| `tests/test_mhc_stages.py` | per-stage fast vs composite vs float64 table at T=4 and T=128 (the gate shapes), both wraps |
| `tests/test_mhc.py::test_mhc_exact_residual` | residual update / collapse of exact = 0, 2, 3 vs the composite on non-bf16-exact fp32 streams (bars 5e-3 / 1e-5 / 1e-6) |
| `tests/test_mhc_perf.py` | one wrap, traced, T=32 and T=1024, both implementations |
| `tests/test_perf_layer.py` (`GLM53_PERF_B`, `GLM53_PERF_EAGER=1`) + `tests/perf_layer_report.py` | layer-4 traced decode step, wall and per-module Tracy attribution |

Knobs: `GLM53_MHC_IMPL=wrap` restores the composite path (`build_mhc(impl=)`); `GLM53_MHC_EXACT=0|2|3` (above).

## Prefill dump A/B: composite wrap vs fast exact path (2026-10-20, rows 4-5 block)

22 layers, 4 users/row, S=128, `GLM53_MHC_DUMP` of every wrap (in / h = sublayer input / y = sublayer output / out),
`tests/mhc_dump_diff.py` (relative rms, `|a-b| / |a|`). Fast path = `GLM53_MHC_IMPL=fast GLM53_MHC_POST_KERNEL=2
GLM53_MHC_PREFILL_EXACT=1 GLM53_MHC_DECODE_EXACT=1`.

| | composite wrap | fast exact |
|---|---|---|
| TTFT warm (22 L, S=128, 2x4) | 417 ms | 294 ms |
| first wrap with rel rms > 1e-6 | L01_attn `h` 2.1e-5 (L00 both wraps exact to 7e-8) | |
| rel rms at L05 / L10 / L15 / L21 (out) | 1.8e-2 / 4.1e-2 / 9.2e-2 / 2.8e-1 | |

Reading: the two pre-blocks differ at the bf16 rounding level once (2e-5 at the L01 attention input); the
sublayers amplify it (L01 attention 2.6e-3, the MoE top-k flips from L01_ffn on), so by layer 21 the two
runs are 28% apart in rel rms. The A/B cannot say which path is closer to HF (both bf16 pipelines sit
inside the model's own sensitivity); the 45-layer token-level gate on the 8x4 is the arbiter (the fast path
flipped the first token before the exact prefill path; with it the token matched). Decision: the 8x4 chain
runs the fast exact path and the gate decides; the composite wrap stays the default until the gate passes.
