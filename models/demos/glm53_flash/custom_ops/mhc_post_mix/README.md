# `mhc_post_mix` -- fused exact fp32 mHC residual update (model-local `ttnn.generic_op`)

One program for the post-block of a GLM-5.3-Flash hyper-connection wrap (`tt/mhc.py::TtMHCFast._post`):

    out[t, j, c] = sum_i comb[t, i*n + j] * x[t, i, c] + post[t, j] * y[t, c]      streams   (n output streams)
    out[t, c]    = (1/n) sum_j out[t, j, c]                                          collapse  (the LM head's mean)

It replaces the composite post-block (block-diagonal matmul + `addcmul` + collapse masks, inexact at bf16 operand
precision through the FPU) with SFPU fp32 arithmetic on operands unpacked straight into the fp32 dest, so the result
matches the composite's exact fp32 chain to ~3e-8 relative rms (`tests/custom_ops/test_mhc_post_mix.py`).

## Contract

Two kernel generations share the op (`MhcPostMix(n, version=1|2)`; `GLM53_MHC_POST_KERNEL=1|2`):

| tensor | shape (logical) | dtype / layout / memory | role |
|---|---|---|---|
| `x` | `[1,1,T,n*C]` | fp32 TILE interleaved | packed residual streams, stream i in columns `i*C:(i+1)*C` |
| `y` | `[1,1,T,C]` | fp32 (v1) or fp32 / bf16 (v2) TILE interleaved | sublayer output (v2 reads the bf16 output directly) |
| `post` | `[T,n]` (padded `[T,32]`) | fp32 TILE interleaved | `H_post` from `mhc_split_sinkhorn` (any row view with `T` rows); v2 also takes the lane layout `[T,32]` |
| `comb` | `[T,n*n]` (padded `[T,32]`) | fp32 TILE interleaved | `H_res` (Sinkhorn), column `i*n+j = comb[i,j]`; v2 also takes the lane layout `[T,n*32]` |
| `out` | `[1,1,T,n*C]` or `[1,1,T,C]` (collapse) | fp32 TILE interleaved | pre-allocated by the caller (`MhcPostMix.allocate_output`) |

`T % 32 == 0`, `C % 32 == 0`, `n*n + n + 4 <= 32` (v1), `n in {2, 4}` (v2). Rows are independent (pad rows just
produce garbage in their own rows), so the model passes its padded `Tp`-row views directly. Inputs are not
modified. Works on a `MeshDevice` (same program on every device) and inside a trace (runtime args carry buffer
addresses; all io tensors of a traced call are allocated inside the trace or persistent).

Lane layout (v2 coefficient tiles, what the fused pre-block will emit; raw tiles are expanded on device once per
token-tile otherwise): column group `m = (c // 16) * 2 + (c % 2)` of a 32-wide tile -- face `m // 2`, column
parity `m % 2`, i.e. the 32 datums one SFPU slot load covers (4 rows x the 8 same-parity columns of a face,
measured in `_logs/optmhc_probe_v2c.log`); the post tile holds `post[:, m]` in group `m`, tile `j` of comb holds
`comb[:, i*n+j]` in group `i`. `tests/custom_ops/test_mhc_post_mix.py::expand_x8` is the host reference.

Pre-mix mode (`MhcPostMix.h_mix(x, pre, out)`, v2 kernels with `HMODE`): `h[t, c] = sum_i pre[t, i] * x[t, i*C + c]`
-- the sublayer input before its norm, exact fp32; `pre [T,n]` raw (padded tile, expanded on device like a post
tile) or its lane layout `[T,32]`; `out [1,1,T,C]` fp32 (`allocate_h`). No y, one output tile per position.

## Program

Work unit = one (token-tile `r`, column-tile `c`) position, `R*Ct` of them dealt column-major over the compute grid
(`split_work`, the first `R*Ct % ncores` cores take one more). Per core: reader (NCRISC) -> compute (TRISC) ->
writer (BRISC).

### v1 (`kernels/{reader,writer,compute}_mhc_post_mix.cpp`)

| CB | tiles | content |
|---|---|---|
| `c_0` X | `2n` | the n stream tiles `x[r, i, c]` of a position (double buffered) |
| `c_1` Y | 2 | `y[r, c]` fp32 |
| `c_2` RAW | 2 | reader scratch: comb and post tiles of token-tile `r` |
| `c_3` COL | `n*n+n` | coefficient column tiles (coefficient k in column 0 of every row), rebuilt per token-tile |
| `c_4` BCAST | `n*n+n` | the same broadcast across 32 columns by the compute kernel (`unary_bcast<COL>`, 32-bit unpack-to-dest) |
| `c_16` OUT | `2*(n or 1)` | output tiles of a position |

Compile-time args: reader `[CB_X, CB_Y, CB_RAW, CB_COL, n, Ct] + TensorAccessorArgs(x, y, post, comb)`; writer
`[CB_OUT, n, Ct, collapse] + TensorAccessorArgs(out)`; compute `[CB_X, CB_Y, CB_COL, CB_BCAST, CB_OUT, n, Ct,
collapse, bits(1/n)]`. Runtime args per core: reader `[x, y, post, comb, start, count]`, writer `[out, start,
count]`, compute `[start, count]` (addresses from `buffer_address()`). Compute config: HiFi4, `fp32_dest_acc_en`,
`unpack_to_dest_mode = UnpackToDestFp32` for X, Y, COL, BCAST (required: the default path rounds fp32 operands to
19 bits in SrcA). Per position the compute kernel copies the n stream tiles and y into the fp32 dest and, per
output stream j, accumulates `sum_i bcast(comb[i,j]) * x_i + bcast(post[j]) * y` with SFPU `mul_binary_tile` /
`add_binary_tile`; `collapse` accumulates all `n*n+n` products into one tile and scales by `1/n`.

### v2 (`kernels/{reader,writer,compute}_mhc_post_mix_v2.cpp` + `kernels/mhc_sfpu_x8.h`)

| CB | tiles | content |
|---|---|---|
| `c_0` X | `2n` | the n stream tiles of a position (fp32) |
| `c_1` Y | 2 | `y[r, c]`, fp32 or bf16 (widened on unpack, exact) |
| `c_2` RAW | 2 | raw comb + post tiles of token-tile `r` (EXPAND mode only) |
| `c_3` COEF | `n+1` | lane-layout coefficient tiles of token-tile `r`: comb of streams 0..n-1, then post (built by the compute kernel from RAW, or read by the reader) |
| `c_16` OUT | `2*(n or 1)` | output tiles of a position |

Compile-time args: reader `[CB_X, CB_Y, CB_RAW, CB_COEF, n, Ct, expand] + TensorAccessorArgs(x, y, post, comb)`;
writer as v1; compute `[CB_X, CB_Y, CB_RAW, CB_COEF, CB_OUT, n, Ct, collapse, bits(1/n), expand, hmode]`
(reader gets `hmode` after `expand`). Runtime args as
v1. Compute config: HiFi4, `fp32_dest_acc_en`, `dst_full_sync_en` (the whole 8-slot fp32 dest per acquire),
`UnpackToDestFp32` for X, RAW, COEF (and Y when fp32); the compute kernel's own include dir is passed through
`compiler_include_paths`. Dest slots: `0..n-1` x_i, `n` y, `n+1` the comb tile of the current stream, `n+2` post,
`n+3` out. Per token-tile (EXPAND) the kernel copies the raw tiles in and runs `mhc_sfpu::expand_comb/expand_post`
(mask + subvector rotate-add broadcasts, two chains interleaved) into dest slots 2..n+2, packing them to COEF. Per
position it copies x_0..x_{n-1}, y and post into the resident slots once; per stream j it copies comb tile j and runs
`mhc_sfpu::stream_mix` (per band: 4 accumulators in LREGs, one coefficient slot load + 4 data loads + 4 FMAs per
term), packs `out`, and releases the dest WITHOUT clearing (`_llk_pack_dest_section_done_<..., false>`) so the
operands stay for the next stream; `collapse` keeps accumulating across streams and scales by `1/n` on the last.
