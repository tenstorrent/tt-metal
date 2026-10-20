# `mhc_post_mix` -- fused exact fp32 mHC residual update (model-local `ttnn.generic_op`)

One program for the post-block of a GLM-5.3-Flash hyper-connection wrap (`tt/mhc.py::TtMHCFast._post`):

    out[t, j, c] = sum_i comb[t, i*n + j] * x[t, i, c] + post[t, j] * y[t, c]      streams   (n output streams)
    out[t, c]    = (1/n) sum_j out[t, j, c]                                          collapse  (the LM head's mean)

It replaces the composite post-block (block-diagonal matmul + `addcmul` + collapse masks, inexact at bf16 operand
precision through the FPU) with SFPU fp32 arithmetic on operands unpacked straight into the fp32 dest, so the result
matches the composite's exact fp32 chain to ~3e-8 relative rms (`tests/custom_ops/test_mhc_post_mix.py`).

## Contract

| tensor | shape (logical) | dtype / layout / memory | role |
|---|---|---|---|
| `x` | `[1,1,T,n*C]` | fp32 TILE interleaved | packed residual streams, stream i in columns `i*C:(i+1)*C` |
| `y` | `[1,1,T,C]` | fp32 TILE interleaved | sublayer output |
| `post` | `[T,n]` (padded `[T,32]`) | fp32 TILE interleaved | `H_post` from `mhc_split_sinkhorn` (any row view with `T` rows) |
| `comb` | `[T,n*n]` (padded `[T,32]`) | fp32 TILE interleaved | `H_res` (Sinkhorn), column `i*n+j = comb[i,j]` |
| `out` | `[1,1,T,n*C]` or `[1,1,T,C]` (collapse) | fp32 TILE interleaved | pre-allocated by the caller (`MhcPostMix.allocate_output`) |

`T % 32 == 0`, `C % 32 == 0`, `n*n + n + 4 <= 32`. Rows are independent (pad rows just produce garbage in their own
rows), so the model passes its padded `Tp`-row views directly. Inputs are not modified. Works on a `MeshDevice`
(same program on every device) and inside a trace (runtime args carry buffer addresses; all io tensors of a traced
call are allocated inside the trace or persistent).

## Program

Work unit = one (token-tile `r`, column-tile `c`) position, `R*Ct` of them dealt column-major over the compute grid
(`split_work`, the first `R*Ct % ncores` cores take one more). Per core: reader (NCRISC) → compute (TRISC) → writer
(BRISC), all CBs fp32 tiles:

| CB | tiles | content |
|---|---|---|
| `c_0` X | `2n` | the n stream tiles `x[r, i, c]` of a position (double buffered) |
| `c_1` Y | 2 | `y[r, c]` |
| `c_2` RAW | 2 | reader scratch: comb and post tiles of token-tile `r` |
| `c_3` COL | `n*n+n` | coefficient column tiles (coefficient k in column 0 of every row), rebuilt per token-tile |
| `c_4` BCAST | `n*n+n` | the same broadcast across 32 columns by the compute kernel (`unary_bcast<COL>`, 32-bit unpack-to-dest) |
| `c_16` OUT | `2*(n or 1)` | output tiles of a position |

Compile-time args: reader `[CB_X, CB_Y, CB_RAW, CB_COL, n, Ct] + TensorAccessorArgs(x, y, post, comb)`; writer
`[CB_OUT, n, Ct, collapse] + TensorAccessorArgs(out)`; compute `[CB_X, CB_Y, CB_COL, CB_BCAST, CB_OUT, n, Ct,
collapse, bits(1/n)]`. Runtime args per core: reader `[x, y, post, comb, start, count]`, writer `[out, start,
count]`, compute `[start, count]` (addresses from `buffer_address()`). Compute config: HiFi4, `fp32_dest_acc_en`,
`unpack_to_dest_mode = UnpackToDestFp32` for X, Y, COL, BCAST (required: the default path rounds fp32 operands to
19 bits in SrcA).

Per position the compute kernel copies the n stream tiles and y into the fp32 dest and, per output stream j,
accumulates `sum_i bcast(comb[i,j]) * x_i + bcast(post[j]) * y` with SFPU `mul_binary_tile` / `add_binary_tile`;
`collapse` accumulates all `n*n+n` products into one tile and scales by `1/n`.

## Measured (chip 1, layer-4 decode, Tracy)

`mhc_post_mix` 47-48 us at B=4 and B=32 (120 cores; ~1 position per core at B=32); the wrap's post-block goes from
8 ops to 2 (typecast + kernel) and the whole wrap from 22 ops / 228 us to 15 ops / 192 us. Doc:
`models/demos/glm53_flash/doc/mhc/README.md` ("Dedicated post-block kernel").
