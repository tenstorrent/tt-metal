# `mhc_pre_block` -- fused exact fp32 mHC pre-block (model-local `ttnn.generic_op`, decode T <= 32)

One program on the whole compute grid for the pre-block of a hyper-connection wrap (`tt/mhc.py::TtMHCFast`):

    mixes = (x @ fn^T) * rsqrt(sum_c x^2 + eps*nC)      projection (sqrt(nC) folded into fn), exact fp32
    pre, post, comb = Sinkhorn(mixes)                    the mhc_split_sinkhorn kernel's tile sequence (one core)
    h = sum_i pre_i x_i                                   exact fp32
    hn = h * rsqrt(mean_c h^2 + rms_eps) * w              the sublayer input, bf16

## Contract

| tensor | shape | dtype / layout | role |
|---|---|---|---|
| `x` | `[1,1,32,n*C]` | fp32 TILE interleaved | packed residual streams, one padded token tile |
| `fn` (held by the op) | `[1,1,n*C,32]` | bf16 | `fn^T * sqrt(nC)`, zero-padded to 32 columns; tile `i*C/32 + c` pairs with stream tile `(i, c)` |
| `consts` | `[8,32,32]` | fp32 | the Sinkhorn constants (`build_consts`) |
| `w` (held by the op) | `[1,1,32,C]` | bf16 | the sublayer norm weight replicated down 32 rows |
| `hn` (out) | `[1,1,32,C]` | bf16 TILE interleaved | normalised sublayer input |
| `post` (out) | `[32,32]` | fp32 | `post` in the post-block kernel's lane layout (`MhcPostMix` v2 input) |
| `comb` (out) | `[32,n*32]` | fp32 | `comb` lane tiles, one per output stream |

`n = 4`, `C % 32 == 0`, `C/32` column tiles must fit the grid (128 on a 13x10 grid). Rows are independent. Outputs
pre-allocated by the caller (`allocate_outputs`). Trace-safe (semaphores are reset by their waiters).

## Precision

Everything after the projection is exact fp32 (SFPU adds / FMAs, `rsqrt` on the SFPU) and the bf16 output is the
only rounding of hn. The projection itself runs on the FPU (HiFi4, fp32 dest accumulation) and is NOT exact:
measured on chip 1 (`_logs/optmhc_probe_mm_prec` via `ttnn.matmul`), the FPU matmul on bf16 x bf16 inputs with fp32
accumulation is 3.2e-4 rel rms off float64 already at K = 32 (one tile pair), the same 3e-4 at K = 256 and 3.0e-4
at K = 16384 (random data), whether the operands arrive as bf16 or as bf16-exact fp32 -- the rounding is inside
the FPU's product/accumulate path, not in the operand unpack, so splitting x into bf16 parts (x = hi + lo + lo2,
which this kernel does) buys nothing beyond what bf16 inputs give. On real layer-4 streams the fused projection's
mixes land 5.9e-5 from float64 (the composite's own `ttnn.matmul` projection: 4.3e-5; both are the same
mechanism). An exact projection therefore needs SFPU FMAs (per core: its 8 stream tiles x 24 outputs through
row-replicated fn tiles, ~25 us at B=32 vs ~1 us on the FPU); it is the planned exact variant of this op.

`MhcPreBlock(..., exact=True)` (`GLM53_MHC_PRE_BLOCK=2`) replaces the FPU projection with `mhc_sfpu::proj_exact_tile`:
per stream tile the kernel reads three row-replicated fn tiles (band r of tile q = four copies of `fn^T * sqrt(nC)`
column `8q + r` over the tile's 32 features) and a constant lane-mask tile (band k = the lane-k mask); per band and
output m it forms the four slot products lane-wise, reduces the 8 lanes with a rotate-add tree and adds the kept
lane into column m of the partial (exact fp32 FMAs/adds; ~28 instructions per band and output).

## Program (one token tile; per core k = x*gy + y, column tile c = k < C/32)

| CB | tiles | content |
|---|---|---|
| X / FN / MASK | 4 (8) fp32 / 4 (8) bf16 (x3 row-replicated when exact) / 1 fp32 | the stream tiles `x[:, i, c]`, the projection tiles, the lane-mask tile (exact) |
| HI / LO / LO2 | 4 bf16 each | `x = hi + lo + lo2` exactly (two exact fp32 residuals) -> every product in the matmul is exact |
| ZERO | 1 | zero tile (dest zeroing; an idle core's contribution) |
| PART | 2 | the partial (mixes columns 0-23 + the rows' sum of squares in column 24) / a leader's column sum |
| GATHER1 / GATHER2 | gy / gx | a leader's / the coordinator's receive buffers (slot y / slot x) |
| PRE / RS2 | 1 | the pre lane tile / the rsqrt tile received by fan-out |
| W | 1 bf16 | the norm weight rows |
| CONSTS, MIXES, SPRE/SPOST/SCOMB, MA/MB/RS/RECIP/TMP | | the Sinkhorn on the coordinator |
| COEF_OUT | 6 | pre, post, comb_0..3 lane tiles (coordinator) |
| H / HSQ | 1 / 2 | this core's h tile / its rows' sums of h^2 (every column) |
| RS2OUT | 1 | the rsqrt tile (coordinator) |
| OUT | 2 bf16 | hn tile |

Phases: (A) every worker splits x, accumulates its 12 exact matmul tile-pairs plus the sum of squares (column
24) into one partial and sends it to its column leader (x, 0) (unicast NoC write into slot y + semaphore
ARRIVE1); (B) leaders sum (exact SFPU adds) and send to the coordinator (0, 0) (ARRIVE2); (C) the coordinator
sums, scales the rows by rsqrt(col 24 + eps*nC), packs mixes, runs the Sinkhorn, expands pre / post / comb into
lane tiles (`mhc_sfpu_x8.h`), writes post / comb to DRAM and sends the pre tile to every leader (SEM_PRE), whose
reader fans it out to the column; (D) every worker mixes h (SFPU FMAs), sends its rows' sums of h^2 up the same
tree; (E) the coordinator turns the total into rsqrt(total/C + rms_eps) and fans it out (SEM_RS2); every worker
writes `h * rs2 * w` as bf16. Compile-time args: the CB ids in the order of `MhcPreBlock.CB`, `n`, `gy`, `gx`,
the Sinkhorn iterations, eps, eps*nC, 1/C, rms_eps (compute); the reader / writer additionally carry the NoC
coordinate tables of the grid rows / columns and the TensorAccessorArgs. Runtime args: buffer addresses, worker /
leader / coordinator flags, the column tile, the NoC coordinates of the leader and the coordinator, the slots.
