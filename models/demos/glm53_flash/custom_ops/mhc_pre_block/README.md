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

`n = 4`, `C % 32 == 0`, `C/32` column tiles over the compute grid at up to two per core (128 tiles: the first 8
cores of a 12x10 grid take two, every core of a 13x10 grid takes one and the last two cores idle, contributing zero
tiles to the tree; `MhcPreBlock.plan_work`). Rows are independent. Outputs pre-allocated by the caller
(`allocate_outputs`). Trace-safe (semaphores are reset by their waiters and re-initialised by the dispatcher at every
launch; the op's construction uploads device tensors, so build it in the eager warm-up, never inside a capture).

Mesh (audit 2026-10-22, `_agents/mhc_a1_audit/AUDIT.md`): the program is identical on every chip. Every NoC target
is a translated worker coordinate from the device's own tables (`worker_core_from_logical_core`; `MeshDevice`
validates them equal across its chips, Blackhole translated coordinates do not depend on harvesting), every remote
address is a circular-buffer or semaphore address of the same program (one address on every core of the range),
the op-held tensors (`fn`, `w`, `mask`, `consts`) are mesh-replicated uploads (one address on every chip), and
the per-launch runtime args (buffer addresses) are re-derived by the generic-op framework on every program-cache
hit (the hash excludes their values; every `MhcPreBlock` instance shares one cached program per accessor
configuration). The grid and the tables are bound to the device object the op was built for (`__call__` asserts
the device id).

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

## Program (one token tile; per core k = x*gy + y, column tiles per `plan_work`)

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

## Roofline knobs (2026-10-26, `_agents/mhc_roofline/STATUS.md`; both default off = the program above)

| constructor | env (`tt/mhc.py`) | kernel define | what |
|---|---|---|---|
| `sinkhorn="fused"` | `GLM53_MHC_PRE_SINKHORN=fused` | `MHC_SINKHORN_FUSED=1` | the Sinkhorn as one dest pass per normalisation (`normalize_fused`: FPU matmul into dest, +eps, recip, m copied in, SFPU multiply, one pack) and one pass per coefficient head: 2*iters + 3 passes instead of ~130 LLK ops with a CB round trip each. The products are fp32 SFPU multiplies where the LLK sequence used 19-bit FPU operands: a bounded rounding change (logits gate before it becomes the default) |
| `split="extras"` | `GLM53_MHC_PRE_SPLIT=extras` | `MHC_MAX_EXTRA=2` | a two-column core projects only its first column; the second column's n stream tiles are projected as single "extra" tiles on single-column, non-leader cores (`plan_work_extras`, `delegate_units`): 12x10 critical path 10 -> 7 cost units; 13x10 / 8x8 unchanged. The h / norm phases keep both columns on the owner; runtime args `[count_proj, n_extra, i, c, ...]` after the reader's base args, `[count_proj, n_extra]` after the compute kernel's; the writer is untouched. FPU projection only |

Host tests: `tests/test_mhc_roofline_host.py` (planner coverage on 6 grids, the arg tails vs these sources, the fused
pass model bit-identical to the LLK sequence); device A/B: `tests/test_mhc_roofline_mesh.py`.
