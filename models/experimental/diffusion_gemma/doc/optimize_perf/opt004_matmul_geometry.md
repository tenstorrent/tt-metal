# OPT-004 — matmul program_config geometry

Status: provenance-only — its subject, the five token-gather-MoE matmuls in `tt/sparse_moe.py`, was
deleted 2026-07-29 (`7417bd7d69d`) together with the tuning flag and the bench scripts. What survives
and is still current is the **TTNN matmul op contract** below and the tuned-vs-auto result it produced.
Owns: the tuned-vs-auto matmul result and the reusable TTNN `program_config` contract.
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).

## The result

| gate matmul `[E,32,2816]@[2816,192]` | ms | effective weight BW |
|---|---:|---|
| auto-config | 4.17 | ~17 GB/s |
| tuned (`in0_block_w` 11/22) | **0.318** | **~235 GB/s ≈ 92% of the 256 GB/s roofline** |

Real-weights full MoE forward at 2 layers: **untuned 10.06 → tuned 2.90 ms/layer, 3.47x, at
PCC(tuned, untuned) = 0.99967.** The practical ~235 GB/s per-chip denominator is owned by the
[non-MoE roofline](nonmoe_roofline/README.md); the landed token-gather MoE result this fed is
provenance in [perf progress](perf_progress.md); why that whole path was then deleted is in
[winter borrow](winter_borrow_20260727.md).

**ROOT CAUSE of the untuned path:** a plain `ttnn.matmul` on a DRAM-interleaved 4D batched tensor with
no `core_grid` **never** auto-selects the batched reuse factory — it falls through
`create_simple_matmul_program_config` to a multicast or the naive `MatmulMultiCore` factory, which does
not block the K-reduction for weight reuse. Consistent with the measured ~46 GB/s effective (415 MB
bf16 / 8.9 ms, ~18% of the 256 GB/s roofline).

## The TTNN op contract (read from source — the reusable part)

`MatmulMultiCoreReuseProgramConfig` (batched reuse):

1. **`per_core_N` MUST equal `Nt`** — `TT_FATAL(N == per_core_N)` in the device op (~L1651). The N
   dimension is never split across cores in this factory.
2. `num_output_blocks_total = (B*Mt/per_core_M) * (Nt/per_core_N)` (factory L174), which with
   `per_core_N == Nt` reduces to `(B*Mt)/per_core_M`.
3. `batch_scale_factor = per_core_M/Mt` when `per_core_M > Mt` (factory L110-111) — a core then loops
   that many experts internally.
4. `split_work_to_cores(grid, units)` uses **exactly `units` cores when `units < grid_cores`**
   (`tt_metal/common/work_split.cpp:345-351`), else all grid cores. That is how 128 experts map to 128
   of Blackhole's 130 cores with zero cross-expert serialization.

`MatmulMultiCoreReuseMultiCastProgramConfig` (2D mcast): **M is parallelized over `grid.y` and N over
`grid.x`** (`llms.md:1497`, `matmul_multicore_reuse_mcast_2d_program_factory.cpp:188-198`), so
`per_core_M = ceil(Mt/gy)` and `per_core_N = ceil(Nt/gx)`; `ceil` is legal because 2D pads the last
block, `in0_block_w` must divide `Kt`, and subblocks need `h|per_core_M`, `w|per_core_N`, `h*w <= 8`.

**DEST-REGISTER BUDGET:** the module default compute-kernel config (HiFi2, `fp32_dest_acc_en=False`,
`packer_l1_acc=True`) gives 8 half-tiles, so every `out_subblock_h * out_subblock_w <= 8`.

**L1 CB model, batched reuse:** `in0_CB = per_core_M_per_batch * in0_block_w * 2`,
`in1_CB = per_core_N * in0_block_w * 2`, `out_CB = interm0_CB = per_core_M * per_core_N`, against
Blackhole's ~1.4 MB usable L1. **2D mcast:** `in0_CB = per_core_M * in0_block_w * 2`,
`in1_CB = per_core_N * in0_block_w * 2`, `out+interm = 2 * per_core_M * per_core_N`.

## Shape and platform facts

- **SHAPE TRAP:** `path_to_100tps.md` quotes `I=96` (3 tiles) — that is the **TP=8 Galaxy** case
  (704/8=88 → pad 96). On the **TP=4 QB2** target, `models/demos/gemma4/tt/experts/weights.py:71-92`
  gives 704/4=176 → pad **192 = 6 tiles**.
- QB2 per-chip compute grid is **13x10 = 130 Tensix cores** for P150, confirmed at
  `tests/ttnn/nightly/.../conv/test_conv2d.py:5253`.
- Expert weights load as `bfloat8_b` by default (`moe.py:26`, `weights.py:38`), so a bfp8 weight tile is
  ~1088 B against a bf16 tile's 2048 B, while activations/intermediates/outputs stay bf16.

## The chosen C=32 geometry, as the record

gate/up `per_core_M` 1, `per_core_N` 6, `in0_block_w` 22, subblock (1,6), 128 cores, 4 K-passes; down
`per_core_M` 1, `per_core_N` 88, `in0_block_w` 2, subblock (1,8), 128 cores, 3 K-passes; gather
`per_core_M` 13, `per_core_N` 7, `in0_block_w` 8, 130 cores; combine `per_core_M` 1, `per_core_N` 7,
`in0_block_w` 16, 104 cores. The combine matmul is narrow (`Mt=8` → `is_narrow_shape` true, W/H = 11 > 8)
so the op auto-picks a 1D config; the 2D config is legal and was compared against a 1D `mcast_in0`
candidate.

**TRAP — this whole L1 budget is computed at C=32 and is not general.** Production capacity moved to the
canvas length (256) on 2026-07-15, at which point the down matmul's per-core output block is 8x88 tiles
= **2.9 MB against ~1.4 MB usable L1, i.e. ILLEGAL rather than merely suboptimal** — see
[winter borrow](winter_borrow_20260727.md).

## Refuted here (one line each; full entries in the [refuted list](../REFUTED.md))

- gate+up fusion into one `[H,2I]` matmul: **0x** extra (0.62 ms fused vs 2 x 0.31 ms separate) — the
  batched matmul is weight-bound at M=32 (1 tile per expert).
- raising `in0_block_w` 11 → 22 → 44: a wash; the geometry already saturates DRAM.
- L1-sharded outputs for these matmuls: impossible — `gathered` and `down` are 128*32*2816 = 11.5 M
  elements ~ 23 MB each, far beyond L1, so they must stay DRAM-interleaved and the only win is
  program-config geometry.

## Signoff bar (reusable methodology)

Tuned MoE PCC >= 0.9997 vs dense and tuned-vs-untuned PCC >= 0.9999; every dominant matmul row in
`tt-perf-report` showing `in0_block_w > 2` and the expected core grid; and the final default run
reproducing the winning candidate. A large op-share headline must be checked against the
[sum-of-device-FW overlap trap](whole_gen_opprofile/README.md) before it is treated as a cost. Weight
bytes, not top-k, set the denoise floor — [work log](work_log.md).
