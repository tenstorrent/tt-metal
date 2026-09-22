# MiniMax-H3 AGMM on the Wormhole galaxy: fused vs unfused, per-step blocking sweeps

Status: complete 2026-09-22. 39 device sessions, 10,571 combos measured OK (43 rejected by the op at program
creation, all L1), every one above PCC 0.99998 against the fp32 reference.

## Question

The three transformer-block projections that gather the K-sharded activation over the 4-device TP ring run as one
fused op, `ttnn.experimental.all_gather_minimal_matmul_async` (`models/tt_dit/layers/linear.py:436-462`). The model
also has an unfused branch, `all_gather_persistent_buffer` then `minimal_matmul` / `minimal_matmul_split` /
`dit_minimal_matmul_addcmul_fused` on the full 8x9 grid (`linear.py:469-522`), which today is reached only when the
activation arrives pre-gathered. Per production shape, what is the fastest fused blocking, and how does the best
unfused pair (all-gather swept on its own hyperparameters, matmul swept on its own blocking) compare?

Shapes are the per-device shapes of the block (`models/tt_dit/tests/models/minimax_h3/tools/minimax_h3_ops.py`):

| op | K (gathered) | K per device | N per device | epilogue | rows per device swept |
|---|---|---|---|---|---|
| to_qkv | 5376 | 1344 | 5376 | 3 output chunks | 4736, 9184, 13664 |
| to_out | 7168 | 1792 | 1344 | addcmul, scalar 1.0 | 4736, 9184, 13664 |
| ff1 | 5376 | 1344 | 7168 (gate and up packed), 3584 out | SwiGLU | 4736, 9184, 13664 |

## Harness

Everything runs from the AGMM unit test, `models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py`.

**Device row.** `wh4x8links4_ring` (`test_all_gather_minimal_matmul_async.py:557-572`) matches what the model runs on
the WH galaxy: (4, 8) mesh, `FABRIC_1D_RING` with a 4096 B payload, Ring, 4 links, 2 workers per link, 8x8 worker
grid, TP on mesh axis 0, 48 buffers per channel, `force_transpose=True`. Sources on the model side:
`models/tt_dit/tests/models/minimax_h3/common.py:132-134,186-191`,
`models/tt_dit/pipelines/minimax_h3/pipeline_minimax_h3.py:208-210`, `models/tt_dit/utils/matmul.py:744-748`.
The test runs on a 4x1 submesh, so M is both the per-device and the global row count.

**Changes made for this study** (same file):
- The separate path's `all_gather_async` hyperparameters became arguments (`ag_num_workers_per_link`,
  `ag_chunks_per_sync`, `ag_num_buffers_per_channel`); the old hardcoded 4 / 16 / 2 stay the defaults. The model's
  unfused gather uses 3 / 16 / 2 (`models/tt_dit/parallel/manager.py:933-945`).
- The separate path's `minimal_matmul` passes `fuse_swiglu`, so ff1 runs unfused.
- The separate path's `dit_minimal_matmul_addcmul_fused` call dropped a `fused_activation` argument the binding does
  not accept (`ttnn/cpp/ttnn/operations/experimental/transformer/dit_minimal_matmul_addcmul_fused/dit_minimal_matmul_addcmul_fused_nanobind.cpp:150-160`).
- Host tensors are uploaded as bf16 when the device dtype is bf16 (fp32 tilize of a 13664-row activation takes
  minutes on this host; the values round identically).
- PCC rows for the H3 shapes: `h3_15s_qkv`, `h3_15s_to_out` in `test_linear`; `h3_15s_ff1` in `test_linear_swiglu`,
  which also gained the Wormhole row and a fused/separate axis.
- `test_h3_agmm_sweep`: one device session per (op, M, mode). Modes: `fused`, `ag`, `mm_ring` (the standalone matmul
  on the AGMM worker grid, 8x8 here) and `mm_full` (on the model's full matmul grid, 8x9 here); the grids and the
  blocking the model would run are resolved from the device at run time. Every combo is
  warmed up and PCC-checked against a torch reference computed once, then the valid combos are trace-captured and
  executed one at a time between Tracy signposts. Combos come from the sweep harness' candidate rules
  (`models/tt_dit/utils/sweep_mm_block_sizes.py:629-804`: ring-safe K_block, even N for SwiGLU, L1 pre-filter);
  the unfused matmul caps K_block at 16 because its candidates are divisors of the full K.
  The all-gather sweeps workers per link {1, 2, 3, 4} x chunks per sync {4, 8, 16, 32} x buffers per channel
  {2, 4, 8} at 4 links. The unfused matmul runs on an activation gathered once with the production 3 / 16 / 2.

**Orchestrator and report:** `models/tt_dit/tests/models/minimax_h3/tools/agmm_unit_sweep.py`. `run` executes the
pytest cases under the Tracy device profiler and joins the ops log with the test's JSON sidecar; `report` prints
the tables below. Durations are device kernel time, mean over the 4 ring devices, one traced execution per combo,
the same statistic as `sweep_mm_block_sizes.py`, so numbers compare directly with the earlier per-op docs.

```bash
cd /home/jameslee/tt-metal && source python_env/bin/activate
TOOL=models/tt_dit/tests/models/minimax_h3/tools/agmm_unit_sweep.py

# smoke: one shipped combo
python $TOOL run --ops to_out --M 13664 --modes fused --combos '[[8,8,6,2,2]]'

# the study, one device session per case (about 20 s per combo at M=13664)
python $TOOL run --ops to_qkv,to_out,ff1 --M 13664 --modes fused,ag,mm_full,mm_ring
python $TOOL run --ops to_qkv,to_out,ff1 --M 9184 4736 --modes fused,ag,mm_full,mm_ring

# tables
python $TOOL report --top 10 --out /tmp/agmm_h3_report.md

# PCC gate for the shipped blockings, fused and separate
python -m pytest models/tt_dit/tests/models/wan2_2/test_all_gather_minimal_matmul_async.py \
  -k "wh4x8links4_ring and h3_15s and check and no_bias"
```

Results land in `agmm_h3_sweep_results.csv` (gitignored) and the profiler logs under
`generated/profiler/reports/agmm_h3_sweep_<row>_<op>_M<rows>_<mode>/`.

## Results

Device kernel time in us, mean over the 4 ring devices, one traced execution per combo; every row below passed PCC
> 0.99998 against the fp32 torch reference. Full tables: `agmm_unit_sweep.py report`.

### M = 13664 rows per device (15 s clip)

| op | best fused (blocking) | best all-gather (workers, chunks/sync, buffers) | best matmul 8x9 | best matmul 8x8 | AG + MM 8x9 | fused vs unfused |
|---|---|---|---|---|---|---|
| to_qkv | 10261 (8, 7, 12) | 1906 (3, 32, 4) | 9605 (12, 7, 8) | 10270 (14, 7, 6) | 11511 | fused 1250 us faster (-10.9%) |
| ff1 | 15156 (8, 6, 10) | 1911 (3, 32, 4) | 14290 (8, 8, 10) | 14872 (14, 8, 6) | 16201 | fused 1045 us faster (-6.4%) |
| to_out | 5065 (12, 8, 6) | 2542 (4, 32, 4) | 4064 (12, 8, 6) | 4099 (12, 8, 6) | 6606 | fused 1541 us faster (-23.3%) |

Fused blockings (8x8): to_qkv 425 combos, to_out 320, ff1 320, all OK.

| op | rank 1 | rank 2 | rank 3 | shipped |
|---|---|---|---|---|
| to_qkv | (8, 7, 12) 10261 | (8, 6, 12) 10271 | (8, 7, 8) 10593 | (8, 7, 12), rank 1 |
| to_out | (12, 8, 6) 5065 | (9, 8, 6, 1, 3) 5185 | (12, 7, 6) 5188 | (8, 8, 6) 5214, rank 5, +2.9% |
| ff1 | (8, 6, 10) 15156 | (8, 7, 10) 15164 | (10, 7, 10) 15477 | (8, 7, 10), rank 2, +0.1% |

All-gather hyperparameters (4 links, 48 points per tensor): the K=5376 tensor (to_qkv, ff1) is fastest at
(3, 32, 4) 1906 us; the K=7168 tensor (to_out) at (4, 32, 4) 2542 us. The production setting (3, 16, 2)
ranks 8th to 12th, 3.4 to 3.9% behind. 32 chunks per sync beats 16 everywhere; 3 vs 4 workers and the buffer
count are within 1%.

Unfused matmul on the gathered activation:

| op | grid | rank 1 | model's blocking on that grid | note |
|---|---|---|---|---|
| to_qkv | 8x9 | (12, 7, 8) 9605 | (8, 7, 12) 9822, rank 6, +2.3% | 446 of 450 OK, 4 L1 rejections |
| to_qkv | 8x8 | (14, 7, 6) 10270 | fused shipped (8, 7, 12) 10547, rank 15 | 591 of 597 OK |
| ff1 | 8x9 | (8, 8, 10) 14290 | (8, 3, 14) 15044, rank 18, +5.3% | 450 of 450 OK |
| ff1 | 8x8 | (14, 8, 6) 14872 | fused shipped (8, 7, 10) 15051, rank 7 | 450 of 450 OK |
| to_out | 8x9 | (12, 8, 6) 4064 | (8, 8, 6) 4066, rank 2, +0.1% | 330 of 332 OK |
| to_out | 8x8 | (12, 8, 6) 4099 | fused shipped (8, 8, 6) 4101, rank 2 | 335 of 338 OK |

What the M=13664 numbers say:

1. **The fused op wins on all three shapes**, by 11% (to_qkv), 23% (to_out) and 6% (ff1) against the best
   possible unfused pair, before the dispatch gap between the two unfused ops is counted.
2. **On equal cores the fused op hides the gather completely.** to_qkv's standalone matmul on 8x8 takes 10270 us;
   the fused op, which also moves 3 x 13664 x 1344 bf16 per device around the ring, takes 10261 us. ff1: 14872
   standalone vs 15156 fused, 284 us for a 1911 us gather.
3. **The 9th row is worth 6.5% (to_qkv) and 3.9% (ff1) to the standalone matmul**, far less than the gather it
   then has to pay for.
4. **Shipped fused blockings hold**: rank 1 for to_qkv, rank 2 within 0.1% for ff1.
5. The model's unfused branch would run its `AGMM_BLOCK_SIZES` entry on 8x9; for ff1 that (8, 3, 14) is 5.3%
   off the 8x9 optimum, for to_qkv 2.3%. Only relevant if the unfused branch is ever taken.
6. The all-gather has 3.4 to 3.9% on the table by moving `chunks_per_sync` from 16 to 32
   (`models/tt_dit/parallel/manager.py:933-945`); this only matters for the unfused path and for the other
   standalone gathers in the block.

### M = 9184 rows per device (10 s clip)

All 12 cases complete; 3,540 of 3,555 combos measured OK (15 L1 rejections on the unfused matmul), every one
above PCC 0.99998.

| op | best fused (blocking) | best all-gather | best matmul 8x9 | best matmul 8x8 | AG + MM 8x9 | fused vs unfused |
|---|---|---|---|---|---|---|
| to_qkv | 7056 (12, 7, 8) | 1281 (3, 32, 2) | 6319 (12, 7, 8) | 6809 (12, 7, 8) | 7600 | fused 544 us faster (-7.2%) |
| to_out | 3376 (12, 8, 6) | 1701 (3, 32, 4) | 2690 (12, 8, 5, 4, 1) | 2714 (12, 8, 6) | 4391 | fused 1015 us faster (-23.1%) |
| ff1 | 10013 (12, 7, 8) | 1283 (4, 32, 2) | 9575 (12, 4, 10) | 9913 (12, 7, 8) | 10858 | fused 845 us faster (-7.8%) |

Fused blockings (8x8) and where the shipped ones land:

| op | rank 1 | rank 2 | shipped |
|---|---|---|---|
| to_qkv | (12, 7, 8) 7056 | (10, 7, 8) 7073 | (8, 7, 12) 7173, rank 5, +1.7% |
| to_out | (12, 8, 6) 3376 | (12, 7, 6) 3450 | (8, 8, 6) 3559, rank 8, +5.4% |
| ff1 | (12, 7, 8) 10013 | (10, 7, 10) 10031 | (8, 7, 10) 10542, rank 12, +5.3% |

All-gather: (x, 32, y) again leads everywhere; production (3, 16, 2) is rank 7, 3.2 to 3.4% behind. Unfused
matmul: the model's `AGMM_BLOCK_SIZES` entries on 8x9 fall to rank 32 to 52 (+4.0% to_out, +9.8% to_qkv,
+10.5% ff1).

What changes against M=13664:

1. **The fused op still wins on every shape** (7%, 23%, 8%), but its margin over the standalone matmul on the
   same 8x8 grid has reversed: at 13664 the fused op matched the standalone matmul (the gather was free); at 9184
   the standalone matmul is 3.5% (to_qkv) and 1.0% (ff1) faster than the fused op. With fewer rows per core there
   is less matmul work to hide the ring traffic behind.
2. **The best blocking moved.** (12, 7, 8) wins to_qkv and ff1 fused and both of their standalone grids;
   (12, 8, 6) wins to_out everywhere. The likely reason is M padding: with 8 columns of workers, the per-core M
   is ceil(M_tiles / 8) tiles, 54 at M=13664 and 36 at M=9184. 36 is divisible by 12 (3 blocks, no padding),
   while 54 is not (12 pads to 60, +11%; 8 pads to 56, +3.7%), so M_block 8 wins at 13664 and 12 at 9184.
   `AGMM_BLOCK_SIZES` (`agmm_config.py:49-53`) is keyed on (K, N) only and cannot express this; M=4736 (19
   tiles per core, prime) will show whether the rule holds.
   The compute kernel clips the last M block, so the padded rows cost relay volume, epilogue passes and block
   count rather than MACs; see "Per-core tile counts" below.

### M = 4736 rows per device (5 s clip)

All 12 cases complete; 2,827 of 2,845 combos OK (18 L1 rejections on the unfused matmul), all above PCC 0.99998.

| op | best fused (blocking) | best all-gather | best matmul 8x9 | best matmul 8x8 | AG + MM 8x9 | fused vs unfused |
|---|---|---|---|---|---|---|
| to_qkv | 3828 (10, 7, 8) | 670 (3, 32, 4) | 3432 (6, 8, 12) | 3811 (8, 7, 12) | 4102 | fused 274 us faster (-6.7%) |
| to_out | 1814 (10, 8, 6) | 884 (3, 32, 2) | 1501 (10, 8, 6) | 1509 (10, 8, 6) | 2385 | fused 571 us faster (-23.9%) |
| ff1 | 5441 (10, 7, 10) | 670 (3, 32, 4) | 4888 (6, 8, 14) | 5416 (10, 7, 10) | 5557 | fused 116 us faster (-2.1%) |

Fused blockings (8x8): M_block 10 wins all three ops, as the padding rule predicts (19 M tiles per core: 10 pads
to 20, 8 and 12 both pad to 24).

| op | rank 1 | rank 2 | shipped |
|---|---|---|---|
| to_qkv | (10, 7, 8) 3828 | (10, 6, 8) 3854 | (8, 7, 12) 4186, rank 8, +9.4% |
| to_out | (10, 8, 6) 1814 | (10, 7, 6) 1854 | (8, 8, 6) 2025, rank 11, +11.6% |
| ff1 | (10, 7, 10) 5441 | (10, 6, 10) 5491 | (8, 7, 10) 6175, rank 21, +13.5% |

All-gather: (3, 32, x) leads again; production (3, 16, 2) is 2.8 to 3.6% behind. Unfused matmul on 8x9: the
model's `AGMM_BLOCK_SIZES` entries are +9.1% (to_qkv), +6.2% (to_out) and +20.7% (ff1, rank 74) off the optimum.

What changes against the longer clips:

1. **The fused op still wins every shape**, but ff1's margin has shrunk to 2%. The gather is small at this length
   (670 us) and the 9th row buys the standalone matmul 10% (4888 vs 5416 us on 8x8), so the two nearly cancel.
   to_out keeps its 24% margin because its gather is 37% of its matmul time at every length.
2. **The shipped fused blockings lose 9 to 14% here**, their widest gap, all from M padding.

### Across the three clip lengths

| op | M | best fused | shipped fused | gap | best AG + MM (8x9) | fused advantage |
|---|---|---|---|---|---|---|
| to_qkv | 13664 | 10261 (8, 7, 12) | 10261 (8, 7, 12) | 0 | 11511 | 10.9% |
| to_qkv | 9184 | 7056 (12, 7, 8) | 7173 | +1.7% | 7600 | 7.2% |
| to_qkv | 4736 | 3828 (10, 7, 8) | 4186 | +9.4% | 4102 | 6.7% |
| to_out | 13664 | 5065 (12, 8, 6) | 5214 | +2.9% | 6606 | 23.3% |
| to_out | 9184 | 3376 (12, 8, 6) | 3559 | +5.4% | 4391 | 23.1% |
| to_out | 4736 | 1814 (10, 8, 6) | 2025 | +11.6% | 2385 | 23.9% |
| ff1 | 13664 | 15156 (8, 6, 10) | 15164 | +0.1% | 16201 | 6.4% |
| ff1 | 9184 | 10013 (12, 7, 8) | 10542 | +5.3% | 10858 | 7.8% |
| ff1 | 4736 | 5441 (10, 7, 10) | 6175 | +13.5% | 5557 | 2.1% |

1. **Keep the fused op.** It beats the best unfused pair on every shape at every length: 7 to 11% for to_qkv,
   23 to 24% for to_out, 2 to 8% for ff1, and the unfused numbers exclude the dispatch gap and the model's
   separate addcmul for to_out. The unfused branch is not a lever.
2. **Make the fused M_block follow M.** The fused op splits M over the 8 worker columns, so each core holds
   ceil(M / 256) M tiles: 54, 36 and 19 for the three clips. The winning M_block is the candidate that pads that
   count least (8 pads 54 to 56; 12 divides 36; 10 pads 19 to 20), and the shipped table, tuned at 54, pays 2 to
   5% at 10 s and 9 to 14% at 5 s. Extending `agmm_block_size(k, n)` (`agmm_config.py:56`) with an M-aware
   M_block pick, keeping the table's K_block and N_block, recovers it with no kernel change. to_out additionally
   wants M_block 12 rather than 8 even at 15 s (+2.9%).
3. **All-gather hyperparameters.** `chunks_per_sync=32` beats the production 16 by 3 to 4% on every tensor and
   length, workers 3 vs 4 and the buffer count are within 1%. Only the standalone gathers in the block would see
   this (`manager.py:933-945`); the fused op does not use these parameters.
4. **The 9th row.** The standalone matmul gains 4 to 10% from the extra row of cores, which is never enough to
   pay for the gather it then needs. If the mux row could be reclaimed for the fused op that gain would apply
   directly, but that is a program-factory change, not a blocking one.

## Per-core tile counts

What one worker core actually computes for the three fused ops, from the shapes above and the program factory's
split. Every core in the 8x8 grid gets the same M and N range on every device, so the counts below hold for all 64
cores; only the share of real vs padded tiles differs by position.

**How the split works.** The factory rounds M and N tiles up to a multiple of 8 and divides evenly
(`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/all_gather_minimal_matmul_async_program_factory.cpp:328-354`);
core (i, j) owns rows `[54 i, 54 (i+1))` and columns `[N_per_core j, N_per_core (j+1))` (`:1277-1280`). K is split
per ring device first, then into K blocks (`:345-348`): 168 gathered K tiles / 4 devices = 42 per device, 6 blocks
of 7, 24 K blocks per output block; to_out 224 / 4 = 56, 7 blocks of 8, 28 K blocks. Every device still contracts
over the full K, so tile-MACs use 168 or 224, not the local 42 or 56. The compute kernel clips the last M and N block
to the core's range and rounds it up to the 2x2 subblock (`kernels/compute.cpp:470-480`, subblock loop `:364-366`).

### M = 13664 rows per device (15 s clip), shipped blockings

| op | M x K x N tiles | per core M x N | blocks (M, K, N) | M blocks | N blocks | K-loop iterations |
|---|---|---|---|---|---|---|
| ff1 | 427 x 168 x 224 (112 out) | 54 x 28 (14 out) | (8, 7, 10) | 8,8,8,8,8,8,**6** | 10,10,**8** | 7 x 3 x 24 = 504 |
| to_qkv | 427 x 168 x 168 | 54 x 21 | (8, 7, 12) | 8,8,8,8,8,8,**6** | 12,**9 (issued 10)** | 7 x 2 x 24 = 336 |
| to_out | 427 x 224 x 42 | 54 x 6 | (8, 8, 6) | 8,8,8,8,8,8,**6** | 6 | 7 x 1 x 28 = 196 |

| op | output tiles computed | tile-MACs issued | useful tile-MACs (device average / 64) | issued / useful | fp32 L1-acc packs | epilogue tiles processed | real tiles the epilogue should touch |
|---|---|---|---|---|---|---|---|
| ff1 | 54 x 28 = 1,512 (756 SwiGLU out) | 54 x 28 x 168 = **254,016** | 251,076 | 1.012 | 36,288 | 1,680 in, 840 out pairs | 756 pairs |
| to_qkv | 54 x 22 = 1,188 | 54 x 22 x 168 = **199,584** | 188,307 | 1.060 | 28,512 | 1,344 | 1,134 |
| to_out | 54 x 6 = 324 | 54 x 6 x 224 = **72,576** | 62,769 | 1.156 | 9,072 | 336 | 324 issued, 283.5 average real |

Reading the columns:

- **Tile-MACs issued** is the sum of the clipped block sizes, not blocks times block size: ff1 is 54 x 28 x 168, and
  a full-block count (56 x 30 x 168) would overstate it by 11%. The ff1 zone study's 254,016 tile-MACs and 36,288 packs
  per core (`ff1.md` §1) are these numbers.
- **fp32 L1-acc packs** are one per issued output tile per K block: 1,512 x 24 for ff1.
- **The epilogues are not clipped.** `swiglu_block`, `copy_block` / `add_bias_block` and `add_bias_and_addcmul_block`
  are called with the full `M_block_tiles` and `N_block_tiles` (`kernels/compute.cpp:543-544, :553, :562-571`), so
  ff1 runs silu and the multiply on 840 gate/up pairs per core against 756 real ones, 11% of its 2.2 ms epilogue.
- **An "out pair"** is one gate tile and its up tile in ff1's interleaved matmul output (column tiles 2p and 2p+1,
  `kernels/compute.cpp:20-23`); the SwiGLU epilogue turns each pair into one bf16 output tile, so the 28 matmul columns
  per core become 14 output columns.

Where the padding sits:

- **M**: 427 = 8 x 53 + 3, padded to 432, so 54 rows per core. The cores in the last M position hold rows 378 to 431,
  49 of them real; the other seven positions are 100% real. No core computes 56 rows: the seventh M block is clipped to 6.
- **to_qkv N**: 21 columns per core split 12 + 9; the 9-wide block is issued as five 2-wide subblocks, so the core
  computes 22 columns. That single padded column is the only N padding of the op. `to_qkv.md` §2 charges its
  "grid padding" row as 21 -> 24 and 54 -> 56 (1.1 ms); with 22 x 54 it is about 0.36 ms of the 6.0 ms compute
  bound, and the remainder belongs to the pipeline-issue row. The measured 29.9 us per K-loop iteration agrees: the
  two N blocks average 8 x 11 x 7 = 616 tile-MACs, which at ff1's pace (27.3 us per 560) is 30.0 us; 24 columns
  would give 32.8 us.
- **to_out N**: 42 columns over 8 core columns is 6 for the first seven and **0 for the eighth**. That whole column of
  8 cores computes only padding, 72,576 tile-MACs each; the in1 reader zero-fills and the writer skips those tiles
  (`kernels/dm_in1_sender_out.cpp:305-309, :442`), so the numerics are unaffected but the cores are. This is the
  12.5% padding in the roofline table below, concentrated on 8 cores rather than spread.

### What M_block changes, per clip length

The compute kernel issues the same number of M rows for every candidate M_block, so the M_block winner in the sweeps
above is not decided by K-loop MACs. What does scale with `ceil(M_per_core / M_block) * M_block` is the in0 volume
moved down the relay chain and unpacked into the CB (the in0 block is always `M_block_tiles x K_block_tiles`,
`kernels/compute.cpp:456, :494`), the intermediate CB reserve, and the epilogue rows; what scales with the number of
M blocks is the in1 re-read from DRAM (once per M block) and the per-block setup. This is the mechanism behind the
padding rule in "Across the three clip lengths", point 2.

| M | M tiles | per core | M_block | blocks | rows issued (MACs) | rows delivered / epilogue | M blocks |
|---|---|---|---|---|---|---|---|
| 13664 | 427 | 54 (last core 49 real) | 8 | 8,8,8,8,8,8,6 | 54 | 56 | 7 |
| | | | 10 | 10,10,10,10,10,4 | 54 | 60 | 6 |
| | | | 12 | 12,12,12,12,6 | 54 | 60 | 5 |
| 9184 | 287 | 36 (last core 35 real) | 8 | 8,8,8,8,4 | 36 | 40 | 5 |
| | | | 10 | 10,10,10,6 | 36 | 40 | 4 |
| | | | 12 | 12,12,12 | 36 | 36 | 3 |
| 4736 | 148 | 19 (last core 15 real) | 8 | 8,8,3 | 20 | 24 | 3 |
| | | | 10 | 10,9 | 20 | 20 | 2 |
| | | | 12 | 12,7 | 20 | 24 | 2 |

At every length the winning M_block is the one with the fewest delivered rows (8 at 13664, 12 at 9184, 10 at 4736),
and among ties the one with fewer M blocks. At 4736 every candidate issues 20 rows of MACs for 19 real, so the 9 to
14% gap of the shipped blockings there is relay volume, epilogue rows and block count, not FPU work. Passing the
clipped `current_M_block_tiles` / `current_N_block_tiles` to the epilogues would remove their share at no numerical
cost; the in0 relay volume needs the reader to clip too.

## Best measured time vs roofline

`agmm_unit_sweep.py roofline`, constants from `transformer_roofline.py` (`tools/transformer_roofline.py:116-129`):
2048 FLOP/cycle/core at HiFi2 and 1.0 GHz, DRAM 288 GB/s, fabric 4 links x 12.5 GB/s, ring of 4. Compute bound =
2 M K N / (cores x 2.048 TFLOP/s); DRAM bound = 2 B (M K + K N) / 288 GB/s; fabric bound = 3/4 of the device's
in0 shard over 8 link-directions at 12.5 GB/s. Cores: 64 for the fused op and the 8x8 matmul, 72 for 8x9.
"compute util" = compute bound / measured; "measured / ideal" is against the largest bound.

Compute utilisation at the best blocking (the full per-row table follows):

| op | mode | cores | M=4736 | M=9184 | M=13664 |
|---|---|---|---|---|---|
| to_qkv | fused | 64 | 55% | 57% | 59% |
| to_qkv | matmul 8x8 | 64 | 55% | 59% | 59% |
| to_qkv | matmul 8x9 | 72 | 54% | 57% | 56% |
| to_out | fused | 64 | 38% | 40% | 40% |
| to_out | matmul 8x8 | 64 | 46% | 50% | 49% |
| to_out | matmul 8x9 | 72 | 41% | 45% | 44% |
| ff1 | fused | 64 | 51% | 54% | 53% |
| ff1 | matmul 8x8 | 64 | 51% | 54% | 54% |
| ff1 | matmul 8x9 | 72 | 51% | 50% | 50% |

Standalone all-gather: 1.73 to 1.75x its fabric bound at every size (58% of 12.5 GB/s per link-direction).

| op | M | mode | cores | best blocking | measured us | compute us | DRAM us | fabric us | limiter | attained TFLOP/s | compute util | measured / ideal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| to_qkv | 4736 | fused | 64 | 10, 7, 8, 2, 2 | 3827.5 | 2088.6 | 377.5 | 381.9 | compute | 71.5 | 55% | 1.83x |
| to_qkv | 4736 | mm_ring | 64 | 8, 7, 12, 2, 2 | 3811.2 | 2088.6 | 377.5 | - | compute | 71.8 | 55% | 1.82x |
| to_qkv | 4736 | mm_full | 72 | 6, 8, 12, 2, 2 | 3432.3 | 1856.5 | 377.5 | - | compute | 79.8 | 54% | 1.85x |
| to_qkv | 4736 | ag | - | 3, 32, 4 | 669.6 | - | - | 381.9 | fabric | - | - | 1.75x |
| to_out | 4736 | fused | 64 | 10, 8, 6, 2, 2 | 1814.3 | 696.2 | 302.6 | 509.2 | compute | 50.3 | 38% | 2.61x |
| to_out | 4736 | mm_ring | 64 | 10, 8, 6, 2, 2 | 1508.6 | 696.2 | 302.6 | - | compute | 60.5 | 46% | 2.17x |
| to_out | 4736 | mm_full | 72 | 10, 8, 6, 2, 2 | 1501.4 | 618.8 | 302.6 | - | compute | 60.8 | 41% | 2.43x |
| to_out | 4736 | ag | - | 3, 32, 2 | 884.0 | - | - | 509.2 | fabric | - | - | 1.74x |
| ff1 | 4736 | fused | 64 | 10, 7, 10, 2, 2 | 5440.9 | 2784.8 | 444.4 | 381.9 | compute | 67.1 | 51% | 1.95x |
| ff1 | 4736 | mm_ring | 64 | 10, 7, 10, 2, 2 | 5416.2 | 2784.8 | 444.4 | - | compute | 67.4 | 51% | 1.94x |
| ff1 | 4736 | mm_full | 72 | 6, 8, 14, 2, 2 | 4887.5 | 2475.3 | 444.4 | - | compute | 74.7 | 51% | 1.97x |
| ff1 | 4736 | ag | - | 3, 32, 4 | 669.5 | - | - | 381.9 | fabric | - | - | 1.75x |
| to_qkv | 9184 | fused | 64 | 12, 7, 8, 2, 2 | 7055.7 | 4050.1 | 543.6 | 740.6 | compute | 75.2 | 57% | 1.74x |
| to_qkv | 9184 | mm_ring | 64 | 12, 7, 8, 2, 2 | 6809.3 | 4050.1 | 543.6 | - | compute | 78.0 | 59% | 1.68x |
| to_qkv | 9184 | mm_full | 72 | 12, 7, 8, 2, 2 | 6318.6 | 3600.1 | 543.6 | - | compute | 84.0 | 57% | 1.76x |
| to_qkv | 9184 | ag | - | 3, 32, 2 | 1281.3 | - | - | 740.6 | fabric | - | - | 1.73x |
| to_out | 9184 | fused | 64 | 12, 8, 6, 2, 2 | 3376.2 | 1350.0 | 524.1 | 987.5 | compute | 52.4 | 40% | 2.50x |
| to_out | 9184 | mm_ring | 64 | 12, 8, 6, 2, 2 | 2713.5 | 1350.0 | 524.1 | - | compute | 65.2 | 50% | 2.01x |
| to_out | 9184 | mm_full | 72 | 12, 8, 5, 4, 1 | 2689.9 | 1200.0 | 524.1 | - | compute | 65.8 | 45% | 2.24x |
| to_out | 9184 | ag | - | 3, 32, 4 | 1700.7 | - | - | 987.5 | fabric | - | - | 1.72x |
| ff1 | 9184 | fused | 64 | 12, 7, 8, 2, 2 | 10013.4 | 5400.2 | 610.5 | 740.6 | compute | 70.7 | 54% | 1.85x |
| ff1 | 9184 | mm_ring | 64 | 12, 7, 8, 2, 2 | 9912.6 | 5400.2 | 610.5 | - | compute | 71.4 | 54% | 1.84x |
| ff1 | 9184 | mm_full | 72 | 12, 4, 10, 2, 2 | 9575.1 | 4800.2 | 610.5 | - | compute | 73.9 | 50% | 1.99x |
| ff1 | 9184 | ag | - | 4, 32, 2 | 1283.2 | - | - | 740.6 | fabric | - | - | 1.73x |
| to_qkv | 13664 | fused | 64 | 8, 7, 12, 2, 2 | 10261.4 | 6025.8 | 710.8 | 1101.9 | compute | 77.0 | 59% | 1.70x |
| to_qkv | 13664 | mm_ring | 64 | 14, 7, 6, 2, 2 | 10269.8 | 6025.8 | 710.8 | - | compute | 76.9 | 59% | 1.70x |
| to_qkv | 13664 | mm_full | 72 | 12, 7, 8, 2, 2 | 9605.2 | 5356.3 | 710.8 | - | compute | 82.2 | 56% | 1.79x |
| to_qkv | 13664 | ag | - | 3, 32, 4 | 1906.1 | - | - | 1101.9 | fabric | - | - | 1.73x |
| to_out | 13664 | fused | 64 | 12, 8, 6, 2, 2 | 5065.4 | 2008.6 | 747.1 | 1469.2 | compute | 52.0 | 40% | 2.52x |
| to_out | 13664 | mm_ring | 64 | 12, 8, 6, 2, 2 | 4098.8 | 2008.6 | 747.1 | - | compute | 64.2 | 49% | 2.04x |
| to_out | 13664 | mm_full | 72 | 12, 8, 6, 2, 2 | 4064.0 | 1785.4 | 747.1 | - | compute | 64.8 | 44% | 2.28x |
| to_out | 13664 | ag | - | 4, 32, 4 | 2541.8 | - | - | 1469.2 | fabric | - | - | 1.73x |
| ff1 | 13664 | fused | 64 | 8, 6, 10, 2, 2 | 15156.0 | 8034.4 | 777.7 | 1101.9 | compute | 69.5 | 53% | 1.89x |
| ff1 | 13664 | mm_ring | 64 | 14, 8, 6, 2, 2 | 14871.6 | 8034.4 | 777.7 | - | compute | 70.8 | 54% | 1.85x |
| ff1 | 13664 | mm_full | 72 | 8, 8, 10, 2, 2 | 14289.9 | 7141.7 | 777.7 | - | compute | 73.7 | 50% | 2.00x |
| ff1 | 13664 | ag | - | 3, 32, 4 | 1911.0 | - | - | 1101.9 | fabric | - | - | 1.73x |

What the roofline says:

1. **Every matmul is compute-limited by the model and runs at 1.7 to 2.6x its compute bound.** DRAM is never
   closer than 5x away. Compute utilisation is flat across clip lengths once the blocking is chosen per M:
   to_qkv 55 to 59%, ff1 50 to 54%, to_out 38 to 50%. This is the same ceiling the per-op studies measured on
   the device: the 2x2 fp32 subblock K loop issues at ~47 cycles per tile MAC against a nominal 32 (68%), the
   epilogues serialize on top of it, and the padding of the per-core output block takes the rest.
2. **The 9th row raises attained TFLOP/s (to_qkv 77 -> 82, ff1 71 -> 74, to_out 64 -> 65) at equal or slightly
   lower utilisation** (to_qkv 59 -> 56%, ff1 54 -> 50%, to_out 49 -> 44%): the 72-core grid pads N a little
   more (168 N tiles over 9 rows, 42 over 9) and the same per-core pipeline pace applies.
3. **to_out is the outlier** and the fused op is where it shows: 38 to 40% fused against 45 to 50% standalone on
   the same cores. Its fabric bound is 73% of its compute bound (N=1344 is only 1.4x above the 982-column
   compute/fabric crossover, `Arch.n_star`), so the ring traffic is not hidden behind the matmul the way it is for
   to_qkv and ff1 (fabric bound 18% and 14% of compute). The 8-column split also pads its 42 output tiles to
   48 (12.5% of the fused op's MACs are padding; 6.7% on 8x9). The earlier finding that the to_out K loop waits on
   the in0 relay a quarter of the time (`to_out.md`) is the same effect seen from inside the kernel.
4. **The standalone all-gather runs at 1.73x its fabric bound at every size**, i.e. 58% of the 12.5 GB/s per
   link-direction, independent of M and K. The fused op's gather is not separately visible, but for to_qkv and
   ff1 it costs at most 3% over the standalone matmul on the same grid at 15 s and about 3.5% at 10 s.

## Caveats

- **to_out's gate is per token.** The model's `addcmul_gate` is one modulation row per token, shape [M, N]
  (`transformer_block_minimax_h3.py:231-234,293-294`). The first to_out pass of this study used a broadcast [1, N]
  gate, which makes the fused epilogue and the standalone addcmul matmul cheaper than the model's (fused shipped
  4676 us vs 5294 us recorded with the full gate in `to_out.md`). Those rows were set aside
  (`generated/agmm_h3_sweep/to_out_M13664_broadcast_gate_rows.csv`) and to_out is being re-measured with the
  full gate (the numbers above are the re-measurement); the all-gather rows do not involve the gate.
- The model's unfused to_out would be a plain matmul followed by a separate `ttnn.addcmul`
  (`attention_minimax_h3.py:615-627`), not the addcmul-fused matmul measured here as the unfused matmul step. The
  measured pair is therefore the best unfused case, not the model's current unfused code.

- Unfused totals add two device kernel durations; the dispatch gap between the two ops and, in the model, the two
  reshapes around the standalone gather (`manager.py:857-860,880-882`) are not included.
- The fused rows here carry no bias; the earlier per-op sweep rows (`ff1.md`, `to_qkv.md`, `to_out.md`) included
  the harness bias add (about 50 us).
- `mm_ring` (8x8 here) is the like-for-like split of the fused op; `mm_full` (8x9 here) is what the model's unfused branch would run.
- The unit test exercises the TP ring alone. In the model the SP axis carries traffic at the same time.
