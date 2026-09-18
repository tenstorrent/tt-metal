# SDXL UNet on Blackhole: full-grid GroupNorm worklog

Goal: use the agent-generated `groupnorm_sc_N_1_HW_C` (Python generic_op) in the SDXL 1024x1024 UNet on
Blackhole p150a, expand the core grid the resnet path runs on (target 10x10 / 10x8 instead of 8x8), remove the
reshards / untilize / copy ops around every GroupNorm, then retune convs and matmuls for OOM and perf.

Worktree: `/localdev/mstaletovic/metal_metal/tt-metal-sdxl-gn`, branch `mstaletovic/sdxl-gn-fullgrid`, based on
`llk_helper_library` @ e77769238a9. Shares the Release+Tracy build, venv and runtime of
`tt-metal-llk_helper_library` via symlinks (`build`, `python_env`, `runtime`; `ttnn/ttnn/_ttnn.so` copied).
The C++ tree is identical to that build, so no rebuild is needed for Python-only work.

Run recipe (from the worktree root):
```
env -u PYTHONPATH TT_METAL_HOME=$PWD PYTHONPATH="$PWD/ttnn:$PWD/tools:$PWD" scripts/run_safe_pytest.sh [--profile] <test>
```

## Baseline (reference model, measured 2026-09-17, one UNet step, Tracy)
- 77.07 ms device kernel time, 2166 ops. Matmul 30.1 ms (80 cores), Conv2d 10.2 ms (80 cores = 10 cols x 8 rows
  output grid; input side inherits the GN's 8x8), SDPA 9.6 (110), GroupNorm 6.7 ms (8x8=64; attention GNs 4x8=32),
  LayerNorm 3.7 (10x8), I2S 2.6, SiLU-after-GN 2.0.
- Every resnet GN is bracketed by S2I -> copy -> untilize -> I2S(8x8, 72 us on 16384x320) -> GN 337 us -> SiLU 80 us
  -> halo -> conv(10x8). Layout ops adjacent to GNs: 4.2 ms/step.
- Reference GN cannot use 10 columns: per-core channels (32) must be a whole number of groups (Cg=10 for C=320).
  The generated op has no such constraint (membership matmul) and its RM direct view is exact for 32-wide shards
  with 32-aligned heights (probe: 16384x320 [1664,32] 10x10 = 47.6 us vs 70.1 us on the 8x8 model shard;
  4096x640 [416,64] 10x10 = 29.9 us vs 46.0).
- Profile CSV copy: scratchpad `sdxl_unet_bh_ops.csv`; how to get it: CI wrapper
  `test_sdxl_perf.py::test_sdxl_perf_device -k test_sdxl_unet_1024x1024` (plain `--profile` on test_unet yields no op CSV).

## Log

### 2026-09-17 13:15 setup
- Created worktree, copied `ttnn/ttnn/operations/groupnorm_sc_N_1_HW_C/` (op + kernels + docs) and its unit tests
  from the dead eval run 1021 clone (refinement 5 WIP state, HEAD a217bc9439 + dirty docs). kernel_lib is identical
  between the op's base (3c1b3ef137d) and e77769238a9 (git diff empty), so the kernels should compile unchanged.
- Initialized `umd` submodule (tt_ops_code_gen is "skipped" by git, checked out manually if the tests need `eval.*`).

## Constraints (from the user, 2026-09-17)
- Do NOT change math fidelity or fp32 dest accumulation (or packer_l1_acc) on any op. Allowed knobs: core grids,
  shard geometries, block / subblock sizes, layouts, op ordering / placement. The generated GroupNorm's own
  numerics are the one accepted change (it runs its statistics path in fp32 DEST / HiFi4 by construction).
- Keep the generated op as a Python generic_op (no C++ port).

## Plan
1. Config switch `use_generated_groupnorm` on ModelOptimisations1024x1024BH (default ON in this branch); GN grid
   derived per shape: 10 columns always (C / 10 is a multiple of 32 for every SDXL C), 10 rows when HW >= 4096,
   8 rows at HW = 1024 (32 tile rows do not split ten ways). Shard = [ceil(HW/32/rows)*32, C/10]; the last
   grid row is a padded shard (16640 > 16384). This is the same geometry conv2d's own parallel-config heuristic
   picks, so the conv should keep the GN's shard (its resnet configs have reshard_if_not_optimal=False).
2. Resnet path: to_memory_config(shard) -> GN (in place) -> SiLU (in place) -> conv1 -> temb add -> GN2 -> SiLU -> conv2.
   Remove the S2I / copy / untilize / I2S around GNs; try conv output_layout=ROW_MAJOR where the consumer is a GN.
3. Validate with tests/pcc/test_module_tt_resnetblock2d.py per shape, then the UNet PCC test, then the CI perf
   wrapper for the per-op profile; iterate on OOM / act_block_h divisibility / matmul program configs.

### 2026-09-17 13:30 step 1: GN swap wired, resnet blocks pass
- Added `sdxl_utility.generated_gn_grid / generated_gn_sharded_memory_config / prepare_generated_gn_beta_gamma /
  run_group_norm`; `ModelOptimisations1024x1024BH(use_generated_groupnorm=...)` (env `SDXL_GENERATED_GN`, default 1)
  returns `{"generated": True}` + (1,1,1,C) bf16 RM gamma/beta; resnet / transformer / unet-out call sites go through
  `run_group_norm` (reference path untouched when the switch is off).
- Sanity: the op's SDXL sharded unit tests pass in the worktree (8/8, shared build).
- `test_module_tt_resnetblock2d.py`: **12/12 pass**, PCC 0.99982-0.99996 (gate 0.998/0.999), first try. No OOM, no
  act_block_h complaint from the convs yet (conv still takes the GN shard as input; whether it kept 10x10 is to be
  confirmed from a profile).
- Transformer model (attention GN): C=640 case failed in proj_in: `per_core_M (16) must equal shard_shape[0]/32
  (13)` — 2D_TM_LINEAR_640 assumes the 4x8/10x8 shard, the generated GN delivers [416,64] on 10x10. Interim fix:
  hand proj_in an L1-interleaved in0 after the generated GN (the C=1280 path already did this). 2/2 pass
  (PCC 0.99876 / 0.99733). TODO: retune 2D_TM_LINEAR_640 to consume the 10x10 shard (per_core_M=13 is awkward:
  subblock_h must divide 13).

### 2026-09-17 13:45 v1 profile (RM path, 10 rows for HW>=4096): 77.07 -> 72.82 ms
- UNet PCC 0.99850 (gate 0.9968). GN 6.68 -> 1.74 ms (46 ops, 100/80 cores). But:
  - I2S DRAM -> 10x10 RM shard on 16384x320 = **300 us** (64 B sticks; old path was DRAM->L1 copy 28 + untilize 17
    + L1->L1 I2S 72). I2S total 2.56 -> 2.91 ms.
  - SiLU on the RM [1664,32] shard: 156 us vs 80 us on the old [2048,40] (Unary on 64 B RM sticks). 2.05 -> 2.50 ms.
  - Conv2d 10.21 -> 10.64 ms: convs on 10x10 are faster for 16384x320 (262 -> 230) and 4096x640 (206 -> 178) but
    the wide-Cin 4096 convs got 1.6x SLOWER (4096x1920->640: 534 -> 853; 1280->640: 357 -> 625; 960->640: 293 -> 479):
    per-core 13 tile rows is prime, and conv2d picks the largest divisor of it <= act_block_h_override -> 1 tile.
  - Layout ops around GNs still 4.36 ms (the structure S2I -> untilize -> I2S at block entry is unchanged because
    every resnet ends with to_memory_config(DRAM) and the next block re-shards).
- Fixes applied for v2: (a) grid rule: 8 rows when HW <= 4096 (16 / 4 tile rows per core, conv act blocks divide),
  10 rows at 16384 (52 -> conv falls back to act_block_h 26 tiles, fine); (b) generated path keeps the incoming
  layout (TILE) instead of forcing RM: the op's TILE shard path is zero-copy, I2S moves 2 KB pages, SiLU runs on
  tiles. conv2d accepted the TILE block-sharded input (resnet PCC 12/12, UNet PCC 0.99855).

### 2026-09-17 13:50 v2 profile (TILE path, 8 rows at HW<=4096): 68.20 ms  (-11.5% vs 77.07)
- GN 1.59 ms (was 6.68), I2S 0.99 (2.56), SiLU 1.50 (2.05; tiles), untilize 0.11 (0.92), conv 10.00 (10.21).
  Layout ops adjacent to GNs 1.49 ms (4.36). conv2d inserts its own small sharded untilize (18 ops, 0.07 ms) when
  fed a TILE shard — accepted, no separate op needed.
- Remaining around each GN: previous op's S2I to DRAM (35 us) + I2S back (27 us) per block boundary.
- v3 changes: (B) temb add on the conv1 output shard in place (drops S2I -> add -> I2S before GN2);
  (F) 2D_TM_LINEAR_640 in0_block_w 5 -> 2 so proj_in consumes the [512,64] 10x8 GN shard directly (S2I dropped);
  (A) resnet outputs stay L1 block-sharded for the next block (`keep_l1_resnet_output`, env SDXL_GN_KEEP_L1);
  run_group_norm in_place="if_copied" so residual inputs that arrive already-sharded are not clobbered
  (resnet GN1, attention GN); up blocks move to DRAM before the skip concat. UNet PCC 0.99852, no OOM.

### 2026-09-17 14:00 v3 = 67.33 ms (-12.6%); v4 (fused SiLU) = 67.55 ms -> fusion reverted to opt-in
- v3 profile: I2S 0.46 + S2I 0.48 ms (from 2.56 + 0.76 baseline), layout ops adjacent to GNs 1.00 ms (4.36),
  BinaryNg 1.53 (temb add now on the shard), GN 1.59, SiLU 1.51, conv 10.03, matmul 30.12 (unchanged).
- Tried: SiLU fused into the GN apply chain (new ckl::Silu element in kernel_lib activations, CT knob 33,
  `activation="silu"` on the op, unit test 6/6). Result: GN 1.59 -> 3.42 ms, Unary 1.51 -> 0.01: **net +0.22 ms**.
  Per cell 16384x320: 36 -> 102 us fused vs 36 + 47.6 unfused. The exact SFPU silu in the op's fp32-DEST chain on
  80-100 cores is slower than the standalone bf16 unary on 110 cores. Kept as opt-in (SDXL_GN_FUSE_SILU=1).
- Tried: 2D_TM_LINEAR_640 out_subblock 2x1 -> 4x2: no change (36.4 -> 36.1 us avg); that matmul is not
  FPU-bound at this size. Kept (harmless).
- Ruled out: sharded nlp_create_qkv_heads (5.3 ms on 32 cores for the 1024-row QKV) — SDPA requires
  interleaved q/k/v (`Operands to SDPA need to be DRAM/L1 interleaved`), so a sharded create_heads would need an
  S2I per tensor afterwards.

### 2026-09-17 14:05 v5 changes: 26-tile conv act blocks on the 10-row grid, block-sharded 640 transformer blocks
- Profiling helper: `fullgrid_worklog/test_profile_probe.py` (PROBE_CMD=...) runs any pytest under
  `python -m tracy -p -r` like the CI wrapper (plain `run_safe_pytest.sh --profile` yields no op CSV on model tests).
- Conv (resnet probe, 100 cores, abh 26 tiles): 16384x640->320 485 -> 370 us (baseline 80-core: 441);
  16384x320->320 215 us unchanged (was already 26 by fallback). up_blocks.2 conv1/conv2 mapped to
  ABH_26T_ADB_WDB_BS when the generated GN is on (per-core 52 tile rows; 8/16-tile overrides fell back to 4/13).
- 640-channel transformer blocks (down_blocks.1 / up_blocks.1): proj_in and attn to_out outputs block-sharded on
  10x8 like the 1280 blocks (`sharded_640_blocks`), so LN1/LN2/LN3 and the residual adds run sharded. Needed
  in0_block_w=2 on 2D_ATTN_QKV_LINEAR_640 / 2D_ATTN_OUT_LINEAR_640 / 2D_GEGLU_LINEAR_640_SPLIT(_GELU) (in0 is the
  [512,64] shard: 2 K-tiles per core; the old 10/4 assumed interleaved in0). Transformer PCC 0.99863 / 0.99733.

### 2026-09-17 14:12 v5 = 65.82 ms (-14.6% vs 77.07 baseline), UNet PCC 0.99846
- LN 4096x640: 42.8 us interleaved/110 cores -> 25.9 us sharded/80 (LN total 3.67 -> 3.17). BinaryNg 1.53 -> 1.17
  (residual adds sharded). Conv 10.03 -> 9.72 (16384x640->320 485 -> 370, 16384x960->320 670 -> 595).
- OOM hit: up_blocks.2.resnets.0.conv1 (Cin=960) with the 26-tile act block: "Statically allocated circular
  buffers ... clash with L1 buffers" (CB region to 1.57 MB). Added ABH_13T_ADB_WDB_BS for that conv.
- Op mix now: matmul 30.0, conv 9.7, SDPA 9.6, create_heads 5.3, LN 3.2, concat_heads 1.9, GN 1.6, SiLU 1.4,
  binary 1.2, concat 0.7, I2S+S2I 0.84.

### 2026-09-17 14:25 v6 = 65.52 ms (-15.0%), UNet PCC 0.99846
- Resnet GN at HW=4096 now on 10 rows too (`resnet_gn_prime_rows`; 13 tile rows per core) with explicit 13-tile
  conv act blocks (ABH_13T_ADB_WDB_BS) where an 8/16-tile override existed. Convs 4096x1280->640 360 -> 304 us,
  4096x960->640 283 -> 242, 4096x640->640 201 -> 178 (x6). Conv total 9.72 -> 9.47 ms.
- Failed attempts (all reverted / carved out):
  - up_blocks.1.resnets.0.conv1 (Cin=1920) with a 13-tile act block: CBs grow to 1.63 MB > 1.5 MB L1.
    act_block_w_div=2 does NOT shrink a block-sharded conv's CBs (identical 1629888 B). That resnet's GN1 stays on
    8 rows (`gn1_prime_rows` carve-out); GN2 (Cin=640) uses 10.
  - up_blocks.0.upsamplers.0 (4096x1280->1280) with 13 tiles: CB clash by 56 KB. Stays 8 rows.
  - up_blocks.1.upsamplers.0 (16384x640->640) with 26 tiles: conv2d reshards the 10x8 upsample output to 10x10 and
    `reshard_reader_diff_width` overflows the 4094 runtime-arg limit (8346). Stays 8 rows (825 us).
- Downsampler conv (16384x320 stride 2) now keeps the 10x10 shard: 94 us on 100 cores vs 72 us on 80 (slightly
  slower kernel, but no reshard in front of it).

### 2026-09-17 14:35 SDPA chunk sweep (transformer-model probes): shipped config is the best
- 4096x4096 self-attn (512_K): q=64 -> 337 us avg (was 198 mixed avg), k=1024 -> 204, q=256 -> 216. Keep 128/512.
- 1024x1024 self-attn (1024_K): q=64 -> 114.5 us (was 76.7); q=256 fails validation. Keep 128/1024.
- Env knobs SDXL_SDPA_Q1024 / Q512 / K512 left in place (defaults = shipped values).
- Tried fewer grid rows for the 1024-row (C=1280) resnets to cut redundant weight reads: rows=4 -> conv1/conv2
  165 -> 317 us (2x slower: the conv is per-core compute/activation bound, not weight-read bound), rows=2 -> CB
  clash. Keep 8 rows (knob SDXL_GN_ROWS_1024 left in, default 8).

### 2026-09-17 15:10 matmul kernel investigation (FF-up 1024x1280x5120, bf16 act / bf8b w, HiFi2, l1_acc)
Probe: `fullgrid_worklog/test_mm_probe.py` under the tracy probe harness (3 calls each, device kernel us):
- shipped 10x8, in0 block-sharded [128,128], per-core 4x16, subblock 1x8: **74.0 us**
- same with in0 L1 interleaved: 82 us; same with in1 (weights) already L1 interleaved: 74.0 (unchanged!)
- padded-M split on 11x10 (transpose_mcast, per_core_M=3 -> 33 tiles for 32, last core ragged): works
  (PCC ok, the 2D factory already handles `last_per_core_M`), but 77.4 us sharded / 83 us interleaved —
  25% less math per core bought nothing.
- in1 width-sharded in L1 with in0 interleaved: device hang (timeout; runner reset the device).
Interpretation: per K block (in0_block_w=4) the in1 sender core reads 64 x 1 KB bf8b tiles (~7 us at the
~9 GB/s a single RISC sustains on 1 KB NoC reads) while the block's math is 4.5-6 us -> the matmul is
in1-fetch-issue bound, which is why fewer output tiles per core or an L1 copy of the weights do not help.
Lever: fewer/larger in1 transactions -> DRAM width-sharded weights read as contiguous rows (the 2D factory has
an `in1_is_sharded and in1_is_dram` path) or a kernel that coalesces tile reads.

### 2026-09-17 15:40 matmul: why 80 cores, and the transposed-mcast weight-reader bottleneck
- Grid divisibility: 2D mcast needs per_core_M x rows = M tiles and per_core_N x cols = N tiles exactly.
  M = 32 (1024 rows) or 128 (4096) tile rows -> rows in {8,4,2}; N = 160/40/120/20/80 tiles -> cols 10 max
  (11 is prime). Hence 8x10 = 80 cores on every big matmul; sharded in0 pins the grid too.
- Diagnostics (FF-up 1024x1280x5120, HiFi2, bf8b weights, us): shipped 10x8 74.0 (LoFi 42.2, HiFi4 134.9 ->
  math-bound at ~82% of the HiFi2 ceiling); no packer_l1_acc 88.0; in0_block_w 4->2 76.8; subblock 2x4 with
  interleaved out 83.6; in0 interleaved 82; weights pre-copied to L1 interleaved 74.0 (weights are not the
  limiter in this layout); N padded over 11 cols (per_core_N 15, interleaved in0) 77.1 vs 82 (scales with math).
- Padded M over the 11 columns (transpose_mcast, per_core_M 3 -> 33 tiles for 32; the factory already handles
  `last_per_core_M`): correct, but 77.4 us, and LoFi 75.9 -> data-movement bound. Per-RISC: BRISC (in1 sender /
  writer) 75.8 = total, NCRISC (in0) 61. Under transpose_mcast the in1 senders are the LEFT COLUMN of cores
  (normal mode: top row): a column of interleaved-DRAM readers is the NoC "column trap" from
  examples/noc_placement (~2.9x slower than a row/diagonal line).
- DRAM-width-sharded weights (factory path IN1_DRAM_WIDTH_SHARDED): correct only when per_core_N == bank shard
  width (8x8, per_core_N 20: 92.1 us = same as interleaved weights); with 10 N blocks over 8 banks the result
  is wrong (PCC 0.11 / 0.0008); forbidden with transpose_mcast; L1 width-sharded weights hung the device.
- Kernel change (this branch, env TT_MM2D_DIAG_IN1_SENDERS=1, block-sharded in0 only), in
  matmul_multicore_reuse_mcast_2d_program_factory.cpp (both the CachedProgram and the descriptor function):
  the in1 sender for N block j is the core at M block j % num_blocks_y (a diagonal), the in1 multicast covers
  the whole line (the NoC excludes the source core), receivers get the diagonal sender's coordinates, and a
  sender that owns the ragged last M block gets the receivers' height-padding writer args. Default unchanged.
  Needs the worktree's own build (`build_own`, started 15:20 with ccache).

### 2026-09-17 15:50 diagonal in1 senders: transposed 11x10 padded-M matmul 77.7 -> 57.4 us (shipped 10x8: 74.1)
- Own build of the worktree (`build_own`, Release+Tracy, ccache; tracy submodule copied from the reference
  worktree because the clone URL is unreachable here). `build` symlink now -> build_own.
- FF-up probe (device kernel us, 3 calls): shipped 10x8 74.1 / with diagonal senders 74.2 (senders were already a
  row); transposed 11x10 per_core_M=3: 77.7 / **57.4 with diagonal senders** (NCRISC 46, TRISCs 57 -> back to
  math-bound; 1.29x over the shipped config, right on the 25%-less-math prediction). PCC gate passed.

### 2026-09-17 16:10 diagonal senders: interleaved-in0 support + probe results
- First interleaved-in0 attempt hung (watcher: right-half in1 senders and top-row in1 receivers stuck in
  noc_async_atomic_barrier). Cause: with interleaved in0 the factory splits receivers into two NoC setups
  (right half rows>=1 use the swapped NoCs); an in1 sender always uses in1_noc, so a right-half sender shared
  its NoC with its own in0 receiver kernel. Fix: senders only on cores whose in0 kernel uses in0_noc (left half
  or first row: transposed -> column j % (half+1); normal -> right-half columns keep the first-row sender), and
  the in1 receiver NoC variant follows the core's in0 variant (predicate x <= half || y == first row).
  Also the in1 multicast rectangle is widened after the transpose swap (my first version widened the in0 one).
- FF-up 1024x1280x5120 (us): t11x10 sharded in0 + diag 57.4 (shipped 74.1); t11x10 interleaved in0 + diag 65.6
  (was 83 without diag); 10x8 interleaved + diag 81.8 (unchanged: its senders were a row already).
- N=1280 (to_q / proj / to_out shape): shipped 10x8 sharded 20.9; t11x10 sharded + diag 17.0; t11x10 interleaved
  32.0 -> 31.6 with diag (the small-N transposed matmul with interleaved in0 stays data-movement bound; the
  model's to_out / proj_in feed interleaved in0 and measured 21.7 on 10x8).
- 1280 transformer block on the transposed grid (SDXL_T1280, PCC 0.99735): FF-up 78.8 -> 61.0, QKV 63.8 -> 51.2,
  LN 13.3 -> 11.0 us; interleaved-in0 projections 21.7 -> 30.3 (regression to resolve), sharded-in0 N=1280
  bucket (FF2 + to_q + proj_out) 25.4 -> 38.4 avg (to be split per op).

### 2026-09-17 16:20 transposed 1280 transformer blocks (SDXL_T1280=1): 10-block model 6.81 -> 6.25 ms (-8.2%)
- Per op (us): FF-up 78.8 -> 61.0, QKV 63.8 -> 51.2, FF2 ~72 -> ~56 (in0_block_w 16: one K block per core),
  to_q/proj_out 25 -> 17, LN 13.3 -> 11.0. to_out stays on the shipped 10x8 config with interleaved output
  (transposed + interleaved in0 was 31.6; 10x8 + interleaved out 25.2 vs 21.4 sharded out); the residual adds on
  mixed layouts cost 5.5 vs 0.9 us. Bug fixed on the way: `self.matmul_configs` aliased matmul_versions["80_cores"],
  so the transposed overrides leaked into the "_IL" fallback (now a dict copy).

### 2026-09-17 16:27 v7 = 62.27 ms (-19.2% vs 77.07), UNet PCC 0.99846
- Transposed 1280 transformer blocks on 11x10 with diagonal weight senders: matmul 30.00 -> 26.38 ms
  (312 matmuls now on 110 cores), LN 3.17 -> 2.76. Side effects: BinaryNg 1.19 -> 1.92 (residual adds on mixed
  sharded/interleaved layouts after to_out), Reshard 0.15 (14 ops: the transformer's final residual onto the
  11x10 shard).
- Fixed: the 640 proj_in override (in0_block_w 2) now also sets a 4x2 subblock — with a block-sharded output the
  subblock must span per_core_N; the shipped 2x1 literal was only legal with the reference's interleaved output.

## Part 2 (2026-09-17 16:45): generated GroupNorm on the VAE decoder (DRAM path)
- Reference VAE decode 1024x1024 profile (reference worktree): 265.8 ms device. GroupNorm 99.1 ms (30 ops on
  8x8; 1048576x128 = 10.2 ms x6, 1048576x256 11.6, 262144x256 2.9 x5), Conv2d 47.7, PaddedSlice+SliceWrite 52.3
  (DRAM conv slicing glue), SiLU 27.6 (~950 us per call, a full DRAM pass), residual adds 10.3, SDPA+matmul 14.6.
- Wiring: `run_group_norm(..., placement="dram")` keeps the activation DRAM-interleaved (no I2S/S2I, no
  reciprocals shard), out of place, SiLU fused (`VAEModelOptimisationsBH.fuse_gn_silu`, env SDXL_VAE_GN_FUSE_SILU;
  here the GN is DRAM-bound so the SFPU pass is free). Resnet / decoder / encoder call sites; reference path intact.
- VAE resnet PCC: 11/11 pass (2 cells still skipped by the test for the reference DRAM-GN PCC issue).
- VAE decode with the generated GN (DRAM path, SiLU fused; attention GN too): **265.8 -> 172.9 ms (-35%)**;
  GN 99.1 -> 35.7 ms, SiLU 27.6 -> 0, I2S/S2I 1.2 -> 0.3. Decoder PCC 0.961 (gate 0.93). Per cell (us):
  1048576x128 10182 -> 2530 (incl. fused SiLU; ~2100 without), 262144x256 ~2930 -> ~1450, 65536x512 1015 -> 709,
  262144x512 3857 -> 2572, 16384x512 164 -> 200 (sharded reference was faster on this small one).
  The fused SiLU is NOT free on the streaming path either (~+400 us on 1048576x128) but far cheaper than the
  separate 950 us pass.

### 2026-09-17 17:15 "temporal" channel rounds — implemented, measured, slower; left opt-in (GN_TEMPORAL_ROUNDS=1)
- Implementation: `create_program_descriptor(channel_tile_offset, channel_tiles)` normalizes a contiguous slice of
  whole groups of the same DRAM tensor (full-tensor tile-row stride, slice-local group ids via a new writer common
  arg `group_base`, absolute channels for gamma/beta); `plan_channel_rounds` picks the fewest rounds whose per-core
  block (all cores on HW, Kt tiles wide) fits L1; the op launches one resident program per round.
  Correct on every VAE shape (PCC >= 0.99999, better RMS than streaming).
- Measured (us, 110 cores, bf16 DRAM): 1048576x128 4 rounds of K=1: 5948 vs streaming 2099; 262144x256 2 rounds
  of K=4: 1352 vs 1078; 262144x512 4 rounds of K=4: 2675 vs 2113; 65536x512 resident either way 502 vs 499.
- Why: the rounds are compute-bound (TRISC = total, NCRISC ~550 of 1460 us per round). Per tile-pass the resident
  three-pass compute costs ~1.6 us against ~0.9 us on the streaming two-pass path, and at K=1 every per-row fixed
  cost (chain / reduce init, expansions) is amortized over one tile. The DRAM saving (3V -> 2V) is only 1.5x, so
  it cannot win while the resident compute is 2-5x slower per tile. Widening K needs 2.4 MB/core for 1048576x128.
  Letting the K=1 chains use 4-tile DEST blocks (compute.cpp `b`) changed nothing measurable.
- Takeaway: the lever for these shapes is the resident compute path's per-tile cost (the op's "Overlap (resident
  pass 1)" and chunk-size lamps), not the traffic; the streaming two-pass stays the default.

### 2026-09-17 17:17 end-to-end demo (generated GN in UNet + VAE)
- `demo.py::test_demo[1024x1024, no_cfg_parallel, device_vae, device_encoders, with_trace]`, 50 steps,
  "An astronaut riding a green horse": PASS, image at output/output1.png (copy in the session scratchpad as
  sdxl_generated_gn_astronaut.png). Denoising loop 6.59 s for the batch, on-device VAE decode 0.18 s
  (reference VAE decode profile: 0.27 s device time). Reference-path VAE decoder PCC 0.9305 vs 0.961 generated.
- Reference demo (SDXL_GENERATED_GN=0), same prompt/seed/settings: denoising loop 8.13 s (vs 6.59 s, -19%),
  on-device VAE decode 0.27 s (vs 0.18 s, -33%), image gen 8.62 s (vs 7.02 s). Both images are the same
  composition (astronaut on a green horse, yellow sun, stars); details differ at the level expected from a
  different normalization numerics (copies: sdxl_generated_gn_astronaut.png / sdxl_reference_gn_astronaut.png in the
  session scratchpad).

## Part 3 (2026-09-17 17:20): "full transposed" UNet
### v8 = 59.66 ms (-22.6% vs 77.07), UNet PCC 0.99847 — 640 transformer blocks transposed too (SDXL_T640)
- 4096-row blocks on 11x10 (per_core_M 12): 4096x640->2560 84 -> 64 us, ->1920 78 -> 61, 640->640 36 -> 30,
  LN 26 -> 20. proj_in now takes the GN output resharded onto the block shard (both 640 and 1280 blocks).
  Matmul total 30.0 -> 24.3 ms (512 of 684 matmuls on 110 cores), LN 3.67 -> 2.59.
- Remaining non-transposed: the resnet path (GN 10x10 / 10x8, convs 100 / 80 cores). Next: COL_MAJOR support in
  the generated GN's host mapping (HW over the 11 columns, C over the 10 rows) + conv transpose_shards.

### 2026-09-17 17:30-18:05 full transposition of the resnet path: v9 59.13 -> v10 (upsample bug) -> v11 58.74 ms
- Generated GN: `_shard_geometry` now accepts COL_MAJOR block shards (core (x, y) holds HW block x / channel block y;
  `hw_splits, c_splits = nx, ny`; per-core `c0 = y*shard_w, s0 = x*shard_rows`). Kernels unchanged.
- Model: `transposed_resnets` (SDXL_T_RESNET=1 default, needs T1280+T640). `generated_gn_grid_transposed(HW, C)`:
  HW over up to 11 columns, C over 10 rows, COL_MAJOR; columns drop while the padding rule fails or the per-core
  tile-row count has no usable divisor (16384 rows -> 47 per core over 11 = prime -> 10 columns x 52; 4096 -> 11 x 12;
  1024 -> 11 x 3 = exactly the transposed transformer shards, so GN -> proj_in and resnet -> attention GN need no
  reshard). Every BLOCK_SHARDED Conv2dConfig gets `transpose_shards=True`; down_blocks.0 convs move to 26-tile
  blocks (32-tile blocks made conv2d re-pick 8 HW cores). Transformer GN runs on the block shard directly.
- v9 = 59.13 ms (UNet PCC 0.99846): conv 9.47 -> 9.01 (1024-row 1280 convs 165 -> 158 us on 110 cores, 4096-row
  640->640 178 -> 152, 1920->640 524 -> 398), Reshard 25 -> 17 calls. All convs are math-bound (BR ~ NC ~ TRISC):
  1280->1280 at 1024 rows ~85% of HiFi2 peak, so the column-of-weight-senders question is moot for them.
  GN 1024x1280 got slower on 110 cores (444 -> 478 us / 16 calls: combine over more cores) - minor.
- Upsamplers: interpolate resharded 4096-row resnet outputs to 8x8 ROW_MAJOR (90 us reshard) and the two upsampler
  convs stayed on 80 cores (825 + 633 us). Upsampling on the COL_MAJOR shard hit two bugs in
  `upsample_program_factory_multicore_sharded.cpp` (fixed, C++ rebuilt): the config tensor put the NHW core index in
  y and channel core in x regardless of orientation, and it was distributed column-major (must be row-major for
  COL_MAJOR data); stick bytes were divided by the x extent (must be the channel extent). Probe: exact for
  ROW_MAJOR 8x8, COL_MAJOR 8x10, ragged COL_MAJOR 11x10 (both model shapes) and ragged ROW_MAJOR 10x10.
- v10 (upsample on the 4096-row grid, 11 columns): the 16384-row upsampler conv output has 47 tiles per core (prime)
  -> act_block_h 1 -> 1838 us. Fix: pick the upsample grid for the UPSAMPLED row count (10 columns x 52 tiles for
  16384 rows, input shard a quarter of that), conv configs ABH_13T (26T clashes with L1 by 135 KB) and ABH_0.
- v11 = 58.74 ms (UNet PCC 0.99846): upsampler convs 825 -> 701 (100 cores) and 633 -> 470 us (110 cores), the
  8x8 reshards gone (Reshard 0.19 -> 0.08 ms). Net for the full transposition vs v8: -0.92 ms (-1.5%).
- Where the step goes now (58.74 ms): matmul 24.24, SDPA 9.64, conv 8.73, create_qkv_heads 5.29 (!), LN 2.59,
  concat_heads 1.92, GN 1.60, SiLU 1.25, adds 1.08, I2S 0.92. create_heads / concat_heads run on 32 cores for the
  1024-row blocks (work split = one tile-row per core) and on 3 cores for the 96-row encoder K/V: ~4.7 ms of pure
  data movement on <= 32 cores -> next target (factory work split).
- Not tried: ABH_26T without act double buffer on the 640 upsampler conv (~80 us at stake).

### 2026-09-17 18:05-18:25 create_qkv_heads / concat_heads on the full grid: v12 = 54.05 ms (-29.9% vs 77.07)
- Both interleaved factories split work by whole tile rows (num_blocks = B*S/32): 32 cores for the 1024-row
  blocks, 3 cores for the 96-row encoder K/V, and the kernels moved one tile per NoC barrier. 4.7 ms/step of pure
  data movement.
- New tile-split kernels (used whenever there is no transpose_k_heads compute stage; the sharded factories are
  untouched): `nlp_create_qkv_heads/.../reader|writer_tm_tile_layout_nlp_create_qkv_heads_tiles.cpp` and
  `nlp_concat_heads/.../reader_tm_tile_layout_nlp_concat_heads_tiles.cpp`. Every core owns a contiguous range of
  the flattened (tile-row, row-tile) index space; reads/writes go 8 tiles per barrier (CB = 2 chunks). KV_TIED and
  READ_FROM_INPUT_TENSOR_KV handled in the reader; the writer scatters (q | k | v) tiles by head. Runtime-arg
  address slots unchanged so override_runtime_arguments still works. concat_heads keeps the shared metal2 writer
  (contiguous output pages) and only the reader/work split changed.
- Unit tests: test_nlp_create_qkv_heads.py 147/147, test_nlp_concat_heads.py 217/217 pass.
- UNet step: create_heads 5.29 -> 1.66 ms (1024x3840 38.7 -> 13.2 us on 110 cores, 96x1280 12.4 -> 1.3 us,
  4096x1920 40.1 -> 24.5), concat_heads 1.92 -> 0.90 (13.6 -> 5.9 us). UNet PCC 0.99846 (unchanged).
- Step breakdown now: matmul 24.24, SDPA 9.58, conv 8.73, LN 2.59, create_heads 1.66, GN 1.60, SiLU 1.25,
  adds 1.08, I2S 0.92, concat_heads 0.90, concat 0.67.
- Gotcha: `VAEModelOptimisationsBH` subclasses the UNet config, so the transposed conv flag leaked into the VAE
  convs (L1 CB clash in the decoder mid-block). Class attribute `TRANSPOSED_RESNETS` (False in the VAE subclass)
  gates it now.
- Demo (50 steps, traced, device VAE + encoders, 4 prompts on 4 chips): PASS, denoising loop 6.31 s (v7: 6.59,
  reference 8.13), VAE 0.18 s, image gen 6.72 s. The loop is ~126 ms/step wall vs 54 ms device time, so host /
  dispatch overhead dominates e2e and only ~0.3 of the 0.4 s device saving shows up (scheduling is out of scope
  per the user). Image: output/output1.png (scratchpad copy sdxl_v12_astronaut.png).

## Part 4 (2026-09-17 18:40-20:40): precision investigation ("image looks slightly worse")
- 50-step UNet loop vs torch fp32 (test_unet_loop, seed 0, per-step PCC; dumps in the session scratchpad
  loop_gen.pt / loop_ref.pt): generated path ends at 0.867 (gate 0.905, FAIL), reference 0.915. Curves identical
  for steps 0-5, generated drifts faster afterwards; generated latents develop a negative bias (-0.03 by step 3,
  -0.09 at the end) and shrink (std ratio 0.85 vs 0.92). Same result without trace (--debug-mode) and with all
  transposition / sharded-640 / matmul-config switches off (0.8735) -> intrinsic to the generated-GN path.
  Image PCC vs the torch decode of the torch latents: generated 0.851, reference 0.904.
- The GroupNorm op itself is NOT the problem: on 46 real GN calls of one UNet step the generated op is 12x more
  accurate than ttnn.group_norm (rms 0.0014 vs 0.017, pcc 0.999998 vs 0.99988, no bias); synthetic probe agrees.
  Every module test is also more accurate with it (resnets rms 0.007-0.028 vs 0.016-0.059, no bias).
- Localized with per-stage dumps in test_module_tt_crossattnupblock (up_blocks.0 fails: 0.9488 vs gate 0.968,
  reference 0.9707): resnet output 0.99984 (ref 0.99958), attention GN 0.99983 (0.99950), proj_in 0.99984
  (0.99960), but after the 10 transformer blocks 0.984 vs 0.992. Inside transformer block 0 every op is more
  accurate on the generated path too. The reference GN output is 3.8% too SMALL (std ratio 0.962); scaling the
  generated GN output by 0.9624 (SDXL_GN_SCALE_HACK) recovers most of the gap (block 0.949 -> 0.965, transformer
  stage 0.984 -> 0.989). The self-attention output is 3.7% too small on every path (attn1 std ratio 0.963) ->
  the transformer blocks have a magnitude-sensitive systematic error that the reference GN's shrinkage was
  partially cancelling ("lucky quantization"). The generated GN removes the compensation.
- Investigation hooks (all env-gated, harmless when unset): SDXL_LOOP_DUMP / SDXL_LOOP_GOLDEN (test_unet_loop),
  SDXL_GN_DUMP (sdxl_utility.run_group_norm), SDXL_UPBLOCK_DUMP (up block / transformer / transformer block /
  upsample), SDXL_UNET_DOUBLE_CALL (test_module_tt_unet), SDXL_GN_SCALE_HACK, BIAS log lines in the module tests.
- Reference (CPU fp32) image: the torch-loop latents of test_unet_loop decoded with the torch VAE
  (scratchpad loop_final_torch.png; TT decodes of the same run: loop_final_tt_gen.png / loop_final_tt_ref.png).
  A standalone diffusers CPU run (scratchpad sdxl_cpu_reference.py, 42 min) produced pure noise - script bug, not
  investigated further since the loop test already provides the fp32 trajectory. Visual: the torch image has the
  sun-with-tower background and a star; the generated-GN image keeps the composition but the visor turns red and the
  background objects shift; the reference-GN image is closer to torch (visor, backpack colour, background).

## Part 5 (2026-09-18 00:00-02:00): root cause of the accuracy loss = two systematic GAIN errors, not the GN
Method (as asked): env-gated fidelity / fp32 knobs on every op class, then op-class sweeps on the failing module
(up_blocks.0, gate 0.968) and synthetic single-op probes reporting the least-squares GAIN of device output vs torch
fp32 (`fullgrid_worklog/test_precision_probe.py`; per-stage transformer-block gains: `stage_gains.py` on the
SDXL_UPBLOCK_DUMP dumps). Knobs (all default to the shipped values): SDXL_SDPA_FIDELITY / SDXL_SDPA_FP32,
SDXL_MM_FIDELITY / _FP32 / _L1ACC (+ automatic out-subblock cap for fp32), SDXL_CONV_FIDELITY / _FP32 / _L1ACC,
SDXL_CONVIO_COMPUTE=<compute config name> (conv_in / conv_out only), SDXL_LN_FIDELITY / _FP32,
SDXL_ATTN_W_DTYPE / SDXL_FF_W_DTYPE, SDXL_QKV_PREROUND=q,k,v bits (host round trip, experiment only).
Also fixed: test_module_tt_crossattnupblock.py was broken by the Part 4 hooks (`_mk` used outside `if _dump`).

### Sweep on up_blocks.0 (PCC, gate 0.968; shipped generated-GN path 0.9488, reference GN 0.9707)
- SDPA LoFi -> HiFi2: **0.9897**. SDPA HiFi4: 0.9912. (+ LN HiFi4 fp32: 0.9934; LN alone: 0.9565.)
- matmul HiFi4: 0.9494 (no change); bf16 attention/FF weights: 0.9492 (no change); conv HiFi4+fp32: L1 clash.
- reference GN + SDPA HiFi2: 0.9944 -> the transformer blocks were the error source on both paths; the reference
  GN's 3.8% shrinkage only hid it (Part 4).
- Per-stage gains in transformer block 0 (shipped): ln1 0.999, **attn1 0.962**, attn2 1.000, ln2/ln3 1.003/1.005,
  **ff 1.013**. With SDPA HiFi2: attn1 0.998, ff 1.017 (see "second error" below).

### Error 1: SDPA in LoFi shrinks the attention output by ~4-6% (systematic, not noise)
- Synthetic (1,20,1024,64) / (1,10,4096,64), random q,k,v: LoFi gain 0.941 / 0.940 whatever fp32_dest_acc or
  exp_approx_mode; HiFi2 1.003 / 1.000; HiFi4 1.008 / 1.006 (fp32 acc: 0.997-0.999). Cross-attention (96 keys) is
  nearly unaffected (flat softmax), which is why attn2 was fine.
- Mechanism (tech_reports/matrix_engine): the FPU is a 5b x 7b multiplier. LoFi uses 1 hidden + 4 MSBs of SrcA
  (= operand 1 / in1) and 1 hidden + 6 MSBs of SrcB (operand 0 / in0) and DROPS the rest (truncation toward zero,
  no rounding). Measured on a plain matmul: rounding in1 to 4 explicit mantissa bits (RNE) and in0 to 5-6 bits
  before a LoFi matmul makes it exact (gain 1.00001); in1 alone accounts for ~1.95% loss, in0 ~0.28%. In SDPA
  in1 is K for Q.K^T and V for P.V (compute_common.hpp matmul_blocks(cb_q_in, cb_k_in) / (cb_qk_im, cb_v_in)):
  scores 2% too small -> flatter softmax, and P.V 2% too small -> attention output ~4% (real data) to 6% (random)
  too small. It is a property of the fidelity, not a kernel bug; unbiased at LoFi would need RNE pre-rounding of
  K and V to 4 bits where they are produced (host emulation SDXL_QKV_PREROUND=8,4,4: up_blocks.0 0.9488 -> 0.979,
  still below HiFi2 because P and Q keep their 6-bit truncation).
- Cost of HiFi2 SDPA on the model shapes (tracy, 110 cores): 1024x1024 83.0 -> 86.5 us, 4096x4096 422 -> 431 us,
  cross-attention 21.0 -> 21.4 / 28.6 -> 28.8 us: the op is not FPU-bound, HiFi2 costs +3-4% of SDPA time
  (~+0.3 ms of the 9.6 ms/step). HiFi4 would be +12% at 1024 rows for no accuracy gain over HiFi2.
- **Fix: SDPA compute config HiFi2** (tt_attention.py knob SDXL_SDPA_FIDELITY; default still LoFi pending the
  user's call — see recommendation).

### Error 2: conv_out / conv_in compute config (HiFi2, fp32 off, packer_l1_acc off) INFLATES the output
- Synthetic conv 3x3 320->32 @128x128 (= conv_out shape class): HiFi2 fp32=0 l1acc=0 gain **1.037** (bias +0.003),
  fp32=0 l1acc=1 0.9998, fp32=1 0.9976, HiFi4 fp32=0 l1acc=0 1.038 (fidelity irrelevant). conv 32->320 (conv_in
  class): 1.0036. Plain matmul K=1280: HiFi2 fp32=0 l1acc=0 gain 1.018 for ANY number of K blocks (1..40), and
  with l1acc=1 the gain follows the tiles accumulated inside DEST per block (in0_block_w 40/10/5/2/1 -> 1.018 /
  1.011 / 1.005 / 1.0006 / 0.999). So the inflation is the bf16 (16-bit) DEST accumulation itself: every MAC
  result rounded into the 16-bit accumulator with a magnitude-increasing bias (~0.05-0.15% per accumulated tile);
  the spill/reload of partials is not the cause. fp32 DEST or packer L1 accumulation (fp32 in the packer) with a
  short in-DEST accumulation avoids it.
- Model: UNet single step (test_module_tt_unet) shipped std_ratio 1.0177 / rms 0.0345 (reference GN 0.975 / 0.047);
  SDPA HiFi2 alone 1.0162 / 0.0280; conv_in+conv_out on CONV_HIFI2_NO_FP32 (l1acc on) 0.9980 / 0.0316;
  **both 0.9965 / 0.0241**; both with fp32 convs 0.9863 / 0.0250. conv_out scales the noise prediction by ~2%
  every denoising step, which is what drove the loop latents off.
- **Fix: conv_in / conv_out on CONV_HIFI2_NO_FP32_COMPUTE_CONFIG (packer_l1_acc=True)** (knob SDXL_CONVIO_COMPUTE).
  Perf cost: see the profile below (two small convs).

### 50-step UNet loop vs torch fp32 (test_unet_loop, seed 0, gate 0.905)
- shipped generated-GN path 0.8674 (reproduced), SDPA HiFi2 only 0.9003, **SDPA HiFi2 + conv io l1acc 0.9621**
  (PASS; reference GN path was 0.915 in Part 4), reference GN + both fixes 0.8992 (the reference GN's own
  shrinkage is now uncompensated) -> generated GN + the two fixes is the most accurate configuration by a wide
  margin, and the GN op was never the problem.

### Remaining systematic error (not fixed, documented): matmul DEST inflation
- All model matmuls run HiFi2 / fp32 off / packer_l1_acc on with in0_block_w 2..16 -> +0.1% to +1.5% gain per
  matmul (FF stage gain 1.013-1.017 in block 0; transformer block outputs std_ratio 1.007-1.019 vs torch).
  Tried SDXL_MM_FP32=1 (fp32 DEST, subblocks capped at 4 tiles): ff gain 1.017 -> 1.003 but attention gain
  0.998 -> 0.990 (the HiFi2 + fp32-DEST path shrinks 0.28% per matmul, srcB truncation) and up_blocks.0 PCC
  0.9897 -> 0.9869, plus sharded-output subblock rules break for some configs and fp32 DEST halves the register
  file -> not worth it. Cheaper alternative if ever needed: smaller in0_block_w (more L1-acc rounds, each in fp32).
- Also measured, minor: LayerNorm HiFi2 gain 0.996 (HiFi4 0.9999; up_blocks.0 +0.004 PCC on top of SDPA HiFi2;
  LN is 2.6 ms/step, HiFi4 cost not measured), fast GELU gain 0.9976 with +0.005 bias.

### Perf cost of the two fixes (UNet step, tracy, same session): 54.01 -> 54.49 ms (+0.9%)
- SDPA 9.59 -> 10.04 ms (140 ops, HiFi2), Conv2d 8.73 -> 8.73 (conv_in/conv_out with packer_l1_acc: free),
  matmul 24.21 -> 24.23. up_blocks.1 with the fixes: PCC 0.9965 (gate 0.993). UNet single step passes.
- Defaults are NOT flipped in the code (the Part 4 constraint forbids fidelity / fp32 / l1acc changes without the
  user's call); to adopt: `SDXL_SDPA_FIDELITY=HiFi2 SDXL_CONVIO_COMPUTE=CONV_HIFI2_NO_FP32_COMPUTE_CONFIG`, or make
  them the defaults in tt_attention.py (sdpa_compute_kernel_config) and get_conv_compute_config (conv_in/out).

### 2026-09-18 06:15 final demo with both precision fixes (SDPA HiFi2 + conv_in/conv_out packer_l1_acc)
- `SDXL_GENERATED_GN=1 SDXL_SDPA_FIDELITY=HiFi2 SDXL_CONVIO_COMPUTE=CONV_HIFI2_NO_FP32_COMPUTE_CONFIG`,
  demo.py::test_demo[... no_cfg_parallel-1024x1024], 50 steps, traced, device VAE + encoders: PASS.
  Image gen 6.78 s (4 prompts), denoising loop 6.36 s (v12 without the fixes: 6.31; reference 8.13),
  on-device VAE decode 0.18 s. Image: output/output1.png, copy fullgrid_worklog/final_astronaut_hifi2_convio_l1acc.png.
  Visually: torch-like composition (yellow sun, stars, clouds, white visor); the red visor of the Part 4 image is gone.
