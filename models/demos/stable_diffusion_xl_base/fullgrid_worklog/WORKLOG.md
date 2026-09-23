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
  on-device VAE decode 0.18 s. Image: output/output1.png, copy fullgrid_worklog/final_astronaut_hifi2_convio_l1acc.jpg.
  Visually: torch-like composition (yellow sun, stars, clouds, white visor); the red visor of the Part 4 image is gone.

### 2026-09-18 13:40 DEST rounding probe (`test_dest_rounding_probe.py`): the 16-bit accumulate is NOT RNE
- Deterministic 1-core matmul with controlled ties (fp32 off, l1acc off): ties round AWAY from zero (256+1 -> 258,
  40 tie adds -> 336), the K-tile is accumulated in two 16-row halves each rounded, products are rounded onto a grid
  6 bits below the bf16 ULP (ties toward +inf) before summation, and the fp32->bf16 pack is ties-away as well.
  Ties-away model 964/1024 cells vs RNE 869; fp32 DEST = exact accumulate (972/1024 with a ties-away pack).
  Random data on the same path: gain 1.0011 / 1.0045 / 1.0172 at 2 / 8 / 40 K-tiles. Details: PRECISION_WRITEUP.md 3.6.

## Part 6 (2026-09-21): why the traced UNet gains 21-26% when the kernel sum says 30% (dispatch gaps)
Question: UNet kernel sum -29.9%, VAE -35%, but demo loop only -22%. Answer: it is inside the traced UNet, not host.
Probes (untracked, this dir): `test_trace_gap_probe.py` (one UNet forward captured in a metal trace, replayed 3x,
wall vs kernel; run via test_profile_probe.py wrapper, env SDXL_GENERATED_GN / SDXL_DIAG_SENDERS), `test_mm_trace_gap_probe.py`
(isolated FF-up matmul, 20 back-to-back in a trace, diag on/off, optional tiny op in between). Raw per-op device rows
incl. trace replays: `generated/profiler/<subdir>/.logs/cpp_device_perf_report.csv` (the tracy merge step asserts on
traced runs because the device profiler buffer overflows mid-replay: only the first ~1500 of 2215 ops are captured; the
`tt_unet.forward` mid-forward `ttnn.ReadDeviceProfiler` calls are host syncs and must be no-op'd during capture).
Gap metric = this op's first kernel start - previous op's last kernel end (FW-based gaps are polluted by idle cores
entering the next op's FW early). No overlap between consecutive ops was observed anywhere (min gap ~440 ns).

| config (1 UNet pass, p150a) | untraced kernel sum | traced replay wall | gap | per-op gap |
|---|---|---|---|---|
| SDXL_GENERATED_GN=0 on this branch (= reference + tile-split heads kernels, C++ unconditional) | 72.45 ms (2166 ops) | 74.61 ms | 2.2 ms | 0.78 us mean / 0.60 us median |
| generated GN, full grid, diag in1 senders (shipped) | 54.07 ms (2215 ops) | 58.63 ms | 4.6 ms | 1.85 us mean / 0.63 us median, p99 19.7 us |
| same, SDXL_DIAG_SENDERS=0 (line senders) | 59.65 ms | 61.93 ms | 2.3 ms | 0.67 us mean / 0.58 us median |
Pure reference (CI kernel sum 77.07) traced ~= 79.2 ms est. -> traced UNet speedup 58.6/79.2 = -26%; vs today's GN=0 run -21%.
Demo loop per step = 2 UNet replays + scheduler/CFG: 162.6 ms (ref) vs 127.2 ms (ours); remainder ~10-13 ms both.

Root cause of the extra 2.3 ms: the diagonal in1 (weight) sender layout. The 2D mcast factory builds the in1
receiver set as ~109 single-core CoreRanges (grid minus a diagonal); kernel groups are keyed per core by kernel set,
so the receiver kernel group can't be a few rectangles (line senders: ~4). Every launch of such a program needs many
more dispatcher writes (launch msgs / binaries per range), ~16-19 us per launch instead of ~0.7 us, and it is exposed
because the ops before these matmuls are short (LayerNorm ~12 us, I2S ~4 us). ~150 such matmuls per pass x ~16 us =
~2.4 ms. Isolated back-to-back (or after a 2 us add) the same program shows only a 0.6-0.9 us gap: the dispatcher
hides the write behind the previous op when it can run ahead. Net: diag senders save 5.6 ms of kernel time per pass but
give back 2.3 ms of dispatch; still a 3.3 ms win. Fix direction: make the in1 role a runtime arg inside one kernel
(one rectangle) or at least merge the receiver ranges per row; not done.
The ~1 us/op baseline (2.2-2.3 ms/pass on every config) is plain per-program dispatch on BH and is the same on the reference.
`model_configs_1024x1024BH.py`: SDXL_DIAG_SENDERS=0 pops TT_MM2D_DIAG_IN1_SENDERS (A/B knob, default unchanged).

### 2026-09-21 12:05 FIX: unified in1 sender/receiver kernel (role = runtime arg 0) -> traced UNet 58.63 -> 56.57 ms
- `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`: under `IN1_UNIFIED_ROLE`, rt arg 0 selects the role; the
  receiver path (`in1_receiver_writer_main`) is the old receiver kernel body reading the sender's compile-time-arg
  layout (in1_block_num_tiles=CT6, num_blocks=7/8/9, sems 10/11, batch 15, out strides 19-24, subblock 25-27, MtNt 28,
  bias block w = CT4, reduce_scatter flag CT31, out TensorAccessor after in1/sparsity accessors at CT33).
- 2D mcast factory (descriptor path only): `in1_unified = diag_in1_senders && !TT_MM2D_IN1_SPLIT_KERNELS`; the sender
  kernel's core set = senders U left-half receivers, no separate left receiver kernel, per-core RTAs get the role
  prepended (variant tensor slots shift +1). The right-half "other NoC" receiver kernel is untouched (needs its own
  NoC config). Whole 11x10 grid = one kernel group = one multicast rectangle (in0 block-sharded kernel is already unified).
- Isolated FF-up matmul: kernel 58.5 us unified vs 58.8 split (same), PCC pass. Traced UNet forward (1024x1024, tracy):

| in1 kernels (diag senders on) | untraced kernel sum | traced wall | gap | gap/op (first 1492 ops) | pairs >5 us |
|---|---|---|---|---|---|
| split (before) | 54.09 ms | 58.63 ms | 4.5 ms | 1847 ns | 97 (1.55 ms) |
| unified (fix)  | 54.03 ms | 56.57 ms | 2.5 ms | 753 ns | 2 (0.01 ms) |
  Reference-GN traced gap for comparison: 2.2 ms / 0.78 us per op. UNet PCC 0.99846 (unchanged), bias/rms identical.
  Kernel time kept (diag senders' 5.6 ms win intact), dispatch overhead recovered (-2.06 ms/pass, ~-4 ms per demo step).

### 2026-09-21 12:15 throughput / CFG modes (demo, 50 steps, traced, device VAE+encoders, unified in1 kernel + precision fixes)
- CFG is hardcoded on (do_classifier_free_guidance=True); two mappings exist: `no_cfg_parallel` = DP=4 (one prompt per
  chip, cond and uncond as two sequential batch-1 UNet passes per step on the same chip), `use_cfg_parallel` = TP=2 x DP=2
  (cond/uncond on two chips, one pass per step each + all_gather of the noise pred, 2 prompts in flight).
- no_cfg_parallel measured: image gen 6.63 s / 4 images (denoising loop 6.20 s = 124 ms/step = 2 x 56.6 ms UNet + ~11 ms
  host/scheduler; VAE 0.18 s) -> 0.60 img/s, 6.6 s latency. Before the dispatch fix: 6.78 / 6.36; reference: 8.62 / 8.13.
- use_cfg_parallel NOT measurable: fabric init fails deterministically ("Fabric Router Sync timeout on Device 2", eth
  chans 4-7 stuck at STARTED, chans 8-11 fine -> one Ethernet link of chip 2 is down). dmesg shows two fatal PCIe AER
  events today (11:57 root port 00:01.1, 12:00 root port 40:01.1 / chip 0000:41:00.0, "recovery failed") that coincide
  with the two crashed device opens (bus error, stale sysmem). PCIe-side dispatch still works on all 4 chips (demo DP run
  passed at 12:11), only the eth link is unhealthy. Needs a chip reset (tt-smi -r) or host attention; not done (shared box).
- 12:18 `tt-smi -r` on all 4 chips (user-approved) fixed the fabric: eth link of chip 2 came back (the AER lines
  printed during the reset are the reset's own link retrain). use_cfg_parallel then measured:
  | mode | config | image gen | loop | per step | img/s | latency/img |
  |---|---|---|---|---|---|---|
  | DP=4 (no_cfg_parallel) | branch + fix | 6.63 s / 4 img | 6.20 s | 124 ms (2 UNet passes) | 0.60 | 6.6 s |
  | TP=2 x DP=2 (use_cfg_parallel) | branch + fix | 3.61 s / 2 img | 3.19 s | 63.8 ms (1 pass + all_gather) | 0.55 | 3.6 s |
  | TP=2 x DP=2 | GN=0 on branch (ref GN, still new heads kernels) | 4.48 s / 2 img | 3.98 s | 79.6 ms | 0.45 | 4.5 s |
  CFG-parallel: 1.8x lower latency for -8% throughput vs DP=4. Per step 63.8 = 56.6 UNet + ~7 ms (all_gather + CFG
  combine + scheduler + host), i.e. the fixed per-step cost is ~11 ms in DP mode and ~7 ms here.

### 2026-09-21 13:30 where the per-step "non-kernel" time really goes (one-chip full-step probe) + guidance-rescale fix
`test_step_trace_probe.py` (1x1 mesh, real TtSDXLPipeline, 50 traced steps; then replays of the pipeline's own step
trace and of a UNet-x2-only trace, plus order/idle/data experiments). Findings, generated-GN branch, unified in1 kernel:
- Host is NOT a factor: execute_trace enqueue 0.02 ms, scheduler's 4 scalar copies 0.02 ms per step; the whole step
  (2 UNet passes + CFG combine + rescale + scheduler.step) is one trace.
- Step trace 117.8 ms = UNet x2 111.8 + 6.0 ms of guidance/scheduler ops. Profiled: the two `ttnn.std` calls of the
  guidance RESCALE block are WelfordReduce on ONE core, 2.98 ms each (5.96 of the 6.0 ms). Same cost on the reference
  path (153.1 = 147.1 + 6.0). The demo runs guidance_rescale = 0.0, so both stds are multiplied by 0 and discarded.
  FIX (test_common.run_tt_image_gen `skip_guidance_rescale`, passed by TtSDXLPipeline when guidance_rescale == 0):
  step trace 117.8 -> 111.9 ms (= exactly UNet x2), bit-identical output (x*1 + y*0). Inpainting loop left as is.
- Loop 123.9 vs step trace 117.8: the SAME trace replays 3.3-3.7 ms slower after ~50 back-to-back executions (also
  after 50 replays with no scheduler copies), and recovers after 10 s idle. Not data (latents restored/zeros/x10: same),
  not capture order, not allocator addresses. AICLK stays 1350 MHz, power ~59-61 W, temp 42->49 C (tt-smi 0.5 s
  samples; only aiclk/voltage/current/power/temp exposed). Looks like sustained-load power management below the
  AICLK granularity (voltage 0.71-0.82 V observed); the reference (80-core matmuls, lower power) shows no such penalty
  (153.28 loop = 153.14 trace). ~3% of our step; not fixable in software here.
- After the rescale fix: one-chip loop 117.5 ms/step (= 111.9 + ~5.6 sustained-load). 4-chip demo (host load avg ~30
  from other eval jobs, so encode/VAE/load numbers are inflated; the traced loop is device-bound and clean):
  DP=4: loop 6.20 -> 6.02 s (120.4 ms/step: 111.9 + ~5.6 + ~3 slowest-of-4-chips); CFG-parallel TP2xDP2: 3.19 -> 2.96 s
  (59.2 ms/step = 56.6 UNet + all_gather/combine + slowest-of-2). Reference DP loop today 7.66 s (153.3 ms/step).
  Per-step accounting now: ref 153.3 = 2x72.45 kernel + 2x2.2 dispatch gaps + 6.0 rescale stds (no throttle);
  ours 120.4 = 2x54.0 kernel + 2x2.5 gaps + 0 + ~5.6 sustained-load + ~3 mesh skew.

### 2026-09-21 13:42 e2e latency, 20 steps, CFG on (TP2 x DP2 = 2 prompts on 4 chips), traced, device VAE + encoders
`test_demo_20steps_probe.py` (STEPS / PROMPTS env), quiet host (load avg < 5). Both prompts finish together.
| path | image gen (2 img) | denoising loop | per step | VAE | encode + latents + misc |
|---|---|---|---|---|---|
| branch (generated GN, unified in1 kernel, rescale skip, HiFi2 SDPA, conv io l1acc) | 1.54 s | 1.13 s (56.5 ms) | 56.5 ms | 0.18 s | ~0.23 s |
| reference GN on branch (GN=0; still has the new heads kernels) | 1.96 s | 1.47 s (73.5 ms) | 73.5 ms | 0.27 s | ~0.22 s |
Latency per image 1.54 s vs 1.96 s (-21%); throughput 1.30 vs 1.02 img/s.

### 2026-09-21 13:55 SDPA / matmul config sweep (isolated, HiFi2, BH p150a; probes test_sdpa_sweep_probe.py, test_mm_sweep_probe.py)
SDPA kernel us (min of 3), 11x10 grid unless noted:
| shape | shipped | best | others |
|---|---|---|---|
| 20h x 1024 self | q128/k1024 85.7 | **q256/k512 72.7** | q256/k1024 80.1, q128/k512 82.4, 10x8 grid 82.2, q64 130-136 |
| 10h x 4096 self | q128/k512 435 | (shipped) | q128/k1024 434, q256/k1024 435, q256/k512 449, q64 761 |
| 20h x 1024 cross (Skv 96) | q128 21.4 | **q256 18.9** | k96 20.4, q512 26.7 |
| 10h x 4096 cross | q128 28.3 | (shipped) | q512 30.0, q256 31.5 |
exp_approx_mode=True: no measurable change anywhere (SDPA is not SFPU-bound here). Attention-level PCC vs torch is
unchanged or slightly better for the picks (0.99969 vs 0.99964). Root cause of the 1024 win: 20 heads x 8 q-chunks =
160 work items on 110 cores -> 2 rounds with half the grid idle; q256 -> 80 items = 1 round.
Matmul (transposed 11x10, M=1024): in0_block_w / out_subblock sweeps around the shipped values found nothing better
(1280x1280 17.1, x3840 44.0, x5120 57.3; FF-down 5120->1280 already at in0_block_w=16 ~56 us; out_subblock_h=3 rejected
by validation with a sharded output). Big matmuls run at 185-240 TFLOP/s; not worth more here.
APPLIED: sdpa_configs 1024_K = q256/k512 (env SDXL_SDPA_Q1024/K1024), new 128_K_S1024 = q256/k128 for cross-attn in the
1024-row blocks (SDXL_SDPA_QX1024); 4096-row blocks unchanged. Result: traced UNet pass 56.57 -> 55.16 ms (-1.4 ms);
UNet PCC 0.99846 -> 0.99918, rms 0.0345 -> 0.0240, std_ratio 1.0177 -> 0.9971 (the k=512 chunk also removes most of
the residual output-gain error; see Part 5).
Not done (bigger change, ~2.5 ms/step): the cross-attention K/V projections of the 77 encoder tokens (96x2048 @
2048x1280/640 on 30-40 cores, 140 matmuls, 2.5 ms) do not depend on the latents and could be computed once per prompt
outside the denoising trace.

## Part 7 (2026-09-21 14:10-14:40): diagonal weight senders for conv2d (the matmul trick, applied to conv)
Question (user): the transposed convs have the same left-column-of-weight-senders geometry as the matmul had
(`transpose_shards` -> `transpose_mcast`, senders at x=0 multicasting along their row); Part 3 called them math-bound
and skipped it. Tried anyway.
- Factory `conv2d_op_sharded_program_factory.cpp` (descriptor path), env `TT_CONV_DIAG_WEIGHT_SENDERS=1`, block-sharded
  2D mcast only (input grid == output grid, one rectangle, no skipped-work cores, weights mcast on): the sender of row y
  is core (y % num_cores_x, y) (normal orientation: column x -> (x, x % num_cores_y)); its multicast covers the WHOLE
  line (NoC excludes the source), num_dests unchanged. Implemented the unified way from the start: ONE writer kernel on
  every core (`WEIGHTS_DIAG_UNIFIED`, 3 extra rt args: role, sender noc x/y), receivers run the old receiver protocol
  inside the sender kernel, no separate receiver kernel -> kernel group = one rectangle (Part 6 lesson). Model config:
  on by default under `transposed_resnets`, `SDXL_CONV_DIAG_SENDERS=0` disables.
- Isolated probe `test_conv_diag_probe.py` (COL_MAJOR 11x10 / 10x10 and the ROW_MAJOR 10x10 path, PCC vs torch fp32
  identical in both modes, 0.9999):
  | conv (3x3, bf8 weights, HiFi2 l1acc) | line senders | diagonal | |
  |---|---|---|---|
  | 1024 rows 1280->1280, 11x10 | 155.7 us (BR 152 NC 148 T 156) | 125.6 us (BR 121 NC 119 T 125) | -19% |
  | 4096 rows 640->640, 11x10 | 132.8 | 132.8 | 0 |
  | 16384 rows 320->320, 10x10, 26T blocks | 210.7 | 210.5 | 0 |
  So "TRISC == total" did not mean math-bound for the 1280 convs: TRISC time fell with the weight path (the compute
  was waiting on weight blocks). The small-weight convs are unaffected.
- Full UNet pass (test_trace_gap_probe under tracy, 1 untraced pass + 3 replays):
  | | untraced kernel sum | conv (40 ops) | traced replay wall |
  |---|---|---|---|
  | line senders (SDXL_CONV_DIAG_SENDERS=0) | 53.09 ms | 8.73 ms | 55.64 ms |
  | diagonal (default now) | 52.57 ms | 8.23 ms | 55.24 ms |
  Per conv on 110 cores: the ten 1024-row 1280->1280 convs 157 -> 126 us, 2560->1280 295 -> 240, 1920->1280 226 -> 183;
  the 4096-row 640 convs (152 us) and every 100-core conv unchanged. -0.5 ms kernel, -0.4 ms traced, gap unchanged.
- UNet PCC (test_module_tt_unet 1024x1024, 4ch, diagonal on by default): 0.99846 (unchanged; the weights path only
  moves the same bytes from a different core). Not tried: the reference (non-transposed) model path keeps line senders;
  the knob only engages when the model transposes its resnets.

## Part 8 (2026-09-21 14:45-15:05): GroupNorm precision isolation (why the reference GN shrinks, how Welford compares)
Question (user): the generated GN is more precise in every case, even against Welford -- is it fp32 DEST all the time?
`test_gn_precision_isolation.py`: the same bf16 input through every configuration, vs torch fp32 on the bf16-rounded
operands (gain = least-squares scale vs torch; rms = abs error on unit-variance output). UNet cells are L1 block-sharded
8x8 ROW_MAJOR in place (model placement), VAE cells DRAM interleaved TILE 8x8 + num_out_blocks. randn input:

| variant | 320@128 sh gain / rms | 640@64 sh | 1280@32 sh | 512@256 dram | 256@512 dram |
|---|---|---|---|---|---|
| ref_default (HiFi4, approx ON, fp32 off) = UNet model | 0.9754 / 0.0285 | 0.9896 / 0.0176 | 0.9822 / 0.0248 | 0.9852 / 0.0214 | 0.9944 / 0.0070 |
| ref_exact_sfpu (approx OFF, fp32 off) | 1.0062 / 0.0082 | 1.0030 / 0.0053 | 1.0028 / 0.0055 | 1.0025 / 0.0049 | 1.0014 / 0.0035 |
| ref_fp32 (approx off, fp32 on) | 1.0065 / 0.0080 | 1.0074 / 0.0091 | 1.0068 / 0.0084 | 1.0009 / 0.0030 | 1.0005 / 0.0027 |
| ref_fp32_approx (approx ON, fp32 on) | 0.9735 / 0.0289 | | | 0.9680 / 0.0355 | |
| ref_welford (default: fp32 forced, approx ON) = VAE DRAM model | 0.9741 / 0.0308 | 0.9767 / 0.0297 | 0.9761 / 0.0295 | 0.9686 / 0.0352 | 0.9669 / 0.0358 |
| ref_welford_exact (approx OFF) | 1.0010 / 0.0036 | | | 1.0006 / 0.0028 | |
| generated (HiFi4, fp32, exact) | 1.0001 / 0.0019 | 1.0000 / 0.0019 | 1.0001 / 0.0019 | 1.0014 / 0.0024 | 1.0015 / 0.0025 |
| generated_approx | 1.0008 / 0.0021 | 1.0005 / 0.0020 | 1.0005 / 0.0020 | 1.0018 / 0.0027 | 1.0019 / 0.0028 |

Findings:
1. **The reference shrinkage is the approximate SFPU rsqrt, not the 16-bit DEST.** Every row with approx ON shrinks
   1.5-3.3% whatever the DEST mode or algorithm (legacy bf16, legacy fp32, Welford); every row with approx OFF has gain
   within +0.1..+0.7%. `ttnn.group_norm` defaults `math_approx_mode=true` (groupnorm.cpp) and both reference kernels call
   `rsqrt_tile<true>` (legacy_compat): APPROX -> `_sqrt_compat_<APPROX,2>` + `_reciprocal_compat_signed_<2>`
   (ckernel_sfpu_rsqrt_compat.h), which is biased low. The model never set a compute config, so every UNet and VAE
   reference GN ran this path. (Part 4's "over-estimated variance in the bf16 statistics" hypothesis is wrong.)
2. **Welford is precise once approx is off** (gain 1.001, rms 0.0036 vs generated 0.0019); its default config sabotaged it.
   The "welford causes PCC drop on Blackhole" comments in the VAE code are most likely this same default.
3. fp32 DEST on the legacy kernel: needed for offset/heavy-tailed inputs on the DRAM path (rms 0.03-0.05 -> 0.009 at
   mean 4 +- 3, PCC 0.9991 -> 1.0), but on the sharded path it adds a +0.017 output mean bias (fp32 x/mean reconfig
   path; 0.0002 without fp32). Even approx off + fp32 off leaves the legacy at +0.3..0.9% gain (bf16 intermediates).
4. The generated op's approx mode costs nothing (ckl::Rsqrt non-compat path): +0.05% gain, rms +0.0002.
Consequence for the reference model: passing `compute_kernel_config=WormholeComputeKernelConfig(HiFi4, approx=False,
fp32_dest_acc_en=False)` to every ttnn.group_norm removes the 3.8% GN shrinkage (which Part 5 showed was compensating
the SDPA/conv_out errors -- so it has to go together with those fixes). Not applied to the model here.

### 2026-09-21 15:20 max-precision path: fp32 activations + fp32 affine + fp32 output (GN_ISO_PREC=fp32)
Reference with fp32 input forces fp32 DEST and fp32 intermediate CBs (im_data_format follows the input). On the 8x8 grid
the 320@128 sharded cell no longer fits (CBs 1.94-2.59 MB > 1.5 MB L1) and the DRAM cells need 4x the model's
num_out_blocks (GN_ISO_FP32_NOB_MULT=4). randn input, gain / rms vs torch fp32 on the same fp32 operands:

| variant | 640@64 sh | 1280@32 sh | 512@256 dram (nob 16) | 256@512 dram (nob 48) |
|---|---|---|---|---|
| ref_default (fp32 DEST forced, approx ON) | 0.9747 / 0.0279 | 0.9730 / 0.0297 | 0.9642 / 0.0390 | 0.9642 / 0.0386 |
| ref_fp32 (approx OFF) | 1.0061 / 0.0068 | 1.0047 / 0.0052 | 0.9990 / 0.0022 | 0.9990 / 0.0023 |
| ref_welford (approx ON) | 0.9725 / 0.0345 | 0.9722 / 0.0343 | 0.9779 / 0.0294 | 0.9800 / 0.0268 |
| ref_welford_exact (approx OFF) | 0.9964 / 0.0044 | 0.9967 / 0.0040 | 0.9962 / 0.0049 | 0.9955 / 0.0056 |
| generated (exact) | 0.9989 / 0.0015 | 0.9989 / 0.0015 | 0.9996 / 0.0009 | 1.0004 / 0.0009 |
| generated_approx | 0.9995 / 0.0011 | 0.9994 / 0.0011 | 1.0003 / 0.0008 | 1.0007 / 0.0011 |
(generated 320@128 sharded fp32: 0.9988 / 0.0015; generated_approx 0.9996 / 0.0009.)
- Same verdict as bf16: approx ON shrinks 2.2-3.6% on every reference kernel even with fp32 everything. So the "max
  precision path" of the reference model (fp32 activations, Welford) is still 2-3% low unless math_approx_mode is
  turned off; fp32 input alone buys nothing against that.
- With approx off the reference DRAM legacy path is now very good (gain 0.999, rms 0.0022) but the sharded legacy path
  keeps +0.5-0.6% gain, and Welford-exact sits 0.35-0.45% LOW in fp32 (it was +0.1% in bf16) with 3x the generated rms.
- The legacy sharded fp32 path's mean bias is real, not a bf16 artefact: +0.018..+0.024 on the offset input (ref_default
  and ref_fp32 alike); Welford (+0.0005) and generated (+0.002) do not have it.
- Generated: rms 0.0009-0.0015 (2x better than its own bf16 run: the bf16 output rounding was the floor there);
  its exact rsqrt path sits 0.11% low and the approx path 0.05% low -- the op's smallest remaining systematic term.
  It is the only implementation that runs the 320@128 cell in fp32 at all on this grid budget.

### 2026-09-21 15:50 APPLIED: exact-SFPU compute config on every reference ttnn.group_norm call
`sdxl_utility.reference_gn_compute_config(use_welford)`: HiFi4, math_approx_mode=False, fp32_dest_acc_en only for
Welford (the sharded legacy fp32 path has the +0.02 mean bias), packer_l1_acc off. Wired into run_group_norm (UNet
reference path) and the five VAE call sites (encoder, attention, resnet x2, decoder norm_out). SDXL_GN_APPROX_RSQRT=1
restores the old default for A/B. Only the reference path is affected (SDXL_GENERATED_GN=0); the generated op is untouched.
| reference-GN path (SDXL_GENERATED_GN=0) | approx rsqrt (old default) | exact rsqrt (now) | generated GN |
|---|---|---|---|
| UNet PCC (test_module_tt_unet 1024, 4ch, gate 0.9968) | 0.99713 | 0.99805 | 0.99846 |
| VAE decoder PCC (test_module_tt_decoder 1024, gate 0.93) | 0.93051 | 0.95176 | 0.961 |
Not re-measured: the 50-step reference loop. Part 5 showed the approx-rsqrt shrinkage was compensating the SDPA LoFi
and conv_out DEST errors there, so on the reference path this fix belongs together with SDPA HiFi2 + conv io l1acc.

### 2026-09-21 16:05 where the reference DOES beat the generated op: mean >> std ("large_offset", 20 +- 10, unit std)
GN_ISO_DISTS=large_offset. gain / bias / rms vs torch:
| | 640@64 sh bf16 | 640@64 sh fp32 | 256@512 dram bf16 | 256@512 dram fp32 |
|---|---|---|---|---|
| ref_fp32 (legacy, exact) | 1.008 / +0.039 / 0.043 | 1.006 / +0.044 / 0.047 | 1.003 / +0.012 / 0.016 | 1.000 / +0.007 / 0.008 |
| ref_welford_exact | 1.003 / +0.008 / 0.011 | 0.997 / +0.001 / **0.0037** | 1.002 / +0.005 / **0.0074** | 0.997 / +0.0002 / **0.0044** |
| generated | 1.000 / +0.005 / **0.0055** | 0.999 / +0.004 / 0.0051 | 1.003 / +0.011 / 0.0121 | 1.002 / +0.010 / 0.0117 |
Welford + exact rsqrt wins 3 of 4 (by up to 2.7x rms) once the input mean is ~20 std away from zero; the generated op's
error is almost entirely a positive output-mean BIAS that scales with the input mean and does not improve with fp32
input. Mechanism (from the kernel structure, not yet probed): the generated apply pass is the fused affine
`y = x * scale_row + shift_full` with shift_full = beta - mean * rstd * gamma (compute kernel apply_chunk); with a
large mean, |shift_full| >> |y| and the shift goes through the FPU src register (tf32 / bf16 operand precision),
so a ~2^-11 relative truncation of a shift of ~20 lands as a ~0.01 bias on a unit-scale output. Welford centers
first ((x - u) * rstd) and, in fp32, reads the mean through the fp32 unpack path, so its bias vanishes with fp32 input.
The legacy sharded kernel has the same class of bias 4-8x larger (+0.04). Relevance for SDXL: post-conv GN inputs have
|mean|/std of a few at most -> generated bias ~0.002-0.003 std (seen on offset_heavy), far below the fixed shrinkage
issues. Fix direction for the generated op: split shift_full into a tf32-exact hi part + lo remainder, or center in DEST
before the affine ((x - m) * s + beta) -- a kernel change, not attempted.

## Part 9 (2026-09-23): generated GN vs reference ttnn.group_norm, head to head, and why it is faster
Probes: `test_gn_compare_probe.py` (+ `parse_gn_compare.py`), `test_gn_ablation_probe.py`; BH p150a, Tracy
DEVICE KERNEL DURATION, min of 3 (UNet) / 2 (VAE). Reference = main's ttnn.group_norm with its DEFAULT compute
config (approx rsqrt), model masks/grids; CSV generated/profiler/reports/2026_09_23_00_56_42. Reference numbers
reproduce the 09-16 baseline to <0.01%. PCC vs torch fp32: generated >= 0.999997 (std ratio 1.000), reference
>= 0.99984 (std ratio 0.983-0.993).

| cell (us) | ref model (8x8 RM) | gen same shard RM | gen same, no direct view | gen RM 11x10 T | gen model TILE 11x10 T | ref/gen same | ref/gen model |
|---|---|---|---|---|---|---|---|
| 16384x320 | 337.1 | 69.5 | 167.5 | 46.5 | 35.8 | 4.85x | 9.43x |
| 16384x640 neg | 381.6 | 110.6 | 199.1 | 69.9 | 60.1 | 3.45x | 6.35x |
| 16384x960 neg | 389.6 | 230.6 | 230.7 | 99.2 | 87.5 | 1.69x | 4.45x |
| 4096x320 | 102.8 | 37.0 | 52.6 | 20.7 | 18.0 | 2.78x | 5.70x |
| 4096x640 | 104.7 | 45.5 | 62.6 | 28.6 | 26.1 | 2.30x | 4.01x |
| 4096x960 | 106.4 | 88.8 | 72.9 | 38.9 | 35.7 | 1.20x | 2.98x |
| 4096x1280 | 95.8 | 64.9 | 80.3 | 47.9 | 44.7 | 1.48x | 2.15x |
| 4096x1920 | 139.0 | 120.7 | 118.4 | 71.4 | 66.1 | 1.15x | 2.10x |
| 1024x640 | 45.1 | 29.4 | 28.3 | 20.2 | 19.1 | 1.53x | 2.36x |
| 1024x1280 | 44.5 | 32.5 | 37.2 | 30.7 | 29.5 | 1.37x | 1.51x |
| 1024x1920 | 54.1 | 68.7 | 52.9 | 41.7 | 40.0 | 0.79x | 1.35x |
| 1024x2560 | 53.0 | 62.0 | 65.2 | 52.7 | 51.1 | 0.85x | 1.04x |
| attn 4096x640 (ref 4x8) | 174.1 | 62.2 | 77.6 | 28.6 | 26.1 | 2.80x | 6.66x |
| attn 1024x1280 | 44.5 | 32.6 | 37.1 | 30.7 | 29.5 | 1.36x | 1.51x |
| VAE 65536x512 | 1,016.9 | | | | 498.3 (DRAM TILE, 110c) | | 2.04x |
| VAE 262144x512 | 3,849.3 | | | | 2,097.7 | | 1.83x |
| VAE 262144x256 | 2,933.1 | | | | 1,055.7 | | 2.78x |
| VAE 1048576x256 | 11,587.8 | | | | 4,121.4 | | 2.81x |
| VAE 1048576x128 | 10,176.3 | | | | 2,104.7 | | 4.84x |

Ablation 1, group count on the model 8x8 RM shard (32 groups / 8 columns = 4 groups per core for the ref):
| us | G=8 | G=16 | G=32 |
|---|---|---|---|
| ref 16384x320 | 148.8 | 211.5 | 337.1 |
| gen 16384x320 | 69.4 | 69.6 | 69.3 |
| ref 4096x640 | 50.9 | 64.6 | 104.7 |
| gen 4096x640 | 45.6 | 45.4 | 45.5 |
| ref 4096x1280 | 54.3 | 68.4 | 95.8 |
| gen 4096x1280 | 64.9 | 64.9 | 64.9 |
The reference is linear in groups-per-core (per-group masked full-block passes + 2 serialized cross-core syncs per
group); the generated op is flat (per-channel column sums, groups formed by a tiny membership matmul). At 1 group
per core the reference is competitive (4096x1280: 54 vs 65 us).
Ablation 2, reference fed TILE shards (out of place, where tile-aligned): 4096x1280 95.8 -> 90.3, 1024x1280
44.5 -> 42.4, 1024x2560 52.9 -> 50.2 (~5%): RM repack/tilize is not the reference's problem.
VAE: both ops move ~3 volumes of DRAM traffic (ref: stats read + apply read + write; gen: two_pass shifted
statistics + apply); gen sustains ~380 GB/s (1048576x128: 805 MB / 2.10 ms), ref 79-210 GB/s on 64 cores with one
barrier per tile and per-group apply passes.
Where the generated op is weak: short, wide shards (1024 rows, C>=1920): its time grows ~10.6 us per 320 channels at
3 tile-rows/core on 110 cores, i.e. per-channel-tile fixed work (stat rows, expansion matmuls, fp32 HiFi4), not data.

## Part 10 (2026-09-23): PR #54786 ("two-pass Welford") A/B on the SDXL GroupNorm cells
Worktree `/localdev/mstaletovic/metal_metal/wt-pr54786` (Release+Tracy, own build): PR head 1d3b5dcf188 vs its merge base
83a21ac4908 (same worktree, incremental rebuild). Probe (untracked there): tests/ttnn/unit_tests/operations/fused/test_pr54786_gn_probe.py,
default compute config (use_welford -> HiFi4 + fp32 DEST, approx rsqrt), min of 3 (UNet) / 2 (VAE) DEVICE KERNEL ns, BH p150a.
Merge-base numbers reproduce Part 9's reference within 0.1%. The PR only touches use_welford=True paths; legacy sharded is
bit-for-bit the control (identical ns). Welford sharded rejects negative_mask (TT_FATAL) and still needs whole groups per core.

| cell (us) | legacy (both) | welford main | welford PR | PR vs main welford | PR welford vs legacy | generated, same shard / model |
|---|---|---|---|---|---|---|
| 16384x320 | 336.7 | 851.7 | 363.6 | 2.34x | 0.93x | 69.5 / 35.8 |
| 4096x320 | 102.5 | 254.3 | 105.6 | 2.41x | 0.97x | 37.0 / 18.0 |
| 4096x640 | 104.3 | 380.2 | 121.5 | 3.13x | 0.86x | 45.5 / 26.1 |
| 4096x960 | 106.1 | 518.3 | 146.2 | 3.55x | 0.73x | 88.8 / 35.7 |
| 4096x1280 | 95.5 | 633.4 | 150.1 | 4.22x | 0.64x | 64.9 / 44.7 |
| 4096x1920 | 138.7 | 913.9 | 208.6 | 4.38x | 0.66x | 120.7 / 66.1 |
| 1024x640 | 44.8 | 138.4 | 49.3 | 2.81x | 0.91x | 29.4 / 19.1 |
| 1024x1280 | 44.1 | 204.8 | 58.1 | 3.52x | 0.76x | 32.5 / 29.5 |
| 1024x1920 | 53.8 | 276.7 | 74.4 | 3.72x | 0.72x | 68.7 / 40.0 |
| 1024x2560 | 52.8 | 338.5 | 80.5 | 4.20x | 0.66x | 62.0 / 51.1 |
| attn 4096x640 4x8 | 173.6 | 729.3 | 209.4 | 3.48x | 0.83x | 62.2 / 26.1 |
| VAE (DRAM, 8x8, model nob) | | main | PR | PR vs main | | generated DRAM 110c |
| 65536x256 (PR's SDXL CI case) | | 780.0 | 764.1 | 1.02x | | - |
| 65536x512 | | 1,016.8 | 990.2 | 1.03x | | 498.3 |
| 262144x512 | | 3,848.4 | 3,793.2 | 1.01x | | 2,097.7 |
| 262144x256 | | 2,933.0 | 3,114.8 | 0.94x | | 1,055.7 |
| 1048576x256 | | 11,588.9 | 12,439.3 | 0.93x | | 4,121.4 |
| 1048576x128 | | 10,175.7 | 9,987.4 | 1.02x | | 2,104.7 |
PR head VAE rerun reproduces within 0.3%. Output std/ref: VAE main 0.983-0.986 -> PR 0.991; sharded welford main
0.986-0.990 -> PR 0.990-0.993 (still the approximate rsqrt default). Reading: the PR's GroupNorm gains are against the OLD
Welford path (2.3-4.4x here, matching its "sharded GroupNorm ~1.9-2.5x / 3.07x geomean" claims); on SDXL they do not beat the
legacy path the UNet uses, and the VAE DRAM path (the only SDXL use of use_welford) is flat (0.93-1.03x).

## Part 11 (2026-09-23): matched compute configs — perf AND accuracy, reference vs generated
Probe `test_gn_equiv_probe.py` (+ `parse_gn_equiv.py`). The generated op refuses fp32_dest_acc_en=False (fp32 stat CBs);
math_fidelity / math_approx_mode / dst_full_sync_en pass through. Reference sharded legacy configs: default (HiFi4, approx
ON, fp32 off) / exact (approx off) / eq (= generated config: HiFi4, exact, fp32 DEST) / HiFi2 / LoFi (exact, fp32 off).
VAE reference = Welford DRAM (fp32 forced). Generated on the SAME shard and at the model placement. BH p150a, min of 4
DEVICE KERNEL ns (perf-only sessions: Tracy keeps ~600 op rows per session, so the matrix ran as 3 sessions:
reports 2026_09_23_15_48_30 / 15_52_38 / 15_56_19). Accuracy: one call per distribution vs torch fp32 on bf16 operands
(run 2026_09_23_15_38_11 log; VAE accuracy only up to 262144 rows).

Perf (us):
| cell | ref_default@ref | ref_exact@ref | ref_eq@ref | ref_hifi2@ref | ref_lofi@ref | gen_default@same | gen_approx@same | gen_hifi2@same | gen_lofi@same | gen_fullsync@same | gen_default@model | gen_approx@model | gen_hifi2@model | gen_lofi@model | gen_fullsync@model |
| res_16384x320 | 337.1 | 340.2 | 340.4 | 251.9 | 251.5 | 69.3 | 69.0 | 66.5 | 66.3 | 76.3 | 35.7 | 35.5 | 34.6 | 33.7 | 38.9 |
| res_4096x320 | 102.8 | 105.9 | 105.8 | 83.7 | 83.3 | 36.8 | 36.6 | 35.7 | 35.7 | 38.9 | 18.1 | 17.8 | 17.6 | 17.4 | 18.7 |
| res_4096x640 | 104.7 | 107.7 | 107.6 | 84.9 | 84.5 | 45.3 | 45.1 | 43.8 | 43.6 | 49.0 | 26.1 | 25.7 | 25.0 | 24.9 | 27.7 |
| res_1024x640 | 45.0 | 48.1 | 47.9 | 42.4 | 42.0 | 29.2 | 28.9 | 28.4 | 28.3 | 30.4 | 19.0 | 18.7 | 18.6 | 18.4 | 19.3 |
| res_1024x1280 | 44.4 | 47.5 | 47.4 | 41.6 | 41.2 | 32.5 | 32.2 | 31.5 | 31.3 | 34.7 | 29.5 | 29.1 | 28.5 | 28.4 | 30.3 |
| res_1024x2560 | 53.0 | 56.0 | 55.9 | 47.0 | 46.6 | 61.7 | 61.9 | 59.7 | 59.5 | 66.3 | 51.1 | 50.8 | 49.5 | 49.4 | 53.0 |
| res_1024x1920 | 54.1 | 57.2 | 57.1 | 47.6 | 47.2 | 69.0 | 68.9 | 66.8 | 66.6 | 73.2 | 40.2 | 40.0 | 39.1 | 39.0 | 41.7 |
| res_4096x1920 | 139.0 | 142.0 | 141.9 | 104.8 | 104.2 | 121.0 | 120.3 | 115.2 | 115.3 | 134.8 | 66.3 | 66.1 | 63.6 | 63.4 | 70.4 |
| res_4096x1280 | 95.9 | 98.9 | 98.8 | 74.6 | 74.3 | 64.9 | 64.7 | 62.0 | 62.0 | 71.9 | 44.5 | 44.3 | 42.6 | 42.4 | 47.8 |
| res_4096x960 | 106.4 | 109.5 | 109.4 | 86.3 | 85.9 | 88.4 | 88.1 | 85.0 | 84.9 | 95.9 | 35.7 | 35.4 | 34.2 | 34.0 | 38.2 |
| resneg_16384x960 | 389.6 | 392.5 | 392.3 | 285.6 | 284.7 | 230.7 | 230.4 | 226.0 | 225.9 | 252.4 | 87.4 | 87.2 | 82.1 | 81.7 | 97.9 |
| resneg_16384x640 | 381.6 | 384.7 | 385.0 | 278.8 | 277.8 | 110.6 | 110.3 | 105.4 | 105.2 | 124.1 | 60.1 | 59.9 | 56.6 | 56.0 | 67.6 |
| attn_4096x640_4x8 | 174.2 | 180.2 | 180.1 | 134.7 | 133.8 | 62.2 | 61.9 | 59.4 | 59.3 | 69.2 | 26.0 | 25.8 | 25.0 | 24.8 | 27.7 |
| attn_1024x1280 | 44.4 | 47.5 | 47.4 | 41.6 | 41.3 | 32.5 | 32.2 | 31.3 | 31.3 | 34.7 | 29.4 | 29.2 | 28.5 | 28.4 | 30.4 |
| cell | ref_welford_default@dram | ref_welford_exact@dram | ref_welford_hifi2@dram | gen_default@dram | gen_approx@dram | gen_hifi2@dram | gen_lofi@dram | gen_fullsync@dram |
| vae_65536x512 | 1,015.9 | 1,018.7 | 974.7 | 498.4 | 498.4 | 495.0 | 492.1 | 510.2 |
| vae_262144x512 | 3,847.4 | 3,850.8 | 3,673.4 | 2,096.9 | 2,095.4 | 2,080.3 | 2,078.2 | 2,171.4 |
| vae_262144x256 | 2,932.0 | 2,934.6 | 2,745.1 | 1,058.6 | 1,057.7 | 1,050.7 | 1,047.5 | 1,098.6 |
| vae_1048576x256 | 11,587.2 | 11,590.1 | 10,828.6 | 4,134.9 | 4,128.5 | 4,082.3 | 4,097.2 | 4,299.5 |
| vae_1048576x128 | 10,173.7 | 10,179.1 | 9,486.2 | 2,071.9 | 2,061.4 | 2,055.7 | 2,069.8 | 2,135.4 |

Accuracy (UNet: 14 cells, VAE: 3 cells; median rms of the output error, unit-variance output; gain range):
| config | randn rms | offset_heavy rms | large_offset rms | gain (randn) | mean bias (large_offset) |
|---|---|---|---|---|---|
| UNet ref_default | 0.0254 | 0.0223 | 0.0296 | 0.975-0.994 | -0.018..+0.005 |
| UNet ref_exact | 0.0063 | 0.0108 | 0.0145 | 1.002-1.006 | -0.018..+0.005 |
| UNet ref_eq (generated's config) | 0.0083 | 0.0210 | 0.0351 | 1.004-1.008 | +0.017..+0.038 |
| UNet ref_hifi2 | 0.0084 | 0.0182 | 0.0287 | 0.998-1.014 | +0.006..+0.060 |
| UNet ref_lofi | 0.0356 | 0.1197 | 0.2069 | 1.021-1.035 | +0.140..+0.240 |
| UNet gen_default (same shard) | 0.0019 | 0.0034 | 0.0054 | 1.000-1.001 | +0.004..+0.009 |
| UNet gen_approx | 0.0021 | 0.0035 | 0.0054 | 1.001-1.002 | same |
| UNet gen_hifi2 | 0.0052 | 0.0232 | 0.0391 | 0.996-0.998 | +0.032..+0.038 |
| UNet gen_lofi | 0.0276 | 0.0668 | 0.0977 | 0.978-0.979 | +0.037..+0.092 |
| UNet gen_fullsync | identical to gen_default | | | | |
| UNet gen_default (model placement) | 0.0019 | 0.0034 | 0.0053 | 1.000-1.001 | +0.004..+0.006 |
| VAE ref_welford_default | 0.0364 | 0.0235 | 0.0262 | 0.966-0.968 | +0.003..+0.005 |
| VAE ref_welford_exact (generated's config) | 0.0027 | 0.0083 | 0.0074 | 1.0005-1.0006 | +0.003..+0.005 |
| VAE ref_welford_hifi2 | 0.0047 | 0.0085 | 0.0090 | 0.997-0.998 | +0.003..+0.005 |
| VAE gen_default | 0.0033 | 0.0095 | 0.0163 | 1.0015-1.0034 | +0.009..+0.024 |
| VAE gen_hifi2 | 0.0047 | 0.0233 | 0.0405 | 0.997-0.998 | +0.031..+0.048 |
| VAE gen_lofi | 0.0273 | 0.0671 | 0.1052 | 0.978-0.979 | +0.068..+0.101 |
Findings: (1) generated perf is flat in the knobs: approx -0.5%, HiFi2/LoFi -1..-6%, full-sync DEST SLOWER (+5..15%) —
it is not FPU-multiply bound; the default is within 6% of its fastest setting. (2) reference sharded is fidelity-bound:
HiFi2 = LoFi = -25% on the long shards (337 -> 252 us), fp32 / exact cost +1..3%; its fastest setting (HiFi2) beats the
generated op on the SAME shard at 1024x1920/1024x2560 (47 vs 67-69 / 60-62 us) and, at 1024x2560, even the generated op at
the model placement (47.0 vs 49.4-51.1). Everywhere else generated at its default still wins: 3.6x (16384x320) same shard vs
ref HiFi2, 7x at the model placement. (3) UNet accuracy win holds at every MATCHED fidelity and is largest at HiFi4:
3.3x / 3.2x / 2.7x lower rms than the best reference setting (ref_exact); the reference at the generated op's exact config
(ref_eq, fp32 DEST) is WORSE than ref_exact on offset data (the legacy fp32 +0.02..0.04 mean bias). At HiFi2 the win flips on
offset inputs (gen 0.023 / 0.039 vs ref 0.018 / 0.029): the generated mean path (colsum x membership matmul, apply mul) is
fidelity-sensitive -> a bias proportional to |mean|. (4) VAE: the reference Welford at the generated op's config
(exact rsqrt, fp32, HiFi4) is MORE accurate than the generated op (rms 0.0027 / 0.0083 / 0.0074 vs 0.0033 / 0.0095 / 0.0163,
gen gain +0.15..0.34%); the generated win on the VAE existed only against Welford's approx-rsqrt default. Generated stays
1.8-5x faster on every VAE cell at any setting.

### 2026-09-23 accuracy error sources of the generated op (host-knob experiment, `test_gn_acc_experiments.py`)
The colsum reduce runs `ReduceFp32Mode::Fast`, which per the helper "keeps fp32 on the FPU path (inputs truncated to
tf32)": the fp32 scratch operands and the reloaded partial accumulators lose 13 mantissa bits (truncation, biased).
Test: same DRAM cells, only the chunk target changed (default 32; chunk128 has no valid split on the 262144 cells):
| cell | chunk8 gain / rms (randn) | chunk32 | chunk128 | three-pass chunk32 | large_offset gain chunk8 / 32 / 3-pass |
|---|---|---|---|---|---|
| 65536x512 | 1.00256 / 0.0034 | 1.00157 / 0.0025 | 1.00157 / 0.0025 | 1.00157 / 0.0025 | 1.0024 / 1.0012 / 1.0012 |
| 262144x512 | 1.00010 / 0.0018 | 1.00338 / 0.0041 | n/a | 1.00317 / 0.0039 | 0.9985 / 1.0065 / 1.0036 |
| 262144x256 | 1.00572 / 0.0064 | 1.00152 / 0.0024 | n/a | 1.00144 / 0.0024 | 1.0117 / 1.0029 / 1.0012 |
The gain error (variance under-estimate) moves 0.01% -> 0.57% with blocking alone (the chunk target also changes the
split K), i.e. it is accumulation rounding, not the algorithm; the two-pass S/U combine adds 0.15-0.3% gain on offset
data only. Mean bias (+0.004..0.009 UNet, +0.01..0.02 VAE on large_offset, 8x larger at HiFi2) is consistent with tf32
truncation of the mean through the three row matmuls (colsum@M, gather@1/n with a non-tf32-exact 1/n, stat@M^T) and the
tf32 srcB shift in the fused apply. Candidate fixes (not implemented): ReduceFp32Mode::Accurate / UnpackToDestFp32
accumulator reload; hi/lo (tf32x2) split of the row-matmul operands + SFPU fp32 1/n scale; centre-then-scale apply
((x - mu) * s + beta) or hi/lo shift.

### 2026-09-23 17:15 APPLIED (uncommitted): accuracy fix 1 — exact cross-chunk statistics accumulation
`config.EXACT_COLSUM_ACCUM` (default True): the column-sum reduce accumulates into a new fp32 CB (`CB_COLSUM_ACC` = 26,
K or 2K pages) tagged UnpackToDestFp32 and reloads it with `AccumulateReloadMode::CopySeedSfpuAdd` (new tiles summed
in a fresh DEST, the accumulator copied losslessly into a 2nd DEST slot, one fp32 SFPU add); `publish_colsum` copies the
finished sums bit-exactly into cb_colsum_rows for the unchanged consumers. Note `ReduceFp32Mode::Accurate` would NOT have
helped: it only affects the ReduceTile datapath, AccumulateViaAdd ignores it. The old reload (CopySeedPairs) copied the
running sum through SrcA -> tf32 at every chunk. Tried and REJECTED: CopySeedZeroPair (faster exact mode, needs a zero
tile in cb_scaler): wrong on 262144x512 chunk8 (gain 0.9935, bias -0.015 on randn) and a -1e-4 bias elsewhere; reverted.
Op unit suites 427 passed / 16 skipped. Runs on card 0 only (TT_VISIBLE_DEVICES=0): card 3 (PCIe c1) dropped off the bus
twice today (0xffffffff reads, eth-core membar timeouts at device init) and broke every multi-card device open.
Matrix rerun (generated variants only; reference unchanged), before -> after:
## accuracy: median rms randn / offset_heavy / large_offset ; gain range randn ; bias range large  (before -> after)
| unet gen_default@same | 0.0019 / 0.0034 / 0.0054 ; gain 0.9999-1.0010 ; bias +0.0037..+0.0086 | 0.0019 / 0.0034 / 0.0052 ; gain 0.9998-1.0001 ; bias +0.0037..+0.0050 |
| unet gen_approx@same | 0.0021 / 0.0035 / 0.0054 ; gain 1.0006-1.0017 ; bias +0.0038..+0.0086 | 0.0021 / 0.0034 / 0.0053 ; gain 1.0006-1.0009 ; bias +0.0038..+0.0050 |
| unet gen_hifi2@same | 0.0052 / 0.0232 / 0.0391 ; gain 0.9964-0.9975 ; bias +0.0319..+0.0375 | 0.0052 / 0.0231 / 0.0386 ; gain 0.9964-0.9975 ; bias +0.0319..+0.0372 |
| unet gen_lofi@same | 0.0276 / 0.0668 / 0.0977 ; gain 0.9783-0.9789 ; bias +0.0370..+0.0917 | 0.0276 / 0.0668 / 0.0977 ; gain 0.9783-0.9789 ; bias +0.0369..+0.0879 |
| unet gen_fullsync@same | 0.0019 / 0.0034 / 0.0054 ; gain 0.9999-1.0010 ; bias +0.0037..+0.0086 | 0.0019 / 0.0034 / 0.0052 ; gain 0.9998-1.0001 ; bias +0.0037..+0.0050 |
| unet gen_default@model | 0.0019 / 0.0034 / 0.0053 ; gain 0.9998-1.0005 ; bias +0.0040..+0.0058 | 0.0019 / 0.0032 / 0.0052 ; gain 0.9998-1.0001 ; bias +0.0040..+0.0049 |
| unet gen_approx@model | 0.0020 / 0.0035 / 0.0053 ; gain 1.0005-1.0010 ; bias +0.0040..+0.0058 | 0.0020 / 0.0033 / 0.0052 ; gain 1.0005-1.0008 ; bias +0.0040..+0.0049 |
| unet gen_hifi2@model | 0.0053 / 0.0231 / 0.0386 ; gain 0.9964-0.9975 ; bias +0.0310..+0.0371 | 0.0053 / 0.0231 / 0.0379 ; gain 0.9964-0.9975 ; bias +0.0310..+0.0371 |
| unet gen_lofi@model | 0.0277 / 0.0646 / 0.0953 ; gain 0.9782-0.9789 ; bias +0.0534..+0.0827 | 0.0277 / 0.0646 / 0.0953 ; gain 0.9782-0.9789 ; bias +0.0534..+0.0827 |
| unet gen_fullsync@model | 0.0019 / 0.0034 / 0.0053 ; gain 0.9998-1.0005 ; bias +0.0040..+0.0058 | 0.0019 / 0.0032 / 0.0052 ; gain 0.9998-1.0001 ; bias +0.0040..+0.0049 |
| vae gen_default@dram | 0.0033 / 0.0095 / 0.0163 ; gain 1.0015-1.0034 ; bias +0.0090..+0.0243 | 0.0018 / 0.0031 / 0.0051 ; gain 1.0001-1.0001 ; bias +0.0039..+0.0046 |
| vae gen_approx@dram | 0.0035 / 0.0096 / 0.0164 ; gain 1.0018-1.0035 ; bias +0.0090..+0.0242 | 0.0019 / 0.0032 / 0.0051 ; gain 1.0003-1.0005 ; bias +0.0039..+0.0045 |
| vae gen_hifi2@dram | 0.0047 / 0.0233 / 0.0405 ; gain 0.9969-0.9977 ; bias +0.0306..+0.0476 | 0.0047 / 0.0214 / 0.0349 ; gain 0.9969-0.9977 ; bias +0.0261..+0.0358 |
| vae gen_lofi@dram | 0.0273 / 0.0671 / 0.1052 ; gain 0.9782-0.9792 ; bias +0.0677..+0.1014 | 0.0273 / 0.0642 / 0.0995 ; gain 0.9782-0.9792 ; bias +0.0579..+0.0897 |
| vae gen_fullsync@dram | 0.0033 / 0.0095 / 0.0163 ; gain 1.0015-1.0034 ; bias +0.0090..+0.0243 | 0.0018 / 0.0031 / 0.0051 ; gain 1.0001-1.0001 ; bias +0.0039..+0.0046 |
| unet ref_exact@ref (reference, unchanged) | 0.0063 / 0.0108 / 0.0145 |
| unet ref_eq@ref (reference, unchanged) | 0.0083 / 0.0210 / 0.0351 |
| vae ref_welford_exact@dram (reference, unchanged) | 0.0027 / 0.0083 / 0.0074 |

## VAE per cell gen_default rms randn/offset/large before -> after, ref_welford_exact
| vae_65536x512 | 0.0025 / 0.0060 / 0.0100 | 0.0019 / 0.0029 / 0.0048 | 0.0028 / 0.0075 / 0.0075 |
| vae_262144x512 | 0.0041 / 0.0116 / 0.0205 | 0.0018 / 0.0030 / 0.0054 | 0.0026 / 0.0083 / 0.0070 |
| vae_262144x256 | 0.0024 / 0.0073 / 0.0121 | 0.0018 / 0.0032 / 0.0047 | 0.0027 / 0.0082 / 0.0074 |
| vae_1048576x128 | 0.0042 / 0.0162 / 0.0289 | 0.0019 / 0.0037 / 0.0059 | 0.0027 / 0.0087 / 0.0077 |

## perf gen (us): before -> after (delta %)
| cell | gen_default@same | gen_default@model / @dram | gen_hifi2@model / @dram | ref best (HiFi2) |
| res_16384x320 | 69.3 -> 74.9 (+8.0%) | 35.7 -> 36.6 (+2.4%) | 34.6 -> 35.4 (+2.4%) | 251.9 |
| res_4096x320 | 36.8 -> 38.0 (+3.3%) | 18.1 -> 18.5 (+2.5%) | 17.6 -> 18.0 (+2.6%) | 83.7 |
| res_4096x640 | 45.3 -> 48.9 (+7.8%) | 26.1 -> 26.6 (+2.3%) | 25.0 -> 25.6 (+2.6%) | 84.9 |
| res_1024x640 | 29.2 -> 30.5 (+4.4%) | 19.0 -> 19.7 (+4.1%) | 18.6 -> 19.2 (+3.5%) | 42.4 |
| res_1024x1280 | 32.5 -> 33.7 (+3.7%) | 29.5 -> 30.6 (+3.7%) | 28.5 -> 29.8 (+4.3%) | 41.6 |
| res_1024x2560 | 61.7 -> 68.4 (+10.9%) | 51.1 -> 53.0 (+3.7%) | 49.5 -> 51.4 (+3.7%) | 47.0 |
| res_1024x1920 | 69.0 -> 72.4 (+4.8%) | 40.2 -> 41.7 (+3.9%) | 39.1 -> 40.9 (+4.5%) | 47.6 |
| res_4096x1920 | 121.0 -> 196.1 (+62.0%) | 66.3 -> 72.9 (+10.0%) | 63.6 -> 70.2 (+10.4%) | 104.8 |
| res_4096x1280 | 64.9 -> 70.5 (+8.6%) | 44.5 -> 47.6 (+6.8%) | 42.6 -> 45.5 (+7.0%) | 74.6 |
| res_4096x960 | 88.4 -> 98.4 (+11.3%) | 35.7 -> 38.0 (+6.5%) | 34.2 -> 36.5 (+6.9%) | 86.3 |
| resneg_16384x960 | 230.7 -> 253.8 (+10.0%) | 87.4 -> 94.0 (+7.5%) | 82.1 -> 88.6 (+7.9%) | 285.6 |
| resneg_16384x640 | 110.6 -> 121.3 (+9.7%) | 60.1 -> 63.1 (+5.0%) | 56.6 -> 59.5 (+5.1%) | 278.8 |
| attn_4096x640_4x8 | 62.2 -> 67.8 (+9.0%) | 26.0 -> 26.7 (+2.6%) | 25.0 -> 25.5 (+2.2%) | 134.7 |
| attn_1024x1280 | 32.5 -> 33.7 (+3.5%) | 29.4 -> 30.6 (+4.1%) | 28.5 -> 29.8 (+4.4%) | 41.6 |
| vae_65536x512 | - | 498.4 -> 518.7 (+4.1%) | 495.0 -> 509.5 (+2.9%) | 974.7 |
| vae_262144x512 | - | 2,096.9 -> 2,231.7 (+6.4%) | 2,080.3 -> 2,212.7 (+6.4%) | 3,673.4 |
| vae_262144x256 | - | 1,058.6 -> 1,097.7 (+3.7%) | 1,050.7 -> 1,091.6 (+3.9%) | 2,745.1 |
| vae_1048576x256 | - | 4,134.9 -> 4,298.1 (+3.9%) | 4,082.3 -> 4,250.7 (+4.1%) | 10,828.6 |
| vae_1048576x128 | - | 2,071.9 -> 2,088.8 (+0.8%) | 2,055.7 -> 2,060.0 (+0.2%) | 9,486.2 |
Summary: VAE (DRAM streaming) median rms 0.0033 / 0.0095 / 0.0163 -> 0.0018 / 0.0031 / 0.0051, gain 1.0015-1.0034 ->
1.0001 — now better than the reference Welford at the same config (0.0027 / 0.0083 / 0.0074) on all three distributions
and every cell. UNet (resident, few chunks): rms unchanged, gain range tightened (0.9999-1.0010 -> 0.9998-1.0001), large-
offset bias max +0.0086 -> +0.0050. Remaining error: the +0.004 mean bias (fix 2: the tf32 mean path). Cost: +2..+10% at
the model placement (worst 4096x1920 +10%), +3..+11% on the same RM shard except 4096x1920 same shard +62% (K = 15 view
tiles x many chunks x extra copy + SFPU add); VAE +0.8..+6.4%. VAE decoder PCC (test_module_tt_decoder 1024): 0.96099 ->
0.96191 (the decoder's error is dominated by other ops). Generated still 1.7-7x faster than the reference at its fastest
setting on every cell except 1024x2560 (53.0 vs ref HiFi2 47.0 us).

### 2026-09-23 17:25 REVERTED: accuracy fix 1 (user decision: not worth the +2..10% / +62% outlier cost)
The three op files (config.py, groupnorm_sc_N_1_HW_C_program_descriptor.py, kernels/groupnorm_sc_N_1_HW_C_compute.cpp)
are back to the committed version; the op again reloads with CopySeedPairs through SrcA (tf32 truncation per chunk).
To re-apply: a dedicated fp32 accumulator CB (index 26, K or 2K pages) tagged UnpackToDestFp32 in the compute config's
unpack_to_dest_mode, `.with_reload(AccumulateReloadMode::CopySeedSfpuAdd)` on the colsum Accumulate, and a
CopyTile -> PackTile publish of the finished K (two_pass: 2K, [S, U] order) tiles into cb_colsum_rows before each
consumer (pass A, pass 1, pass 2). Measurements above stay valid as the record of what it buys and costs.

## Part 12 (2026-09-23): branch cleanup — the full-grid path is the only BH path, no switches
Squashed onto `mstaletovic/sdxl-gn-fullgrid-clean` (7 code commits + this worklog, base e77769238a9); the original
`mstaletovic/sdxl-gn-fullgrid` is kept untouched as the history of everything above.
- Every model-side env switch is gone (SDXL_GENERATED_GN, SDXL_T1280 / T640 / T_RESNET / SHARDED_640, SDXL_GN_*,
  SDXL_SDPA_*, SDXL_MM_* / CONV_* / LN_* precision knobs, SDXL_CONVIO_COMPUTE, SDXL_*_W_DTYPE, SDXL_QKV_PREROUND, all
  dump hooks, SDXL_GN_SCALE_HACK, SDXL_GN_APPROX_RSQRT, GN_TEMPORAL_ROUNDS); each was pinned to its shipped value.
  `ModelOptimisations1024x1024BH` has two class attributes instead: FULL_GRID (the refiner sets False and keeps its
  reference path; it had silently inherited the full-grid flags before) and TRANSPOSED_RESNETS (False in the VAE).
- The precision fixes are now the BH defaults: SDPA HiFi2 (model_config.sdpa_math_fidelity, LoFi elsewhere) and
  conv_in / conv_out CONV_HIFI2_NO_FP32_COMPUTE_CONFIG (packer_l1_acc). The reference-GN exact-rsqrt change (Part 8)
  is reverted: the Wormhole / refiner reference paths are as on main.
- C++ env knobs became config fields: `MatmulMultiCoreReuseMultiCastProgramConfig.diagonal_in1_senders` (unified in1
  kernel always on with it; TT_MM2D_IN1_SPLIT_KERNELS dropped) and `Conv2dConfig.diagonal_weight_senders`. The env
  vars used to be process-wide, so the UNet config sets them on all its 2D matmul configs / block-sharded convs.
- Deleted: every probe script and image in this directory, the op's debug / probe / perf-measurement tests and
  eval-run artifacts; the upsample probes became tests/.../pool/test_upsample_block_sharded_col_major.py; new unit
  tests test_matmul_2d_diagonal_in1_senders.py and test_conv2d_diagonal_weight_senders.py (bit-identical vs the
  line senders). The VAE resnet / up-block tests skipped on BH "due to PCC issue with DRAM group_norm" now pass
  (17/17) and are unskipped.
Verification (p150a, card 0/1; card 3 flaky today, see Part 11):
- Unit: matmul+linear 1069 passed / 317 skipped, conv2d 161 / 48, upsample 323 / 96, nlp heads 367 / 1, generated GN
  441 / 16, new tests 14 / 0.
- PCC: UNet single step 0.99988 (1024), resnet blocks >= 0.99982, transformer models 0.9991 / 0.9979, up_blocks
  0.989 / 0.9965, VAE decoder 0.96099, VAE encoder 0.9871, autoencoder 0.9413 / 0.9857.
- Device kernel time (CI wrapper, targets updated in test_sdxl_perf.py): UNet step 53.13 ms (main target 76.89),
  VAE decode 172.74 ms (267.50), VAE encode 87.36 ms (141.18); refiner UNet unchanged, inside its 114.15 ms range.
- Traced demo (demo.py::test_demo, 1024x1024, 50 steps, CFG on, device VAE + encoders, with_trace), cards selected with
  TT_VISIBLE_DEVICES. Ethernet-linked pairs on this box: 0-2, 0-3, 1-2, 1-3 (0-1 and 2-3 are not; a 0,1 mesh silently
  downgrades to 1x1):
  | cards | CFG mode | images | image gen | denoising loop | per step | VAE | img/s |
  |---|---|---|---|---|---|---|---|
  | 1 | no_cfg_parallel (cond + uncond on one chip) | 1 | 6.39 s | 5.84 s | 116.8 ms | 0.19 s | 0.16 |
  | 2 (0,2) | no_cfg_parallel (DP=2) | 2 | 6.23 s | 5.84 s | 116.8 ms | 0.19 s | 0.32 |
  | 2 (0,2) | use_cfg_parallel (TP=2) | 1 | 3.26 s | 2.88 s | 57.6 ms | 0.19 s | 0.31 |
  | 4 | no_cfg_parallel (DP=4) | 4 | 6.33 s | 5.90 s | 118.0 ms | 0.19 s | 0.63 |
  | 4 | use_cfg_parallel (TP=2 x DP=2) | 2 | 3.31 s | 2.88 s | 57.6 ms | 0.19 s | 0.60 |
  (Part 6, before the precision fixes and this cleanup: DP=4 loop 6.02 s, TP2xDP2 2.96 s.)
- UNet loop vs torch (test_unet_loop 1024): 50 steps seed 0 PCC 0.9549 (gate 0.905, pass). 10 steps seed 42 (gate
  0.93) FAILS at 0.9229; bisect at step 9: base on main (reference GN, LoFi SDPA, conv io without L1 acc) 0.9425;
  reference GN + LoFi + conv io L1 acc 0.9361; generated GN + LoFi + L1 acc 0.9328; this branch (generated GN + HiFi2
  + L1 acc) 0.9229; generated GN output scaled by 0.975 (mimicking the reference's approx-rsqrt shrinkage) 0.9033.
  Each precision change that brings a component closer to torch fp32 (and raises the 50-step PCC from 0.867 to ~0.96)
  lowers this 10-step endpoint by 0.003-0.01; not a layout effect (the diagonal senders are bit-exact, see the unit
  tests). Open: more seeds, or a recalibrated 10-step gate.
