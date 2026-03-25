# Conv3D Blocking Sweep — Wan 2.2 VAE Decoder

## Goal
Find optimal conv3d blockings (C_in_block, C_out_block, T_out_block, H_out_block, W_out_block) for every conv3d layer in the Wan 2.2 VAE decoder across BH production meshes (2x2, 2x4, 4x8) at 480p and 720p.

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| Staged sweep: spatial → C_out → C_in | O(H*W + C_out + C_in) vs brute-force O(H*W*C_in*C_out) | Brute-force: 6732 combos for one shape, ~hours per case |
| Run on 1x1 physical mesh with target mesh local dims | Can't open 4x8 mesh on 8-device box; conv3d kernel is per-device | Running on actual multi-device mesh (not available) |
| Skip W=13, W=52 in spatial candidates | These cause device hangs on BH p150b (kernel never returns) | Timeout approach: signal.alarm can't interrupt C++ kernel |
| Dedup layers with same (C_in_padded, C_out, kernel, H, W) | Same shape = same optimal blocking regardless of position in model | Sweep every layer instance: redundant work |
| fp32_dest_acc_en=True for all sweeps | Branch adds fp32 intermediate CB; blockings should be optimal with it enabled | Sweeping with fp32 off: not representative of branch behavior |
| H-row interleaved DRAM gather in reader | Start vol2col as soon as kH shard rows are gathered, not all H_shard rows. 7-15% speedup on top layers. | Gather all → barrier → vol2col (original): simpler but higher startup latency |
| Fused tilize+matmul per M tile-row | Process one tile-row at a time: tilize 32 patches, matmul immediately, pop. Shrinks cb_vol2col_tiled from M_t*K_t to K_t tiles, saving 324 KB L1. | Tilize all → matmul all (original): needs M_t*K_t tiles in L1 |
| Reduced cb_vol2col_rm from 2*TILE_HEIGHT to TILE_HEIGHT pages | Fused loop only needs one tile-row of RM patches at a time. Saves 162 KB L1 for large blockings. | Double-buffered RM CB: needed when tilize consumes all M rows at once |

## Constraints & Workarounds
- **Hardware:** BH Loud Box, 8x p150b (130 cores, 13x10 grid per device)
- **Device hangs:** Non-power-of-2 H or W block sizes cause kernel hangs (W=13, W=52, H=23, H=46 confirmed). Only power-of-2 spatial blocks are safe.
- **L1 limit:** 1,572,864 bytes per core. Large spatial blocks + large C_out_block → L1 OOM (caught as TT_THROW)
- **matmul_N_t subblock:** C_out=192 with certain spatial blocks fails with "matmul_N_t must be divisible by out_subblock_w"
- **L1 prefetch budget:** Shard must fit in remaining L1 after CBs. Fused tilize+matmul frees ~486 KB, enabling larger blockings (e.g. W=16, H=32) that previously OOM'd.
- **Weight preparation:** `prepare_conv3d_weights` must be called with matching `C_in_block` — manual weight reshaping produces wrong results for multi-C_in-block configs.

## Surprises & Discoveries
- Compute (tilize+matmul) is fully hidden behind reader pipeline with optimal (32,8) blocking — commenting out all compute changes wall time by <1%.
- The reader DRAM gather is the true bottleneck; larger blocks reduce total gather iterations and give linear speedup.
- Non-power-of-2 spatial blocks hang even when they exactly divide the output dims (e.g. H=23 divides H_out=184 exactly but hangs).

## Open Questions
- [ ] Why do non-power-of-2 H/W block sizes cause hangs? Is it a core assignment or NOC addressing bug?
- [ ] Should the blocking table be mesh-specific (keyed on mesh shape) or universal?
- [ ] How much do 720p optimal blockings differ from 480p for the same (C_in, C_out, kernel)?

## State
- [x] Built sweep script with staged strategy + mesh-aware local dims
- [x] Consolidated sweep script with Stephen's valid_cin/cout, external padding, deallocate
- [x] bh_4x8 sweep: 62/62 configs (480p+720p, cached+uncached) — on 1x1 simulated dims
- [x] bh_4x32 sweep: 51/52 configs — on 1x1 simulated dims
- [x] bh_2x4 sweep: 30 configs (consolidated script) — on actual 2x4 mesh
- [x] wh_4x8 sweep: 61/62 configs — on BH hardware (WH has different core grid)
- [x] Made blocking table mesh-aware: keyed on (h_factor, w_factor, C_in, C_out, kernel)
- [x] Updated get_conv3d_config() to accept h_factor/w_factor from parallel_config
- [x] Added missing time conv entries (384→768 (3,1,1) and 192→384 (3,1,1)) — 19-635x speedup
- [x] VAE decoder end-to-end on bh_2x4: **2.09x faster** (45.6s → 21.8s)
- [x] H-row interleaved reader: 7-15% speedup on top 3 bottleneck layers
- [x] Fused tilize+matmul: bit-identical, saves 324 KB L1, enables larger blockings
- [x] Reduced cb_vol2col_rm: saves 162 KB L1
- [x] Tracy device profiling: identified reader DRAM gather as true bottleneck (compute fully overlapped at optimal blocking)
- [x] Brute-force re-sweep with fused kernel: found (32,8) = 5515 us for up3_res (32% vs baseline)
- [ ] **Validate bh_4x8 blockings on actual BH Galaxy** (swept on simulated 1x1 dims)
- [ ] **Validate bh_4x32 blockings on actual BH Galaxy 6U** (swept on simulated 1x1 dims)
- [ ] **Sweep wh_4x8 on actual WH hardware** (BH 130 cores vs WH 64 cores)
- [ ] bh_2x2 sweep
- [ ] Run full Wan pipeline (transformer + VAE) end-to-end with new blockings
- [ ] Investigate non-power-of-2 spatial block hangs

## Key Measurements

### up3_res optimization progression (96x96 k333, bh_4x32 720p, 1x1 mesh 12x10 grid)

| Change | Blocking | Per-call (us) | vs baseline |
|--------|----------|---------------|-------------|
| Baseline (original main) | (96,96,1,8,8) | 8,135 | — |
| + h-row interleaved reader | (96,96,1,8,8) | 7,276 | 11% |
| + fused tilize+matmul | (96,96,1,8,8) | 7,201 | 11% |
| + reduced cb_vol2col_rm | (96,96,1,8,8) | 6,519 | 20% |
| + optimal blocking | (96,96,1,8,16) | 5,956 | 27% |
| + best blocking | **(96,96,1,32,8)** | **5,515** | **32%** |

Decoder total for up3_res (6x calls): 48,810 us → 33,090 us = **32% faster**

### up2_res sweep results (192x192 k333, bh_4x32 720p)

Best blocking with fused kernel: **(96,96,1,32,4) = 6,220 us** (vs 7,629 us baseline = 18% faster)

### Tilize/matmul ablation (up3_res, blocking (96,96,1,8,16))

| Config | Per-call (us) | Delta from full |
|--------|---------------|-----------------|
| Full (tilize + matmul) | 5,924 | — |
| Matmul only (no tilize) | 6,723 | +799 |
| Tilize only (no matmul) | 6,215 | +291 |
| No compute (reader+writer) | 5,991 | +67 |

Compute is fully hidden behind the reader pipeline. The op is reader-bound.

### bh_4x8 480p (h_factor=4, w_factor=8, per-device H/W = global/factor)

| Case | Shape | H_local×W_local | Baseline | Best | Speedup |
|------|-------|-----------------|----------|------|---------|
| conv_in | (32,384,(3,3,3)) | 15×13 | (32,96,1,2,32) 106us | **(32,32,1,15,4) 90us** | 1.19x |
| mid_r0c1 | (384,384,(3,3,3)) | 15×13 | (96,96,1,8,4) 351us | **(64,32,1,8,8) 168us** | 2.10x |
| up0_sconv | (384,192,(1,3,3)) | 30×26 | (192,96,1,32,4) 178us | **(192,32,1,32,4) 143us** | 1.25x |
| up0_tconv | (384,768,(3,1,1)) | 15×13 | default 384,32,1,1,1 | **(384,32,1,5,8) 112us** | N/A |
| up1_r0c1 | (192,192,(3,3,3)) | 30×26 | (96,96,1,8,4) 372us | **(96,32,1,3,16) 262us** | 1.42x |
| up1_sconv | (192,96,(1,3,3)) | 60×52 | (192,96,1,4,8) 323us | **(192,96,1,32,4) 265us** | 1.22x |
| up1_tconv | (192,384,(3,1,1)) | 30×26 | N/A | **HUNG** — skip W=26 at H=5 | - |
| up2_r0c1 | (96,96,(3,3,3)) | 60×52 | (96,96,1,8,8) 652us | **(96,32,1,3,32) 495us** (partial) | 1.32x |
| conv_out | (96,3,(3,3,3)) | 60×52 | (96,32,1,16,8) | **(96,32,1,3,26) 518us** (partial) | - |

Commands:
```bash
source python_env/bin/activate && export PYTHONPATH=$(pwd)
python models/tt_dit/tests/models/wan2_2/sweep_conv3d_blockings.py --mesh bh_4x8 --skip-done
```

Results saved to: `.cache/wan_conv3d_blocking_sweeps/bh_4x8/*.json`
