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

## Constraints & Workarounds
- **Hardware:** BH Loud Box, 8x p150b (130 cores, 13x10 grid per device)
- **Device hangs:** W=13 and W=52 cause kernel hangs. After hang: `kill -9`, `tt-smi -r 0,1,2,3,4,5,6,7`
- **L1 limit:** 1,572,864 bytes per core. Large spatial blocks + large C_out_block → L1 OOM (caught as TT_THROW)
- **matmul_N_t subblock:** C_out=192 with certain spatial blocks fails with "matmul_N_t must be divisible by out_subblock_w"
- **Missing from blocking table:** (384, 768, (3,1,1)) and (192, 384, (3,1,1)) time convs have no tuned blocking

## Open Questions
- [ ] Why do W=13 and W=52 cause hangs? Is it a core assignment bug?
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
- [ ] **Validate bh_4x8 blockings on actual BH Galaxy** (swept on simulated 1x1 dims)
- [ ] **Validate bh_4x32 blockings on actual BH Galaxy 6U** (swept on simulated 1x1 dims)
- [ ] **Sweep wh_4x8 on actual WH hardware** (BH 130 cores vs WH 64 cores)
- [ ] bh_2x2 sweep
- [ ] Run full Wan pipeline (transformer + VAE) end-to-end with new blockings

## Key Measurements

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
