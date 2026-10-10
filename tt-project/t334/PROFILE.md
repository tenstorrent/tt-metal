# t349 (continues t334): per-layer device profile, LTX conv VAE decode, 4x8, 1080p / 145 frames

## Run
- blx01 (g15blx01) broker job **452** (2026-10-10 11:38-11:40 PDT), full (4,8) mesh, Tracy with `-r`, one decode.
  Test `test_vae_ltx_prof_4x8.py` passed (1 passed, 108 s incl. JIT). Earlier job 448 ran the same decode
  but without op names (tracy `-r` missing), so it is not used. No drops.
- Code: fresh Tracy build of ba3636df5ee (ttp/t48-ltx25-integrated incl. #339 residual-in-conv3d epilogue).
  Decoder blockings as in t48; HALO_ONLY, FOLD_TIME_PAD, EXACT_SHARD on; LTX_FUSE_YUV_OUTPUT=1.
  Random-init decoder weights (timing does not depend on weights), latent 128x19x34x60 (1088x1920, 145 f).
- **Clock: 900 MHz on all 32 chips (blx01 clamp). All numbers are relative only (user #230).**
- Files: `results/t349fix_4x8_table.txt` (full table), `results/t349fix_4x8_conv3d.csv`,
  `results/t349fix_4x8_ops.csv` (slowest chip, every op), raw op report `ops_perf_results_2026_10_10_18_41_15_job452.csv.gz` (887 KB, over the repo size hook) kept uncommitted in `results/` on g15blx02 and in blx01 `/var/tmp/fasth3/t349/res2/`.
  `t349_4x8_table.txt` is the same run with the old FLOP count (upsampler convs counted at the d2s output grid, 2-8x too high);
  `conv_table.py` is fixed.

## Totals (device kernel time, slowest chip 15, 140 ops)
Per-chip op sum min/median/max **478.7 / 480.5 / 482.2 ms** at 900 MHz.

| class | ms | ops | % |
|---|---:|---:|---:|
| conv3d | 367.7 | 42 | 76.3 |
| layout (final Reshape + Permute + misc) | 47.4 | 11 | 9.8 |
| LayerNorm | 40.0 | 37 | 8.3 |
| halo / neighbor-pad CCL | 14.6 | 42 | 3.0 |
| other (AllBroadcast 8.6, RgbToYuv 2.7) | 11.4 | 5 | 2.4 |
| eltwise | 1.1 | 3 | 0.2 |

## Conv3d per layer group (ms = max over chips; all HiFi2, bf16; peak = 1024 FLOP/cyc/core at HiFi4, 2048 at HiFi2)

| layers | in THWC (per chip) | blocking T,H,W,Cout,Cin | ms each | ms total | % decode | % HiFi2 peak | % DRAM BW (blocked est.) |
|---|---|---|---:|---:|---:|---:|---:|
| conv_in | 19x9x8x128 -> 1024 | 7,8,4,128,64 | 0.40 | 0.4 | 0.1 | 11 | 8 |
| res1024 x4 | 19x9x8x1024 | 5,4,8,64,128 | 2.90 | 11.6 | 2.4 | 12 | 12 |
| up_all/2 (1024->4096, d2s) | 19x9x8x1024 | 5,4,8,64,128 | 7.87 | 7.9 | 1.6 | 18 | 18 |
| res512a x4 | 37x17x15x512 | 1,4,8,256,64 | 2.64 | 10.6 | 2.2 | 23 | 10 |
| up_all_x1 (512->4096, d2s) | 37x17x15x512 | 5,2,16,64,128 | 23.95 | 24.0 | 5.0 | 20 | 18 |
| **res512b x8** | 73x34x30x512 | 1,8,4,256,64 | 17.2 | **137.6** | **28.5** | 28 | 11 |
| up_time (d2s T) | 73x34x30x512 | 1,8,4,256,64 | 17.27 | 17.3 | 3.6 | 28 | 11 |
| **res256 x12** | 145x34x30x256 | 1,8,4,256,64 | 6.6 | **79.1** | **16.4** | 36 | 15 |
| up_space (d2s HW) | 145x34x30x256 | 1,8,4,256,64 | 12.76 | 12.8 | 2.6 | 37 | 16 |
| **res128 x8** | 145x68x60x128 | 6,4,8,64,128 | 8.6 | **69.0** | **14.3** | 27 | 21 |
| conv_out (128->48) | 145x68x60x128 | 6,2,16,64,128 | 4.87 | 4.9 | 1.0 | 18 | 21 |
| **all 42** | | | | **374.8** | 77.7 | **28** (56.5 % of HiFi4) | 9-22 |

23.4 TFLOP/chip of conv3d. Conv3d is neither compute-bound (28 % of HiFi2 peak) nor DRAM-bound (at most ~22 % of
512 GB/s even with the re-read estimate). The limit is the reader side: vol2col gather of the 3x3x3 window
into L1 and per-block overhead (same conclusion as #332 / #335).

Other ops worth naming:
- Final unpatchify after conv_out: `ReshapeView` 591600x48 -> 7099200x4 **16.3 ms** and `Permute` 60x3x4x4 -> 4x60x4x145
  **29.6 ms** = **45.9 ms (9.5 %)** for 145x1088x1920x3 output, row-major with a last dim of 4 (and T last after permute).
- LayerNorm before every res conv: 0.93 (res256), 1.04 (res512b), 2.05 ms (res128). The res128 ones move
  ~150 MB in+out per chip in 2.05 ms, ~74 GB/s each way, ~30 % of DRAM BW.
- AllBroadcast x2 (37x18x16x512 2.6 ms, 37x17x16x512 6.0 ms) around up_all_x1 / res512a: 8.6 ms.
- Halo exchange per chip (42 ops): min/median/max 12.2 / 13.6 / 15.4 ms (includes CCL wait).

## Data-movement levers, ranked by expected gain (at this clock)
1. **Unpatchify into conv_out's writer (~40-45 ms, ~9 %).** conv_out (Cout 48 = 3x4x4) is followed by a row-major
   reshape to a last dim of 4 and a permute that puts T last. The upsampler convs already write depth-to-space output
   from the conv3d writer; reuse that path for conv_out (patch 4x4, 3 ch) and, with LTX_FUSE_YUV_OUTPUT, feed
   RgbToYuv directly. Pure layout, should be bit-identical. Lowest risk, clearest win.
2. **Temporal reuse in conv3d for the T_block=1 layers (res512b, up_time, res256, up_space, res512a: 257 ms, 53 %).**
   With T_out_block=1 each output frame re-gathers all 3 input frames, so the reader moves each input frame 3x
   (plus Cout/256 passes). Options: T_out_block 2-4 where L1 allows (sweep per layer on a (2,4) submesh of the full
   mesh), or a rolling 3-frame window kept in L1 across consecutive T blocks. res128 (T_block 6) runs at the same
   ~27 % of peak, so the gain is not certain: expect 10-30 % of 257 ms (25-75 ms); a per-layer blocking sweep decides.
3. **LayerNorm (+ the following SiLU) out of DRAM (15-40 ms, 3-8 %).** 37 standalone LNs at ~30 % of DRAM BW.
   Cheapest: sharded / better-blocked LN (~half, 15-20 ms). Larger: fold the normalize + SiLU into the conv3d reader
   (stats from a small reduction op), which removes one full read+write of every res-block activation (~40 ms).
4. **Replace the two AllBroadcasts with a halo-only exchange (~6-8 ms, ~1.5 %).** They move full 37x18x16x512
   tensors to all chips around up_all_x1; only the H halo rows should be needed.
5. **Blocking for the two all-axis upsampler convs (~10 ms, ~2 %).** up_all/2 and up_all_x1 run at 18-20 % of peak
   vs ~28 % for the res convs (32 ms together); their d2s output scatter and Cout 4096 with Cout_block 64
   (64 passes over the input) look like the cause. Try Cout_block 128-256.

Not repeated (rejected before): fused neighbor_pad+conv3d (#98), LoFi / fidelity changes (#335).

## blx01 cleanup
Driver drv349b deleted the Tracy build (`t349/b`), JIT cache and raw traces (profile_log_device.csv 2.0 GB,
tracy host trace 0.19 GB) after extracting the tables. `/var/tmp/fasth3/t349` is 14 MB (logs + small results).
df / before 52 % (431 G free), after 52 %. No t349 or tracy process left on blx01.
