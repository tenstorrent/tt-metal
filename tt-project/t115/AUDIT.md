# #116: conv3d blocking audit against prefetch_shard_fits()

CPU only (2026-10-03). In halo-only mode, a blocking whose L1 prefetch shard does not fit
fell back to the direct reader. That reader drops the halo, so the output at chip seams is
wrong and any timing taken on it is not valid. Checked with `prefetch_shard_fits()`
(bruteforce_conv3d_sweep.py:152).

Key: (mesh_r, mesh_c, C_in, C_out, T incl. causal pad, H/chip, W/chip), kernel (3,3,3).
Value: (Cin_blk, Cout_blk, T, H, W).

## (a) t100 winners on t48: all fit
s4_res (128,64,6,4,8) and s1_up (128,64,5,2,16) fit. The old values and the other best arms
fit too. s2_res and s3_res are unchanged and fit.

## (b) Notes
| Source | Blocking | Fits | Effect |
|---|---|---|---|
| t114 exact_s2_res "best so far" | (64,256,1,8,8) | no | Its 13464 us (-1.1%) was measured on the fallback, so it is invalid. It was never accepted (below the 3% bar). |
| t114 BUG.md combo 143 | (64,128,6,8,8) | no | Most likely cause of the job 484 hang |
| t114 BUG.md combo 144 | (64,128,6,16,4) | no | Most likely cause of the job 484 hang |
| t114 combos 141,142,145-150 | | yes | |
| t93 PLAN.md | none recorded | n/a | |

## (c) t48 _BLOCKINGS: LTX keys that fail (131 entries pass)
| Key | Blocking | Fits | Affects | Recommended replacement (all fit) |
|---|---|---|---|---|
| (4,8,128,1024,21,5,4) | (128,128,3,2,4) | no | s0 conv_in / ups_initial at 544x960/145f. It is on the timed decode path of t96/t100/#99/#111. | (128,128,1,2,4) or (64,128,3,4,4) |
| (4,8,128,1024,22,5,4) | (128,128,3,2,4) | no | Same layer, 1080p 22-T | (128,128,1,2,4) or (64,128,3,4,4) |
| (4,8,1024,1024,22,5,4) | (128,128,5,2,2) | no | 1080p 22-T | (128,64,7,2,4), the value of the 21-T sibling |
| (4,8,1024,1024,22,10,8) | (64,256,2,8,4) | no | 1080p 22-T | (128,64,5,4,8), the value of the 21-T sibling |
| (2,4,128,128,147,136,120) | (64,128,12,4,8) | no | s4_res on 2x4 | (128,64,12,4,8), which keeps T=12, or (64,128,10,4,8) |

Non-LTX keys (C_in 192/384, H3/Wan) also fail: about 20 of them across 2x4, 4x8 and 4x32,
for example (4,8,384,384,43,46,40)->(96,128,5,16,2) and
(4,8,192,384,43,30,26)->(96,128,5,8,4). The H3/Wan owner needs to look at these. They are
out of scope for LTX.

## Do the earlier timing claims hold?
- #100 519.7->506.2 ms still holds as a relative delta. Both arms used the same s0 blocking,
  their outputs were md5-identical, and both changed layers fit.
- But every halo run that hit (4,8,128,1024,21,5,4) has wrong seams at s0 conv_in, and that
  conv's ~95 us was timed on the fallback. Comparisons that use the same blocking on both sides
  (PSNR 51.58, md5 matches) cannot catch this. Absolute decode time and quality need a device
  re-check with the replacement.
- Timings for the 1080p 22-T keys and the 2x4 s4_res key were taken on the fallback, so they
  are unreliable.

## Next
1. Put the replacements into t48 before the TT_FATAL guard (5bce3778127) lands.
2. When device work is allowed, re-time the replacements and check seam PSNR against a
   single-chip or torch reference.
