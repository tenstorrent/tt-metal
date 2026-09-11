# LTX 2.3 Fast — Dynamic-Length Trace Bucketing Plan

Branch: `rsalman-ltx-dynamic-trace-sept-10`

## Goal

Reuse a single captured `ttnn` trace per denoise stage across many request shapes (resolution / FPS / duration) instead of recapturing per shape. A ttnn trace bakes tensor addresses, program configs, and Python-derived scalars as constants, so a naive trace is locked to one sequence length. We break that by (1) padding every request up to a small set of fixed **bucket** lengths and (2) feeding the *real* logical length to ring-joint SDPA as an on-device tensor read every replay, so masking follows the request without recapturing.

## Config space (256)

`8 canvases (4 resolutions x 2 orientations) x 4 FPS x 8 durations = 256`. Portrait is the transpose of landscape, so its token count (H*W) is identical — the 256 configs collapse to **128 distinct sequence lengths** (each realised in both orientations).

- Resolutions (landscape H x W): 720p `704x1280`, 1080p `1088x1920`, 1440p `1472x2560`, 4K `2176x3840`.
- FPS: 24, 25, 48, 50.  Durations (s): 6, 8, 10, 12, 14, 16, 18, 20.

> **Note (open item):** the code currently encodes 1440p as `1440x2560`, which is not divisible by 64 (half-res 720 is not divisible by 32). The correct canvas is `1472x2560`, used throughout this document. Fix `LTX_FAST_CANVASES` before enabling any 1440p bucket.

## Sequence-length formulas

```
num_frames  = ceil((fps*seconds - 1)/8)*8 + 1      # VAE-compatible frame count
LF          = (num_frames - 1)//8 + 1              # latent frames (temporal /8)
rows_s1     = (H/2/32) * (W/2/32)                  # stage 1 runs at half resolution
rows_s2     = (H/32)   * (W/32)   = 4 * rows_s1     # stage 2 runs at full resolution
seq_real    = LF * rows_stage                      # logical tokens
seq_trace   = ceil(seq_real / 256) * 256           # SP(8) x TILE(32) alignment
per_device_M= seq_trace / 8                         # SP=8 shard (single GLX, 4x8 mesh)
```

### Latent frames (LF) by FPS x duration

| FPS \\ dur | 6s | 8s | 10s | 12s | 14s | 16s | 18s | 20s |
|---|---|---|---|---|---|---|---|---|
| **24** | 19 | 25 | 31 | 37 | 43 | 49 | 55 | 61 |
| **25** | 20 | 26 | 33 | 39 | 45 | 51 | 58 | 64 |
| **48** | 37 | 49 | 61 | 73 | 85 | 97 | 109 | 121 |
| **50** | 39 | 51 | 64 | 76 | 89 | 101 | 114 | 126 |

### Rows per latent frame

| Resolution | rows_s1 | rows_s2 |
|---|---|---|
| 720p (704x1280) | 220 | 880 |
| 1080p (1088x1920) | 510 | 2040 |
| 1440p (1472x2560) | 920 | 3680 |
| 4K (2176x3840) | 2040 | 8160 |

## Bucketing scheme (8 buckets)

Configs are partitioned by stage-1 trace length; each stage's bucket size is the max trace length among the configs in that bucket (guarantees `bucket_N >= seq_real` for both stages simultaneously). `M = N/8` is the per-device sequence length on the 4x8 (SP=8) GLX.

| Bucket | #configs (x2 orient) | s1 N | s1 M | s2 N | s2 M | s1 span (real trace) |
|---|---|---|---|---|---|---|
| 0 | 20 | 8704 | 1088 | 34560 | 4320 | 4352..8704 |
| 1 | 32 | 14080 | 1760 | 56320 | 7040 | 9472..14080 |
| 2 | 44 | 25088 | 3136 | 100352 | 12544 | 15872..25088 |
| 3 | 40 | 39680 | 4960 | 158464 | 19808 | 26112..39680 |
| 4 | 40 | 58880 | 7360 | 235520 | 29440 | 40960..58880 |
| 5 | 24 | 81920 | 10240 | 327680 | 40960 | 61952..81920 |
| 6 | 36 | 130560 | 16320 | 522240 | 65280 | 87808..130560 |
| 7 | 20 | 257280 | 32160 | 1028352 | 128544 | 148992..257280 |

Mean padding: s1 23.1%, s2 23.5%. Worst-case padding: s1 100%, s2 105% (the smallest config in a bucket; e.g. bucket 0 spans 4352..8704, so 720p/24fps/6s runs at ~2x its needed length). Splitting the widest buckets is the lever to cut worst-case padding.

**Bucket 0 (lowest tier) — implemented:** s1 `N=8704 (M=1088)`, s2 `N=34560 (M=4320)`, covering 20 configs (10 distinct lengths x 2 orientations): all 720p at 24fps@{6,8,10,12}s, 25fps@{6,8,10,12}s, 48fps@6s, 50fps@6s.

## Full sequence-length list (128 distinct; x2 orientations = 256)

Sorted by stage-1 trace length. `s1/s2 N` are the bucket sizes actually allocated.

| res | fps | dur | LF | s1 real | s1 trace | s1 N | s2 real | s2 trace | s2 N | bkt |
|---|---|---|---|---|---|---|---|---|---|---|
| 720p | 24 | 6 | 19 | 4180 | 4352 | 8704 | 16720 | 16896 | 34560 | 0 |
| 720p | 25 | 6 | 20 | 4400 | 4608 | 8704 | 17600 | 17664 | 34560 | 0 |
| 720p | 24 | 8 | 25 | 5500 | 5632 | 8704 | 22000 | 22016 | 34560 | 0 |
| 720p | 25 | 8 | 26 | 5720 | 5888 | 8704 | 22880 | 23040 | 34560 | 0 |
| 720p | 24 | 10 | 31 | 6820 | 6912 | 8704 | 27280 | 27392 | 34560 | 0 |
| 720p | 25 | 10 | 33 | 7260 | 7424 | 8704 | 29040 | 29184 | 34560 | 0 |
| 720p | 24 | 12 | 37 | 8140 | 8192 | 8704 | 32560 | 32768 | 34560 | 0 |
| 720p | 48 | 6 | 37 | 8140 | 8192 | 8704 | 32560 | 32768 | 34560 | 0 |
| 720p | 25 | 12 | 39 | 8580 | 8704 | 8704 | 34320 | 34560 | 34560 | 0 |
| 720p | 50 | 6 | 39 | 8580 | 8704 | 8704 | 34320 | 34560 | 34560 | 0 |
| 720p | 24 | 14 | 43 | 9460 | 9472 | 14080 | 37840 | 37888 | 56320 | 1 |
| 1080p | 24 | 6 | 19 | 9690 | 9728 | 14080 | 38760 | 38912 | 56320 | 1 |
| 720p | 25 | 14 | 45 | 9900 | 9984 | 14080 | 39600 | 39680 | 56320 | 1 |
| 1080p | 25 | 6 | 20 | 10200 | 10240 | 14080 | 40800 | 40960 | 56320 | 1 |
| 720p | 24 | 16 | 49 | 10780 | 11008 | 14080 | 43120 | 43264 | 56320 | 1 |
| 720p | 48 | 8 | 49 | 10780 | 11008 | 14080 | 43120 | 43264 | 56320 | 1 |
| 720p | 25 | 16 | 51 | 11220 | 11264 | 14080 | 44880 | 45056 | 56320 | 1 |
| 720p | 50 | 8 | 51 | 11220 | 11264 | 14080 | 44880 | 45056 | 56320 | 1 |
| 720p | 24 | 18 | 55 | 12100 | 12288 | 14080 | 48400 | 48640 | 56320 | 1 |
| 1080p | 24 | 8 | 25 | 12750 | 12800 | 14080 | 51000 | 51200 | 56320 | 1 |
| 720p | 25 | 18 | 58 | 12760 | 12800 | 14080 | 51040 | 51200 | 56320 | 1 |
| 1080p | 25 | 8 | 26 | 13260 | 13312 | 14080 | 53040 | 53248 | 56320 | 1 |
| 720p | 24 | 20 | 61 | 13420 | 13568 | 14080 | 53680 | 53760 | 56320 | 1 |
| 720p | 48 | 10 | 61 | 13420 | 13568 | 14080 | 53680 | 53760 | 56320 | 1 |
| 720p | 25 | 20 | 64 | 14080 | 14080 | 14080 | 56320 | 56320 | 56320 | 1 |
| 720p | 50 | 10 | 64 | 14080 | 14080 | 14080 | 56320 | 56320 | 56320 | 1 |
| 1080p | 24 | 10 | 31 | 15810 | 15872 | 25088 | 63240 | 63488 | 100352 | 2 |
| 720p | 48 | 12 | 73 | 16060 | 16128 | 25088 | 64240 | 64256 | 100352 | 2 |
| 1080p | 25 | 10 | 33 | 16830 | 16896 | 25088 | 67320 | 67328 | 100352 | 2 |
| 720p | 50 | 12 | 76 | 16720 | 16896 | 25088 | 66880 | 67072 | 100352 | 2 |
| 1440p | 24 | 6 | 19 | 17480 | 17664 | 25088 | 69920 | 70144 | 100352 | 2 |
| 1440p | 25 | 6 | 20 | 18400 | 18432 | 25088 | 73600 | 73728 | 100352 | 2 |
| 1080p | 24 | 12 | 37 | 18870 | 18944 | 25088 | 75480 | 75520 | 100352 | 2 |
| 1080p | 48 | 6 | 37 | 18870 | 18944 | 25088 | 75480 | 75520 | 100352 | 2 |
| 720p | 48 | 14 | 85 | 18700 | 18944 | 25088 | 74800 | 75008 | 100352 | 2 |
| 720p | 50 | 14 | 89 | 19580 | 19712 | 25088 | 78320 | 78336 | 100352 | 2 |
| 1080p | 25 | 12 | 39 | 19890 | 19968 | 25088 | 79560 | 79616 | 100352 | 2 |
| 1080p | 50 | 6 | 39 | 19890 | 19968 | 25088 | 79560 | 79616 | 100352 | 2 |
| 720p | 48 | 16 | 97 | 21340 | 21504 | 25088 | 85360 | 85504 | 100352 | 2 |
| 1080p | 24 | 14 | 43 | 21930 | 22016 | 25088 | 87720 | 87808 | 100352 | 2 |
| 720p | 50 | 16 | 101 | 22220 | 22272 | 25088 | 88880 | 89088 | 100352 | 2 |
| 1080p | 25 | 14 | 45 | 22950 | 23040 | 25088 | 91800 | 91904 | 100352 | 2 |
| 1440p | 24 | 8 | 25 | 23000 | 23040 | 25088 | 92000 | 92160 | 100352 | 2 |
| 1440p | 25 | 8 | 26 | 23920 | 24064 | 25088 | 95680 | 95744 | 100352 | 2 |
| 720p | 48 | 18 | 109 | 23980 | 24064 | 25088 | 95920 | 96000 | 100352 | 2 |
| 1080p | 24 | 16 | 49 | 24990 | 25088 | 25088 | 99960 | 100096 | 100352 | 2 |
| 1080p | 48 | 8 | 49 | 24990 | 25088 | 25088 | 99960 | 100096 | 100352 | 2 |
| 720p | 50 | 18 | 114 | 25080 | 25088 | 25088 | 100320 | 100352 | 100352 | 2 |
| 1080p | 25 | 16 | 51 | 26010 | 26112 | 39680 | 104040 | 104192 | 158464 | 3 |
| 1080p | 50 | 8 | 51 | 26010 | 26112 | 39680 | 104040 | 104192 | 158464 | 3 |
| 720p | 48 | 20 | 121 | 26620 | 26624 | 39680 | 106480 | 106496 | 158464 | 3 |
| 720p | 50 | 20 | 126 | 27720 | 27904 | 39680 | 110880 | 111104 | 158464 | 3 |
| 1080p | 24 | 18 | 55 | 28050 | 28160 | 39680 | 112200 | 112384 | 158464 | 3 |
| 1440p | 24 | 10 | 31 | 28520 | 28672 | 39680 | 114080 | 114176 | 158464 | 3 |
| 1080p | 25 | 18 | 58 | 29580 | 29696 | 39680 | 118320 | 118528 | 158464 | 3 |
| 1440p | 25 | 10 | 33 | 30360 | 30464 | 39680 | 121440 | 121600 | 158464 | 3 |
| 1080p | 24 | 20 | 61 | 31110 | 31232 | 39680 | 124440 | 124672 | 158464 | 3 |
| 1080p | 48 | 10 | 61 | 31110 | 31232 | 39680 | 124440 | 124672 | 158464 | 3 |
| 1080p | 25 | 20 | 64 | 32640 | 32768 | 39680 | 130560 | 130560 | 158464 | 3 |
| 1080p | 50 | 10 | 64 | 32640 | 32768 | 39680 | 130560 | 130560 | 158464 | 3 |
| 1440p | 24 | 12 | 37 | 34040 | 34048 | 39680 | 136160 | 136192 | 158464 | 3 |
| 1440p | 48 | 6 | 37 | 34040 | 34048 | 39680 | 136160 | 136192 | 158464 | 3 |
| 1440p | 25 | 12 | 39 | 35880 | 36096 | 39680 | 143520 | 143616 | 158464 | 3 |
| 1440p | 50 | 6 | 39 | 35880 | 36096 | 39680 | 143520 | 143616 | 158464 | 3 |
| 1080p | 48 | 12 | 73 | 37230 | 37376 | 39680 | 148920 | 148992 | 158464 | 3 |
| 1080p | 50 | 12 | 76 | 38760 | 38912 | 39680 | 155040 | 155136 | 158464 | 3 |
| 4K | 24 | 6 | 19 | 38760 | 38912 | 39680 | 155040 | 155136 | 158464 | 3 |
| 1440p | 24 | 14 | 43 | 39560 | 39680 | 39680 | 158240 | 158464 | 158464 | 3 |
| 4K | 25 | 6 | 20 | 40800 | 40960 | 58880 | 163200 | 163328 | 235520 | 4 |
| 1440p | 25 | 14 | 45 | 41400 | 41472 | 58880 | 165600 | 165632 | 235520 | 4 |
| 1080p | 48 | 14 | 85 | 43350 | 43520 | 58880 | 173400 | 173568 | 235520 | 4 |
| 1440p | 24 | 16 | 49 | 45080 | 45312 | 58880 | 180320 | 180480 | 235520 | 4 |
| 1440p | 48 | 8 | 49 | 45080 | 45312 | 58880 | 180320 | 180480 | 235520 | 4 |
| 1080p | 50 | 14 | 89 | 45390 | 45568 | 58880 | 181560 | 181760 | 235520 | 4 |
| 1440p | 25 | 16 | 51 | 46920 | 47104 | 58880 | 187680 | 187904 | 235520 | 4 |
| 1440p | 50 | 8 | 51 | 46920 | 47104 | 58880 | 187680 | 187904 | 235520 | 4 |
| 1080p | 48 | 16 | 97 | 49470 | 49664 | 58880 | 197880 | 197888 | 235520 | 4 |
| 1440p | 24 | 18 | 55 | 50600 | 50688 | 58880 | 202400 | 202496 | 235520 | 4 |
| 4K | 24 | 8 | 25 | 51000 | 51200 | 58880 | 204000 | 204032 | 235520 | 4 |
| 1080p | 50 | 16 | 101 | 51510 | 51712 | 58880 | 206040 | 206080 | 235520 | 4 |
| 4K | 25 | 8 | 26 | 53040 | 53248 | 58880 | 212160 | 212224 | 235520 | 4 |
| 1440p | 25 | 18 | 58 | 53360 | 53504 | 58880 | 213440 | 213504 | 235520 | 4 |
| 1080p | 48 | 18 | 109 | 55590 | 55808 | 58880 | 222360 | 222464 | 235520 | 4 |
| 1440p | 24 | 20 | 61 | 56120 | 56320 | 58880 | 224480 | 224512 | 235520 | 4 |
| 1440p | 48 | 10 | 61 | 56120 | 56320 | 58880 | 224480 | 224512 | 235520 | 4 |
| 1080p | 50 | 18 | 114 | 58140 | 58368 | 58880 | 232560 | 232704 | 235520 | 4 |
| 1440p | 25 | 20 | 64 | 58880 | 58880 | 58880 | 235520 | 235520 | 235520 | 4 |
| 1440p | 50 | 10 | 64 | 58880 | 58880 | 58880 | 235520 | 235520 | 235520 | 4 |
| 1080p | 48 | 20 | 121 | 61710 | 61952 | 81920 | 246840 | 247040 | 327680 | 5 |
| 4K | 24 | 10 | 31 | 63240 | 63488 | 81920 | 252960 | 253184 | 327680 | 5 |
| 1080p | 50 | 20 | 126 | 64260 | 64512 | 81920 | 257040 | 257280 | 327680 | 5 |
| 1440p | 48 | 12 | 73 | 67160 | 67328 | 81920 | 268640 | 268800 | 327680 | 5 |
| 4K | 25 | 10 | 33 | 67320 | 67328 | 81920 | 269280 | 269312 | 327680 | 5 |
| 1440p | 50 | 12 | 76 | 69920 | 70144 | 81920 | 279680 | 279808 | 327680 | 5 |
| 4K | 24 | 12 | 37 | 75480 | 75520 | 81920 | 301920 | 302080 | 327680 | 5 |
| 4K | 48 | 6 | 37 | 75480 | 75520 | 81920 | 301920 | 302080 | 327680 | 5 |
| 1440p | 48 | 14 | 85 | 78200 | 78336 | 81920 | 312800 | 312832 | 327680 | 5 |
| 4K | 25 | 12 | 39 | 79560 | 79616 | 81920 | 318240 | 318464 | 327680 | 5 |
| 4K | 50 | 6 | 39 | 79560 | 79616 | 81920 | 318240 | 318464 | 327680 | 5 |
| 1440p | 50 | 14 | 89 | 81880 | 81920 | 81920 | 327520 | 327680 | 327680 | 5 |
| 4K | 24 | 14 | 43 | 87720 | 87808 | 130560 | 350880 | 350976 | 522240 | 6 |
| 1440p | 48 | 16 | 97 | 89240 | 89344 | 130560 | 356960 | 357120 | 522240 | 6 |
| 4K | 25 | 14 | 45 | 91800 | 91904 | 130560 | 367200 | 367360 | 522240 | 6 |
| 1440p | 50 | 16 | 101 | 92920 | 92928 | 130560 | 371680 | 371712 | 522240 | 6 |
| 4K | 24 | 16 | 49 | 99960 | 100096 | 130560 | 399840 | 399872 | 522240 | 6 |
| 4K | 48 | 8 | 49 | 99960 | 100096 | 130560 | 399840 | 399872 | 522240 | 6 |
| 1440p | 48 | 18 | 109 | 100280 | 100352 | 130560 | 401120 | 401152 | 522240 | 6 |
| 4K | 25 | 16 | 51 | 104040 | 104192 | 130560 | 416160 | 416256 | 522240 | 6 |
| 4K | 50 | 8 | 51 | 104040 | 104192 | 130560 | 416160 | 416256 | 522240 | 6 |
| 1440p | 50 | 18 | 114 | 104880 | 104960 | 130560 | 419520 | 419584 | 522240 | 6 |
| 1440p | 48 | 20 | 121 | 111320 | 111360 | 130560 | 445280 | 445440 | 522240 | 6 |
| 4K | 24 | 18 | 55 | 112200 | 112384 | 130560 | 448800 | 449024 | 522240 | 6 |
| 1440p | 50 | 20 | 126 | 115920 | 115968 | 130560 | 463680 | 463872 | 522240 | 6 |
| 4K | 25 | 18 | 58 | 118320 | 118528 | 130560 | 473280 | 473344 | 522240 | 6 |
| 4K | 24 | 20 | 61 | 124440 | 124672 | 130560 | 497760 | 497920 | 522240 | 6 |
| 4K | 48 | 10 | 61 | 124440 | 124672 | 130560 | 497760 | 497920 | 522240 | 6 |
| 4K | 25 | 20 | 64 | 130560 | 130560 | 130560 | 522240 | 522240 | 522240 | 6 |
| 4K | 50 | 10 | 64 | 130560 | 130560 | 130560 | 522240 | 522240 | 522240 | 6 |
| 4K | 48 | 12 | 73 | 148920 | 148992 | 257280 | 595680 | 595712 | 1028352 | 7 |
| 4K | 50 | 12 | 76 | 155040 | 155136 | 257280 | 620160 | 620288 | 1028352 | 7 |
| 4K | 48 | 14 | 85 | 173400 | 173568 | 257280 | 693600 | 693760 | 1028352 | 7 |
| 4K | 50 | 14 | 89 | 181560 | 181760 | 257280 | 726240 | 726272 | 1028352 | 7 |
| 4K | 48 | 16 | 97 | 197880 | 197888 | 257280 | 791520 | 791552 | 1028352 | 7 |
| 4K | 50 | 16 | 101 | 206040 | 206080 | 257280 | 824160 | 824320 | 1028352 | 7 |
| 4K | 48 | 18 | 109 | 222360 | 222464 | 257280 | 889440 | 889600 | 1028352 | 7 |
| 4K | 50 | 18 | 114 | 232560 | 232704 | 257280 | 930240 | 930304 | 1028352 | 7 |
| 4K | 48 | 20 | 121 | 246840 | 247040 | 257280 | 987360 | 987392 | 1028352 | 7 |
| 4K | 50 | 20 | 126 | 257040 | 257280 | 257280 | 1028160 | 1028352 | 1028352 | 7 |

## Device mechanism (implemented)

Ring-joint SDPA gained an optional `logical_n_tensor` input: a one-element `UINT32` `ROW_MAJOR`
DRAM tensor, replicated across the mesh, read on-device on every invocation. When present it
overrides the baked scalar `logical_n` for masking and ring-work derivation:

- Reader/writer derive `logical_nt = ceil(logical_n / 32)`, the active ring-iteration mask, and
  K-chunk pruning from the tensor value each replay.
- The writer builds the **sub-tile partial-column mask** from `logical_n % 32` (full column when
  tile-aligned), so lengths that end mid-tile mask exactly — `ceil(logical_n/32)` alone would
  silently change softmax normalisation.
- Restricted to non-causal streaming compute; independent of and mutually exclusive with the
  chunked-cache metadata path (`slot_id` / `kv_actual_isl`). Scalar and chunked behaviour unchanged.
- `logical_n` is excluded from the program hash, so one capture replays across lengths.

On the LTX side each `(stage, bucket)` owns a persistent `LTXTransformerState`; the video latent,
RoPE, cross-PE, and padding masks are padded to the bucket `N`, and a persistent uint32
length StateTensor (the real token count) is threaded into both the video self-attention and the
V->A ring cross-attention SDPA calls.

## Status

| Item | State |
|---|---|
| Ring-SDPA `logical_n_tensor` device path (C++ + kernels) | Implemented; full Release build green |
| LTX bucket routing + `(stage,bucket)` trace keys + length StateTensor | Implemented |
| Host unit tests (routing, 20/256 count, boundaries, RoPE padding) | 11 pass |
| Device PCC test (720p lowest bucket, self-attention) | Written (`test_ltx_transformer_block_lowest_bucket`); not yet run on HW |
| Device PCC / trace replay on 4x8 GLX | Pending hardware |

## Rollout plan

1. **Bucket 0 bring-up (in progress).** Land the device path + routing (done). Run the device PCC
   test on the 4x8 GLX for both stages (aligned + non-tile-aligned lengths). Then a full
   capture -> replay -> release lifecycle at 720p/24fps/6s and confirm output parity vs untraced.
2. **Generalise capture to all 8 buckets.** Warmup allocates the per-bucket states; verify the
   trace-region memory budget holds for the largest bucket (bucket 7 s2 M=128544) on the single GLX.
3. **Fix 1440p canvas** `1440 -> 1472` and add its buckets (2-7) to routing + tests.
4. **V->A cross-attention numerical coverage.** Extend PCC to AV mode so the cross path's
   `logical_n_tensor` is validated, not just video self-attention.
5. **Engineering gaps (deferred):** audio FPS handling (currently 24 baked in AudioLatentShape),
   dynamic frame normalisation, VAE/upsampler shape-locked components that may block switching all
   configs in one process, and I2V (needs a separate trace class).

## Validation commands

Host (no device):
```
pytest models/tt_dit/tests/models/ltx/test_fast_bucket_ltx.py
```
Device PCC — lowest bucket, single GLX (4x8):
```
pytest "models/tt_dit/tests/models/ltx/test_transformer_ltx.py::test_ltx_transformer_block_lowest_bucket" --timeout 1800
```
