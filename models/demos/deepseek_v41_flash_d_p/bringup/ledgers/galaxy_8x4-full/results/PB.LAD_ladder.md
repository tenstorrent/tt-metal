# PB.LAD: the full performance ladder (batching branch, head 688e78f5e1a + the 4x1792s test case)

Owner task 2026-11-22. 8x4 Galaxy, layers 0-20 (perf load), BRINGUP_CHUNK=8192, DECODER_SKIP=decode, engram resident,
every packed switch on (the gate's env). PB.9C's method (tests/packed/ladder_common.py): at each history h every segment
sits at h in its own slot; steady = mid-request chunk (write_start = start), resumed = first chunk after a prefix hit
(write_start = start + 128, encoder replay); the real previous chunk runs untimed before every timed one; warm-up until
no new program, 10 timed runs with program-cache misses refused, median; device ms = one profiled run (slowest chip);
exposed host = wall - device. tok/s = new tokens / wall (steady: 8192 on every path; resumed: 8192 - 128 x segments).
Source: gate `PB.LAD --check` (PASS), log runs/galaxy_8x4-full/run1/logs/PB.LAD.check.log.

## Single request, 8192 tokens (test_skip_positions)

| history | steady wall ms (spread) | steady tok/s | resumed wall ms | resumed tok/s | steady device ms | steady exposed host ms | resumed device ms | resumed exposed host ms | serving wall ms |
|---|---|---|---|---|---|---|---|---|---|
| 51200 | 99.8 (4.9) | 82084 | 105.5 | 76436 | 84.4 | 15.4 | 86.2 | 19.3 | 102.1 |
| 131072 | 100.9 (90.5) | 81189 | 103.2 | 78140 | 88.6 | 12.3 | 87.8 | 15.4 | 105.1 |
| 139264 | 99.8 (6.0) | 82084 | 104.2 | 77390 | 85.9 | 13.9 | 87.6 | 16.6 | 103.2 |
| 262144 | 102.2 (17.5) | 80157 | 105.9 | 76147 | 90.7 | 11.5 | 92.5 | 13.4 | 106.6 |
| 524288 | 104.9 (2.5) | 78093 | 109.8 | 73443 | 94.6 | 10.3 | 96.8 | 13.0 | 109.5 |
| 786432 | 116.5 (3.2) | 70318 | 118.5 | 68051 | 105.1 | 11.4 | 108.4 | 10.1 | 121.5 |
| 1024000 | 120.5 (5.6) | 67983 | 123.6 | 65243 | 109.0 | 11.5 | 110.5 | 13.1 | 124.2 |

## Chip-aligned 4 x 2048 (path lanes4x2k: four requests, 8 chips each; test_packed_walls)

| history | steady wall ms (spread) | steady tok/s | resumed wall ms | resumed tok/s | steady device ms | steady exposed host ms | resumed device ms | serving wall ms | gathers |
|---|---|---|---|---|---|---|---|---|---|
| 51200 | 109.4 (14.5) | 74881 | 111.8 | 68694 | 91.4 | 18.0 | 91.0 | 122.1 | 6 |
| 131072 | 112.4 (12.2) | 72883 | 110.7 | 69377 | 92.4 | 20.0 | 92.6 | 111.8 | 6 |
| 139264 | 112.1 (11.2) | 73078 | 109.9 | 69882 | 93.2 | 18.9 | 93.3 | 115.1 | 6 |
| 262144 | 116.1 (25.8) | 70560 | 117.1 | 65585 | 98.0 | 18.1 | 97.8 | 119.2 | 9 |
| 524288 | 124.8 (9.2) | 65641 | 123.0 | 62439 | 106.9 | 17.9 | 106.6 | 131.3 | 12 |
| 786432 | 128.9 (23.0) | 63553 | 129.2 | 59443 | 115.7 | 13.2 | 115.7 | 130.1 | 14 |
| 1024000 | 134.0 (26.0) | 61134 | 134.8 | 56973 | 120.9 | 13.1 | 120.9 | 137.5 | 18 |

## 4 x 1792 + 1024 split (path lanes4x1792s: four requests of 7 whole chips + a 5th request's first 1024 tokens on 4 chips; 8192 real tokens, no pad row; test_packed_walls)

| history | steady wall ms (spread) | steady tok/s | resumed wall ms | resumed tok/s | steady device ms | steady exposed host ms | resumed device ms | serving wall ms | gathers |
|---|---|---|---|---|---|---|---|---|---|
| 51200 | 114.9 (49.2) | 71297 | 114.1 | 66188 | 92.0 | 22.9 | 92.7 | 116.9 | 6 |
| 131072 | 113.2 (6.9) | 72367 | 114.1 | 66188 | 93.3 | 19.9 | 93.2 | 120.4 | 6 |
| 139264 | 116.4 (7.8) | 70378 | 114.0 | 66246 | 94.6 | 21.8 | 95.0 | 119.8 | 6 |
| 262144 | 119.4 (15.7) | 68610 | 120.2 | 62829 | 99.7 | 19.7 | 99.4 | 125.8 | 9 |
| 524288 | 127.8 (12.8) | 64100 | 126.7 | 59605 | 108.0 | 19.8 | 109.4 | 133.8 | 12 |
| 786432 | 133.1 (1.9) | 61548 | 131.9 | 57255 | 118.1 | 15.0 | 118.6 | 133.4 | 14 |
| 1024000 | 137.8 (3.5) | 59448 | 137.3 | 55004 | 125.7 | 12.1 | 124.6 | 139.2 | 18 |

## Cold strict no-rebuild (test_no_rebuild, NO_REBUILD_PACKED_STRICT=1, first in a fresh process)

PASS on every path, 0 program builds after warm-up: single8192, one4k, one1600, lanes4x2k, lanes4x1600s, lanes6mix,
lanes2x3k (fresh positions, slots and layouts each chunk). Determinism max |diff| 0.0; on vs off pulled-back chunk 0.0.

## Reading

- With no padding, 4x1792s still runs 1-4 ms slower than 4x2k at every history (steady, both 8192 rows), and its tok/s
  is 1-6 % lower. It has five segments instead of four: device +0.6..+4.8 ms (most at 1M: 125.7 vs 120.9), exposed host
  +0..+5 ms (one more segment's per-segment programs, DS41-364).
- Packed vs single at 8192 tokens: +10..+17 ms steady (139264: 112.1 / 116.4 vs 99.8), mostly exposed host (19-22 vs
  14 ms) and device +7..+17 ms.
- The resumed and steady walls of the packed paths are within noise of each other (DS41-363: replay runs in the one call).
