# Packed prefill walls (test_packed_walls.py)

8192-row chunk, 4 lanes of 2048; slots at one capacity 1048576; layers 0-20 run (DECODER_SKIP=decode). Wall = engine device input -> ids readback -> sync after the last layer (median of the warm runs). Gathers: each at its least chip (the reference = the same chunk's gathers serial). DRAM: the allocator's peak on the worst chip while the chunk runs, every live slot's state and history buffers held; the 40-layer column adds the layers the performance load does not hold (weights + each live slot's state, per layer kind, from a loaded layer of the same kind).

| case | positions | new tokens | wall ms (spread) | device ms | host exposed ms | serving new tok/s | gather ms: reference / exposed / hidden | per segment: exposed / hidden ms | DRAM GiB a chip: loaded layers peak / 40-layer model |
|---|---|---|---|---|---|---|---|---|---|
| 4xp50_p139264 | 139264/139264/139264/139264 | 6400 | 109.2 (31.1) | 97.1 | 12.1 | 58608 | 2.28 / 0.25 / 2.02 | segtails 0.24 / -0.24; seg3 0.00 / 0.54; seg0 0.00 / 0.57; seg1 0.01 / 0.57; seg2 0.00 / 0.58 | 13.26 / 22.99 |
| 4x2k_p1m | 1040384/1036288/1032192/1028096 | 8192 | 203.6 (4.5) | 192.0 | 11.6 | 40236 | 12.67 / 11.94 / 0.73 | segtails 0.00 / -0.00; seg0 2.09 / 1.00; seg1 3.91 / -0.76; seg2 4.02 / -0.80; seg3 1.92 / 1.29 | 13.27 / 23.00 |
| 4x2k_p139264 | 139264/139264/139264/139264 | 8192 | 109.8 (27.9) | 99.4 | 10.4 | 74608 | 2.26 / 0.22 / 2.04 | segtails 0.21 / -0.21; seg1 0.00 / 0.52; seg2 0.00 / 0.57; seg3 0.00 / 0.58; seg0 0.00 / 0.58 | 13.27 / 23.00 |
| mixed | 61440/139264/307200/546816 | 8192 | 126.3 (34.5) | 117.2 | 9.1 | 64861 | 3.63 / 0.45 / 3.18 | segtails 0.15 / -0.15; seg2 0.23 / 0.71; seg3 0.07 / 1.65; seg0 0.00 / 0.38; seg1 0.00 / 0.59 | 13.28 / 23.01 |
| 1x2k_p139264 | 139264 | 2048 | 88.5 (33.7) | 78.7 | 9.8 | 23141 | 0.52 / 0.16 / 0.36 | segtails 0.16 / -0.16; seg0 0.00 / 0.52 | 12.83 / 21.87 |

DRAM capacity a chip (allocator): 31.83 GiB.
- 4xp50_p139264: peak 13.26 GiB measured (layer 13 embed, 520 samples), 40-layer 22.99 GiB, headroom 8.84 GiB (of the allocator's 31.83), 9.01 GiB of 32
- 4x2k_p1m: peak 13.27 GiB measured (layer 13 embed, 545 samples), 40-layer 23.00 GiB, headroom 8.83 GiB (of the allocator's 31.83), 9.00 GiB of 32
- 4x2k_p139264: peak 13.27 GiB measured (layer 13 embed, 520 samples), 40-layer 23.00 GiB, headroom 8.83 GiB (of the allocator's 31.83), 9.00 GiB of 32
- mixed: peak 13.28 GiB measured (layer 13 embed, 524 samples), 40-layer 23.01 GiB, headroom 8.82 GiB (of the allocator's 31.83), 8.99 GiB of 32
- 1x2k_p139264: peak 12.83 GiB measured (layer 13 embed, 633 samples), 40-layer 21.87 GiB, headroom 9.96 GiB (of the allocator's 31.83), 10.13 GiB of 32
- not loaded here (projected per layer kind): held layers' own weights 8.297 GiB; layers 21-39 (15) as L3: weights 0.354 GiB, slot state 0.001 GiB a layer and slot; layers 24-36 (4) as L20: weights 0.873 GiB, slot state 0.055 GiB a layer and slot

## Per-segment overhead (history 139264)

| chunk | new tokens | wall ms (spread) | device ms | host exposed ms | new tok/s |
|---|---|---|---|---|---|
| single request, 8192 rows (PACKED=off) | 8192 | 102.5 (40.8) | 89.5 | 13.0 | 79922 |
| 1x2k_p139264 | 2048 | 88.5 (33.7) | 78.7 | 9.8 | 23141 |
| 4x2k_p139264 | 8192 | 109.8 (27.9) | 99.4 | 10.4 | 74608 |
| 4xp50_p139264 | 6400 | 109.2 (31.1) | 97.1 | 12.1 | 58608 |

- 1 x 2k vs the single 8192 request: 88.5 vs 102.5 ms wall (-14.0), device 78.7 vs 89.5 (-10.8): a quarter of the tokens costs 86% of the wall
- 4 x 2k vs the single 8192 request (the same 8192 new tokens): +7.3 ms wall, +9.9 ms device; per extra segment +2.4 ms wall, +3.3 ms device
