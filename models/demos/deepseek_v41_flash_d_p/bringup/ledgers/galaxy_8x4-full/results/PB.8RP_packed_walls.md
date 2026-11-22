# Packed prefill walls (test_packed_walls.py)

8192-row chunk, 4 lanes of 2048; slots at one capacity 1048576; layers 0-20 run (DECODER_SKIP=decode). Wall = engine device input -> ids readback -> sync after the last layer (median of the warm runs). Gathers: each at its least chip (the reference = the same chunk's gathers serial). DRAM: the allocator's peak on the worst chip while the chunk runs, every live slot's state and history buffers held; the 40-layer column adds the layers the performance load does not hold (weights + each live slot's state, per layer kind, from a loaded layer of the same kind).

| case | positions | new tokens | wall ms (spread) | device ms | host exposed ms | serving new tok/s | gather ms: reference / exposed / hidden | per segment: exposed / hidden ms | DRAM GiB a chip: loaded layers peak / 40-layer model |
|---|---|---|---|---|---|---|---|---|---|
| 4x2k_p139264 | 139264/139264/139264/139264 | 8192 | 124.2 (16.0) | 104.0 | 20.2 | 65958 | 2.34 / 0.24 / 2.10 | segtails 0.24 / -0.24; seg3 0.00 / 0.54; seg0 0.00 / 0.60; seg1 0.00 / 0.60; seg2 0.00 / 0.60 | 14.58 / 24.32 |
| mixed | 61440/139264/307200/546816 | 8192 | 136.7 (5.6) | 116.8 | 19.9 | 59927 | 3.64 / 0.57 / 3.07 | segtails 0.25 / -0.25; seg2 0.13 / 0.83; seg3 0.15 / 1.52; seg0 0.00 / 0.39; seg1 0.04 / 0.57 | 14.58 / 24.32 |

DRAM capacity a chip (allocator): 31.83 GiB.
- 4x2k_p139264: peak 14.58 GiB measured (layer 13 embed, 564 samples), 40-layer 24.32 GiB, headroom 7.51 GiB (of the allocator's 31.83), 7.68 GiB of 32
- mixed: peak 14.58 GiB measured (layer 13 embed, 586 samples), 40-layer 24.32 GiB, headroom 7.51 GiB (of the allocator's 31.83), 7.68 GiB of 32
- not loaded here (projected per layer kind): held layers' own weights 8.319 GiB; layers 21-39 (15) as L3: weights 0.356 GiB, slot state 0.001 GiB a layer and slot; layers 24-36 (4) as L20: weights 0.873 GiB, slot state 0.055 GiB a layer and slot
