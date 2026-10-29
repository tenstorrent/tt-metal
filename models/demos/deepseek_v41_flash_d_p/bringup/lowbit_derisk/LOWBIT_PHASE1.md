# Window KV, index keys, index query at the reference's width: phase 1 (2x2, 2026-10-29)

Owner task 2026-10-29 (supervision.md 08:00): the window KV to the reference's fp8 width, the index keys and the index
query to its fp4 width, the way the compressed KV went to 4 bits. Phase 1 = forks, codecs, tests, single-card numbers;
no model file touched (tt/attention.py, sparse_attention.py, compressor.py, indexer.py are the Galaxy's until P.10).
Branch dnijemcevic/model_bringup_lowbit. Ids DS41-150.

## 1. What the reference does (reference/ds41_ref.py)
| tensor | reference | today on device |
|---|---|---|
| window KV (`_window_kv`, :769) | e4m3, power-of-two scale per 32 (`act_quant_dequant(kv, 32)`) | bf16 |
| index keys (`_compressor`, :797) | e2m1, power-of-two scale per 32 (`fp4_quant_dequant(k, 32)`) | bf16 (served cache, re-tiled every chunk) |
| index query (`_indexer`, :817) | e2m1, power-of-two scale per 32, before scoring | bf16 |

## 2. Accuracy (CPU, `tools/lowbit_study.py`, 2x2 golden s4096 chunk 1, 256 queries, vs the reference's native values)
Window KV, layers 0 1 2 3 14 20 21 24 (attention output over the window + native compressed rows):

| format | rows rel L2 | attention out rel |
|---|---|---|
| bf16 (today) | 0.0017 | 0.0006-0.0014 |
| reference e4m3 + 2^k per 32 | 0.026 | 0.010-0.023 |
| **bfloat8_b, round to nearest even** | 0.008 | 0.003-0.006 |
| bfloat8_b, half away (the packer alone) | 0.008 | 0.004-0.008 |

Index keys + query (top-512 selection missed vs native, mean / worst row; scores unmasked except causality and, L24, the
candidate pool):

| format (q and k) | L2 | L14 | L20 | L24 |
|---|---|---|---|---|
| bf16 (today) | 0.1 % | 0.1 % | 0.2 % | 0.3 % |
| reference fp4 + 2^k per 32 | 8.3 / 11.9 % | 8.0 / 12.7 % | 11.3 / 17.2 % | 16.6 / 23.2 % |
| **bfloat4_b, round to nearest even** | 7.6 / 11.7 % | 8.3 / 12.3 % | 11.4 / 19.7 % | 15.7 / 21.5 % |
| bfloat4_b truncated (the packer alone) | 12.9 % | 12.8 % | 18.1 % | 25.7 % |
| bfloat8_b | 0.6 % | 0.6 % | 0.8 % | 1.1 % |

Keys alone cost about 2/3 of it (bfp4 keys + bf16 q: 6.8-14.3 %), the query alone 2.6-6.6 %.
So bfloat4_b RNE lands where the reference's own fp4 lands; bfloat8_b for the window is 3-4x closer to native than the
reference's e4m3.

## 3. Byte layouts
- Window, bfloat8_b row record, 576 B (`ttnn.bringup.ckv_bfp4(dtype=ttnn.bfloat8_b)`): `[32 shared exponents (one a
  16 values) | the same 32 | 512 sign-magnitude bytes]` (the bf16 row is 1024 B). As the 320 B bfloat4_b compressed
  record: the second exponent copy makes any gathered row a row-major bfloat8_b block for the unpacker.
- Index keys / query: bfloat4_b tiles (576 B a 32x32 tile: 64 exponents, 512 nibble bytes; bf16 2048 B), one exponent
  per 16 values of a tile row = per 16 of a head's 128 dims (the reference: per 32).

## 4. Device work (forks, all opt-in, defaults unchanged; f5d169873aa + this commit)
- `ttnn.bringup.ckv_bfp4(dtype=bfloat8_b)`: the window records; `ttnn.bringup.bfp_round`: bf16 TILE -> bfloat4_b /
  bfloat8_b TILE, new or in place at a tile row (the key table). Round to nearest even on the SFPU (the packer
  truncates bfp4 and rounds bfp8 half away); now |x| + 2^23 LSB - 2^23 LSB (the SFPU adder rounds to even), which also
  makes the 4-bit compressed-KV write ~30 % faster. Byte-exact vs the host rule.
- `indexer_score`: q / k bfloat4_b. 2x2 ring cases at the DS41 geometry pass.
- `sparse_sdpa`: kv_window as bfloat8_b records (the compressed layers), kv_format ROW_BFP8 (the window-only layers
  0-1). Bit-exact vs the bf16 window holding the same values; deterministic.

## 5. Kernel time (one p150b, device profiler, median of 5; 2x2 per-chip shapes, 8x4 where noted)
| op | bf16 (today) | low-bit | delta |
|---|---|---|---|
| index score, ratio 2, T 27648, 512 q | 787 us | 774 us (bfp4 q / k) | -13 us |
| index score, ratio 1, T 55296, 512 q | 1519 us | 1507 us | -12 us |
| index query round [32, 512, 128] | - | 16-28 us | + |
| index key round, a chunk's 512 / 1024 keys into the table | - | 2 us | + |
| today's per-chunk re-tiling of the whole key slab (13824 / 27648 rows) | 32 / 65 us | gone with a persistent bfloat4_b key table | - (grows with context) |
| window write, 2176 rows | 13 us (untilize) | 28 us (bfloat8_b records) | +15 us |
| sparse_sdpa, compressed layers, 160 q (8x4) / 512 q (2x2) | 351 / 1250 us | 362 / 1286 us | +3 % |
| sparse_sdpa, window-only layers, 160 q / 512 q | 133 / 494 us | 138 / 567 us (ROW_BFP8) | +4 / +15 % |
| e4m3 window, lower bound (plain fp8 rows, no scale multiply) | 133 / 494 us | 119 / 484 us | -10 / -2 % |
| sparse_sdpa, compressed layers, bfloat8_b window blocks (consecutive window ids), 160 q / 512 q | 352 / 1242 us | 339 / 1188 us | -4 / -4 % |
| sparse_sdpa, window-only layers, bfloat8_b blocks, 160 q / 512 q | 132 / 487 us | 110 / 404 us | -16 / -17 % |
| window write as blocks, 2176 rows | 13 us | 28 us | +15 us |

Index path per index layer at 55k on the 2x2: about -20 us (ratio 2) and -50 us (ratio 1), more at longer context.
Window as 576 B records: the cost was the second NoC read a row (the exponents; switching it off took the window-only
layers 567 -> 461 us), not the conversion (batched: no change), the ring depth or the address lookups. As blocks of 32
consecutive rows a page (ckv_bfp4 blocks=True; two reads a block) the 8-bit window is faster than bf16: per layer at
512 q, read + write, compressed layers 1255 -> 1216 us (-3 %), window-only 500 -> 432 us (-14 %). Part of that is the
bulk read itself, which a bf16 window in blocks would also get (not built). The 4-bit compressed rows pay the same
per-row exponent read (switching it off: compressed layers 1229 -> 986 us); their gather is random, so that needs the
exponents beside the values in the unpacker's read (not built). Diagnosis: the same exponent read from local L1
(same transaction count) gives 970 us, so the cost is DRAM serving 32 B reads. Fixed (default, bit-identical): one DRAM
read of the whole 320 B record into a scratch, then two local copies: compressed layers 512 q 1242 -> 978 us (-21 %),
160 q 352 -> 293 us (-17 %).
Window as records (first measurement): +15 us write + 3 % (compressed layers) / +4-15 % (window-only layers) read, no memory gained: the
window table is per-chunk scratch (2 MB); the served window cache (0_window_kv, the contract) stays bf16 unless the
owner / server change it.

## 6. Owner choices (phase 2 waits for them and for P.10)
1. Index keys + query: bfloat4_b (recommended: the reference's own selection error, slightly faster, 3.6x less ring
   traffic and key memory); or bfloat8_b (near bf16 selection); or bf16 (today, above the reference).
2. Window KV:
   - bfloat8_b blocks (implemented; -3 % / -14 % a layer vs bf16, above) or rows (F208 "the reference's width in the device's own format"): 3-4x closer to native than
     the reference, costs +3 % / +15 us a layer, gains no memory;
   - e4m3 + 2^k scale per 32 (not built): the reference's numbers exactly, cost at least the fp8 lower bound plus 16
     broadcast multiplies per 32 keys in compute, and an SFPU RNE (the packer truncates e4m3);
   - bf16 (today, above the reference, fastest).
3. Phase 2 wiring (after P.10): INDEX_K_FORMAT=bfp4 as a persistent bfloat4_b key table per KV source written by
   `bfp_round(out=, row0=)` (served bf16 cache unchanged, so the contract stays), ring_indexer_score_dsa on it;
   INDEX_Q_FORMAT=bfp4 as `bfp_round` after the query's rope; WINDOW_KV_FORMAT=bfp8_rows as `ckv_bfp4(dtype=bfloat8_b)`
   of [carry | chunk] and kv_window / ROW_BFP8. Framework: the noise floor's input dtype for the index keys / query
   needs the power-of-two-scale fp4 cell (F223 has the e4m3-scale one): F224.

## 7. What one card / the 2x2 cannot show
The ring's key traffic (SP 8 on the 8x4), memory at 1M tokens, host time per call (the Python descriptor ops rebuild
their programs each call, DS41-135), and end-to-end accuracy (phase 2: component / swap tests and the ladder).
