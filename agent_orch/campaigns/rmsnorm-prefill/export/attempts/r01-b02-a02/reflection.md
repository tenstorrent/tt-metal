# r01-b02-a02 result: 1.0338 (ok)

## What happened vs expected
Valid on all shapes. PCC 0.9999985 and max_abs 0.021-0.024 are unchanged (the data path is bit-identical).
Per shape: h3584 1.036, h4096 1.021, h6144 1.041, h7168 1.039. Every shape is above the ±1% noise band, but the
gain is ~0.4-1 µs, under the 1-2.5 µs I expected. Only the reader half of the mechanism paid off. The drain half did
nothing measurable.

## Why (profiler evidence, reports/r01-b02-a02, device 1, µs from kernel start; parent r01-b02-a01 in brackets)

| zone (end) | h3584 call 8 | h7168 call 48 |
|---|---|---|
| R_INPUT | 2.5-3.8 (3.7-4.7) | 4.7-6.7 (7.5-8.5) |
| W_PUSH (PRE + stick done) | 4.1-5.7 (5.0-6.1) | 7.0-9.2 (8.7-9.8) |
| F_COLLECT (last stick in) | 5.6 (6.0) | 9.1 (9.7) |
| W_AGWAIT | 7.4-7.8 (8.8-9.1) | 12.4-12.7 (12.7-13.1) |
| TRISC end | 12.9-13.6 (14.2-14.9) | 21.4-22.1 (21.7-22.5) |
| W_DRAIN | 13.7-15.5 (15.2-16.7) | 22.4-25.8 (22.5-26.2) |

1. **The deep trid read works.** At h7168 the read is ~2.3 µs shorter (2.24 MB/chip in ~5.5 µs, ~400 GB/s, which
   matches b03's 60-core figure, so this is now about the aggregate DRAM read limit). The per-core spread grew to
   2 µs: cores now compete for DRAM bandwidth instead of each being latency bound.
2. **PRE compute is now the pre-AG bottleneck.** On every core W_PUSH ends ~2.4 µs after R_INPUT ends (it was ~1.2 µs).
   PRE (mul_tiles x*x HiFi4 + pack_tile with L1 accumulation into one tile, per tile) runs at ~125 ns/tile, about
   7 µs for 56 tiles, so it can't keep up with a 5.5 µs read. The AG starts on the *slowest* worker's stick
   (F_COLLECT), which moved only 0.6 µs on h7168 and 0.4 µs on h3584. The small shapes gained more downstream
   (AG wait end -1.3 µs) because their read+PRE are shorter and the AG was waiting less on the tail core.
3. **The one-flush-per-row drain did nothing.** On h7168 the drain still ends 0.3-3.7 µs after compute, the same tail
   as the parent. So the per-block `async_writes_flushed` was not what serialized the drain. The tail is DRAM-write /
   NoC contention among the 20 writers. The b04 reflection's hypothesis is refuted. The change is harmless; keep it or
   drop it.
4. Cross-chip skew (xchip table) looks like the parent/baseline: F_FABRIC waits of 0.5-6.8 µs on early chips in
   some calls, which inflates the chip-mean metric a bit. No new skew problem, because the core count is unchanged (21).

## Classification
win (small: +3.4% geomean, all shapes > noise). The reader part is real; the drain part is neutral.

## What a child of this node should try next
1. **Stack with x*gamma (r01-b01-a01's compute change, score 1.109).** It is orthogonal: b01 shortens POST, this node
   shortens the read. Port b01's compute `pre_ag_weight` path plus its single-barrier weight read (with the deep
   input read, gamma should land even earlier). That is the most likely new best.
2. **Make PRE cheaper. It now gates the AG start.** Instead of a pack_tile<l1_acc> per input tile (56 L1
   read-modify-write packs of a 4 KB fp32 tile), accumulate x^2 in DST across the block/row (e.g. square into DST, then
   ELWADD with dest reuse or `mul_tiles` + `binary_dest_reuse_tiles` accumulation) and pack once per block. Or check
   whether PRE's math fidelity can drop for x*x, but watch the accuracy gate (pcc_min 0.99999).
3. The drain tail is contention, not flush serialization. Try spreading the writes over both NoCs: the reader
   (NCRISC, NOC1) is idle after the weight read, but that read is still per-block barriered and ends ~14 µs, so batch it
   first. Alternatively, stagger each worker's starting column (rotate write order by tile_row) so the 20 cores don't
   hit the same DRAM banks in lockstep.
