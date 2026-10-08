# r05-b03-a02: four-wave row pipeline on 80 quarter-row workers (k=4 column split, 4 AG waves of 20 cores, 8-bit wave fields), with tile-row-0 stick slots so each worker's post-go gather stays one read per chip

## Motivation
Both 2-wave nodes (this parent r05-b03-a01 at 1.5174, and r05-b01-a01 at 1.5696) fit one simple model:

    kernel ≈ (first wave's read) + C + (total drain)

Here C is one wave's chain from read end to drain start (PRE tail, push, fabric AG including chip skew, go, gather,
combine, first POST block). It is ~5.2-5.5 µs. The total drain runs at the DRAM aggregate rate once it starts.
- r05-b01-a01, h7168: A read end 3.59 + C 5.46 + drain window 6.54 (A drain start 9.05 -> B drain end 15.59) = 15.6,
  measured 15.74.
- h3584: 2.08 + 5.16 + 3.67 = 10.9, measured 11.05.

Two consequences:
- With N equal waves, the first-wave read term is R/N. Going from 2 to 4 waves cuts R/2 to R/4: about -1.5 µs at
  h7168 and -0.8 µs at h3584.
- With 4 waves at h7168, wave 0's drain starts at ~R/4 + C ≈ 7.5 µs, after the whole read has landed (R ≈ 6.3). So
  the waves' reads and drains still don't fight for DRAM, and the drains run back to back at the aggregate rate.

A wave needs ~20 cores to read at the aggregate DRAM rate: r03-b04-a02's 10-core waves lost 5% on per-core caps,
while both k=2 nodes showed that 20-core waves keep the full rate. So 4 waves need 4 x 20 = 80 workers: a k=4 column
split (quarter rows: 7/8/12/14 tiles per core) plus the forwarder. That is 81 cores, the size r01-b03-a02 already ran
in one kernel group.

The parent's post-go gather is already too slow (go -> C_POST start 1.5/1.9 µs vs 1.25 in its own parent), because
it reads every partial as 2 x 64 B face rows. With k=4 that would become 32 reads. r05-b01-a01 showed the fix: lay
each slot out as fp32 tile row 0 (`L(j) = (j/16)*2048 + (j%16)*64`, face_01 row 0 at +1024). Then a row's
consecutive slots are contiguous, so one read per chip brings all of them. With k=4 the 4 quarter slots of a row are
256 B contiguous in each face row, so one 1280 B read per chip gives 4 reads total, the same count as b01's 2-wave
node.

## Mechanism
Everything stays inside the existing wave plumbing of the parent, generalized from 2 to N waves (N = col_split = 4):

1. **Sizing (`compute_sizing`).** Pick 4 waves with col_split 4 when rows % 4 == 0, cols % 4 == 0, 4*rows+1 cores
   fit, and the per-wave stick region fits one packet. Otherwise fall back to the parent's 2 waves with col_split 2.
   One scratch page per wave (4 rounds). The page stays sticks_per_packet*128 = 4352 B. A wave's region is
   `L(slots-1) + 1088` = 3328 B for 20 slots.
2. **Decomposition (factory).** This is the parent's formula. Worker w belongs to wave w / 20. Its slot j = w % 20
   owns row `wave*5 + j/4` and column quarter `j%4`. Kernels get num_tile_cols = W/4. Compute gets
   stats_tiles_cols = ring*4 = 16, so `1/(num_tile_cols*32*16)` is still 1/H_full.
3. **Wave gate chain (reader).** wave_role becomes a bitmask: bit0 = signal the partner (worker + 20), bit1 = wait
   for own start_sem. A middle wave waits, then signals. The signal fires once at least half of the core's tiles have
   landed. That is block 1 for 14 or 12 tiles, and block 0 for 7 or 8 tiles; the parent's "2 blocks left" rule would
   start every wave almost at once on 2-block rows.
4. **Stick push (writer).** The stick goes to `packet_buf[wave] + L(j)` (face_00 row 0) and `+1024` (face_01 row 0).
   The arrival inc is `1 << (8*wave)`. The packet CB is `max(2, num_waves)` packets deep, so each wave has its own
   buffer.
5. **Gather (writer).** After go, one `1024 + 4*64` = 1280 B read per chip, from page (d, wave) at offset
   `L(4*(j/4))`, into a 64 B-page gathered CB at d*1280. That is 4 reads instead of 32.
6. **Combine (compute).** Partial t (device t/4, quarter t%4) is the fp32 tile view at 64 B page `(t/4)*20 + t%4`
   (r05-b01-a01's validated overlapping-view trick). The combine is the existing pairwise ELWADD loop, now with 8 adds.
   Everything after that is unchanged: HiFi4, no approx mode.
7. **Forwarder (`dit_rmsnorm_wave_forwarder.cpp`).** N ≤ 4 waves in 8-bit fields of arrival and out_ready. Each wave
   sends its 3328 B region (a new CT arg) and releases its own 20 go-sems. The poll loop is unchanged.

Files: factory + sizing type, reader, worker writer, compute (gathered-tile indexing only), wave forwarder.

## Why this is not a repeat
- r05-b03-a01 (parent) and r05-b01-a01 run 2 waves of 40 half-row workers. This node runs 4 waves of 80 quarter-row
  workers, which halves the exposed first-wave read. Both nodes' reflections name "more waves" (#2) as the next step.
  The parent's says to fix the gather first, so this node does both: without the slot layout, k=4 would need
  32 reads.
- r01-b03-a01/a02/a03 (k=3/4 split, all cores at once) lost on a leader-combine hop and on 80 concurrent drainers.
  Here there is no leader hop (partials meet in the combine), and only ~20 cores read or drain at any moment.
- r03-b04-a02 (waves of 10 cores) was per-core capped. Every wave here has 20 cores.

## Expected effect and risk
Model `R/4 + C + W` (C ~5.4 plus ~0.1 for the 16-partial combine). h7168 ≈ 2.0 + 5.5 + 6.5 ≈ 14.0 µs (vs 15.7-16.0).
h3584 ≈ 1.2 + 5.3 + 3.7 ≈ 10.2 (vs 11.0-11.5). Score ~1.65-1.7 if the model holds.

Risks:
- **Per-wave fixed costs.** The PRE tail is ~0.8 µs per core however short the row, and each wave pays an AG. If C
  grows (e.g. cross-chip skew compounds over 4 sends), the gain shrinks. Check with per-wave zones.
- **Dispatch / launch skew** with 81 cores (more RT args). The narrow shapes are already host-bound (b01 #3).
- **Hang** from the 8-bit fields, the wave chain, or the 4-deep packet CB. **Accuracy** from slot offsets or the
  gathered page indexing would show up at once as PCC ≪ 0.99999.
- Drains of consecutive waves overlap and become aggregate-bound. That is expected; it is the W term.
