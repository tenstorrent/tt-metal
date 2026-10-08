# r05-b01-a01 result: 1.5696 (ok)

## What happened vs expected
Valid on every shape, and it ran first time: no hang, no JIT error. PCC is 0.9999985 and max_abs 0.0205-0.0240, the
root's values. So these all check out on HW:
- the 40-worker column split;
- the two-wave forwarder fork (16-bit per-wave fields);
- the tile-row-0 slot layout with one 1152 B pair read per chip;
- the 64 B-page gathered CB with 8-partial ELWADD;
- the ragged 14-tile half rows (h3584).

Score 1.5696 vs root r04-b04-a02 1.3997 (**+12.1%**). This is the new campaign best by a wide margin. Every shape is
>10x the noise band. µs, chip mean:

| shape | root | this | change |
|---|---|---|---|
| h3584 | 11.96 | 11.19 | -6.4% |
| h4096 | 13.62 | 12.08 | -11.3% |
| h6144 | 16.91 | 14.59 | -13.8% |
| h7168 | 17.93 | 15.84 | -11.6% |

I predicted 9.5/10.5/12.8/13.7 µs if ideal, and said I'd take half of the gain. I got a bit over half on the wide
shapes and less on h3584. The reasons are below.

## Why (profiler evidence)
`analysis/waves.py` and `analysis/fwd.py` (run each as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/r05-b01-a01`). Outputs:
`waves_out.txt` and `fwd_out.txt`. Values are medians over measured calls x 4 chips of the per-wave max, in µs from
the chip's first worker start.

| shape | A read end | B read start/end | A push | B push | A go | B go | A drain s->e | B drain s->e | end |
|---|---|---|---|---|---|---|---|---|---|
| h3584 | 2.08 | 1.61 / 3.43 | 3.13 | 4.46 | 6.73 | 7.86 | 7.24-9.71 | 8.43-10.91 | 11.05 |
| h4096 | 2.33 | 1.73 / 3.88 | 3.43 | 4.93 | 6.88 | 8.43 | 7.37-10.36 | 9.03-11.95 | 12.09 |
| h6144 | 3.15 | 2.65 / 5.48 | 4.33 | 6.59 | 8.29 | 10.13 | 8.80-12.95 | 10.72-14.43 | 14.57 |
| h7168 | 3.59 | 2.87 / 6.29 | 4.51 | 7.24 | 8.53 | 10.66 | 9.05-14.21 | 11.31-15.59 | 15.74 |

(Root, h7168: R_INPUT end max 6.33, push end max 7.75, AG-wait end max 10.84, drain end max 17.58.)
- **Total read time is unchanged.** B's last read lands at 6.29 µs vs the root's 6.33 at h7168. 20 cores per wave
  read at the root's aggregate rate, so a 20-core wave is not per-core capped (r03-b04-a02's 10-core waves were).
- **Wave A's stat is ready 3.2 µs earlier** at h7168 (push 4.51 vs 7.75). Half-row reads plus half-row HiFi4 PRE do
  that. A's AG overlaps B's read, as designed (F_SEND#0 at 5.3, B's read runs to 6.3).
- **go -> drain start fell to 0.50-0.57 µs** (root ~0.6) with 4 pair reads instead of 8 face-row reads.
- **What's left on the table:**
  1. The two waves are too close together. B's read starts 0.5-0.7 µs *before* A's ends (lead = 2 blocks), and both
     share DRAM. So B's push is only ~1.0-2.7 µs behind A's, B's go comes 1.1-2.1 µs after A's (F_GO#1 - F_GO#0), and
     **the two drains overlap**: B starts draining 1.3-2.9 µs before A finishes. Overlapped drains are
     aggregate-bound, so per-core drain time is 2.25-3.8 µs for 14-28 tiles (~135-160 ns/tile vs the root's ~95).
     The write window, A drain start -> B drain end, is 3.7/4.6/5.6/6.5 µs. That is ~350 GB/s, and the drain is
     back to being the tail.
  2. DRAM is still idle from B's read end to A's drain start: 3.8/3.5/3.3/2.8 µs. A's chain from read end to drain
     start (~5.2-5.5 µs: PRE tail, push, fabric AG ~3 µs incl. skew, go, combine) is longer than one wave's read
     (~1.4-2.9 µs). So only part of the gap is hidden.
  3. Host launch skew grew on the short shapes. Per-device kernel time at h3584 is 13.06 / 11.80 / 10.25 / 9.65 µs
     (dev0..3), at h4096 13.59 / 12.66 / 11.46 / 10.59, but at h7168 only 16.02-15.71. The device op is now as short
     as the host enqueue cadence (~13.5 µs/call at h3584, d0..d3 enqueued ~1 µs apart). So dev0 waits ~3 µs in its
     AG for dev3. The min-chip kernel (9.65 at h3584) is close to my ideal model, but the chip-mean metric can't see
     it. 41 cores vs 21 also makes dispatch a bit heavier.

## Classification
win (+12.1% over the best node; new campaign best). This confirms r03-b04-a02 #4: waves pay once each wave has
enough cores to keep the DRAM-bound phases at full aggregate rate.

## What a child of this node should try next
1. **Separate the waves more** (cheap, kernel-only: `kWaveSignalLeadBlocks` in the reader). With lead 0, wave B starts
   at A's last block push, A's read gets the whole DRAM, and A's chain moves earlier. B's drain has 1.3-2.9 µs of
   overlap to give back, so a later B is free until its drain start reaches A's drain end. Alternative: gate B's read
   start on A's *push* instead. Watch `waves.py`: target A drain end ≈ B drain start.
2. **Three waves** (k=2 split, rows in thirds: 13/13/14 rows -> 26-28 workers per wave, or k=3 with 60 workers).
   Hidden gap = A chain − (waves − 1) × wave read. With three waves, ~2 more µs of the 2.8-3.8 µs idle-DRAM gap is
   covered. Needs a 3-field forwarder (10-bit fields fit) and a 3-region page. Try only after (1).
3. **The drain is aggregate-bound again** (~350 GB/s over A+B). Revisit the per-core NoC0 share (r02-b02-a01 #1)
   now that 40 cores drain, and stagger the drain starts within a wave.
4. **Cut each wave's fixed chain** (each ~0.1-0.4 µs, they now count twice):
   - the HiFi4 PRE backlog. Matmul-accumulate x*x^T (r02-b02-a02) removes it, but needs a cheap diagonal
     extraction;
   - the 0.5 µs post-go read + push;
   - the combine.
5. Launch skew on h3584/h4096 is host-side (op shorter than the enqueue cadence). Don't chase it in the kernel. Judge
   kernel changes by the min-chip duration too.
6. Production caveats carried over: the posted drain needs a per-bank fence (r04-b04-a01 #2). The 64 B-page
   gathered-CB trick (overlapping tile views) relies on the eltwise unpack addressing tiles by fifo page size.
