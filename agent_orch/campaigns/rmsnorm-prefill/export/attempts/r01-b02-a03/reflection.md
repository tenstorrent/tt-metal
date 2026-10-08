# r01-b02-a03 result: 1.1721 (ok)

## What happened vs expected
Valid on all shapes. PCC is 0.9999985 and max_abs is 0.022-0.024, identical to r01-b01-a01. Per shape: h3584 1.167
(14.56 µs), h4096 1.159 (15.75), h6144 1.190 (19.73), h7168 1.172 (22.20). This is the new best, ahead of r01-b01-a01
(1.109) and the parent (1.034). It lands at the top of the expected range (~1.13-1.16), above it on the wide shapes.
The two mechanisms stack slightly better than additively: b01-a01 alone gave 1.077 on h7168, and here h7168 gets 1.172.

## Why (profiler evidence, reports/r01-b02-a03, device 1, µs relative to each call's kernel start; zones.py logic)
| zone end | h3584 (call idx 12) | h7168 (call idx 51) |
|---|---|---|
| R_INPUT (deep trid read) | ~2.3-2.5 | ~4.7-5.3 |
| W_PUSH (PRE + stick done) | ~4.1-4.5 | ~7.0-7.6 |
| NCRISC end (gamma batch landed) | ~4.4 | ~8.2-8.6 |
| W_AGWAIT end | ~7.0-7.8 | ~11.1 |
| TRISC end | ~11.0 | ~16.8-17.3 |
| W_DRAIN end | ~12.2-12.9 | ~18.5-20.6 |
- The deep input read moves gamma earlier. The single-barrier gamma batch starts right after the input lands and
  arrives ~0.3-1 µs after PRE ends. In r01-b01-a01 it arrived ~3 µs after PRE. At h3584 the x*gamma pass now hides
  completely under the AG wait.
- At h7168 x*gamma (56 tiles, ~5 µs) starts at ~8.5 and still runs ~2.5 µs past the AG end (11.1). Post-AG compute is
  ~6 µs: the remaining x*gamma plus the single x*rsqrt pass. Wide shapes still have ~2-3 µs of compute that could
  hide under the AG, if gamma arrived earlier or the pre-pass were cheaper.
- The drain tail is unchanged in kind: W_DRAIN ends 1.2-3.5 µs after TRISC on h7168. It is DRAM-write contention among
  20 writers, the same tail every 20-core node shows. It is now the largest single exposed cost after POST.
- PRE still trails the read by ~2.3 µs (pack with L1 accumulation per tile), so the AG start (F_COLLECT end) is gated
  by PRE, not by the read.

## Classification
win (+17.2% geomean vs baseline, +5.7% vs best previous node r01-b01-a01, +13.4% vs parent; all shapes far outside ±1%
noise). It combines two validated mechanisms (deep trid input read + x*gamma under the AG wait).

## What a child of this node should try next
1. **Column split on this lineage (port r01-b03-a02's k | num_tile_cols, 80-worker decomposition).** It is the third
   orthogonal win (1.094 alone) and cuts read, PRE, x*gamma, POST and drain per core ~4x. The b02 plumbing (col_offset,
   row_stride) is already here. Keep one kernel group (equal slices). Watch the drain: b03-a02 found ~200 GB/s
   aggregate write throughput regardless of core count, so de-phase bank access as well (the h4096 outlier).
2. **Get gamma off NCRISC so it lands during the input read.** Have BRISC (idle until the stick push) issue the
   gamma face-row batch on NOC0 at kernel start, or read gamma once on one core and multicast. That hides the
   remaining ~2.5 µs of x*gamma on h6144/h7168. Don't interleave it into the input stream on NCRISC (b01-a02, b04-a02).
3. **Cheaper PRE**: accumulate x^2 in DST across a block and pack once per block instead of an L1-acc pack per tile.
   PRE gates the AG start by ~2.3 µs after the read ends.
4. Drain: rotate each worker's write start column (by tile_row) so the 20 writers don't walk the DRAM banks in
   lockstep, or split the writes across both NoCs (NCRISC is idle after ~8.5 µs).
