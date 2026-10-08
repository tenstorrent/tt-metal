# r01-b01-a03 result: 1.1868 (ok)

## What happened vs expected
Valid on all shapes. PCC is 0.9999985 and max_abs 0.022-0.024, bit-identical to the parent (same data, same compute).
Per shape: h3584 14.51 us (1.171), h4096 15.52 (1.176), h6144 19.47 (1.206), h7168 21.80 (1.194). That is the new
best: +7.0% over r01-b01-a01 (1.109) and +32% over the parent r01-b01-a02 (0.899). It landed at the top of the
predicted ~1.15-1.2 range, and on all four shapes, not just the wide ones.

## Why (profiler evidence)
Zone timelines come from `reports/<node>/.logs/profile_log_device.csv`, device 3, measured runs 13315 (h3584) and
53251 (h7168), using the parent's /tmp/r01b01a02/zones.py logic. Times are µs relative to kernel start, min-max over
the 20 workers.

| zone end | a03 h3584 | parent a02 h3584 | a03 h7168 | parent a02 h7168 | b02-a02 h7168 (deep read, gamma on NCRISC after) |
|---|---|---|---|---|---|
| R_INPUT | 2.3-3.6 | 8.1-9.2 | 5.0-6.2 | 15.8-17.1 | 5.0-6.4 |
| W_PUSH (PRE + stick) | 4.1-5.5 | 8.2-9.4 | 7.1-9.9 | 15.9-17.2 | 7.2-8.7 |
| W_AGWAIT | 7.9-8.3 | 12.6-13.0 | 12.6-12.9 | 19.5-19.9 | 12.5-12.9 |
| TRISC end | 11.8-12.2 | 16.5-16.9 | 18.2-18.7 | 25.2-25.7 | 21.8-22.1 |
| W_DRAIN | 12.5-14.5 | 17.3-19.0 | 19.5-22.4 | 26.4-29.5 | 22.6-26.3 |

- **The input read is no longer held hostage.** R_INPUT is back to the deep-trid speed b02-a02 measured
  (h7168 ~5-6 us, ~400 GB/s/chip). So the gamma reads on BRISC/NoC0, de-phased across workers, do NOT measurably slow
  the NCRISC input stream. The parent's slowdown was NCRISC issue/queue serialization, not the DRAM itself.
- **Gamma is resident by PRE end.** post-AG (AG wait end -> TRISC end) is ~5.7 us at h7168 and ~3.9 us at h3584, the
  same as the parent, whose gamma was early. So x*gamma hides fully under the AG. Versus b02-a02 (deep read, two-pass
  POST), TRISC ends ~3.5 us earlier at h7168.
- The weight push only happens after the stick push, so it costs nothing on the AG start.

## Classification
win (repair of r01-b01-a02: the bug was gamma traffic on the NCRISC input path; +18.7% geomean vs baseline, all
shapes far above the ±1% noise).

## What a child of this node should try next
The remaining critical path, h7168, µs rel. kernel start: input 0->5-6, PRE tail -> stick 7-10 (spread 2.8 us; the AG
waits on the slowest worker), AG ~2.7 -> 12.7, single POST pass ~5.7 -> 18.5, drain tail -> 19.5-22.4.
1. **Stack the column split (r01-b03-a02, 1.094: k=4, 80 workers, equal slices, leader FPU combine) onto this
   node.** Its per-core read, PRE, x*gamma and POST all shrink 4x. The gamma read per core also shrinks to the
   slice, and the BRISC gamma read should start at the slice offset. This is the biggest structural lever left.
   Keep one kernel group (k | num_tile_cols).
2. **PRE now gates the AG start.** W_PUSH ends 2-3.7 us after R_INPUT ends, at ~125 ns/tile (mul x*x HiFi4 +
   pack with L1 accumulation per tile). Accumulate x^2 in DST across a block and pack once per block, or use a
   lower fidelity for x*x if the 0.99999 PCC gate allows. A uniform PRE also shrinks the W_PUSH spread.
3. **The drain tail (1-4 us after compute) is still DRAM-write contention** (b02-a02 showed flush batching is
   neutral). Try rotating each worker's output write order by my_slot, the same de-phasing trick used here for
   gamma (and for the input read: 56/32/48 cols are 0 mod 8 banks, so all rows march across banks in lockstep).
   Or split the drain across NCRISC/NoC1, which is idle after R_INPUT ends.
4. POST x*rsqrt at ~100 ns/tile is HiFi4 fp32. Check whether a lower math fidelity passes the accuracy gate.
