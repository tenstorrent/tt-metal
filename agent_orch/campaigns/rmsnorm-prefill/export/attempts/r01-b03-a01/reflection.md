# r01-b03-a01 result: 0.9867 (ok)

Per shape: h3584 0.909, h4096 0.956, h6144 1.076, h7168 1.014. PCC is unchanged (0.9999985) and max_abs is the same as baseline.
The protocol works (no hang, no accuracy loss) on 60 workers + 1 forwarder. us_min/us_max
per shape spread far wider than baseline (h3584: 13.3-27.6 µs vs 14.9-18.3).

## What happened vs expected
The local work shrank as planned. Device 0 timeline (µs from kernel start, from
reports/r01-b03-a01 profile_log_device.csv, compared with baseline_1):
- R_INPUT end: h3584 2.4-4.0 (was 3.8-4.9); h7168 4.5-6.3 (was 7.3-9.0). Above 3x,
  because 60 cores reading 2.2 MB start to hit aggregate DRAM bandwidth on the widest shape.
- PRE + leader peer combine (W_PUSH end): h3584 3.2-5.5 (was 5.1-6.1); h7168 5.4-8.0 (was 8.4-10.1).
  The follower -> leader hop costs about 1-2 µs on the leader. Followers finish ~2 µs before leaders.
- POST + drain (W_DRAIN): h3584 ~4.3 µs (was ~6.5); h7168 ~7.3 µs (was ~11.2).
- **F_FABRIC grew** from ~2.4 µs to 7.6-14 µs on SOME chips. That wiped out the gain in the
  chip-mean metric.

## Why (best explanation, with profiler evidence)
The fabric wait is cross-chip launch skew, not a slower gather. Per-call, per-chip
F_FABRIC duration and kernel end (/tmp/xchip.py over both profile CSVs):

| call idx (h3584 measured) | d0 fab / end | d1 | d2 | d3 |
|---|---|---|---|---|
| attempt #12 | 12.9 / 25.8 | 9.9 / 23.2 | **0.5 / 13.3** | 4.6 / 17.5 |
| baseline #12 | 2.4 / 17.1 | 2.0 / 16.4 | 2.2 / 16.7 | 2.4 / 17.0 |
| attempt #56 (h7168) | 12.3 / 32.5 | 9.3 / 29.4 | **0.5 / 20.1** | 3.8 / 23.6 |
| baseline #56 | 3.0 / 26.5 | 2.9 / 26.3 | 1.7 / 25.4 | 1.7 / 25.3 |

The last-launched chip (d2) never waits. Its kernel is 13.3 µs (h3584) and 20.1 µs (h7168) vs
16.4-17.5 / 25.3-26.5 at baseline, so the op's critical path did get **~3-5 µs faster**.
The other chips launch earlier and sit in F_FABRIC waiting for d2's packet. The skew also *grows*
over consecutive calls (attempt d0 fab: 7.9 -> 9.5 -> 12.9 µs at calls 4/8/12). That fits the op
now being shorter than the per-call dispatch cadence across the 4 chips. Once the device
outruns host/dispatch, each chip starts when its program is dispatched. Early chips burn the
difference waiting in the AG, and the metric (mean over chips of kernel duration) counts that wait.
The baseline at ~17 µs was slow enough that launches stayed back-to-back and the waits balanced at ~2 µs.
This attempt also made dispatch heavier: 61 cores instead of 21, 2 kernel groups (h3584/h4096/h7168 have
uneven slices 10/9/9, 11/11/10, 19/19/18) so 6 kernel binaries instead of 3, and ~12-17 per-core writer
RT words. That likely slowed the per-call dispatch and made the skew worse. h6144 (48 = 3x16, a single
kernel group) is the only shape whose waits stayed balanced (calls 36-44: fab 1.2-3.9 µs), and it gained 7.6%.

## Classification
neutral overall (0.987, inside/near noise, mixed per shape). The mechanism is a real
critical-path win (~3-5 µs/call on the un-skewed chip) masked by cross-chip launch skew /
dispatch cost. It is not a flawed idea, but the metric punishes it unless dispatch overhead is kept down.

## What a child of this node should try next
1. Keep the column split but make dispatch cheap. Pick col_splits so slices are equal-width,
   giving one kernel group (k | num_tile_cols, or pad the last slice instead of a second
   group). Move per-core writer args into common args where possible: col_start/peer_idx can
   be derived from a single per-core index, and follower coords from a leader-adjacent layout.
   Check whether k=2 (41 cores) beats k=3 once skew is accounted for.
2. Whatever the mechanism, judge it with the per-chip table above (/tmp/xchip.py logic: per call
   index, per device, F_FABRIC duration and kernel end). The min-over-chips kernel end is the
   true critical path; the mean over chips includes the skew wait.
3. Orthogonal: reduce the per-call fixed cost the skew exposes (fewer cores/kernels/RT args
   for the baseline layout too). Or shrink the forwarder's go/collect latency so a late chip
   costs less.
4. The leader-combine hop (~1-2 µs) could be removed by having followers write their partial
   stick straight into the leader's stats_local tile path earlier. Or the leader could start
   its forwarder push only on its own data plus an asynchronous second packet, but the 34-stick
   packet limit (4352 B fabric payload) is the binding constraint.
