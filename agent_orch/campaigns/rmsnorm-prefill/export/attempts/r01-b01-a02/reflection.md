# r01-b01-a02 result: 0.8994 (ok)

## What happened vs expected
Valid (PCC 0.9999985, max_abs 0.022-0.024, same as parent), but slower than the parent on every shape:
h3584 18.44 us (parent 15.13), h4096 20.04 (16.13), h6144 26.44 (21.25), h7168 29.65 (24.17). The parent scored
1.1089, this node 0.8994, a ~19% regression. I expected the input read to get faster and gamma to land with the
input. Gamma did land early, but the input row finished much LATER.

## Why (best explanation, with profiler evidence)
Zones from `reports/r01-b01-a02` vs `reports/r01-b01-a01`, device 3, measured call idx 12/25/38/51, µs
(min-max over 20 workers). /tmp/r01b01a02/zones.py logic: per run host id, zone start/end relative to the first
kernel/FW marker.
| | a01 h3584 | a02 h3584 | a01 h7168 | a02 h7168 |
|---|---|---|---|---|
| R_INPUT duration (a02 also includes the interleaved gamma reads) | ~3.7 | ~8.0 | ~7.4 | ~15.7 |
| NCRISC end - R_INPUT start (input + gamma) | ~5.6 | ~8.0 | ~10.5 | ~15.8 |
| W_PUSH end (PRE done) - kernel start | ~5.0 | ~8.1 | ~8.6 | ~15.8 |
| post-AG (W_AGWAIT end -> TRISC end) | ~3.9 | ~3.9 | ~8.1 | ~5.7 |
- **The gamma-early part works.** At h7168 the parent's gamma landed at about the same time as the AG finished, so
  x*gamma partly ran after it (post-AG 8.1 us). Here gamma is resident at PRE end, x*gamma hides fully under the
  AG wait, and post-AG drops to 5.7 us (-2.4 us).
- **The input completion got 3-7 us slower.** That more than cancels the post-AG gain. The input + gamma read
  combined takes ~1.45x LONGER than the parent's sequential input-then-gamma (8.0 vs 5.6 µs, 15.8 vs 10.5 µs).
  Best explanation: the 2 x num_tile_cols 32 B gamma face-row reads are issue-bound on NCRISC (the parent measured
  ~3 µs for 112 reads alone, ~27 ns each). Interleaving them between input-block waits put that issue time ON the
  input push path: block b+1 is pushed only after the reader has issued block b's quota of gamma reads. With
  ~1100 deep 2 KB input reads in flight per chip, each read request is also slower to inject (NoC request
  backpressure / DRAM queueing), which inflates the gamma issue cost further. PRE (W_PUSH end) tracks the last
  push, so PRE, the stick push and the whole AG shifted later by the same amount.
- The profile cannot separate the deep-trid input read from the gamma interleave (there is one R_INPUT zone). So
  whether deep input reads alone help is UNTESTED. They may also be hurt by DRAM/NoC contention with 20 cores x 56
  reads in flight.

## Classification
repairable failure (execution: the issue-bound gamma reads were interleaved onto the input push critical path).
The early-gamma half is confirmed useful (post-AG -2.4 µs at h7168). The deep-input half is unmeasured.

## What a child of this node should try next
1. Take the gamma read OFF the NCRISC input path entirely. Have the WRITER (BRISC, NoC0) issue the broadcast gamma
   face-row reads into weight_cb at kernel start. It is idle until the stick push, and runs on the other NoC, so
   gamma lands during the input read without delaying it. Keep the parent's plain per-block input read. That alone
   should get the -2.4 µs post-AG gain at h6144/h7168 with no input penalty (weight_cb producer moves reader ->
   writer; the reader must then skip its weight block).
2. Separately, test the trid deep input read WITHOUT any interleaved work (gamma after, or on BRISC as in 1) to see
   whether depth alone helps or just congests DRAM. Instrument with a zone around the input-only part.
3. Cheaper gamma reads: compute the gamma page's NoC address once per tile and issue both face-row reads with the
   raw address. Or have one core read gamma and multicast it to the 20 workers, so 20x fewer DRAM requests.
