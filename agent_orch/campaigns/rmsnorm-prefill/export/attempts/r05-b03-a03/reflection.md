# r05-b03-a03 result: 1.4881 (ok)

## What happened vs expected
Valid on every shape on the first run. PCC 0.9999985 and max_abs 0.0204-0.0237 are the parent's values, as they should
be: the change is writer scheduling only. µs, chip mean:

| shape | parent r05-b03-a02 (4 waves, gated) | this (4 waves, un-gated) | r05-b03-a01 (2 waves) | r05-b01-a01 (2 waves, best) |
|---|---|---|---|---|
| h3584 | 13.05 | **12.28** (-5.9%) | 11.47 | 11.19 |
| h4096 | 13.59 | **12.62** (-7.1%) | 12.76 | 12.08 |
| h6144 | 15.52 | **15.12** (-2.6%) | 15.25 | 14.59 |
| h7168 | 16.84 | **16.50** (-2.0%) | 16.02 | 15.84 |

Score 1.4881 vs parent 1.4219 (+4.7%, every shape outside the ±1% noise). So the repair is a real win over its parent.
But 4 waves still don't beat 2 waves: it is roughly level with r05-b03-a01 on h4096/h6144, worse on h3584/h7168, and
worse than r05-b01-a01 on every shape. I hoped for 1-2 µs on the wide shapes. I got 0.3-0.4.

## Why (profiler evidence)
Scripts: the parent's `analysis/push.py` and `wavesN.py`, run with `$DREAM_HOME/rmsnorm-prefill/reports/r05-b03-a03 4`.
Per-device forwarder zones come from an ad-hoc script that lists each F_SEND#k / F_GO#k start and end per device,
relative to that chip's kernel start.

1. **The un-gate works exactly as designed.** W_PUSH end max per wave now tracks R_INPUT end + 0.6-1.0 µs, not
   W_GAMMA end:

   | shape | push end per wave (W0..W3), this | parent | read end per wave |
   |---|---|---|---|
   | h3584 | 2.56 / 3.19 / 4.24 / 5.07 | 4.50 / 4.46 / 4.56 / 5.08 | 1.35 / 2.46 / 3.52 / 4.27 |
   | h7168 | 2.96 / 4.58 / 6.06 / 7.41 | 2.96 / 6.55 / 6.82 / 7.41 | 2.07 / 3.54 / 5.23 / 6.39 |

   The forwarder sends are now staggered: F_SEND#0..3 end at 3.47 / 3.95 / 4.60 / 5.48 µs at h3584 (parent 4.89..5.64)
   and 3.93 / 5.13 / 6.40 / 7.81 µs at h7168.
2. **But each wave's all-gather is ~4.3 µs from send to out_ready while the read traffic is running, so go#0 barely
   moved.** At h7168 the chips are in sync: every device sends wave 0 at 3.5-3.7 µs after its own kernel start.
   Even so, every device's F_GO#0 starts at 7.9-8.3 µs. r05-b04-a01 measured a synced fabric round at ~2.3 µs. So
   the extra ~2 µs is the AG slowing down while the other waves' input reads (and later the drains) load the NoC/DRAM.
   - The stats scratch is L1-interleaved over the worker grid, so the EDM's landing writes cross the same busy NoC.
   - The EDM's reads of the forwarder's packet do too.
   - The 2-wave nodes show the same thing: A push 4.51 -> A go 8.53 at h7168 in r05-b01-a01.

   So go#0 lands at ~8.8 µs (wavesN "go max") no matter when wave 0 pushes. That is about the same as the 2-wave
   node's A go (8.53).
3. **After go#0 the forwarder's release chain sets the pace.** Each F_GO is 0.55 µs (async_write_barrier + 20 serial
   go incs). The gos come out ~0.6-1.0 µs apart: h7168 8.82 / 9.50 / 10.20 / 10.96, h3584 7.19 / 8.19 / 8.78 / 9.38.
   The last wave's go (10.96 at h7168) is later than the 2-wave node's B go (10.66).
   - The drains then overlap and turn aggregate-bound: per-core drain 3.0-3.8 µs for 14 tiles at h7168.
   - The write window (first drain start -> last drain end) is 9.30 -> 16.30 = 7.0 µs, vs 6.5 µs for r05-b01-a01's
     2 waves.
   - DRAM is still idle from the last read end (6.39) to the first drain start (9.30).
4. **Narrow shapes are host-launch-skew bound.** At h3584 the per-device kernel medians are 13.94 / 12.62 / 11.02 /
   11.65 µs (dev0..3). F_GO#0 starts at 8.76 on dev0 vs 4.90-5.82 on dev2/dev3. dev0 waits for the others. The
   min-chip kernel (11.02) is about the 2-wave node's chip mean. The chip-mean metric charges 4 waves for the ~1 µs
   longer read chain (start-sem chain: last read end 4.27 vs 3.43).
5. go -> C_POST is still 1.36-1.82 µs per wave: the 1280 B gather read plus the 16-partial combine plus the fixed
   ~0.36. The parent measured this.

## Classification
win over the parent (repair: the gamma-trid barrier no longer gates the stick push; +4.7%, every shape outside noise).
As evidence about waves, it is now a real test, and it says 4 waves ≤ 2 waves on this hardware. A wave's AG under
load is ~4 µs (not ~2.3). So no wave can go before ~8.5 µs, and the extra waves only add serial go releases and
overlapping drains. Don't add more waves.

## What a child of this node should try next
1. **Port this exact 5-line writer fix to the best node r05-b01-a01 (2 waves, 1.5696).** That node's wave A push
   (4.51 vs read end 3.59 at h7168) and its wave B push (on W_GAMMA end) are gated the same way.
   - Expect ~0.2-0.4 µs on the shapes where B's push sits on W_GAMMA end.
   - Expect little on A's go, because of #2.
   - This is the cheapest likely new best.
2. **Shorten the per-wave AG under load (biggest lever for any wave design).** Send -> out_ready is ~4.3 µs here vs
   ~2.3 µs synced and idle.
   - Land the gathered pages in the forwarder's own L1 (height-sharded on the forwarder, as r04-b02-a01 did), not
     L1-interleaved across the busy worker grid. The EDM's landing writes then hit one core near the ERISC instead of
     crossing the grid.
   - Or move the forwarder core next to the active ethernet core and off the input-read rows.
   - Measure F_SEND#0 end -> F_GO#0 start per device: target ≤ 2.5 µs.
3. **Cheaper go release.** Use flush-then-inc instead of `noc.async_write_barrier()` before the incs. The local
   page landed via fabric, so the barrier only covers the forwarder's own writes. Or multicast the go to the wave's
   worker rectangle. Each F_GO is 0.55 µs, and they serialize: 3 × 0.55 on the last wave's path.
4. If waves are revisited, use 2 waves with fewer, faster stages, not 4. The 4-wave plumbing is correct (8-bit fields,
   start-sem chain, 1280 B row read, 16-partial combine), but its chain cost (+0.8 µs read end on narrow shapes, 4 go
   releases, 16-partial combine) isn't repaid while the AG is ~4 µs.
5. Production caveat carried over: the posted drain needs a per-bank fence (r04-b04-a01 #2).
