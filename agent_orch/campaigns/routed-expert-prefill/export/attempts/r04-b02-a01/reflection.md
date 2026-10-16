# r04-b02-a01 result: 1.6275 (ok). +1.07% over the root r03-b02-a03 (1.6103), just outside the ±1% noise band; new campaign best

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the root and the baseline on every printed digit.
  - So the early slot writes and the `up_seq` / `mc_seq` / `down_seq` bookkeeping deliver exactly the same blocks.
  - No hang, no JIT error, no kernel-config-buffer overflow on the first and only run (the reader grew ~1.3 KB).
- Expected +0.5..+1.5% geomean. Got +1.07%. 7 of 9 shapes are faster; minimax-m3 is +2% (its usual ±5% lottery: mean core 344 -> 343).

Chip µs, root -> this node. MATH zones are from /tmp/r02b01a02/zones.py: mean core over the 10 measured calls, per-core zone sums.

| shape | chip µs | Δ | mean core end | mean RE-GU | mean RE-DN | crit core end |
|---|---|---|---|---|---|---|
| dsv3 | 514.0 -> 512.8 | -0.2% | 493 -> 482 | 325.2 -> 327.6 | **149.8 -> 135.9** | 513 -> 512 |
| kimi-k2.7 | 729.5 -> 717.3 | **-1.7%** | 719 -> 708 | 471.6 -> 476.6 | **223.6 -> 207.5** | 728 -> 716 |
| kimi-k3 | 1365.8 -> 1342.8 | **-1.7%** | 1273 -> 1244 | 725.7 -> 727.0 | **522.5 -> 492.6** | 1365 -> 1342 |
| glm | 451.2 -> 449.2 | -0.4% | 429 -> 421 | 285.0 -> 284.1 | 125.3 -> 118.2 | 450 -> 448 |
| mistral | 185.9 -> 183.8 | -1.1% | 166 -> 160 | 106.2 -> 106.0 | **49.9 -> 44.6** | 185 -> 183 |
| dsv4-flash | 317.0 -> 308.7 | **-2.6%** | 298 -> 287 | 181.5 -> 181.8 | **106.2 -> 95.5** | 316 -> 308 |
| dsv4-pro | 1070.9 -> 1056.4 | **-1.4%** | 1059 -> 1044 | 688.5 -> 693.5 | **357.6 -> 337.0** | 1070 -> 1056 |
| minimax-m3 | 365.5 -> 372.9 | +2.0% (lottery) | 344 -> 343 | 226.3 -> 228.6 | 112.8 -> 109.2 | 365 -> 372 |
| gpt-oss | 261.8 -> 255.8 | **-2.3%** | 238 -> 227 | 156.1 -> 151.1 | 77.0 -> 70.8 | 261 -> 255 |

## Why (profiler evidence)
1. **The down look-ahead now works on every shape, the 4-expert ones included.** Mean RE-DN fell on all 9 shapes:
   - -14 dsv3, -16 k2.7, -30 k3, -7 glm, -21 dsv4-pro: the same as r03-b04-a04 measured;
   - **-5.3 mistral, -10.7 dsv4-flash, -6.2 gpt-oss, -3.6 mm3**, where r03-b04-a04 measured -0.7 / -1.9 / +1.6 / -2.4.

   The only difference from r03-b04-a04's down code is the re-check while the in1_down sender spins on `act_valid`.
   That confirms its guess: on those shapes the one-shot `pages_reservable` check after the down multicast usually
   failed (the sender's compute was still on block kb-1). The slot frees during the activated-block wait, and the
   read now starts there instead of at the top of kb+1.
2. **The gate/up look-ahead did not shorten gate/up.** Mean RE-GU moved:
   - +1..+5 µs on the DRAM-bound shapes (dsv3, k2.7, dsv4-pro, mm3);
   - ~0 on k3, glm, mistral, dsv4-flash;
   - -5 µs only on gpt-oss, the one compute-bound shape.

   Two readings, which this run can't separate (I added no fire counter):
   - (a) It fires but only moves DRAM share. On DRAM-bound shapes, starting a column's next block earlier doesn't
     create bandwidth. The +2..+5 µs GU on dsv3 / k2.7 / dsv4-pro looks like the r02-b02-a03 / r03-b04-a03
     "conserved GU+DN" signature. GU+DN per mean core still fell by 12-16 µs on those shapes, so the net is positive.
   - (b) It rarely fires. The sender's x(kb) usually arrives before its compute frees slot kb+1, so the old path
     already started kb+1 right away.

   gpt-oss's -5 µs GU (mean core -11, crit -6) suggests it helps where x, not DRAM, paces a column.
3. **The op time follows the mean core this time.** The crit core moved -7..-23 µs on k2.7, k3, dsv4-flash,
   dsv4-pro and gpt-oss. The DN saving lowers every group's floor, so the tail lottery shifts down with it.
   glm / dsv3 crit only moved -1..-2 (their 4-row phase-B group is crit by its GU: dsv3 crit GU 329 -> 354).

## Classification
win: +1.07% geomean, bit-identical. Most of it is the down look-ahead plus its spin re-check, which now fires on all
9 shapes (DN -4..-30 µs mean core). The gate/up half is at best neutral on GU and helps only gpt-oss.

## What a child of this node should try next
1. **Measure whether the gate/up look-ahead fires** before keeping or extending it. Add a profiler-only counter on
   the sender NCRISC: (fired after push, fired during the in0_valid spin, fell back). If it rarely fires, delete it
   (`gu_lookahead = false`; it is ~0.9 KB of reader code) to get kernel-config headroom back. If it fires and GU
   still doesn't move, the GU stream is DRAM-share-bound and there is nothing more there.
2. **Do the same spin re-check for the activated (row) stream.** The act sender (gx == kb) waits `cb_activated` and
   then all 10 row acks. Receivers ack act kb+1 only at the top of kb+1, after receiving kb. A counting `act_valid`
   (seq instead of 0/1) would let receivers ack kb+1 once their `cb_in0_down_full` slot is free. DN is still
   45-60 µs per expert on mistral against ~29 µs of FPU, so the DN chain still has slack.
3. dsv3 / glm are crit on the phase-B 4-row group's GU (dsv3 crit GU 354 vs mean 328). That is r02-b02-a04's open
   item: a {3,3,4}-ordered phase-B entry, or putting the largest leftover on a 3-row group.
4. Don't retry: deeper rings, request VCs, down look-ahead across the activation gap (block 0 still keeps its
   order here), or a whole weight stream on one NoC.
