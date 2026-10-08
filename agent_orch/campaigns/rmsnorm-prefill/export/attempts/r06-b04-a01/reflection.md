# r06-b04-a01 result: 1.4264 (ok)

## What happened vs expected
Valid on every shape, and bit-identical in accuracy to the parent: PCC 0.9999985, max_abs 0.0205-0.0240. Same
bytes to the same pages; only the NoC and the issue order changed. But it is clearly slower on every shape. µs,
chip mean, parent r05-b01-a01 → this node:

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 11.19 | 12.47 | +1.28 (+11%) |
| h4096 | 12.08 | 13.25 | +1.17 (+10%) |
| h6144 | 14.59 | 15.95 | +1.36 (+9%) |
| h7168 | 15.84 | 17.39 | +1.54 (+10%) |

Score 1.4264 vs 1.5696 (-9.1%), far outside the ±1% noise. I expected -0.3..-0.6 µs from shorter straggler tails.
Instead every core's drain got slower, and the left-half cores got much slower.

## Why (profiler evidence)
Scripts are in `analysis/`; run each on `$DREAM_HOME/rmsnorm-prefill/reports/<node>`:
- `readpos.py <report> shape dev`: per-core medians on one chip. `wdur` is the time from POST start (or drain
  start, if later) to drain end.
- `chain.py <report>`: per-device wave chain.

Outputs are `readpos_s{0,3}_d{2,3}.txt` (+ `readpos_parent_*`), `readpos_h7168_dev0.txt`, `chain_out.txt` and
`chain_parent_out.txt`.

1. **Drain time per core, left of the x=9 DRAM column vs right of it** (median / max, µs):

   | | this left | parent left | this right | parent right |
   |---|---|---|---|---|
   | h3584 dev2 | 3.30 / 4.62 | 1.68 / 1.95 | 2.27 / 2.30 | 1.63 / 1.80 |
   | h3584 dev3 | 3.25 / 4.52 | 1.69 / 1.97 | 2.28 / 2.47 | 1.63 / 1.73 |
   | h7168 dev2 | 4.80 / 7.32 | 2.98 / 3.62 | 3.74 / 4.57 | 3.00 / 4.09 |
   | h7168 dev3 | 4.77 / 7.24 | 3.08 / 3.46 | 3.73 / 4.09 | 3.21 / 4.25 |

   On dev0 h7168, wave-A cores at x=2..5 drain in 6.2-7.3 µs (parent 2.7-3.4). Cores at x≥10 take 3.7-4.0, which is
   the parent's range.
   - Left cores roughly doubled.
   - Right cores got ~0.5-0.7 µs slower.
   - The position gradient the mechanism was meant to remove is now much larger, and it is the r01-b02-a04 /
     r02-b03-a01 left-vs-right signature again.
2. **The policy degenerated to "almost all eligible tiles on NoC0".** The backlog metric adds +1 while the write
   command buffer is busy. Right after BRISC issues a 2 KB tile on NoC1, that buffer is busy, so NoC1 always looks
   more backed up than an idle NoC0, and the next eligible tile goes to NoC0.
   - Result: eligible tiles alternated N/E with almost every E tile on NoC0. That is ~50% of a core's tiles on NoC0,
     vs the static rule's ~25%.
   - For left cores the eligible (x=9-column) banks are reached on NoC0 by going east along the worker row, then
     south down the x=9 DRAM column.
   - With every left core doing that, the x=9 column's NoC0 ingress saturates. Local injection state can't see that:
     the cost is downstream, at the merge into the DRAM column. That is r02-b03-a01's per-router arbitration
     finding.
3. **The slow drains feed the cross-call launch skew.**
   - Late-draining cores restart late in the next call: on dev0, wave A's read start is up to 0.9-1.0 µs late at
     h7168.
   - Per-device kernel end now spans 15.2-19.1 µs at h7168 (parent 15.5-15.9) and 11.9-13.6 at h3584 (parent
     9.7-13.1).
   - So the chip mean pays twice: once for the slower drain, again for the skewed next call.
   - On dev3 (latest launch, little skew) h7168 ends at 15.22 vs the parent's 15.53. That chip's AG wait was short,
     so this is not evidence that the drain itself is better.
4. **The out-of-order, single-flush part can't be judged separately.** It was bundled with the NoC policy. Nothing
   suggests it helped: right-side cores, which mostly keep their NoC1 path, also got slower. Most likely that is
   their x=0-column eligible tiles also shifting to NoC0 (the 16→0 wrap, then south down column 0).

## Classification
flawed idea as implemented. The injection-side backlog signal (NIU unsent count + busy command buffer) is blind to
where the drain congests: the merge into the DRAM column, downstream. It pushed the NoC0 share from ~25% to ~50%,
and more NoC0 share is worse on this machine, especially for the left half. The static path-aware rule with 50% of
eligible visits (r02-b02-a01) is better than a greedy local policy.

## What a child of this node should try next
1. **Don't retry local-congestion NoC selection for the drain.** Revert to the parent's drain loop.
   - Evidence now spans four nodes: r01-b02-a04, r02-b03-a01, r06-b04-a01 (all bad when the NoC0 share grows) and
     r02-b02-a01 (good at ~25%).
   - The one untested direction this suggests is a **lower** static NoC0 share on the left half. For example, x<9
     cores send only 1 of 4 eligible visits on NoC0, and right cores keep 1 of 2.
   - Expect ≤0.3 µs; it's a one-line change in `use_alt` (`(out_idx / NUM_DRAM_BANKS) & 3`).
2. If a reordering drain is retried, keep the parent's exact NoC assignment and change only the order and the flush:
   one flush+pop per row instead of per block. That isolates the out-of-order effect this node couldn't separate.
   Expect a small gain at most, since the per-core rate is capped downstream.
3. The bigger levers are still upstream of the drain, from the parent's timeline:
   - **Wave A's push → F_SEND lag** is 0.5-1.0 µs at h7168 (`analysis/push.py`): the last A push ends ~4.4 µs, but
     the forwarder only sees all arrivals at 4.9-5.4. That is a NoC1 delivery delay of the stick write + arrival inc
     to the forwarder at (11,5) while wave B's read is in flight. Try sending the push on NoC0, or moving the
     forwarder core.
   - **The ~1.1 µs from go to POST start** per wave: the 4 × 1152 B pair reads (~0.5 µs) plus the combine
     (~0.55 µs).
