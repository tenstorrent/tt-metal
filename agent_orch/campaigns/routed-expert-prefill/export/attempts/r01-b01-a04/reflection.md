# r01-b01-a04 result: 1.4576 (ok), neutral vs r01-b01-a02 (1.4716)

## What happened vs expected
All 9 shapes are valid and bit-identical to a02 and the baseline (PCC/max_abs/mean_abs equal to
every digit). This confirms that reader and compute agree on the cross-unit look-ahead (next unit,
pre-tilized block 0, next per_core_M), with no hang and no wrong rows. The node also reverts the
parent's (a03) pacing code to a02, so a02 is the right comparison. Score 1.4716 -> 1.4576 (-1.0%).
The per-shape changes are noise:

| shape | a02 µs | a04 µs | units/group | mechanism active? |
|---|---|---|---|---|
| dsv3 | 563.1 | 556.3 | 2 | yes |
| kimi-k2.7 | 771.5 | 773.4 | 3 | yes |
| kimi-k3 | 1478.8 | 1487.2 | 7 | yes |
| glm | 486.9 | 493.7 | 2 | yes |
| mistral | 185.9 | 195.1 | 1 | NO (code path identical to a02) |
| dsv4-flash | 340.7 | 343.7 | 2 | yes |
| dsv4-pro | 1148.9 | 1152.8 | 3 | yes |
| minimax-m3 | 378.2 | 382.1 | 1 | NO (code path identical to a02) |
| gpt-oss | 364.5 | 366.6 | 2 | yes |

mistral (+4.9%) and minimax-m3 (+1.0%) run exactly a02's code here (one unit per group, so there is
never a next unit), yet they moved the most. Run-to-run noise on the 1-expert-per-group shapes is
about ±5%, well above the campaign's ±1% band. The -1% overall is within that noise.

## Why (best explanation, with profiler evidence)
Per-group TRISC1 end times (max over a 2-row group), averaged over the 10 measured calls
(/tmp/r01b01a04/cmp.py on profile_log_device.csv), as min / MEAN / max µs:

| shape | a02 | a04 | mean Δ |
|---|---|---|---|
| dsv3 | 499/529/562 | 505/530/554 | +0.0% |
| kimi-k2.7 | 698/740/770 | 718/745/771 | +0.8% |
| kimi-k3 | 1382/1433/1477 | 1403/1444/1486 | +0.8% |
| glm | 432/460/486 | 444/466/492 | +1.3% |
| mistral (unchanged code) | 158/169/184 | 157/173/193 | +2.5% |
| dsv4-flash | 297/319/339 | 297/319/342 | 0.0% |
| dsv4-pro | 1086/1120/1147 | 1058/1112/1151 | -0.7% |
| minimax-m3 (unchanged code) | 335/356/377 | 340/359/380 | +0.7% |
| gpt-oss | 358/361/363 | 354/360/365 | -0.4% |

- The mean group finish, i.e. the work actually done without the tail lottery, did not move on
  any shape where the look-ahead fires. kimi-k3 has 6 unit boundaries per group, so it should
  have shown the gain most clearly, and it is flat. The x(0) read + tilize + multicast at a unit
  start is therefore NOT on the critical path, or it costs less than ~1 µs per boundary. Possible
  reasons: the reader already ran far enough ahead during the down phase for the up-front read to be
  hidden, and the row's cores reach the next unit staggered (down tail + copy pass), so one owner's
  ~3 µs tilize is absorbed. The tilize I moved into the previous unit's last gate/up block costs that
  owner the same work there, so the change is at best a wash.
- Together with r01-b03-a03 (weight side of the boundary, neutral), the unit/expert boundary is now
  closed on both input paths.
- Side analyses from this node, for the next worker:
  * Static expert->group balance is fine on every shape. Replaying the test's seed-0 routing
    (/tmp/r01b01a04/counts.py, sched.py) gives equal or near-equal modelled loads per group.
    Tile counts per expert: dsv3/glm [6,5,6,5,5,5,5,6], kimi-k2.7 3-5, kimi-k3 3-4 (28 experts),
    mistral/minimax [5,6,6,5], dsv4-flash 4-5, dsv4-pro all 3, gpt-oss [8,9,9,8].
  * In calls where two groups finish ~15% before the other two (e.g. dsv3 call 10: 477/570/572/489),
    the slow pair does not finish any faster than its earlier pace would predict, even though it
    should then have the whole DRAM. The tail looks like a per-call "slow mode" of the slow groups,
    not an unfair DRAM share. That supports a03's conclusion that pacing is the wrong fix.
  * R=1 (one-row groups) for kimi-k3: by the cost model it is about 15-20% better, but 28 experts on
    8 rows means 4 experts on some rows and 3 on others. Those rows read 33% more weight bytes, and
    the worst row's MAC rows (13-14) equal R=2's (14). b02-a01 also measured G=1 equal to R=2 there.
    Not promising.

## Classification
neutral (within noise): flawed idea. The x(0) boundary bubble is not a measurable cost. The
implementation is correct (bit-identical, no hang). It is harmless and can stay or be reverted; a
child should prefer a02's simpler code.

## What a child of this node should try next
- Start from r01-b01-a02 (or keep this node; same performance). Don't spend more attempts on
  expert/unit-boundary feeds (weights: b03-a03, x: this node) or on DM read timing.
- The noise floor on mistral/minimax-m3 is about ±5% per run with identical code. Judge
  single-expert-per-group shapes by the per-group mean in profile_log_device.csv, not by the
  summary µs.
- Per-core serial work per expert is what's left: gate/up matmul, then the SFPU activation, then
  the down matmul and copy pass, at ~29-45 cycles per tile-mult.
  The activation is pure serial SFPU time: about 18-27 tiles per core per expert at
  ~1-3k cycles/tile, roughly 7% of group time on dsv3, ~17% on kimi-k3 (SiTU-GLU, 7 experts) and
  ~15% on gpt-oss. It cannot overlap with the down phase, because down block j needs column j's
  activation and every column finishes at the same time. The only way to hide it is SFPU-FPU
  concurrency: run the SFPU on the PACK thread as moe_gpt/compute.cpp does with swiglu, either
  overlapped with the last gate/up K-block per subblock (a partial hide), or with cross-expert
  pipelining (act(e) || gu(e+1), which needs double partials and a DM phase reorder). That is the
  largest untried lever. It is high-risk and compute-kernel heavy.
- A measurement node with a few DeviceZoneScopedN zones per phase on TRISC1 (gate/up, activation,
  down, copy) is still missing and would settle the phase split. Keep the zone count per op small,
  for the profiler buffer.
