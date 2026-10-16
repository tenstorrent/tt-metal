# r03-b03-a01 result: 1.5751 (ok). Neutral (+0.3% vs the root r02-b02-a04's 1.5697, inside the ±1% noise)

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the root and the baseline.
- That was expected:
  - The planner produces the same plan. A host fuzz of the root's plan_launch against this node's
    gave 0 mismatches in 1.2M cases on 10- and 8-row grids with 16- and 32-bit counts. A planted bug
    gave 4224 mismatches out of 50k cases, so the check works.
  - The math is untouched.
- Kernel-only change: no host build, and no JIT error on the first run.

Per shape vs the root (µs, chip mean). Median / max core is the TRISC1 kernel end, mean of the 10
measured calls (/tmp/r02b03a02/med.py):

| shape | root | this | Δ | median/max core root -> this |
|---|---|---|---|---|
| kimi-k3 | 1405.9 | 1385.8 | **-1.4%** | 1321/1405 -> 1316/1385 |
| minimax-m3 | 381.6 | 374.3 | -1.9% (lottery range) | 347/381 -> 339/373 |
| mistral | 195.6 | 189.7 | -3.0% (lottery range) | 167/195 -> 166/189 |
| dsv4-flash | 327.9 | 326.2 | -0.5% | 303/327 -> 303/325 |
| dsv3 | 529.1 | 531.1 | +0.4% | 500/528 -> 489/530 |
| kimi-k2.7 | 736.5 | 737.0 | 0 | 733/735 -> 734/736 |
| dsv4-pro | 1080.7 | 1082.1 | +0.1% | 1076/1079 -> 1077/1081 |
| gpt-oss | 267.0 | 270.0 | +1.1% | 245/267 -> 241/270 |
| glm | 453.5 | 463.4 | +2.2% (lottery; median 433 -> 435) | 433/453 -> 435/462 |

I expected about +1% geomean, with kimi-k3 -1.2% and the 8-12-expert shapes -1..-2%. kimi-k3 got its
gain. The rest is noise.

## Why (profiler evidence)
**New measurement: RE-PLAN zone on the reader (NCRISC)**, i.e. plan_launch alone, after this node's
changes. Mean over the 10 measured calls, every core (/tmp/r03b03a01/plan.py on
reports/r03-b03-a01/.logs/profile_log_device.csv):

| shape | RE-PLAN µs (new) | my model of the root planner | my model of the new one |
|---|---|---|---|
| kimi-k3 (28 experts, single phase, mixed layouts tried) | **20.4** | 30.0 | 14.3 |
| dsv4-flash (8, mixed 3+3+2+2) | 7.4 | 9.9 | 5.0 |
| kimi-k2.7 (12, two-phase) | 6.0 | 9.1 | 4.8 |
| dsv4-pro (12, two-phase) | 5.8 | 8.9 | 4.7 |
| dsv3 / glm (8, two-phase + mixed B) | 4.75 | 6.4 | 3.5 |
| gpt-oss (4, mixed) | 2.9 | 3.4 | 2.1 |
| mistral / minimax-m3 (4) | 2.25 | 2.5 | 1.4 |

- **The planner is still expensive.** Caching the counts (656 -> 28 `tiles_of` calls on kimi-k3) and
  removing the divides (~820 -> 2) cut it far less than modelled.
  - What remains is plain loop work: ~10.7k executed source lines on kimi-k3 (gcov, -O0). By the
    measurement that is ~1.9 cycles per line, so the per-line cost in my model was too low.
  - Divides on BH are cheap (the old-picker calibration implies a few cycles), so they were never
    the cost.
  - The saving vs the root is therefore only ~5-9 µs on kimi-k3 and ~1-3 µs elsewhere. It shows on
    kimi-k3's max core (-20 µs, part of it lottery) and on the medians of dsv3 (-11) and mm3 (-8).
    That is about 1% of these shapes' time, which is noise on all but kimi-k3.
  - Remaining hot spots (gcov on kimi-k3, /tmp/r03b03a01/gc):
    - `group_cost`'s ceil loop: 252 calls, 434 loop iterations.
    - The least-loaded group search in `schedule_in_order` / `GroupScheduler` (5 groups, ~640
      iterations).
    - `schedule_lpt`'s per-distinct-value rescans of all experts for 3 mixed layouts plus the owner
      pass (~500).
    - `model_phase`'s per-expert streams loop (~580).
- **The UNPACK RE-PLAN zone did not record**: no TRISC_0 rows in the CSV. The compute-side plan cost
  is therefore inferred only: it runs the same code, so it should be about the same.
- kimi-k3's 20 µs is still ~1.5% of its time. A child that wants it has to cut work, not latency:
  - Have the uniform scheduler record per-layout owners, so `model_phase(single)` and the final
    owner pass reuse them.
  - Precompute the LPT order once.
  - Skip mixed layouts with an exact lower bound: Σ_e min_g cost > groups × cap means schedule_lpt
    must fail. That skips (4,3,3) on kimi-k3.
  - Most effective: plan once per core and share the result via L1 with the other 4 RISCs, so it is
    paid once per core. That doesn't shorten the critical path either, because every RISC waits for
    it. The real fix is a planner with fewer layout trials (e.g. only score layouts whose groups ≥
    active experts / per-group cap).
- I can't tell whether the residual cost is on the critical path for the DRAM-bound shapes. Their
  medians did not move outside noise (k2.7 / dsv4-pro +1 µs), so a few µs of prologue is invisible
  next to the run-to-run variance there.

## Classification
neutral (within noise). The mechanism is correct and exact (same plan, bit-identical) and harmless to
keep. It cut the planner by roughly a third, not by the modelled half or more. kimi-k3 is the only
shape that clearly gained (-1.4%).

## What a child of this node should try next
1. **Combine with the PACK-thread activation pipeline** (r03-b01-a01 / r03-b02-a01 are porting
   r02-b03-a04 onto the root). This node only touches adaptive_chunk.hpp and two call sites (reader,
   writer), plus one UNPACK line in fused_swiglu.cpp, so it merges trivially.
2. Don't spend another node on planner speed unless it removes whole layout trials. The residual is
   20 µs on kimi-k3 and 2-7 µs elsewhere, which is worth at most ~0.5% geomean.
3. The RE-PLAN zone on NCRISC is inert and can stay. The UNPACK one records nothing (TRISC_0 zone
   inside the UNPACK(({ })) statement expression), so drop it or move it.
4. Tools for planner changes:
   - /tmp/r03b03a01/{old,new}.cpp + main.cpp: old-vs-new exact-plan fuzz. Rename the namespace per TU
     with -Dadaptive_chunk=..., or the linker merges the two inline definitions.
   - /tmp/r03b03a01/gc/est.sh: gcov line counts on the test routing. Use ~1.9 cycles per -O0 line as
     the cost per line.
