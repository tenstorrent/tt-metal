# r08-b02-a01: combine r07-b01-a01 (plan once per core + idle-row tie pass) with r07-b02-a01 (wrap-free gate/up multicasts)

## Motivation
- Round 7 produced two independent wins over the same parent, r06-b04-a01:
  - r07-b02-a01 (this round's root, 1.7387): wrap-free gate/up multicasts. Reader/writer K-loop only.
  - r07-b01-a01 (1.7196): the reader plans once and publishes the plan in a one-page L1 CB, and the
    freed kernel-config space carries the idle-row tie pass. Its gain is mistral -9.8%: 3+3+2+2 rows
    instead of 2x5 with rows y10-11 idle.
  - The two diffs touch disjoint code: kernel start-up and the planner vs the K-loops. r07-b02-a01
    next #2 and r08-b03-a01 next #3 both ask for this combination. No round-8 branch does it: b01 is a
    bottom-first planner deal, b03 and b04 are cross-group pacing.
- On the root, mistral and minimax-m3 still leave a fifth of the grid idle. Per-row ends from
  reports/r07-b02-a01 (/tmp/r06b01a01/ends.py):
  - mistral: y2-3 167, y4-5 157, y6-7 140, y8-9 127, y10-11 idle (4 µs);
  - mm3: 333 / 311 / 280 / 265 / idle.
- Delivered traffic shows they are the shapes furthest from the DRAM limit. Weights + bf16 x + bf8
  out over the chip time:
  - mistral ~390 GB/s, mm3 ~420 GB/s;
  - kimi-k2.7 / dsv4-pro ~470 GB/s, against the 490-512 GB/s BH all-core DRAM microbenchmark.
  So the extra rows can still raise throughput on these two shapes.
- Their lower 2-row groups are close to compute-bound at per_core_M 3. mistral bottom-group GU is
  83 µs against a ~72 µs LoFi matmul floor. The 3-row groups of 3+3+2+2 run the 6-tile experts at
  per_core_M 2 instead of 3, so there is less per-core work where the rows are starved.

## Mechanism
Apply r07-b01-a01's 5-file diff (vs r06-b04-a01) onto r07-b02-a01. It applies cleanly with git's 3-way
merge, with no conflict:
- `kernels/adaptive_chunk.hpp`:
  - the tie pass in `plan_groups`: when the uniform winner leaves groups idle, take a mixed layout with
    the same modelled makespan, one expert per group and a strictly lower summed cost;
  - `store_plan` / `load_plan` for the one-page plan CB;
  - the per-RISC planner pragmas (TRISC -O2, BRISC -Os) are removed, because only the reader plans.
- Reader: plans, stores and pushes the plan page. The `start` page read is issued before planning and
  barriered after it.
- Writer and compute UNPACK: `wait_front` the plan CB and `load_plan` instead of `plan_launch`.
- Factory: CB c_21 (`cb_plan_scratch`, 32-64 B) and the named compile-time arg on all three kernels.

r07-b02-a01's K-loop changes compose with this without edits. The top-sender (`my_mt == 0`) reader
multicast and the bottom-sender writer multicast predicates come from the group height and row of
whichever layout the plan picks. Corner placement already covers the 3-row groups of 3+3+2+2
(r06-b04-a01).

## Why this is not a repeat
This is a new combination of two measured wins. r07-b01-a01 ported plan-once + tie pass onto
r06-b04-a01, the common parent, so it never had wrap-free gate/up multicasts. r07-b02-a01 never had the
tie pass. The tie pass on mistral/mm3 was measured three times (r04-b03-a03, r05-b01-a01, r07-b01-a01),
always on lineages where the 3-row groups' gate/up multicasts still wrapped.

## Expected effect and risk
- The 7 shapes whose plan does not change (dsv3, kimi-k2.7, kimi-k3, glm, dsv4-flash, dsv4-pro,
  gpt-oss) should stay within noise. Plan-once is latency-neutral (r05-b01-a01: first-GU start
  +0.2-0.5 µs).
- mistral: about -6..-10% (r07-b01-a01 got -9.8% from the same plan change).
- mm3: about -2..-6%. On r07-b01-a01 its 2-row per_core_M-3 groups (y8-11) became the tail at
  311-315 µs. The root's bottom groups end ~30 µs earlier than r06-b04-a01's, so that tail should
  shrink.
- Geomean: about +1..+1.5%.
- Kernel config: local -O2 profiler compile gives reader +1.2 KB and writer -3.6 KB vs the root. UNPACK
  also loses its planner copy (~4 KB per r05-b01-a01). Net several KB smaller.
- Risks:
  - Host rebuild for the new CB.
  - A hang if a consumer waits on a plan page that is never pushed. The reader pushes it
    unconditionally.
  - Accuracy: bit-identical expected, because every expert still runs whole on one group in K order.
- How to read the eval: chip µs and per-row ends on mistral / mm3 (rows y10-11 busy), and the other 7
  shapes within their usual spread.
