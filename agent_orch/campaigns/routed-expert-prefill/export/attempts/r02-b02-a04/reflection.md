# r02-b02-a04 result: 1.5697 (ok). New campaign best: +1.7% over r02-b03-a03 (1.5435), whose code this node carries plus the planner change

## What happened vs expected
- All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every
  printed digit).
- The new plan ran on dsv3 and glm on the first try, with no hang: 2x5 phase A (one expert per group),
  then the 4+3+3 phase B. So the existing kernels' per-phase layout handling (topology set = layout
  index, phase-A sem pair, layout_group) works for a mixed phase-B layout, as expected from the code.
- There was a host rebuild only because the parent's factory change (r02-b02-a03's dw_slots) was
  reverted to r02-b03-a03's.

Per shape vs r02-b03-a03 (µs; noise ±1%). Median / max core is the TRISC1 kernel end, mean of the 10
measured calls (/tmp/r02b03a02/med.py):

| shape | r02-b03-a03 | this | Δ | median/max core before -> after | plan |
|---|---|---|---|---|---|
| dsv3 | 565.0 | 529.1 | **-6.4%** | 512/564 -> 500/528 | 2x5 -> two-phase 2x5 + (4,3,3) |
| glm | 492.2 | 453.5 | **-7.9%** | 445/492 -> 433/453 | same change |
| dsv4-pro | 1093.2 | 1080.7 | -1.1% | 1086/1092 -> 1076/1079 | unchanged (2x5 -> 5+5) |
| kimi-k2.7 / kimi-k3 / mistral / dsv4-flash / minimax-m3 / gpt-oss | | | +0.3 / 0 / -1.2 / +0.5 / +1.3 / -0.2% | | unchanged |

- Predicted -8..-10% on dsv3 and glm (model said -14%). Measured -6.4% and -7.9%.
- The other seven shapes run byte-identical plans: the host replay matches the old header, and the
  8-row-grid plans are identical too. Their deltas are run-to-run noise.

## Why (profiler evidence)
TRISC1 kernel end per physical row y2..y11, mean of the 10 measured calls (/tmp/r01b04a04/rows.py):

| shape | r02-b03-a03 (2x5, 2/2/2/1/1 experts) | this node (phase B rows 0-3 / 4-6 / 7-9) |
|---|---|---|
| dsv3 | 539 539 538 538 530 530 289 287 284 282 | 515 516 516 515 496 497 495 481 481 480 |
| glm | 465 465 466 465 473 473 257 257 252 250 | 443 443 443 442 432 432 431 418 419 417 |

- **The tail is gone.** Before, rows 6-9 idled for the last ~45% of the call (ending ~285 vs ~539).
  Now every row ends within ~35 µs of the others. The 3 leftover experts use the 4 idle rows at
  per_core_M 2 instead of running at per_core_M 3 on 2-row groups.
- Phase A (five 2-row groups, one expert each at pcM 3) ends at about where the old light groups
  ended, ~285 µs. So the phase-B expert takes ~200-230 µs (dsv3) vs ~250 µs for the old second
  expert at pcM 3. That is -15..-20% on the tail expert. The model's c-units predicted -32% for pcM 3 -> 2.
  The shortfall fits b03-a01's finding: per-group stream rate falls for taller groups, and a
  phase-B group starts at the LATEST of the phase-A groups it spans (lottery max).
- The 4-row group (rows 0-3) is the critical one, ~20-35 µs behind the 3-row groups, on both shapes.
  It runs the same pcM 2 as the others but has 3 phantom tile-rows (5 tiles on 8 slots). It spans
  phase-A groups 0+1 (6- and 5-tile experts). A 4-row group may also stream slower than a 3-row one
  (one more receiver row per column multicast). It is unclear which dominates without zones.
- Total bytes: dsv3 ~234 MB in 529 µs = 442 GB/s. That is now at the top of the measured DRAM range
  (410-455 GB/s), so dsv3 and glm are at the DRAM floor too.

## Classification
win (+1.7% geomean, new campaign best). Two shapes moved -6..-8%, far outside noise. Numerics are
bit-identical, the change is planner-only (one header), and every other plan is unchanged.

## What a child of this node should try next
1. **dsv3/glm are now DRAM-bound (~440 GB/s), so don't spend more schedule attempts there.** The only
   residual is the 4-row group trailing by ~20-35 µs. A cheap probe: order the phase-B layout
   candidates so 3 leftovers prefer (3,3,4)-like placement on the earliest-finishing phase-A groups.
   That needs a new table entry {3,3,4} plus host topology (automatic). Expect <= 2% on two shapes.
2. **Combine with the compute-side wins from other branches** (all orthogonal to the planner):
   - r02-b01-a03's PACK-thread pipelined activation (glm -2.3%, kimi-k2.7 -1.5%, minimax -5%,
     gpt-oss -3% on its lineage). Merge its fused_swiglu.cpp K-loop changes with r02-b03-a03's
     binary_act_sfpu.h (b01-a03 deleted binary_activation_phase, which b03-a03 edited).
   - SiLU SFPU treatment (r02-b03-a03 next #1) for dsv3/kimi-k2.7/glm/mistral.
3. **Planner prologue trim** (r02-b03-a02 next #1) is still untried. kimi-k3's median core (1321) is
   still ~15 µs above r02-b04-a01's (1306). Cache tiles_of into a local array once per RISC.
4. Remaining non-DRAM-bound shapes: gpt-oss (one expert per group, compute/activation-bound),
   minimax-m3 and mistral (4 experts on 5 groups, 2 rows idle). For the latter, only a sub-expert split
   (N or M across the idle rows) can use the idle rows. A mixed phase B can't, because there is no
   phase A when experts < groups.
5. Tools: /tmp/routed_r02b02a04/pt.cpp (test-routing replay) and fuzz.cpp (deal-invariant fuzz, 300k
   cases on 10- and 8-row grids). Run both after any planner change.
