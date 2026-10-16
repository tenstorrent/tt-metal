# r02-b03-a02 result: 1.5278 (ok), neutral vs best r02-b04-a01 (1.5314) and +2.0% vs parent r02-b03-a01 (1.4973)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every printed
digit). The combined plan ran on device on the first try: two-phase on kimi-k2.7 / dsv4-pro, mixed (3,3,2,2)
on gpt-oss / dsv4-flash, uniform 2x5 elsewhere. No hang, and the host build and JIT were clean. So the
per-layout topology sets, the moved phase-A semaphore offset (after `num_layouts` sets) and the
`owner[2][E]` map agree across reader, writer and compute.

Per shape (µs; noise ±1%):

| shape | r01-b04-a04 | r02-b04-a01 (best) | parent r02-b03-a01 | this | plan here |
|---|---|---|---|---|---|
| kimi-k2.7 | 764.3 | 767.5 | 744.8 | **736.0** | two-phase (as parent) |
| dsv4-pro | 1128.0 | 1131.8 | 1083.6 | **1083.5** | two-phase (as parent) |
| gpt-oss | 338.8 | 277.1 | 340.5 | **281.1** | (3,3,2,2) (as b04) |
| dsv4-flash | 331.2 | 322.4 | 336.2 | 330.5 | (3,3,2,2) (as b04) |
| dsv3 | 547.5 | 555.7 | 565.0 | 555.9 | uniform 2x5 (repair: no longer two-phase) |
| glm | 487.6 | 486.3 | 492.5 | 500.3 | uniform 2x5 (repair) |
| kimi-k3 | 1378.8 | 1397.0 | 1396.6 | 1404.5 | uniform 2x5 |
| minimax-m3 | 382.0 | 388.8 | 381.7 | 393.4 | uniform 2x5 |
| mistral | 193.0 | 192.6 | 196.0 | 196.7 | uniform 2x5 |

Each schedule's gains survived the combination. kimi-k2.7 (736 µs) and dsv4-pro are campaign bests, and
gpt-oss stays at b04's level. The repair worked: dsv3 is back to b04's 556 µs (parent 565). I expected
~1.54. The score is 0.2% under the best, because the shapes whose plan did not change came in 1-3% slower.

## Why (profiler evidence)
TRISC1 kernel end, median core / max core, mean of the 10 measured calls (µs from op start;
/tmp/r02b03a02/med.py on reports/<node>/.logs/profile_log_device.csv):

| shape | r02-b04-a01 | this |
|---|---|---|
| dsv3 | 504 / 555 | 504 / 555 |
| glm | 450 / 486 | 451 / 500 |
| mistral | 163 / 192 | 165 / 196 |
| minimax-m3 | 348 / 388 | 355 / 393 |
| kimi-k3 | 1306 / 1396 | **1327** / 1404 |
| dsv4-flash | 303 / 321 | **313** / 330 |
| gpt-oss | 245 / 277 | 250 / 281 |
| kimi-k2.7 | 615 / 766 | 732 / 735 (tail removed) |
| dsv4-pro | 1107 / 1130 | 1078 / 1082 (tail removed) |

- glm and mistral: the median core is unchanged, so their losses are only in the per-call max. That is the known
  group-tail lottery (±3-5% on these shapes), not an effect of this change.
- kimi-k3 and dsv4-flash: the MEDIAN core moved too (+21 µs, +10 µs), on the same deal as b04. That is a shift of
  the whole call, which fits **planner prologue cost**. Every RISC now runs `make_plan` and then `plan_groups` before
  its first DRAM read. On these two shapes `make_plan` gets as far as modelling: kimi-k3 (28 experts: n_a 25, 3 left
  over) and dsv4-flash (8: n_a 5, 3 left over) both run the distinct-count thresh loop and two `model_phase` passes
  (each walks every expert twice, with software divides in num_chunks / expert_cost). Only then does the new
  single-round rule reject every rows_b. The parent already showed +10 µs on kimi-k3 from `make_plan` alone
  (1316 median). Here kimi-k3 pays that plus `plan_groups`' six layout schedules. The 4-expert shapes return at
  n_a == 0 before modelling and moved only +2..7 µs (noise level).

## Classification
neutral (within noise vs the best node; +2.0% over the parent). The combination works and keeps both
schedules' wins. A measurable prologue cost on the many-expert shapes eats them.

## What a child of this node should try next
1. **Cut the planner prologue (cheap, kernel-only, likely about +0.5-1%).** In `make_plan`, reject right after
   computing `n_a`: if `active - n_a > kGridY / (2 * rows_a)` (no larger uniform size can take the leftover in one
   round), return `single` before the thresh loop and before any `model_phase`. That skips all modelling on kimi-k3,
   dsv3, glm and dsv4-flash. Also cache `tiles_of(e)` once into a local `uint8_t tiles[E]` and pass a lambda over the
   array, since every planner pass re-reads the counts L1 and re-derives tiles. Finally, have `plan_launch`
   reuse the uniform pick (`pick_group_rows` and `plan_groups` both schedule the 3 uniform layouts). Judge it by the
   median-core shift on kimi-k3 (target: back to ~1306 µs) and dsv4-flash (~303).
2. Phase-B is still restricted to uniform layouts. A mixed phase B (e.g. dsv3/glm's 3 leftovers on 4+3+3 rows, one
   round, 3 streams) would extend the two-phase win to those shapes. The phase loop already takes any layout index,
   but phase-A ring alignment rules still apply.
3. kimi-k2.7 / dsv4-pro are now tail-free (all rows end within ~3 µs) and DRAM-bound at ~430-455 GB/s. Don't spend
   more scheduling attempts on them.
