# r04-b03-a03: put the idle rows to work on a tie: when the best uniform row-group layout leaves groups idle (fewer active experts than groups), the planner takes a mixed-height layout with one expert per group and the same modelled makespan if it lowers the summed expert cost, so mistral / minimax-m3 run 3+3+2+2 rows (the two 6-tile experts at per_core_M 2 on 3 rows) instead of 2+2+2+2 with rows y10-11 idle

## Motivation
- mistral and minimax-m3 route 4 experts (5, 6, 6, 5 tile-rows) to the chip. The planner's uniform 5 x 2-row layout
  wins on modelled makespan (4 units: per_core_M 3 + 1), so one group (rows y10-11, 22 cores, 20% of the grid) idles for
  the whole call (rows.py on reports/r04-b03-a02: y10/y11 end at 4-5 µs). Every node since r01-b04-a04 has noted this.
- The mixed 3+3+2+2 layout gives the 6-tile experts per_core_M 2 on 3 rows (cost 3) and the 5-tile ones per_core_M 3 on
  2 rows (cost 4). Same makespan, so `plan_groups` (mixed must be STRICTLY better, cap = best - 1) never takes it.
- But the chip time is not the modelled makespan; it is the max over the groups' end times, and each group's end
  carries a large per-call lottery (/tmp/w_r04b03a03c/grp.py on reports/r04-b03-a02, 10 measured calls):
  - mistral group ends: means 169.5 / 168.2 / 164.0 / 160.0 µs; per-call max 183 µs mean. Single-group spikes of
    +45 µs (call 7: y4-5 DN 94 µs with 51-60 µs activated wait; call 9: y6-7 GU 170 µs with 80 µs x wait).
  - The mean per-call max over only the two 5-tile groups is 173 µs (mistral) and 356 µs (minimax-m3) vs 183 / 368
    over all four.
- per_core_M costs U ≈ 25 µs per tile-row on mistral (r03-b04-a01's T = F + U·pcM fit) and ≈ 29 µs on gpt-oss
  (r02-b04-a01: its 3-row pcM-3 groups end ~25 µs before its 2-row pcM-4 groups). So the two 6-tile groups should end
  ~25 µs earlier and drop out of the max, which then runs over two groups instead of four, and the two 5-tile groups
  get the DRAM / NoC to themselves for their last ~20-25 µs.

## Mechanism
`kernels/adaptive_chunk.hpp`, `plan_groups` only (kernel-only, no host rebuild; every RISC runs the same planner):
- After the existing strict pass, if the winner is still a uniform layout AND it leaves groups idle (active experts
  < groups), try every mixed layout with at least `active` groups under cap = best (ties allowed). Accept it only if
  every active expert lands on its own group (no group runs two experts, so no extra per-expert fixed cost on any
  group) and the summed expert cost (sum of per_core_M + chunks over the experts) is strictly lower than the uniform
  deal's. Among several, the lowest sum wins (table order on a tie).
- mistral / minimax-m3 (test routing): 2x5 (sum 16) -> 3+3+2+2 (sum 14): 6-tile experts on the 3-row groups (rows
  y2-4, y5-7) at pcM 2, 5-tile experts on y8-9 and y10-11 at pcM 3.
- No other test shape changes plan: k3 / dsv4-flash / gpt-oss have no idle uniform group (dsv4-flash and gpt-oss are
  already 3+3+2+2), the two-phase shapes (dsv3, glm, k2.7, dsv4-pro) never reach `plan_groups`. Checked with a host
  replay of `plan_launch` on the test routing (old vs new) and a fuzz over random count vectors that the new plan
  differs only when the uniform layout had idle groups.
- Numerics: bit-identical. Each expert still runs whole on one group with the same K order; mixed layouts with 3-row
  groups are already exercised by dsv4-flash and gpt-oss.

## Why this is not a repeat
- r02-b02-a01 / r02-b04-a01 introduced mixed layouts but take them only when strictly better in the model; they left
  mistral / minimax-m3 on 2x5 by design. r02-b02-a04 / r02-b03-a02 changed two-phase / phase-B choices; mistral and
  minimax-m3 have no phase A (4 experts < 5 groups). Nobody has run the 4-expert shapes on all 10 rows.
- r02-b02-a04 next #4 says only a sub-expert split can use the idle rows. That holds for the critical per_core_M (3),
  but not for the end-time distribution: two groups at pcM 2 leave the max.
- Not a DM handshake change (the last three r04 attempts on that line were neutral).

## Expected effect and risk
- mistral -3..-6%, minimax-m3 -2..-4% chip time; others unchanged (identical plan and code path) => geomean
  about +0.5..+1%.
- Risks: (1) the 2-row groups at rows y10-11 may be a slower position (gpt-oss's 2-row groups at y8-11 are its slowest),
  which would eat the gain; (2) 3-row groups move more x rows (3 x-owner streams); (3) planner code grows ~0.2 KB on
  reader, writer and UNPACK (h7168 programs had ~4.7 KB margin at the root). Tell from rows.py / grp.py per-group ends.

## Addendum after run 1 (runtime_error, not scored): kernel-config-buffer overflow, repaired
- Run 1 failed program creation on 6 shapes: `Program size (70736..71648) too large for kernel config buffer (70656)`,
  worst kimi-k2.7 at +992 B. kimi-k3, dsv4-pro and minimax-m3 ran and were bit-identical to the parent (minimax-m3 with
  the new 3+3+2+2 plan). My margin estimate (~4.5 KB, from r04-b03-a01's root figure) was wrong: the parent was only
  ~0.8 KB under the limit, and the tie pass adds ~0.6 KB to each of the 3 planning RISCs (reader, writer, UNPACK).
- Repair (size only, same plan): the planner region is built at -Os instead of -O2 on the writer (BRISC) and the
  compute UNPACK; the reader keeps -O2 because its plan releases the first DRAM reads. Local riscv-tt-elf-g++ size of
  plan_launch: parent 4832 B per RISC at -O2; now reader 5428, writer 4024, UNPACK 4024, i.e. about -1.0 KB per core vs
  the parent and -2.2 KB vs run 1. r04-b01-a02 measured the writer -Os planner as cost-free (first-GU-start unchanged).
  The UNPACK planner only has to finish before the reader's first multicast lands.
