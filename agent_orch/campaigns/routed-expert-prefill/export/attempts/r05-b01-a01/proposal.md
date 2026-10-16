# r05-b01-a01: plan once per core (the reader publishes the launch plan in an L1 page for the writer and compute) and spend the freed kernel-config space on r04-b03-a03's idle-row tie pass, so minimax-m3 / mistral run 3+3+2+2 rows instead of 2x5 with two rows idle

## Motivation
- Per-shape DRAM floor (bytes / ~460 GB/s, the rate kimi-k2.7 already sustains) against r04-b02-a01 (best, 1.6275):
  dsv3 1.00x, kimi-k2.7 0.98x, glm 1.03x, dsv4-pro 1.03x. Those four are at the floor.
  The headroom is on the shapes that are not DRAM-bound: minimax-m3 1.20x, mistral 1.26x, gpt-oss 1.73x,
  kimi-k3 1.11x, dsv4-flash 1.10x.
- minimax-m3 and mistral route 4 experts (6/6/5/5 tile-rows) to the chip. The planner takes the uniform 5 x 2-row
  layout, so rows y10-11 (22 cores, 20% of the grid) idle for the whole call. In r04-b02-a01's profile, y10/y11 end
  at 4 us on every call of both shapes.
- r04-b03-a03 measured a tie pass that takes 3+3+2+2 instead: the 6-tile experts run at per_core_M 2 on 3 rows, and
  the 5-tile experts at per_core_M 3 on 2 rows.
  - It was bit-identical, and minimax-m3 went 368.8 -> 352.0 us (-4.6%), mean core end 341.6 -> 324.9. All four
    groups then ended within 329-333 us.
  - It scored neutral only because its size repair compiled the UNPACK planner at -Os. That pushed first-GU-start
    +2.3..+7.5 us on every shape.
  - Its next #1/#2: keep the tie pass, find the size elsewhere ("plan once per core"), and port it onto
    r04-b02-a01.
- The size problem is real on this lineage. r04-b02-a01's kimi-k2.7 program is within a few hundred bytes of the
  70656 B kernel config buffer (r04-b04-a01, r04-b01-a02 measurements). The tie pass costs ~0.6 KB per planning
  RISC, and today three RISCs per core plan: reader, writer and compute UNPACK.
- r03-b01-a02 removed two of the three planner copies on another lineage by having the reader publish its plan in a
  one-page L1 CB:
  - ~12-14 KB of headroom;
  - bit-identical;
  - latency-neutral, because compute waits for the reader's first data, which needs the same plan.

## Mechanism
- `kernels/adaptive_chunk.hpp`:
  - `plan_page_bytes` / `store_plan` / `load_plan`, a port of r03-b01-a02.
  - `plan_groups` gets r04-b03-a03's tie pass verbatim. It applies when the best layout is uniform and leaves groups
    idle (active experts < groups). It then tries the mixed layouts under cap = best (ties allowed), and accepts one
    only if every active expert gets its own group and the summed expert cost is strictly lower.
  - The planner keeps its current -O2 build. Only the reader instantiates it now, so the TRISC/BRISC opt-level
    question disappears.
- Program factory: new 1-page CB `CB_PLAN_SCRATCH` (c_21, 32-64 B), passed to the reader, writer and compute as the
  named compile-time arg `cb_plan_scratch`.
- Reader:
  - Issue the `start` page read, then plan, then store and push the plan page, then barrier the start read.
  - Today the start read is barriered before planning, so the plan now reaches the writer and compute one DRAM
    latency sooner than the reader's plan used to be ready.
- Writer: `wait_front` the plan CB and `load_plan` instead of building `tiles_of` and running `plan_launch`.
- Compute: `wait_front` the plan CB, and UNPACK runs `load_plan` instead of `plan_launch`. The per-expert count
  scan (`tiles_of` for count_tiles) is unchanged.

The plan is a deterministic function of the same L1 counts that all three RISCs computed independently before.
Every code path (TILE x, bias, every activation, 0 tokens, any experts_per_chip) goes through the same hand-off,
and the numerics are unchanged: each expert still runs whole on one group with the same K order.

## Why this is not a repeat
- r04-b03-a03 introduced the tie pass but paid for its size with an -Os UNPACK planner on the critical path, which
  cost ~3 us on every shape. Here no planner copy is made slower; two are removed.
- r03-b01-a02 introduced plan-once on branch b01 of round 3. Its descendants never reached the current best
  lineage (r03-b02 -> r04-b02-a01), which still carries three planner copies.
- Neither is a DM-handshake / ack-timing change (the r04 dead end).

## Expected effect and risk
- minimax-m3 -3..-5% chip (r04-b03-a03's measurement), mistral ~0 (neutral there). The other 7 shapes keep their
  plan, maybe 1-2 us earlier first-GU from the overlapped start read. Geomean about +0.5..+0.8%. Plus ~9 KB of
  kernel-config headroom for later nodes.
- Risks:
  - The host rebuild (new CB + named args).
  - A hang if a consumer waits on a plan page that is never pushed. The reader pushes it unconditionally right
    after planning.
  - Wrong plan bytes would show as a hang or a PCC failure.
- How to read the eval:
  - PCC / max_abs / mean_abs bit-identical to the baseline on every shape.
  - minimax-m3 / mistral per-row ends show y10-11 working.
  - First-GU-start (zones2.py) on the others is equal or earlier.
