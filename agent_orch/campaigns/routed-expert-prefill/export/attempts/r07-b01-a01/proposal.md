# r07-b01-a01: combine r05-b01-a01 (plan once per core + idle-row tie pass) with r06-b04-a01's wrap-free down multicasts

## Motivation
- The best node r06-b04-a01 (1.6931) and r05-b01-a01 (1.6457, +1.1% over their common ancestor r04-b02-a01) sit on
  different branches of the r04-b02-a01 root. Their changes are on different axes:
  - r06-b04-a01 (via r05-b02-a01): weight delivery on the NoC (60/40 NoC split, wrap-free down multicasts with corner
    senders). Its gain is DN -7..-14% on every shape.
  - r05-b01-a01: the plan. The reader plans once and publishes the plan in an L1 page (writer and compute UNPACK
    lose their planner copies). The freed kernel-config space carries r04-b03-a03's idle-row tie pass, so
    minimax-m3 / mistral run 3+3+2+2 (rows y10-11 work) instead of 2x5 with 22 cores idle. Measured mm3 -5.9%,
    mistral -4.1%, others flat, bit-identical.
- On r06-b04-a01, mistral and minimax-m3 still run 5x2 with y10-11 idle (r06-b04-a01 reflection, point 1). So the
  tie pass's gain is not on the best lineage yet.
- r06-b04-a01's reflection, next #2, names exactly this port, and next #4 says the lineage has "only a few hundred
  bytes of headroom left on h7168 without plan-once". Its writer planner already had to go to -Os to fit. Plan-once
  removes two planner copies (~4-5 KB each), which also unblocks next #1 (gate/up wrap-free multicasts on the reader,
  whose multicast path is compiled out today for size).

## Mechanism
Port r05-b01-a01's code diff onto r06-b04-a01 (5 files):
- `kernels/adaptive_chunk.hpp`: `plan_page_bytes` / `store_plan` / `load_plan`; r04-b03-a03's tie pass in
  `plan_groups`. Resolve the conflict with r06-b04-a01's pragma block: the writer (BRISC) no longer instantiates the
  planner, so its `-Os` branch becomes moot; the planner is built -O2 on the reader only (as r05-b01-a01).
- Program factory: 1-page `CB_PLAN_SCRATCH` (c_21) + named compile-time arg `cb_plan_scratch` to all three kernels.
- Reader: issue the `start` page read, plan, store + push the plan page, then barrier the start read.
- Writer: `wait_front` the plan CB + `load_plan` instead of planning.
- Compute: UNPACK `load_plan` instead of `plan_launch`.
- Check that r06-b04-a01's corner-sender placement (`sender_row_in_group`) and x-owner key cover the tie pass's
  3-row groups (3+3+2+2 already exists as a mixed layout for dsv4-flash on this lineage, so the host placement and
  the reader/writer predicates handle it).

## Why this is not a repeat
- A combination of two measured wins on different branches; neither is on the other's lineage. r05-b01-a01 was the
  port of r03-b01-a02 + r04-b03-a03 onto r04-b02-a01; this is its port onto r05-b02-a01 + r06-b04-a01's NoC changes.
- Not a DM handshake / ack-timing change, not a NoC split change.

## Expected effect and risk
- minimax-m3 -3..-6%, mistral -2..-5% (their 3-row groups now get corner senders, so r06-b04's DN gain should
  stack). Other 7 shapes: same plans, first-GU-start +0.2..+1.5 µs (compute waits on the reader's plan page), i.e.
  flat. Geomean about +0.7..+1.2%.
- Kernel-config headroom: several KB back on the h7168 programs.
- Risks: host rebuild (new CB); a hang if a consumer waits on a plan page never pushed; a conflict in the
  x-owner/pragma region of adaptive_chunk.hpp resolved wrongly (would show as a hang or PCC change).
- Reading the eval: bit-identical PCC / max_abs / mean_abs; mm3 / mistral per-row ends show y10-11 busy.
