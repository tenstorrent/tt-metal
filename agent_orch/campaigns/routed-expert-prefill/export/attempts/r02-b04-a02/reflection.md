# r02-b04-a02 result: 1.3612 (ok), a clear regression from the parent r02-b04-a01 (1.5314)

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs match the parent to every printed digit, so the
writer-owned down stream delivers exactly the same weight blocks into the same slots, with no hang.
The first device run failed to JIT-compile because of an unused `DRuns` typedef left in the reader
(`-Werror=unused-local-typedefs`). I removed the typedef and re-ran once; the result below is from
that run.

The speed went the wrong way. Per shape vs the parent (noise ±1%):

| shape | parent µs | this µs | change |
|---|---|---|---|
| kimi-k2.7 | 767.5 | 934.7 | **+21.8%** |
| dsv4-pro | 1131.8 | 1365.6 | **+20.7%** |
| dsv3 | 555.7 | 642.2 | **+15.6%** |
| glm | 486.3 | 555.7 | **+14.3%** |
| kimi-k3 | 1397.0 | 1571.8 | **+12.5%** |
| dsv4-flash | 322.4 | 362.4 | **+12.4%** |
| minimax-m3 | 388.8 | 429.8 | **+10.5%** |
| mistral | 192.6 | 202.6 | +5.2% |
| gpt-oss | 277.1 | 280.3 | +1.2% (noise) |

I expected -5..-15% on the few-expert shapes. Instead every DRAM-heavy shape lost 10-22%. The losses
are largest on the DRAM-bound shapes, and gpt-oss (compute/activation-bound) is unchanged.

## Why (profiler evidence)
**1. Phase split, measured for the first time.** These are the new MATH-thread zones (RE-GU = fused
gate/up K-loop, including tilize and SiLU copy; RE-ACT = activation / multiply phase; RE-DOWN = down
K-loop). Values are per active core per call in µs, mean of the 10 measured calls
(`/tmp/r02b04a02/zones.py` on reports/r02-b04-a02/.logs/profile_log_device.csv). Absolute values
are from this slower node. The ACT column is pure SFPU work, so it should carry over to the parent
almost unchanged.

| shape | experts/core | GU | ACT | DOWN | ACT per expert |
|---|---|---|---|---|---|
| dsv3 | 1.6 | 322 | 0.9 | 175 | ~0 (SiLU runs inside GU's copy pass) |
| kimi-k2.7 | 2.4 | 503 | 1.0 | 267 | ~0 |
| kimi-k3 (SiTU-GLU) | 5.6 | 745 | **221** | 488 | **~40 µs** (18 tiles at pcM 2, ~3k cycles/tile) |
| glm | 1.6 | 278 | 0.9 | 148 | ~0 |
| mistral | 1.0 | 122 | 0.5 | 55 | ~0 |
| dsv4-flash (clamped SiLU-GLU) | 2.0 | 187 | 31 | 110 | ~15 µs |
| dsv4-pro (clamped SiLU-GLU) | 3.2 | 632 | 46 | 477 | ~14 µs |
| minimax-m3 (SwiGLU-OAI) | 1.0 | 218 | 39 | 152 | ~39 µs (27 tiles, ~1.9k cycles/tile) |
| gpt-oss (SwiGLU-OAI + bias) | 1.0 | 135 | **55** | 64 | **~55 µs** (27-36 tiles) |

- Down is ~1/3 of the matmul time and gate/up ~2/3, in proportion to their MACs and weight bytes.
  Neither phase is dominated by a fixed handshake cost.
- The ACTIVATION is serial SFPU time and is large on exactly the shapes with a big "fixed cost F":
  gpt-oss ~21% of a core's busy time (55 of 254 µs), kimi-k3 ~15% (221 of 1454), minimax-m3 ~9%,
  dsv4-flash/pro ~5-9%. For SiLU shapes it is hidden inside GU (silu_tile in the partials copy pass).
  This is a big part of the gpt-oss F ~150 µs the parent measured, and it scales with per_core_M,
  not with the expert count.

**2. Why the regression.** I have no zones for the parent, so this is inferred:
- Row end times (rows.py) moved up by the same fraction on heavy and light groups: dsv3 534 -> 619
  and 300 -> 370; dsv4-pro 1113 -> 1358 and 750 -> 900. That is a uniform drop in stream rate, not
  a tail effect.
- The shapes that lost most are the DRAM-bound ones (kimi-k2.7, dsv4-pro, dsv3, glm). gpt-oss, the
  least DRAM-bound shape, did not move. So the change made weight delivery less efficient overall. Two
  things changed on the sender side:
  (a) All down-weight DRAM reads moved from NoC 0 (the reader read half) to NoC 1, where the
      `up` reads, the gate/up multicast, the down multicast and the output writes already go. The
      op previously kept weight reads on NoC 0 on purpose (`noc_read(0)`).
  (b) One RISC now does read -> barrier -> ready-wait -> multicast for the whole block, serially.
      Before, the two RISCs read halves concurrently, and the factory comment measured the split
      at 1.13x vs 1.03x for a single-RISC handoff.
- The run-ahead itself also started the down reads during gate/up. Those reads compete for DRAM
  and NoC 1 with the same group's gate/up stream (`up` reads and multicast), which is the critical
  phase. The down blocks then wait in the ring anyway.
- The premise was wrong. The down phase is ~1/3 of time and proportional to its MACs and bytes, so
  the act/weight coupling in the reader was not a measurable bottleneck. Decoupling gained nothing
  and cost delivery efficiency.

## Classification
flawed idea. The down-weight / activated coupling in phase 4 is not a bottleneck. Moving the down
stream to the writer on NoC 1 costs 10-22% of delivery efficiency on the DRAM-bound shapes. The
implementation is correct (bit-identical, no hang). The measurement zones are the useful part of
this node.

## What a child of this node should try next
- **Start from r02-b04-a01 (the parent), not this node.** If you want the phase zones, copy only
  the `RE_PHASE_ZONE` macro and the three zone scopes in compute/fused_swiglu.cpp. They are
  numerically inert and cheap (3 zones per expert on MATH).
- **Activation is the measured, untried lever:** ~55 µs per expert on gpt-oss, ~40 on kimi-k3, ~39 on
  minimax-m3, ~15 on dsv4. It is serial MATH-thread SFPU work between gate/up and down, and no DM is
  in flight while it runs. Options, without touching approximation modes or formats:
  (a) overlap it with the down phase: down K-block kb needs only column kb's activated tiles, so
      activate in an order that lets core kb push its block first, or interleave the activation
      per down block;
  (b) run the SFPU activation on the PACK thread, as moe_gpt does for swiglu, so it overlaps the
      next subblock's FPU work;
  (c) for SiTU-GLU (~3k cycles/tile), check whether the tile op re-inits or reloads constants per
      tile.
  Expected: up to -15..-20% on gpt-oss and kimi-k3 if fully hidden.
- Don't move weight DRAM reads off the reader's NoC 0, and don't serialise a whole block's read and
  multicast on one RISC. If the down stream is revisited, keep the reader/writer split read and only
  decouple the receivers' ready acks, though this node suggests there is little to gain.
- Caveat for anyone reusing this code: the down multicast moves to a NoC-1 worker multicast, which
  is the fabric hazard class the factory comment describes for IN1_WRITER_MCAST, and it is not
  disabled by DS_NO_WRITER_MCAST.
