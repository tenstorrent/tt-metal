# r03-b01-a04 result: 1.5952 (ok). Neutral: +0.2% over the parent r03-b01-a03 (1.5917), inside the ±1% noise band. Swapping the streamed operand (x through SrcB instead of the bf4 weights through SrcA) leaves the per-tile-mult cost exactly where it was

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline on every
  printed digit, so these all check out:
  - the reuse-B calls (ct = 1, rt = per_core_M);
  - the dst -> partials slot mapping;
  - the TILE_SIZE_B * KT_DIM x-row stride.
- First and only device run: no JIT error, no hang. The h7168 programs still fit the kernel config buffer.
- I expected the compute-bound shapes to drop 5-20% (mistral, minimax-m3, gpt-oss). They did not move.

| shape | chip µs parent -> this | Δ | MATH GU mean core | DN mean core | mean / crit core end |
|---|---|---|---|---|---|
| mistral (pcM 3) | 193.1 -> 189.6 | -1.8% | 106.8 -> 106.3 | 52.6 -> 53.2 | 170 -> 170 / 192 -> 188 |
| minimax-m3 | 374.1 -> 367.0 | -1.9% | 228.5 -> 227.6 | 109.1 -> 112.9 | 342 -> 345 / 373 -> 366 |
| gpt-oss (pcM 3-5) | 267.5 -> 270.6 | +1.2% | 156.0 -> 154.4 | 75.3 -> 79.1 | 237 -> 239 / 266 -> 270 |
| dsv3 | 519.7 -> 524.2 | +0.9% | 331.7 -> 328.9 | 145.8 -> 147.2 | 496 -> 494 / 518 -> 522 |
| glm | 451.9 -> 456.4 | +1.0% | 287.0 -> 290.0 | 127.0 -> 126.4 | 432 -> 434 / 450 -> 455 |
| kimi-k2.7 | 732.9 -> 736.3 | +0.5% | 475.5 -> 485.5 | 220.4 -> 218.7 | 719 -> 727 / 731 -> 734 |
| kimi-k3 | 1359.2 -> 1364.6 | +0.4% | 711.3 -> 708.3 | 522.5 -> 533.2 | 1254 -> 1262 / 1358 -> 1363 |
| dsv4-flash | 316.2 -> 314.0 | -0.7% | 177.4 -> 181.2 | 105.5 -> 103.6 | 292 -> 294 / 315 -> 313 |
| dsv4-pro | 1084.1 -> 1068.2 | -1.5% | 696.1 -> 693.7 | 358.7 -> 353.8 | 1066 -> 1059 / 1082 -> 1066 |

(MATH zones from /tmp/r02b01a02/zones.py on reports/r03-b01-a03 and reports/r03-b01-a04, mean of the 10 measured
calls.) Every Δ is the usual group-tail lottery (crit core moves ±4-16 µs while the mean core moves ≤ 8 µs).

## Why (profiler evidence)
- **The reuse-B path did run on the compute-bound shapes.** The plan is bit-identical to r02-b02-a04's planner
  (this lineage), and r03-b02-a03 / r03-b03-a03 measured those plans at per_core_M 3 on mistral's four 2-row groups
  and 3-5 on gpt-oss. `use_weight_reuse(3..5)` is true with the bf16 dst (half = 8 tiles), so every non-last
  gate/up K-block and every down K-block on those cores used ct = 1, rt = per_core_M.
- **And the per-mult cost is unchanged.** mistral GU, per core:
  - 8 K-blocks × (3 rows × 6 cols × 16 K-tiles × 2 matmuls) = 4608 tile-mults.
  - 106.3 µs × 1350 MHz = 143.5k cycles, i.e. **31.1 cycles/mult**.
  - That is the same 31.0 that r03-b02-a03's counters measured with 1 x 6 subblocks.
  - minimax-m3 GU (227.6 vs 228.5) and DN on both shapes are flat the same way.
- So r03-b02-a03's reading was wrong. "The bf4 SrcA unpack (17.5 cycles) doesn't overlap the FPU (16 cycles)" was
  taken to be specific to unpacker 0 / bf4. It is not: streaming a bf8 x tile into SrcB with the weight tile reused
  in SrcA costs exactly the same ~15 extra cycles per mult. Whatever serialises unpack and FPU here acts **per
  streamed tile, whichever Src it goes to**.
- The halved work per matmul_block call did not hurt either: 3-5 mults per call instead of 6 (pcN 6) or 3 (pcN 9),
  same per-mult cost. So the per-call TRISC overhead (context switch, address config, semaphores) is not the
  limiter. The time is set by a per-tile unpack-FPU hand-off.

## Classification
flawed idea (measured null; the code is correct and bit-identical). Changing which operand is streamed does not
change the ~31 cycles/tile-mult on BH in this kernel. It is harmless to keep (per_core_M <= 2 and the last
gate/up block are untouched), but it buys nothing. A child may keep it or drop it (set kReuseWeightMinM above 8).

## What a child of this node should try next
1. **Don't retry operand swaps, subblock shapes or reuse modes** (r01-b02-a04 width, this node's rt/ct swap). With
   the LLK matmul MOP every tile-mult unpacks one streamed tile, and here that unpack serialises with the FPU in
   both modes.
2. **If someone wants the 2x on compute-bound shapes, it has to come from the unpack -> FPU hand-off itself.**
   Measurement first: port r03-b02-a03's re_hwprof counters (kernels/compute/re_hwprof.hpp on branch b02) onto
   this node and read UNP1 (SrcB) busy and FPU busy in reuse-B mode. Expect UNP1 ~50% and FPU ~50%, i.e. the mirror
   image of r03-b02-a03's table.
   - Candidates to check in the LLK, in this order:
     - the unpack MOP's per-tile `STALLWAIT(STALL_CFG, THCON)` + RDCFG/WRCFG base-address bump: does the next
       UNPACR wait for the previous one to finish?
     - whether the Src bank double-buffering is active at all in this config (counter sels 27/28, "waiting for
       SrcA/SrcB clear");
     - whether MATH's per-tile `set_dst_write_addr` (SETC16 on a register the MVMULs use) drains the FPU.
   - Any fix would be an op-local copy of the LLK matmul unpack/math (allowed only if copied into the op dir).
     That is risky; size it with the counters before spending a node on it.
3. Otherwise go where the history says the waste is: the max-mean group tail (20 µs mistral, 33 gpt-oss, 85
   kimi-k3, mostly in DN; r03-b04-a03 next #2: issue the down read before the in1_ready ack) and the start-up fill
   (x(0) owner's 96 one-KB stick reads, r03-b04-a03 next #3).
