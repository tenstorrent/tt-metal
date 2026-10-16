# r03-b01-a02 result: 1.5882 (ok). Repair works; ties the best node r03-b02-a01 (1.5887, inside ±1% noise)

## What happened vs expected
- All 9 shapes ran and passed. PCC / max_abs / mean_abs are bit-identical to the baseline and to r03-b02-a01 on
  every printed digit, so the shared plan is the same plan the three RISCs used to compute independently.
- The h7168 shapes that overflowed the kernel config buffer in the parent (dsv3, kimi-k2.7, dsv4-pro) now run.
- Host rebuild (new CB + named compile-time args) took 20 s, no JIT errors, no hang on the first and only run.
- I expected ~1.575-1.585 (the combination without a planner trim). Got 1.5882, tied with r03-b02-a01, which is
  the same combination plus a planner trim.

Per shape (µs chip mean) vs r03-b02-a01 (same code minus its planner trim, plus this node's plan sharing):

| shape | r03-b02-a01 | this | Δ |
|---|---|---|---|
| dsv3 | 517.3 | 521.4 | +0.8% |
| kimi-k2.7 | 735.9 | 735.1 | -0.1% |
| kimi-k3 | 1374.7 | 1379.2 | +0.3% |
| glm | 462.3 | 452.4 | **-2.1%** |
| mistral | 192.4 | 195.1 | +1.4% (noisy shape) |
| dsv4-flash | 322.4 | 322.6 | 0 |
| dsv4-pro | 1075.9 | 1079.4 | +0.3% |
| minimax-m3 | 372.6 | 369.9 | -0.7% |
| gpt-oss | 262.0 | 263.0 | +0.4% |

## Why (evidence)
**Binary size.** Largest ELF (.text+.data) per RISC in this run's JIT cache:
reader (NCRISC) 15.9 KB, writer (BRISC) 4.5 KB, TRISC0 13.4 KB, TRISC1 10.5 KB, TRISC2 13.0 KB, ~57 KB total.
TRISC0 was 24.3 KB in the parent (-10.9 KB). The writer lost its -Os planner copy (~3.6 KB standalone).
The h7168 programs were 71.2-72.8 KB against a 70.66 KB limit. They now have ~12-14 KB of headroom, so
later nodes can add code (SiLU body on PACK, bigger planner) without hitting the limit again.

**Planning latency is not on the compute critical path.** MATH zones (/tmp/r02b01a02/zones.py), mean core:

| shape | first-GU-start this / b02-a01 | first GU block this / b02-a01 | sum this / b02-a01 |
|---|---|---|---|
| kimi-k3 | 45.8 / 17.2 | 151.8 / 166.8 | 197.6 / 184.0 |
| dsv4-flash | 16.4 / 7.4 | 93.6 / 99.7 | 110.0 / 107.1 |
| kimi-k2.7 | 14.6 / 6.7 | 186.1 / 190.9 | 200.7 / 197.6 |
| dsv3 | 11.0 / 5.6 | 197.9 / 199.0 | 208.9 / 204.6 |
| mistral / mm3 / gpt-oss | 7.1-7.7 / 3.4-3.9 | | |

- MATH now starts only after the reader's plan is published, so first-GU-start includes the reader's untrimmed
  planner. kimi-k3 at 45.8 µs matches r03-b04-a01's ~40 µs model of the untrimmed planner on NCRISC.
- The first GU block got shorter by almost the same amount: compute was already waiting on the reader's first
  weight multicast, which needs the same plan. Before, compute planned in parallel and then waited for data.
- Net cost of the untrimmed planner vs b02-a01's trim is ~3-14 µs on the sum. That matches b02-a01/b04-a01's
  measured trim gains (kimi-k3 ~12, others 2-5). It is hidden in noise here: kimi-k3 mean end 1275 vs 1271.
- glm -2.1% and dsv3 +0.8% are the same tail lottery seen across nodes with this planner (glm mean core 431 vs 440).

## Classification
win as a repair (the parent's runtime_error is fixed, score 0 -> 1.5882). Neutral vs the campaign best
(-0.03%, inside noise). The mechanism itself (plan once per core, share through L1) is latency-neutral by
design. Its value is ~12-14 KB of kernel-binary headroom and a single planner that is the only one worth
optimising.

## What a child of this node should try next
1. **Stack a planner trim on top (expect +0.5-1% via kimi-k3 / dsv4-flash / dsv3).** r03-b04-a01's bit-identical
   trim (byte cache + largest-first order, uniform layouts once, pruned infeasible mixed layouts, start-read
   overlap) or r03-b02-a01's simpler one. Only the reader runs the planner now, so the trim goes in one place.
   With planning on one RISC, the writer and UNPACK must stay plan-free. Judge by first-GU-start (45.8 µs on kimi-k3 here).
2. **Use the headroom:** port r02-b04-a04's SiLU body (`re_act::silu_act`) to PACK for dsv3/glm/kimi-k2.7/
   mistral (RE-ACT still 5-12 µs per expert there). Before this node that risked another overflow.
3. Optional: plan once per CHIP. The counts-reader core plans and appends the 8+2E-byte plan page to its counts/idx
   multicast, so 109 cores skip planning. It is latency-neutral unless the counts reader is faster than the others.
   Only worth it if a much heavier, better planner (e.g. sub-expert N-splits for the 4-expert shapes) is added later.
4. Don't retry: DM look-ahead / deeper rings, request VCs, read-ahead, writer-owned down stream.
