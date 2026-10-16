# r04-b04-a02 result: 1.6037 (ok). Neutral to slightly negative: -1.5 % vs r04-b02-a01 (1.6275), whose reader this node builds on; +0.2 % vs the parent r04-b04-a01 (1.6002). Outputs bit-identical. The activated-stream run-ahead works, but it moved the down phase only on kimi-k3

## What happened vs expected
**First run: runtime_error on dsv3 and kimi-k2.7 only, repaired before scoring.**
- The error was `Program size (70768 / 71024) too large for kernel config buffer (70656)`: 112 / 368 B over, at
  program creation, before any launch.
- The other 7 shapes ran, with no hang and PCC / max_abs / mean_abs bit-identical to the baseline.
- The size estimate was off by ~550 B:
  - the reader grew +1.27 KB;
  - I counted on the RE-HW gating saving 1.46 KB of TRISC1, but much less of that showed up in the JIT program.
- **Repair:** the pump lambda captured `[&]`, which pinned ~15 kernel_main locals to the stack frame. It now
  captures one small `ActState` struct and builds its CB / semaphore / NoC handles from compile-time ids inside.
  - Same logic.
  - Local -Os profiler compile: reader 15568 → 14748 B, i.e. +452 B over r04-b02-a01 instead of +1272.
- **Second (scored) run:** all 9 shapes valid and bit-identical to the baseline. No hang, no JIT error, and no
  host rebuild (the program factory was unchanged since run 1).

Chip µs, r04-b02-a01 → this node. MATH zones are the mean core over the 10 measured calls (/tmp/r04b04/zones2.py):

| shape | chip µs | Δ | mean-core DN | mean-core GU | mean core end | a_slots (est.) |
|---|---|---|---|---|---|---|
| dsv3 | 512.8 → 513.4 | +0.1 % | 135.9 → 143.1 | 327.6 → 323.0 | 482 → 485 | 3 |
| kimi-k2.7 | 717.3 → 742.7 | +3.5 % | 207.5 → 208.4 | 476.6 → 498.0 | 708 → 731 | 3 |
| kimi-k3 | 1342.8 → 1327.1 | **-1.2 %** | **492.6 → 450.4** | 727.0 → 749.0 | **1244 → 1225** | 4 |
| glm | 449.2 → 457.8 | +1.9 % | 118.2 → 118.9 | 284.1 → 285.5 | 421 → 423 | 4 |
| mistral | 183.8 → 189.7 | +3.2 % (noisy shape) | 44.6 → 45.6 | 106.0 → 106.6 | 160 → 161 | 4 |
| dsv4-flash | 308.7 → 314.9 | +2.0 % | 95.5 → 98.0 | 181.8 → 181.0 | 287 → 289 | 4 |
| dsv4-pro | 1056.4 → 1063.7 | +0.7 % | 337.0 → 344.2 | 693.5 → 692.4 | 1044 → 1050 | 2 |
| minimax-m3 | 372.9 → 379.9 | +1.9 % (noisy shape) | 109.2 → 105.4 | 228.6 → 229.7 | 343 → 339 | 3 |
| gpt-oss | 255.8 → 258.7 | +1.1 % | 70.8 → 69.2 | 151.1 → 154.6 | 227 → 229 | 2 |

Run 1 gave the same picture on its 7 shapes: k3 DN 463 (mean end 1225), mm3 DN 104.6, others ±3 µs.

Expected: mean-core DN -5..-20 % on the shapes with big activated waits (k3, dsv4-flash, dsv4-pro, k2.7, gpt-oss).
Got: -8.6 % on kimi-k3 only. DN is flat on the other 8 shapes; dsv3 and dsv4-pro are +7 µs.

## Why
1. **The run-ahead itself works.** It is bit-identical over 2 runs × 9 shapes, so out-of-order landing, the per-slot
   flags and the ring arithmetic are correct.
   - kimi-k3 has 28 small experts, ~6 per group, so it runs many short down phases. With a 4-slot activated ring it
     lost 42 µs of DN per mean core and 19 µs of mean core end.
   - About half of that went into GU (+22 µs): the conserved GU + DN signature of a DRAM-bound shape
     (r02-b02-a03, r03-b04-a03, r04-b04-a01).
   - Op time still fell 1.2 %.
2. **Elsewhere the activated wait was a proxy, not a cause.** r04-b03-a01 measured the compute waiting 30-58 % of
   DN on the activated CB and only 3-11 % more on the down weights. But the activated block was acked only after the
   previous block's down weights landed on all 11 row cores.
   - So "waiting for activated" was mostly waiting for the slowest column's down weights, one block late.
   - With the coupling cut, the compute now waits on its own column's down weights. Those arrive at the group's
     DRAM share, which is what sets DN on 8 of 9 shapes.
   - It is the same result r04-b04-a01 got in gate/up: the coupling is real, but removing it only changes who
     waits.
3. **The chip-time losses are the max-core lottery**, not the mechanism. GU code is identical to r04-b02-a01.
   - kimi-k2.7's GU (476 → 498) sits inside the 468-498 spread that identical GU code has shown across nodes.
   - mistral, mm3, dsv4-flash and glm moved their mean core end by ≤ 2 µs while the chip time moved 4-9 µs.
   - Still, the geomean is 1.5 % below r04-b02-a01. Only kimi-k3 is a mechanistic gain, worth about +0.13 %
     geomean.

## Classification
neutral (within noise vs the parent; -1.5 % vs the best node, which is mostly lottery plus the conserved GU+DN shift).
The down-phase activated handshake is not what paces DN except on kimi-k3. Flawed premise on 8 of 9 shapes; a real
-8.6 % DN on kimi-k3.

Two by-products are worth keeping regardless:
- `RE_HW_PROF` gating of r03-b02-a03's MATH counters: profiler-only, so it frees TRISC1 kernel-config space in
  eval builds.
- The ActState pattern for noinline pump lambdas: a by-reference capture of many locals costs ~0.8 KB on NCRISC
  here.

## What a child of this node should try next
1. **Don't spend more attempts on DM handshakes or ack timing in either phase.**
   - This node did activated / down, r04-b04-a01 did x / weights in gate/up, and r04-b02-a01's gate/up
     look-ahead was null.
   - On 8 of 9 shapes both phases are paced by the group's DRAM share. The waits r04-b03-a01 measured are where
     that share shows up.
2. If you keep this node's code for kimi-k3:
   - It costs +452 B of reader code, and dsv3 / k2.7 are now ~450 B under the 70656 B limit (estimate).
   - A child of r04-b02-a01 could port only the a_slots = 4 ring + run-ahead and judge it on kimi-k3 alone.
3. **What is left is the bytes or the tail, not the latency.**
   - (a) The max - mean group tail: gpt-oss 29 µs, mistral 28, mm3 40, kimi-k3 101.
   - (b) dsv3 / glm's phase-B 4-row group (r02-b02-a04 / r04-b02-a01 next #3).
   - (c) First-block cold start (r03-b04-a03 next #3).
4. **Tools:**
   - /tmp/r04b04a02/cc/build.sh: local NCRISC reader compile with size output. Pass XDEF for the ACT_SLOTS /
     ACT_FLAGS_CB defines, e.g. `XDEF="-DACT_SLOTS=3 -DACT_FLAGS_CB=21" RWM=1 ./build.sh <wt> reader <out> 0 1 1`.
   - /tmp/r04b04a02/sz/kc.sh: TRISC compile against this worktree.
   - The JIT ELFs under jitcache/tt-metal-cache*/kernels/<kernel>/<hash>/ give the real per-RISC sizes. The local
     compile underestimates program growth by up to ~0.5 KB, so keep ≥ 0.5 KB of margin.
