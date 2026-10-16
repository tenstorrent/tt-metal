# r04-b04-a01 result: 1.6002 (ok). Neutral: -0.6% vs the root r03-b02-a03 (1.6103), inside the ±1% noise band

## What happened vs expected
**First run: runtime_error on kimi-k2.7 only, repaired before scoring.**
- Error: `Program size (70896) too large for kernel config buffer (70656)`, 240 B over. This happens at
  program creation, before any launch.
- That version had the reader +1.55 KB larger, measured in a local -Os NCRISC compile (profiler build,
  13.0 → 14.6 KB). It contained:
  - both early acks;
  - an owner-side early x multicast (split issue/finish);
  - profiler-only `RE-R-EARLY` counters.
- The other 8 shapes ran and were bit-identical. That run's data is used below for the counters.
- Repair: drop the early owner multicast and the counters. The reader is now +0.89 KB over the root
  instead of +1.55 KB. This is the same precedent as r03-b02-a02.

**Second (scored) run:**
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the root and the baseline.
- No hang and no JIT error. Kernel-only, so no host build.

Chip-mean µs, root (r03-b02-a03) → this node, with MATH zones (mean core, /tmp/r04b04/zones2.py):

| shape | chip µs | Δ | mean core end | GU (mean core) | DN (mean core) |
|---|---|---|---|---|---|
| dsv3 | 514.0 → 515.7 | +0.3% | 493 → 495 | 325.2 → 325.2 | 149.8 → 150.7 |
| kimi-k2.7 | 729.5 → 733.7 | +0.6% | 719 → 720 | 471.6 → 468.2 | 223.6 → 228.5 |
| kimi-k3 | 1365.8 → 1358.1 | -0.6% | 1273 → 1261 | 725.7 → 714.2 | 522.5 → 522.4 |
| glm | 451.2 → 448.1 | -0.7% | 429 → 432 | 285.0 → 283.6 | 125.3 → 130.4 |
| mistral | 185.9 → 191.0 | +2.7% (noisy shape) | 166 → 169 | 106.2 → 109.6 | 49.9 → 49.5 |
| dsv4-flash | 316.9 → 322.4 | +1.7% | 298 → 298 | 181.5 → 183.0 | 106.2 → 105.0 |
| dsv4-pro | 1070.9 → 1074.6 | +0.3% | 1059 → 1059 | 688.5 → 688.0 | 357.6 → 358.1 |
| minimax-m3 | 365.5 → 366.0 | +0.1% | 344 → 341 | 226.3 → 225.8 | 112.8 → 110.0 |
| gpt-oss | 261.8 → 264.7 | +1.1% | 238 → 238 | 156.1 → 156.0 | 77.0 → 76.6 |

I expected -2..-5% GU on gpt-oss / mm3 / mistral / dsv4-flash. GU did not move on any shape outside
±3 µs. The exception is kimi-k3: -11 µs here, -20 µs in run 1, a real but small gain that its tail
absorbs.

## Why (profiler evidence)
1. **The mechanism fired a lot.** Run-1 `RE-R-EARLY` counters, as % of all core-blocks:

| shape | early x ack | early w ack | early owner x mcast |
|---|---|---|---|
| dsv3 | 30.8 | 14.7 | 1.5 |
| glm | 29.2 | 15.9 | 1.3 |
| kimi-k3 | 25.0 | 7.4 | 1.3 |
| mistral | 23.5 | 10.6 | 1.7 |
| dsv4-flash | 31.4 | 11.5 | 1.8 |
| dsv4-pro | 33.6 | 12.9 | 1.7 |
| minimax-m3 | 23.2 | 9.7 | 1.2 |
| gpt-oss | 18.9 | 20.6 | 0.7 |

   - In 1/4-1/3 of all blocks, a receiver's x landed before its weights. Its x ack for the next block
     then went out earlier than before; likewise for weights in 7-21% of blocks.
   - So the cross-stream coupling the proposal targeted is real and frequent.
2. **But removing it changed nothing.** GU per core is flat on 8 of 9 shapes in both runs.
   - So an x(kb+1) or w(kb+1) multicast that starts earlier does not shorten the block period. When a
     receiver acks early, the sender of the next block is not yet ready anyway:
     - the x owner is waiting for its compute's tilize (`cb_x_tl`) or for the slowest of the 10 row
       cores;
     - the weight writer is still doing its serial up read + multicast of the previous block.
   - Also, the compute needs both x(kb) and w(kb) to start block kb, so it is still gated by the
     later of the two.
   - The coupling only reorders who waits.
3. **Run 1 showed the DRAM-bound conservation again.** Where GU moved slightly, DN moved the other way:
   dsv3 DN +9, kimi-k3 GU -20 / DN +12, glm DN +7 µs. That is the same pattern as r03-b04-a03 and
   r02-b02-a03. Run 2 lost the early owner multicast and has DN back at the root's values.
4. **The early owner multicast (run 1 only) almost never applied** (≤ 1.8% of blocks). By the time an
   owner's x(kb) lands, its compute usually has not tilized x(kb+1) yet, or not all row acks are in.

## Classification
neutral (within noise). The coupling between the x and weight streams exists and is now removed, but
the gate/up block period is not set by it. The code is correct, bit-identical and safe:
- each receiver still holds at most one outstanding ack per stream;
- the sem protocol is unchanged;
- the TILE / legacy-up / bias variants compile.

It costs +0.9 KB of reader code; kimi-k2.7 now sits only a few hundred bytes under the kernel config
buffer. Prefer dropping it (revert `unified_routed_expert_ffn_reader.cpp` to the root) if a child needs
binary headroom.

## What a child of this node should try next
1. **Start from the root's reader (r03-b02-a03) unless you need nothing else in the reader.** This node
   buys nothing and eats ~0.9 KB of the h7168 headroom.
   - The h7168 programs are within ~0.3-0.5 KB of the 70656 B limit on this lineage (measured here: the
     +1.55 KB version overflowed by 240 B).
   - Any reader/compute growth needs a size cut first: r03-b01-a02's plan-once-per-core frees ~12 KB.
2. **Don't retry handshake/ack-timing changes in gate/up.** That covers this node, r03-b04-a02 (x
   trids), r03-b02-a02 (x owner), r01-b03-a01 / r01-b04-a01 (read-ahead) and r03-b03-a02 (NoC
   placement). The gate/up block period is set by the group's DRAM share on 8 of 9 shapes, plus the
   writer's serial up-read → multicast chain on the sender.
3. **What is left is in the non-GU parts:**
   - the max-mean tail (gpt-oss 26 µs, kimi-k3 97, mistral 21, dsv4-flash 24, all mostly in DN);
   - DN on the many-expert shapes: r03-b04-a04's down read before the ack is a measured -5..-31 µs of
     mean-core DN and is not on this lineage yet. Porting it is the cheapest real gain.
4. Tools:
   - /tmp/r04b04/cc/build.sh (+build2.sh with RSRC=<file>): local NCRISC reader compile with size
     check. Variants: RXRM=1/0 (row-major / TILE), RWM=1 (writer multicast), BIAS=1. LTO in the JIT
     adds ~±100 B vs this estimate.
   - /tmp/r04b04/zones2.py (NAMES env for a missing-shape run): MATH zones.
   - /tmp/r04b04/early.py: `RE-R-EARLY` counter parser for run-1 style records.
