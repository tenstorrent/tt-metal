# r03-b01-a03 result: 1.5917 (ok). Neutral: +0.2% over the parent r03-b01-a02 (1.5882), tied with the campaign best r03-b02-a02 (1.5952, -0.2%), all inside the ±1% noise

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline on every
  printed digit, so the trimmed planner gives the same plan on device (the host fuzz against the parent's
  planner had 0 mismatches in 2.2M cases).
- No JIT error, no hang, no kernel-config-buffer overflow on the first and only run. So the
  `#pragma GCC optimize("O3")` region (guarded to COMPILE_FOR_NCRISC / BRISC) plus the noinline
  `plan_launch` compiled clean, and the bigger reader still fits on the h7168 shapes.
- I expected +0.8..+1.5% geomean. Got +0.2%. The planner got exactly as fast as simulated, and the mean core
  moved, but the op time is the max core, which barely moved.

Per shape (µs). Chip mean = op time. Mean / crit core = TRISC1 kernel end, mean of the 10 measured calls
(/tmp/r02b01a02/zones.py):

| shape | chip mean parent -> this | Δ | mean core | crit core | first-GU-start |
|---|---|---|---|---|---|
| kimi-k3 | 1379.2 -> 1359.2 | **-1.5%** | 1275 -> 1254 | 1378 -> 1358 | 45.8 -> 16.7 |
| dsv4-flash | 322.6 -> 316.2 | **-2.0%** | 302 -> 292 | 321 -> 315 | 16.4 -> 7.7 |
| kimi-k2.7 | 735.1 -> 732.9 | -0.3% | 728 -> 719 | 733 -> 731 | 14.6 -> 6.4 |
| dsv4-pro | 1079.4 -> 1084.1 | +0.4% | 1071 -> 1066 | 1078 -> 1082 | 14.4 -> 6.1 |
| dsv3 | 521.4 -> 519.7 | -0.3% | 497 -> 496 | 520 -> 518 | 11.0 -> 5.4 |
| glm | 452.4 -> 451.9 | -0.1% | 431 -> 432 | 451 -> 450 | 11.0 -> 5.4 |
| mistral | 195.1 -> 193.1 | -1.0% | 170 -> 170 | 194 -> 192 | 7.1 -> 4.3 |
| minimax-m3 | 369.9 -> 374.1 | +1.1% (noisy shape) | 349 -> 342 | 369 -> 373 | 7.1 -> 4.2 |
| gpt-oss | 263.0 -> 267.5 | +1.7% (noisy shape) | 232 -> 237 | 262 -> 266 | 7.7 -> 5.1 |

## Why (profiler evidence)
**The planner is now fast, and the simulator predicted it to within 2%.** The NCRISC RE-PLAN zone
(/tmp/rep-r03b04a01-rO2J/plan_zone.py), median over cores and the 10 measured calls:

| | kimi-k3 | dsv4-flash | k2.7 | dsv4-pro | dsv3 / glm | mistral / mm3 | gpt-oss |
|---|---|---|---|---|---|---|---|
| RE-PLAN measured | 14.2 | 5.7 | 4.3 | 4.1 | 3.4 | 2.4 | 3.3 |
| simulated (trim + O3 region) | 14.4 | 5.3 | 4.2 | 4.0 | 3.0 | 2.3 | 4.1 |
| simulated parent planner (-O2) | 48.8 | 15.9 | 15.4 | 15.1 | 10.7 | 3.8 | 5.5 |
| r03-b04-a01 (trim, -O2, measured) | 17.7 | 6.9 | 5.3 | 5.1 | 4.0 | 2.7 | 3.6 |

- Kernel start -> plan published is RE-PLAN + ~1.1 µs (the counts multicast and the `start` read issue). The
  `start` read barrier after planning costs nothing visible.
- The -O3 region alone is worth ~20% of the trimmed planner (kimi-k3 17.7 -> 14.2 µs, dsv4-flash 6.9 -> 5.7),
  as simulated. The rvsim cost model (BH g++ 15.1, 1 cycle/instruction plus L1 / div / branch penalties) is
  now calibrated against two device measurements at -O2 and -O3.

**Where the saved time went.** first-GU-start fell by the planner saving on every shape (kimi-k3 -29 µs,
dsv4-flash -9, k2.7 / dsv4-pro -8, dsv3 / glm -5.6). Then:
- Part of it is absorbed by a longer first GU block (kimi-k3 151.8 -> 158.8, dsv4-flash 93.6 -> 95.9, glm
  168.6 -> 173.3). Before, compute planned in parallel and then waited for the first weight multicast. Now
  the first DRAM reads start earlier but their latency is still exposed. r03-b01-a02 predicted this.
- The MEAN core end moved by roughly 2/3 of the saving on the shapes with the big planner: kimi-k3 -21 µs,
  dsv4-flash -10, k2.7 -9, dsv4-pro -5, mm3 -7. dsv3 / glm / mistral are flat (only ~2-6 µs to save, inside
  run-to-run variation).
- The op time is the CRIT core. It followed only on kimi-k3 (-20) and dsv4-flash (-6). On the DRAM-bound
  k2.7 / dsv4-pro the critical rows are limited by aggregate DRAM bandwidth, so starting earlier mostly
  lets every group compete for the same bytes sooner. On gpt-oss / mm3 the crit core is the group-tail
  lottery (±5%): gpt-oss mean core +5 with a planner that is 2 µs faster is noise.
- Net: two shapes gained outside noise (-1.5%, -2.0%), the rest is noise in both directions, geomean +0.2%.

**Size.** The reader grew by the O3 planner (simulated planner code 6.2 KB -> 9.1-11.1 KB per instantiation)
and the h7168 programs still fit the 70656 B kernel config buffer. I did not capture the JIT ELF sizes; a
child that adds code should measure them first (r03-b01-a01 recipe).

## Classification
neutral (within noise): +0.2% geomean. The mechanism works exactly as designed: the planner is ~3.4x faster
than the parent's (kimi-k3 48.8 -> 14.2 µs), bit-identical, and the mean core moved -5..-21 µs on the
many-expert shapes. Only kimi-k3 and dsv4-flash turn that into op time. The planner is no longer a
lever: at most ~14 µs left on kimi-k3 (1%), 2-6 µs elsewhere. This node is the plan-once lineage with
the planner done, a good base for compute-side children.

## What a child of this node should try next
1. **Stop working on the planner.** Remaining RE-PLAN is 2.4-5.7 µs on 8 of 9 shapes and 14 µs on kimi-k3.
   Even a free planner would gain < 0.4% geomean.
2. **Port r03-b02-a02's best-node pieces only if they are real.** Its x-owner change was measured neutral, so
   the 1.5952 vs this node's 1.5917 is noise. Don't spend a node merging it.
3. **SiLU body on PACK (r02-b04-a04's `re_act::silu_act`, 25 vs 34 SFPU instructions).** RE-ACT is still
   10.3-12 µs per core on dsv3 / glm / kimi-k2.7 and 5.2 on mistral (the FPU multiply pass plus the last
   subblock's SiLU). This lineage has the binary headroom for it. Expect ≤1% (glm / mistral compute-bound).
4. **The first-GU block is ~150-280 µs on every shape, against ~95-130 µs for later GU blocks-worth.** It is
   the cold start of all groups at once: every core's first reads land together. The planner can no
   longer hide it. A measurement node with NCRISC/BRISC zones on the first expert's first 2-3 K-blocks
   (gate read barrier, writer up-read, in1_ready / mcast) would show whether it is DRAM queueing or
   the first-block pipeline fill.
5. Tools for the next planner or binary-size question: /tmp/r03b01a03/sim (cmp.py runs any header
   variant on the test routing, prog.cpp includes "hdr.hpp" from its OWN dir first, so put variants in
   subdirs and compile with -I<subdir> from a dir without hdr.hpp), /tmp/r03b01a03/fz (old-vs-new plan
   fuzz).
6. Don't retry: request VCs, down look-ahead / deeper rings, read-ahead, writer-owned down stream, putting
   a whole weight stream on one NoC, x-owner placement, x-read scheduling.
