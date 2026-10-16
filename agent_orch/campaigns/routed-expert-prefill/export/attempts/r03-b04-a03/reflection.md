# r03-b04-a03 result: 1.6010 (ok). Neutral: +0.36% over the code it builds on (r03-b02-a02, 1.5952), inside the ±1% noise band

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to r03-b02-a02 and the baseline on every printed
  digit, so these all check out:
  - the two-block subblock walk keeps each tile's L1_ACC order;
  - the wrapped ring offsets on UNPACK address the right block;
  - the split activation parts hand off correctly.
- No hang. Kernel-only, so no host build. The h7168 shapes fit the kernel config buffer (+1.0-1.5 KB of compute code
  in a local compile).
- This node also brings branch b04 onto r03-b02-a02's op code. Its four kernel files are copied from that node, and
  the b04 planner trim / x-trid reader are dropped. So r03-b02-a02 is the right comparison, not the parent
  (1.5761).
- I expected +1-2% geomean, with gpt-oss -4..-6%. I got +0.36%.

Per shape (µs chip mean) and MATH zones. Values are the mean over busy cores of the per-call zone sums, mean of the
10 measured calls, r03-b02-a02 -> this node:

| shape | µs | GU | ACT | DN | mean core end | max core end |
|---|---|---|---|---|---|---|
| dsv3 | 519.7 -> 516.4 | 325.5 -> 335.2 | 10.4 -> 7.3 | 147.1 -> 145.6 | 491.8 -> 496.9 | 518.1 -> 514.8 |
| kimi-k2.7 | 733.0 -> 735.8 | 477.5 -> 484.3 | 12.1 -> 7.5 | 225.3 -> 223.0 | 727.4 -> 727.4 | 730.7 -> 733.8 |
| kimi-k3 | 1371.5 -> 1358.7 | 730.2 -> 742.4 | 0 | 519.8 -> 503.9 | 1275.7 -> 1272.0 | 1370.3 -> 1357.4 |
| glm | 457.5 -> 450.7 | 285.1 -> 290.0 | 10.4 -> 7.2 | 127.2 -> 127.3 | 431.3 -> 433.1 | 455.9 -> 449.3 |
| mistral | 191.5 -> 189.8 | 106.0 -> 111.7 | 5.2 -> 3.7 | 51.2 -> 49.0 | 166.7 -> 168.7 | 190.3 -> 188.6 |
| dsv4-flash | 318.5 -> 315.9 | 181.1 -> 186.8 | 0 | 105.2 -> 98.5 | 296.1 -> 295.2 | 317.1 -> 314.6 |
| dsv4-pro | 1069.2 -> 1075.2 | 688.2 -> 701.5 | 0 | 358.9 -> 342.6 | 1060.2 -> 1057.6 | 1067.4 -> 1073.7 |
| minimax-m3 | 366.1 -> 372.1 | 227.5 -> 234.5 | 0 | 111.9 -> 112.7 | 343.7 -> 351.5 | 364.8 -> 370.8 |
| gpt-oss | 266.1 -> 263.3 | 170.1 -> 167.0 | 0 | 58.7 -> 58.4 | 233.2 -> 229.8 | 265.2 -> 262.4 |

## Why (profiler evidence)
1. **The mechanism did what it was built for locally.** On the SiLU shapes RE-ACT is the multiply pass waiting for
   PACK's silu backlog. It fell 3-4.6 µs per call (dsv3 10.4 -> 7.3, kimi-k2.7 12.1 -> 7.5, glm 10.4 -> 7.2,
   mistral 5.2 -> 3.7), so the activation backlog left after the K-loop shrank. On the one compute-bound shape,
   gpt-oss, GU fell 3 µs and the mean core end 3.4 µs.
2. **Everywhere else RE-GU grew 5-13 µs and RE-DN shrank by about as much.** GU+ACT+DN per core is conserved within
   ±1-2%: kimi-k3 1250 -> 1246, dsv4-pro 1047 -> 1044, dsv3 483 -> 488, mistral 162 -> 164, mm3 339 -> 347. This
   is the DRAM-bound signature seen in r02-b02-a03: compute time saved in one phase is time the group then waits in
   the next phase for its DRAM share. The mean core end equals aggregate weight bytes / DRAM bandwidth on these
   shapes, so a compute-side saving cannot show up.
3. **The walk is not free.** Each subblock now alternates activation and matmul twice instead of once. That is an
   extra enter_act / leave_act reconfig (unpack srcA format, copy / matmul MOP init, packer format + L1_ACC) and a
   packer drain per subblock. Block N-2 also lost its plain K-loop form. On the shapes where the activation was
   not SFPU-bound, or was already hidden, this overhead is the GU growth. My cost model assumed ~1.5k cycles of
   SFPU-overlappable matmul saved per subblock. The measured gain on gpt-oss (15 subblocks) is ~3 µs, about
   300 cycles per subblock, so either the activation was less SFPU-bound than modelled, or the extra transition
   eats most of the gain.
4. **Measurements made while choosing this node** (they matter more than the score):
   - **mistral's gate/up phase is already at the DRAM peak.** In r03-b04-a02's RE-GW zones the 4 groups issue
     1.18 MB weight blocks at a ~10.5 µs median period, ~450 GB/s aggregate. The "2x the LoFi ideal per K-block"
     seen by r03-b02-a02 / r03-b04-a01 is the group's DRAM share, not a DM latency chain.
   - **First-block fill.** The first gate read on a group's sender row starts at 5.6 µs, but column gx0 (block-0 x
     owner) starts it at 15-24 µs: its up-front x(0) read of 96 one-KB sticks takes 11-20 µs in the start-up
     burst. It is staggered by row group, y8 first and y2 last, 3 µs apart, the same every call. The x(2) owner's
     96 tagged reads take ~13 µs just to issue behind the gate reads (NoC read credits).
   - **The tail is random per call and is mostly in the DOWN phase.** In r03-b02-a02 mistral (4 equal experts on
     4 groups of 2 rows), groups take 83-141 µs of GU and 35-130 µs of DN for identical work. The slow group
     changes from call to call. In call 7 the y8-9 group's DN took 130 µs, and it kept going ~45 µs after every
     other group had finished. The down loop issues each block's DRAM read only after all column receivers have
     acked (no read-ahead), so DRAM latency is exposed once per down block. The tail is still the largest waste:
     max - mean = 20 µs on mistral (12%), 19 on mm3, 33 on gpt-oss, 85 on kimi-k3, 19 on dsv4-flash, 18 on dsv3.

## Classification
neutral (within noise). The mechanism is correct, bit-identical and general (fused / non-fused subblocks,
num_blocks 1 / 2 fall back to the one-block walk). It removes 3-5 µs of activation backlog on the SiLU shapes and
3 µs on gpt-oss, but the DRAM-bound shapes absorb any compute saving, and the extra act/matmul transitions cost
about as much in GU. Keep or drop freely. It is harmless, and it gives a later node a ready two-block walk if
the activation ever becomes the limit.

## What a child of this node should try next
1. **Stop spending attempts on compute-side activation hiding** (this node, r02-b01-a03 and its ports). On 8 of 9
   shapes GU+ACT+DN per core is set by the group's DRAM share. Only gpt-oss is compute-bound, and its activation
   is now nearly hidden.
2. **Attack the max - mean tail, which is now the biggest waste on every shape except kimi-k2.7 / dsv4-pro.**
   - Measure first. Add NCRISC zones in the phase-4 loop on the column sender: the in1_ready wait, the down read
     barrier, the act_ready / act multicast. Then compare the slow group's DN blocks with a fast group's in the
     same call. mistral (one expert per group) is the cleanest shape.
   - If the slow group's DN blocks are read-latency-bound, issue the down read for block kb into the sender's own
     ring slot BEFORE waiting for the receivers' in1_ready (the read touches only the sender's slot; only the
     multicast needs the acks). That cuts the exposed DRAM latency per block without reading ahead across the
     activation gap (r02-b02-a03 did that and widened the race).
3. **Start-up fill (~10-15 µs on the 1-expert-per-group shapes):** x(0) for every row is read as 96 one-KB stick
   reads by one owner core, in the same burst as every group's block-0 weights. Options:
   - read x(0) with fewer, larger requests (2 K-blocks per stick read);
   - split block 0's sticks across the row's cores;
   - or have the weight senders issue block 0 a little after the x(0) reads (all bit-identical).

   Judge it by the first RE-GW start on column gx0 (15-24 µs today vs 5.6 µs elsewhere) and by first-GU-start.
4. The SiLU op-local body on PACK (r02-b04-a04's re_act::silu_act) is still untried on this lineage, but per (1)
   expect nothing outside noise.
5. Tooling: a local TRISC compile harness (3 TRISCs, hand-built CT args, old-vs-new object size) is in
   /tmp/routed-r03b04a03 (run.sh / kc.sh). Use it to check kernel-config-buffer growth before an eval.
