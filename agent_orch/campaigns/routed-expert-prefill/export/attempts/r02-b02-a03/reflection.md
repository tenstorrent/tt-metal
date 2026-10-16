# r02-b02-a03 result: 1.5059 (ok): a regression vs the parent r02-b02-a02 (1.5191, -0.9%) and vs r02-b02-a01, whose code this node otherwise restores (1.5279, -1.4%)

## What happened vs expected
- All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every
  printed digit). So:
  - the prologue / look-ahead weight protocol delivers the same blocks into the same slots;
  - the writer's DOWN_SPLIT slot index (`down_blk % IN1_D_SLOTS`, now = cb_in1_down depth) agrees
    with the reader;
  - there was no hang.
- The host built and the kernels JIT-compiled on the first try. The parent's request-VC change is
  reverted here, so the a01 code is the right comparison.

Per shape vs r02-b02-a01 (µs; noise ±1%):

| shape | a01 | a02 (parent) | this | vs a01 |
|---|---|---|---|---|
| dsv3 | 565.0 | 561.2 | 563.0 | -0.4% |
| kimi-k2.7 | 765.9 | 768.8 | 778.5 | +1.6% |
| kimi-k3 | 1391.7 | 1398.3 | 1395.0 | +0.2% |
| glm | 492.9 | 489.0 | 491.0 | -0.4% |
| mistral | 190.6 | 193.6 | 202.7 | **+6.3%** |
| dsv4-flash | 332.8 | 337.3 | 327.7 | -1.5% |
| dsv4-pro | 1125.6 | 1127.5 | 1132.2 | +0.6% |
| minimax-m3 | 392.7 | 387.7 | 387.0 | -1.5% |
| gpt-oss | 269.0 | 280.1 | 291.9 | **+8.5%** |

I predicted -3..-8% on dsv4-flash, -2..-6% on kimi-k3 and noise elsewhere. Instead nothing improved
outside noise, and the two 1-expert-per-group shapes with free L1 or a tight lock-step got worse.

## Why (profiler evidence)
Per-core TRISC1 kernel end, mean of the 10 measured calls. Columns are mean core / median core /
max core, in µs from op start. Script: /tmp/routed_r02b02a03/med.py on
reports/<node>/.logs/profile_log_device.csv, cores ending < 20 µs excluded.

| shape | a01 | a02 | this |
|---|---|---|---|
| dsv3 | 447 / 508 / 564 | 439 / 505 / 561 | 442 / 513 / 562 |
| kimi-k2.7 | 648 / 614 / 764 | 652 / 627 / 767 | 663 / 629 / 777 |
| kimi-k3 | 1280 / 1308 / 1391 | 1293 / 1321 / 1398 | 1299 / 1317 / 1394 |
| glm | 383 / 443 / 492 | 386 / 440 / 488 | 386 / 444 / 490 |
| mistral | 174 / 177 / 190 | 174 / 179 / 193 | 173 / 170 / 202 |
| dsv4-flash | 266 / 279 / 332 | 275 / 289 / 336 | 268 / 284 / 327 |
| dsv4-pro | 967 / 1097 / 1124 | 967 / 1100 / 1126 | 968 / 1099 / 1130 |
| minimax-m3 | 360 / 367 / 392 | 356 / 355 / 387 | 357 / 360 / 386 |
| gpt-oss | 249 / 252 / 268 | 252 / 253 / 279 | 255 / 252 / 291 |

- **The MEAN core end did not move on any shape** (within ~1-2%; kimi-k3 is even +19 µs). The
  look-ahead fetched down blocks during the activation, yet the cores' total busy time did not
  shrink. So the DRAM bandwidth a group leaves idle during its own activation is not lost: the
  other groups, which are still in gate/up or down, absorb it. On these shapes the op is bound by
  aggregate weight bandwidth shared across the groups, not by any one group's stream gap. The
  per-group "DN is stream-bound" reading (≈7 µs per down block on kimi-k3 against ≈3 µs of FPU)
  reflects the group's SHARE of DRAM, not a latency bubble that pulling reads forward could remove.
- **What changed is the dispersion.** Per call, per row group (/tmp/routed_r02b02a03/grp.py):
  - mistral (1 expert per 2-row group, 4 groups) group ends went from ~158-202 µs to ~142-226 µs.
  - gpt-oss went from ~222-282 to ~217-306.
  - Some groups now finish EARLIER than any a01 group, and others later.

  The group that leaves gate/up first starts its down stream ahead (up to Dw-2 blocks), taking
  DRAM share from groups still in gate/up. That widens an already random race. The slowest group
  sets the op time, so max went up while the mean stayed put. gpt-oss probably got a 3rd down slot
  (its L1 margin is small, and my fitter replica assumes ~100 KB reserved). It shows the same
  signature.
- dsv4-flash (-1.5%) and minimax-m3 (-1.5%) are inside the known run-to-run spread of those shapes.
  minimax-m3 almost certainly ran the unchanged Dw = 2 loop.

## Classification
flawed idea. The activation gap is not idle DRAM time at the device level: groups share DRAM, so one
group's gap is filled by the others' streams. Prefetching across it only redistributes bandwidth,
which increases the group-tail spread on 1-expert-per-group shapes. The implementation is correct
(bit-identical, no hang), but a child should not keep it.

## What a child of this node should try next
1. **Start from r02-b02-a01's code** (or r02-b04-a01, the best node). Revert this node's reader
   phase-4 restructure and the factory's `dw_slots`. The VC revert here is already a01's code.
2. **Lesson for the whole campaign:** per-core / per-group stream timing is no longer the lever.
   The mean core end equals the aggregate weight bytes / DRAM bandwidth on dsv3, kimi-k2.7,
   kimi-k3, glm and dsv4-pro, and the remaining waste is **max − mean**:
   - dsv3 117 µs (26%), glm 109 (28%), kimi-k2.7 116 (18%), dsv4-pro 157 (16%), dsv4-flash 66
     (25%), kimi-k3 111 (9%), gpt-oss 19-36 (8-14%).
   - On dsv3 / glm / kimi-k2.7 / dsv4-pro this is mostly whole-expert imbalance: groups with one
     fewer expert idle at the end.
   - The untried way to recover it without extra DRAM bytes, and without fewer concurrent streams
     (r02-b01-a01 and r02-b03-a01 showed both lose), is to **split a last-wave expert by the
     intermediate dim N across two groups**: group A runs gate/up N-columns [0, I/2) and down K-rows
     [0, I/2), group B the other half. Each reads half the weights (same total bytes, 2 streams).
   - The cost is a cross-group reduction of the two partial down outputs. Options: B multicasts
     its bf16 partial into A's partials CB before A's final pack, or A and B write disjoint halves
     of a K-split down and one adds through DRAM.
   - Big change, but it is the only lever left that attacks the measured 16-28% max-mean gap on
     four shapes.
3. Don't retry down-weight look-ahead, deeper down rings, or any "use the activation gap for DRAM"
   variant. Also don't retry request VCs (r02-b02-a02). Both only move bandwidth between groups.
4. For the compute-bound gpt-oss (mean ≈ max − 8%), the activation's SFPU time (~26 µs per expert)
   can only be hidden by FPU work running concurrently on the same core, which needs a PACK-thread
   SFPU plus independent matmul work. With one expert per group there is none, short of re-blocking
   the down K loop, which changes the L1-acc rounding. I don't see a precision-neutral way to do it.
