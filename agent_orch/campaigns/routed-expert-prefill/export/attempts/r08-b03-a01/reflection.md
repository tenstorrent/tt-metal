# r08-b03-a01 result: 1.7478 (ok). Neutral overall: +0.5% over the root r07-b02-a01 (1.7387), inside the ±1% noise. The pacing works as built: on every shape the row groups now end within a few µs of each other. It is a clear win on the shapes whose phases give each group 1-2 experts (dsv3 -6.0%, glm -4.0%, kimi-k2.7 -3.1%, dsv4-pro -2.3%, mm3 -2.1%), and a loss where groups run many experts (kimi-k3 +9.2%, dsv4-flash +2.3%). Bit-identical outputs

## What happened vs expected
- One run. The host build was OK, and there was no JIT error, no hang and no kernel-config overflow. The h7168 programs fit: reader +1.0 KB in a local -O2 profiler compile, offset by the RE_HWPROF opt-in (~-1 KB TRISC1).
- All 9 shapes are valid. PCC, max_abs and mean_abs equal the root's and the baseline's on every printed digit.
- Expected: mm3 / mistral -8..-15%, two-phase shapes -3..-8%, kimi-k3 -3..-8%, gpt-oss neutral. Got:

| shape | root chip µs | this | Δ | per-row end mean / max / min, root -> this (/tmp/r06b01a01/ends.py, busy rows) |
|---|---|---|---|---|
| dsv3 | 510.1 | 479.7 | **-6.0%** | 456 / 509 / 387 -> 472 / 476 / 466 |
| kimi-k2.7 | 700.7 | 678.7 | **-3.1%** | 673 / 699 / 646 -> 674 / 675 / 672 |
| kimi-k3 | 1212.3 | 1324.0 | **+9.2%** | 1071 / 1210 / 883 -> 1251 / 1323 / 1152 |
| glm | 441.0 | 423.3 | **-4.0%** | 395 / 440 / 340 -> 417 / 420 / 414 |
| mistral | 167.1 | 166.4 | -0.4% | 147 / 167 / 126 -> 160 / 165 / 156 |
| dsv4-flash | 287.0 | 293.6 | +2.3% | 263 / 287 / 231 -> 287 / 292 / 284 |
| dsv4-pro | 997.3 | 974.0 | **-2.3%** | 951 / 996 / 906 -> 966 / 972 / 959 |
| minimax-m3 | 333.7 | 326.6 | -2.1% | 297 / 333 / 265 -> 314 / 325 / 309 |
| gpt-oss | 229.2 | 235.4 | +2.7% | 219 / 228 / 202 -> 222 / 229 / 213 |

## Why (profiler evidence: /tmp/r06b01a01/zones.py on reports/{r07-b02-a01,r08-b03-a01})
1. **The groups' spread collapsed everywhere.** Max minus min row end went from 50-330 µs to 3-170 µs, and to
   under 12 µs on 7 of 9 shapes. kimi-k3's remaining gap is its 5-expert groups finishing early, which is
   correct. The board, the per-peer totals and the one-column pacing behave as designed. Nothing hung, so the
   min-group-never-waits argument holds in practice.
2. **The top group really is starved in a synchronized burst, and pacing gives it the bandwidth back.**
   - First gate/up per group, top -> bottom (µs):
     - dsv3 phase A: 295/263/200/138/130 -> 225/207/195/194/194;
     - kimi-k2.7 phase A: 364/307/185/124/119 -> 206/190/181/181/180.
   - The top group's GU0 fell 24% / 43%. So this gradient is a share effect, not a fixed per-position latency.
   - Phase B (one expert per group) is now balanced too: dsv3 4+3+3 groups end at 466-476 (root 387-509),
     k2.7 5+5 at 672-675.
   - Those shapes' chip time is the top group's end, so they gained 2-6%.
3. **Pacing costs aggregate throughput, and that cost wins on multi-expert phases.** The mean busy-row end rose
   on every shape: dsv3 +16, glm +22, mm3 +17, mistral +13, dsv4-flash +24, kimi-k3 +180 µs.
   - Lock-stepping block counts with a 2-block slack forces all groups into the same phase at the same time:
     all in gate/up, then all in down.
   - In the root, groups drift out of phase after the first expert. One group's down phase (lighter on DRAM,
     paced by the activated row stream) then overlaps another group's gate/up, and that mix uses DRAM better.
   - kimi-k3 shows it most clearly. Per-expert GU is ~145 µs on every group here, against 100-140 in the root.
     The per-expert period is ~225 µs here, against ~170-210 on the root's groups. All 5 groups now run at a
     rate slower than the root's slowest group.
   - k2.7 phase A shows the same on its second expert: GU1 ~185 µs everywhere vs 103-173. Its first-expert
     gain was larger, so the net was positive.
   - dsv4-flash (2 experts per group, 3+3+2+2): the top group gained nothing (end 286 -> 292), and the bottom
     groups were held to it (231 -> 285).
4. **mm3 / mistral gained less than modelled.** Their top group's GU0 moved only 247 -> 237 (mm3) and
   122 -> 119 (mistral). The lower groups were held: mm3 y8-9 GU0 176 -> 195, DN 81 -> 105; mistral lower-group
   DN 33 -> 55, because they wait in down for the top group's gate/up tail.
   - With rows y10-11 idle (4 of 5 groups busy), freeing bandwidth helps the top group much less than in the
     5-group phase A of dsv3 / k2.7.
5. gpt-oss's +2.7% is mostly its spread: its row ends are flat (max 228 -> 229), and its compute-bound 2-row
   groups set the pace either way.

## Classification
neutral (within noise): +0.5%, bit-identical, host+kernel. It is a real mechanism, and it moves the shapes in
opposite directions. It wins on phases where every group runs 1-2 experts (dsv3 / glm / k2.7 / dsv4-pro / mm3,
-2..-6%). It loses on many-expert phases (kimi-k3 +9%, dsv4-flash +2%), where whole-phase lock-step destroys
the GU/DN overlap between out-of-phase groups.

## What a child of this node should try next
1. **Pace only the first expert-chunk of each phase (one-line scope change, most likely net win).**
   - In GroupPacer, cap pacing at `limit = num_blocks_gu + num_blocks_d` blocks per phase:
     `total[g] = min(total[g], limit)`, and `take()` returns true without waiting or counting once
     `taken >= limit`.
   - Every 1-expert-per-group phase keeps this node's behaviour: dsv3 / glm phase A and B, k2.7 / dsv4-pro
     phase B, mm3, mistral, gpt-oss.
   - The multi-expert phases (kimi-k3, dsv4-flash, k2.7 / dsv4-pro phase A) get the synchronized first expert
     equalized (r06-b01-a01: "the gradient is a first-expert phenomenon") and then drift out of phase as in
     the root.
   - If kimi-k3 and dsv4-flash return to the root's numbers and the rest hold, the score is about 1.765-1.77
     (+1.5-2% over the root).
2. **Alternatively make the slack phase-aware.** Keep S = 2 blocks while a group is on its first expert, and use
   S = one expert's blocks after it. That bounds the long-run lag without forcing GU/DN lock-step. Tune only
   after (1) is measured.
3. **Stack with r07-b01-a01's plan-once + idle-row tie pass.** It gives mm3 / mistral 3+3+2+2 with all rows busy,
   where pacing should do better than with an idle group. It also frees kernel-config space: this node's reader
   is +1.0 KB, paid for by the RE_HWPROF opt-in.
4. Don't retry: pacing every block of a multi-expert phase with a small slack (this node, kimi-k3 +9%).
   Pacing via a multicast board on a sender (r01-b01-a03).
5. **Implementation notes for children.**
   - The board is the 16th program semaphore's 16-B slot: 4 words, A/B phases in the 16-bit halves.
   - It is cleared by the reader at kernel start. Only word 0 is re-inited by the dispatcher.
   - Peers are addressed by NoC y = own y + row delta in column 0. The host checks that it is linear.
   - DS_NO_PACE=1 turns pacing off.
   - `GroupPacer` is built at -Os: take 368 B, begin 296 B.
   - All increments are non-posted NoC-0 atomics, issued only at DRAM-read issue points, never inside a
     linked multicast. The final `noc.async_atomic_barrier()` covers them.
