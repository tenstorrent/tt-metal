# r05-b02-a01 result: 1.6667 (ok). Win: +2.4% over the parent r04-b02-a01 (1.6275), outside the ±1% noise band; new campaign best. Bit-identical outputs

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs equal the baseline on every printed digit, so the split reads
  land the same bytes in the same slots: up head rows via the gate-layout accessor, and the down head/tail tiles.
- First and only run. Host build OK, no JIT error, no hang, no kernel-config overflow. The reader is +240 B and the
  writer +4 B (local -O2 profiler compile).
- Expected -3..-10% on the DRAM-bound shapes. Got it in the mean core on every DRAM-bound shape. The chip time (max
  core) took a bit less of it, because the row-group spread widened.

| shape | chip µs parent -> this | Δ chip | mean row end | max row end | mean GU | mean DN |
|---|---|---|---|---|---|---|
| kimi-k3 | 1342.8 -> 1243.6 | **-7.4%** | 1246 -> **1096 (-12%)** | 1307 -> 1216 | 727 -> 625 | 493 -> 445 |
| dsv4-pro | 1056.4 -> 993.2 | **-6.0%** | 1045 -> **958 (-8.3%)** | 1052 -> 990 | 694 -> 649 | 337 -> 296 |
| dsv4-flash | 308.7 -> 298.4 | -3.3% | 288 -> 270 (-6.3%) | 298 -> 294 | 182 -> 165 | 96 -> 94 |
| glm | 449.2 -> 436.7 | -2.8% | 423 -> 398 (-5.9%) | 437 -> 435 | 284 -> 271 | 118 -> 107 |
| minimax-m3 | 372.9 -> 361.9 | -2.9% | 349 -> 325 (-6.9%) | 366 -> 359 | 228 -> 222 | 109 -> 95 |
| kimi-k2.7 | 717.3 -> 701.4 | -2.2% | 709 -> 676 (-4.7%) | 713 -> 698 | 476 -> 473 | 208 -> 180 |
| dsv3 | 512.8 -> 504.9 | -1.5% | 484 -> 459 (-5.2%) | 498 -> 503 | 328 -> 317 | 136 -> 122 |
| mistral | 183.8 -> 190.1 | +3.4% (noisy shape) | 164 -> 160 | 177 -> 177 | 106 -> 108 | 45 -> 40 |
| gpt-oss | 255.8 -> 261.0 | +2.0% (noisy shape) | 229 -> 232 | 246 -> 247 | 151 -> 151 | 71 -> 73 |

The table comes from /tmp/r04b01a02/rows.py on reports/{r04-b02-a01,r05-b02-a01}: per physical row, mean of the 10
measured calls. The mean is over busy rows.

## Why (profiler evidence)
1. **The weight stream was NoC-limited, not DRAM-limited.** Moving ~10% of the weight bytes from NoC 1 to NoC 0
   (no byte, order or handshake change) shortened both phases on every DRAM-bound shape:
   - GU -3..-14% (k3 -14%, dsv4-flash -9%, dsv4-pro -6.5%);
   - DN -2..-13% (dsv4-pro -12%, k2.7 -13%, mm3 -13%, dsv3 -10%).

   So the ~420-450 GB/s "DRAM ceiling" earlier nodes assumed was the NoC 1 response links. Measured whole-op weight
   bandwidth:
   - dsv4-pro 422 -> 449 GB/s;
   - kimi-k3 388 -> 418 GB/s;
   - its mean-core rate is ~475 GB/s.

   That matches the link model (/tmp/r05b02a01/noc4.py):
   - NoC 1 responses run north along the DRAM column first, so the four banks of a column share those links;
   - the hottest link was 65-79 GB/s at a 50/50 split, against 45 on NoC 0;
   - at ~60/40 both sit at ~52-56 GB/s.
2. **The gain is uneven across row positions, so the max core took less of it than the mean.** The lower row groups
   (y7-y11) gained much more than the upper ones:
   - kimi-k3 rows y8-11: 1138-1184 -> 936-950 µs, while y2-7 went 1294-1307 -> 1171-1216;
   - dsv3 y9-11: 460 -> 409, while the y2-5 phase-B 4-row group stayed at ~503 and is now the critical group by 50-90 µs;
   - kimi-k2.7 and dsv4-pro: y7-11 end 40-60 µs before y2-6.

   The model's NoC 1 hot spots sit on the column links that serve rows above the bank endpoints; the details are
   not resolved. Whatever the cause, the static plan now leaves 50-280 µs of tail on the top groups.
3. **mistral / gpt-oss** (one expert per group, close to compute-bound) moved only in the mean (mistral DN -11%). The
   chip time is their usual ±3-5% max-core lottery.

## Classification
win (+2.4% geomean, bit-identical, host+kernel). Structurally new: the first change that raises the weight delivery
ceiling itself rather than reordering who waits.

## What a child of this node should try next
1. **Rebalance the plan for the new row speeds (largest, cheapest next gain).** The row groups no longer run at equal
   speed.
   - **kimi-k3:** the static deal puts 6 experts on groups y2-3 / y4-5 / y6-7 and 5 on y8-9 / y10-11. Those two lower
     groups now end ~270 µs early, more than one expert (~190 µs). Deal the extra experts to the LAST (lower, faster)
     groups instead of the first. Expect kimi-k3 max ~1216 -> ~1130 (-6%).
   - **dsv3 / glm:** phase B is (4,3,3) with the 4-row group on y2-5, which is now the critical group (dsv3 503 vs
     450 / 409). Order phase B as (3,3,4) with the 4-row group at the bottom, or move the largest leftover expert down.
     r02-b02-a04 next #1 already proposed a {3,3,4} table entry.
   - Planner-only (adaptive_chunk.hpp). Keep the h7168 kernel-config budget in mind: the reader grew 240 B here.
2. **Sweep the split once, separately per phase.**
   - GU: try up_head_rows 4 of 16 (s0 0.625).
   - DN: try d_head_tiles ~0.65 (dsv3 14/21).
   - The model has the GU optimum at ~0.6 and DN at 0.55-0.62 for 2-row groups. For 5-row groups NoC 0's down
     multicast is hotter, so a layout-aware d_head may help dsv4-pro / k2.7 phase B.
   - One knob per node. Both values are in the program factory (`up_head_rows`, `d_head_tiles`).
3. **Unload the hot links further without moving bytes.**
   - The down multicast (reader, NoC 0) and the gate/up multicast (writer, NoC 1) wrap the whole column whenever the
     sender is on the far row of its group, because each NoC can only go one way.
   - Send each multicast on the NoC that reaches the group in 1-4 hops (sender at top -> NoC 0, at bottom -> NoC 1).
     That needs the other RISC to do it, signalled like IN1_WRITER_MCAST.
   - Re-run noc4.py with your layout to size it first.
4. Don't revert to 50/50, and don't move a whole stream onto one NoC (r03-b03-a02, r02-b04-a02).
5. Tools:
   - /tmp/r05b02a01/noc*.py: link-load model. Edit `heights` and `s0`; noc4 includes the weight multicasts.
   - /tmp/r05b02a01/b.sh / b2.sh: local reader / writer compile with size. DH = reader CT 30 (d_head_tiles),
     XDEF=-DRE_UP_HEAD_ROWS=n.
   - rows.py (above) for per-row GU / DN / end.
