# r06-b01-a01 result: 1.6619 (ok). Neutral: -0.3% vs the parent r05-b02-a01 (1.6667), inside the ±1% noise. Bit-identical. The tilt sped up the bottom rows, not the top rows, so the sign of the lever is the other way round

## What happened vs expected
- All 9 shapes are valid on the first and only run. PCC, max_abs and mean_abs equal the parent and the baseline on
  every printed digit, so the per-row head/tail split lands the same bytes in the same slots.
- No JIT error and no hang. Reader +28 B, writer +72 B (local -O2).
- Chip µs vs the parent:

| shape | parent | this |
|---|---|---|
| dsv3 | 504.9 | 509.9 |
| kimi-k2.7 | 701.4 | 696.6 |
| kimi-k3 | 1243.6 | 1241.1 |
| glm | 436.7 | 438.3 |
| mistral | 190.1 | 183.9 |
| dsv4-flash | 298.4 | 298.8 |
| dsv4-pro | 993.2 | 993.8 |
| minimax-m3 | 361.9 | 373.4 |
| gpt-oss | 261.0 | 266.5 |

  Every move is the usual per-shape spread. Mistral, mm3 and gpt-oss are the noisy max-core lottery shapes.
- Expected: the top groups' first gate/up window falls back toward its 50/50 value (dsv3 263 -> ~225, k2.7
  314 -> ~220). It did not.

## Why (profiler evidence)
Per-row RE-GU of each group's FIRST expert, top group -> bottom group (/tmp/r06b01a01/zones.py, mean of the 10
measured calls):

| shape | r04-b02-a01 (50/50 everywhere) | r05-b02-a01 (h = 3 everywhere, ~60/40) | this node (h = 0 top ... 6 bottom) |
|---|---|---|---|
| dsv3 | 226 218 201 167 184 | 263 263 196 154 134 | 262 244 202 160 137 |
| kimi-k2.7 | 219 214 198 166 163 | 314 295 203 141 108 | 328 297 198 149 94 |
| dsv4-pro | 293 292 280 273 269 | 363 324 249 217 214 | 370 356 271 219 160 |
| glm | 194 195 180 146 132 | 235 216 174 129 113 | 230 206 175 134 113 |
| mm3 (4 groups) | 242 234 224 211 | 257 234 193 194 | 243 234 200 190 |
| mistral (4 groups) | 120 112 97 93 | 127 126 97 84 | 117 116 104 85 |

1. **A top-row group's speed does not follow its own NoC split.**
   - The top rows went from 59% to 50-53% NoC 0, the same split they had at r04-b02-a01, and their first-expert
     GU stayed at the r05 value (dsv3 262, k2.7 328, dsv4-pro 370), not the r04 one (226 / 219 / 293).
   - The bottom rows went to 66-69% NoC 0 and got faster again (k2.7 y10 108 -> 94, dsv4-pro y10-11 214 -> 160).
2. **So what slowed the top rows from r04 to r05 is the OTHER rows' NoC-0 traffic, not their own.**
   - With all groups streaming at once, NoC 0 serves the bottom rows first: here and in r05-b02-a01, every extra
     NoC-0 byte on the bottom rows came out of the top rows' gate stream.
   - The top rows' gate reads are NoC 0 at any split, because gate is wholly on the reader.
   - That matches the DRAM-column request-merge picture: NoC-0 requests go south down x = 0 / 9, and two of the
     four endpoint rows (11 and 2) come right after the bottom rows.
   - It does not tell NoC-0 request arbitration apart from response-path arbitration.
3. **Why the chip time didn't move:** the slowest (top) group still ends at the same time (dsv3 y2-5 503, k2.7
   695, dsv4-pro 992, glm 436). The faster bottom groups only lengthen the idle tail. mm3 / mistral top rows gained
   ~10-14 µs in the first expert (243 vs 257, 117 vs 127), which their max-core lottery hides.

## Classification
flawed idea as signed (neutral, within noise). The mechanism works as built. It shifted per-row speed in the
direction the row split predicts for the rows whose split went UP. The premise that a row's own NoC-0 share sets
its speed is refuted. The finding is useful: the top groups are slowed by the bottom groups' NoC-0 load.

## What a child of this node should try next
1. **The opposite tilt (the direct test of point 2).** Bottom rows lighter on NoC 0, top rows heavier.
   - Change one line in `up_head_rows_for_row` (weight_runs.hpp): use (G - 1 - gy) instead of gy. That gives
     h = 6 on the top row and 0 on the bottom row, with the same 0.594 mean.
   - If the bottom rows' NoC-0 traffic is what starves the top, the top's first expert should move toward the
     r04 values (dsv3 ~225, k2.7 ~220) while the bottom rows slow toward r04's (~165). That cuts the end-of-call
     tail on dsv3 / glm / k2.7 / dsv4-pro / mm3 / mistral.
   - Judge it by the table above (zones.py) and by the per-row end times (ends.py), not only by µs.
2. If (1) is also neutral, the row gradient is not steerable by byte placement. Use it in the planner instead,
   as r05-b02-a01's next #1 says: deal the extra experts / bigger work to the bottom groups. Concretely:
   - kimi-k3: 5/5/6/6/6 instead of 6/6/6/5/5;
   - mm3 / mistral: put the 4 experts on y4-11 and leave y2-3 idle;
   - two-phase shapes: put phase B's leftovers on the groups made of the early phase-A groups.
   r05-b01-a01's plan-once page frees the kernel-config space this needs.
3. The gradient is a first-expert phenomenon. Later experts (kimi-k3 GU1..5, k2.7 GU1) run at nearly equal speed
   on all rows, because the groups are out of phase by then. Any fix only has to cover the first ~100-300 µs.
4. Tools in /tmp/r06b01a01:
   - `zones.py <csv> <shape>`: per-row start + duration of every RE-GU / RE-ACT / RE-DN instance;
   - `percore.py <csv> <shape> RE-GU 0`: per-core table;
   - `ends.py <csv> <shapes...>`: per-row kernel end;
   - `pt.cpp`: plan replay for the test routing;
   - `counts.py`: test token counts;
   - `tilt.py` / `fair.py`: link-load model with the profiler's real columns.
