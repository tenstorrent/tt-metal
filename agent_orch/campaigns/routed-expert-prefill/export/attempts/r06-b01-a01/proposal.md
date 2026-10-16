# r06-b01-a01: tilt the gate/up NoC split by grid row: the reader's NoC-0 share of each `up` block grows from 0 rows on the top grid row to 2 x the host's head-row count on the bottom row (same ~0.6 average), so top-row senders lean on NoC 1 and bottom-row senders on NoC 0

## Motivation
The row groups of the best node (r05-b02-a01, 1.6667) do not run at equal speed, and the gap is almost all in the
FIRST expert's gate/up window, when every group streams gate/up at once. Per-row mean RE-GU zone of each group's
first expert (TRISC_1, 10 measured calls; /tmp/r06b01a01/zones.py), top group -> bottom group:

| shape | r04-b02-a01 (50/50 split) | r05-b02-a01 (~60/40 split) |
|---|---|---|
| dsv3 | 226 218 201 167 184 | **263 263** 196 154 134 |
| kimi-k2.7 | 219 214 198 166 163 | **314 295** 203 141 108 |
| dsv4-pro | 293 292 280 273 269 | **363 324** 249 217 214 |
| glm | 194 195 180 146 132 | **235 216** 174 129 113 |
| mm3 (4 groups) | 242 234 224 211 | **257 234** 193 194 |
| mistral (4 groups) | 120 112 97 93 | **127 126** 97 84 |
| dsv4-flash (3,3,2,2) | 112 99 83 82 | 112 103 77 70 |

- The work is identical across groups here (same expert size and per_core_M), and all 11 columns of a group show
  the same time (percore.py), so this is a per-group stream rate, set by the group's grid ROWS.
- r05-b02-a01's move of ~10% of the bytes from NoC 1 to NoC 0 made the bottom groups much faster (k2.7 y10-11
  163 -> 108) and the top groups much slower (219 -> 314). The aggregate rate in that window stayed ~450 GB/s.
  So the rate a group gets on a NoC depends on its row: NoC 0 serves the bottom rows first, and NoC 1 is at least
  less hostile to the top rows. A plausible cause is request arbitration in the DRAM columns: NoC-0 requests run
  east along the requester's row, then south down x = 0 / 9 to the bank endpoints (rows 11, 2, 9, 5). Requests
  from rows just above an endpoint merge late, and two of the four endpoints (11 and 2, the latter reached through
  the wrap) sit right after the bottom rows. Counting merges per row on that path gives y2-5 >> y6-7 > y8-11, the
  measured order.
- The slow top groups set the op time. They end last on 8 of 9 shapes: dsv3 503 vs 409 on y9-11, kimi-k3 1216 vs
  936, mm3 358 vs 298, mistral 176 vs 142, dsv4-flash 294 vs 238.

## Mechanism
- Make UP_SPLIT's `up_head_rows` (the leading up K-rows of each gate/up block the reader issues on NoC 0)
  depend on the core's grid row gy instead of being one constant:
  `h(gy) = min(w, (2 * H * gy + (G - 1) / 2) / (G - 1))`, where H is the host's current value (`(w + 2) / 5`,
  3 for w = 16) and G = kGridY.
  - For w = 16 on the 10-row grid: gy 0..9 -> 0 1 1 2 3 3 4 5 5 6.
  - So the NoC-0 share of the gate/up bytes is 0.50 on the top row and 0.69 on the bottom row. Its mean over the
    rows is the same 0.594 as today.
- Reader and writer of a core compute the same h from their own `my_gy`:
  - the reader issues gate + up rows [0, h) on NoC 0;
  - the writer reads up rows [h, w) on NoC 1.
  - No cross-core agreement is needed, and the bytes land in the same slots.
- Down split unchanged. DN shows no row gradient, and one knob per node.
- Files:
  - `kernels/weight_runs.hpp`: `up_head_rows_for_row` helper;
  - reader / writer: `up_head_rows` becomes a runtime value, and the reader's `gu_rows_read` with it.
  - Kernel-only, no host change.
- The link-load model (/tmp/r06b01a01/tilt.py, r05-b02-a01's model with the profiler's real columns) gives the
  same hottest NoC-0 / NoC-1 link loads for tilt and uniform (50 / 53-54 GB/s at 440 GB/s). So the tilt does not
  load the links more. It only changes which rows use which NoC.

## Why this is not a repeat
- r05-b02-a01 set one global 60/40 split. Its model was symmetric in rows, and it noted the row spread widened.
  Its next #2 (sweep the split) and #1 (rebalance the plan for row speeds) both treat the row speeds as given.
  This node changes the split per row to remove the row speed difference at its source.
- r03-b03-a02 (all gate/up on NoC 0) and r02-b04-a02 (all down on NoC 1) moved whole streams. Here every row keeps
  both NoCs.
- r02-b02-a02 (request VCs by column + group) changed arbitration classes and mixed every row into every VC.

## Expected effect and risk
- **If the row-dependence holds:** the top groups' first gate/up window falls back toward its 50/50 value (dsv3
  263 -> ~225, k2.7 314 -> ~220, dsv4-pro 363 -> ~300), and the bottom groups slow a little. Because the top
  groups end last, the chip time should drop:
  - dsv3 / glm / k2.7 / dsv4-pro / mm3 / mistral / dsv4-flash: -2..-6%;
  - gpt-oss: ~0 (compute-bound, its bottom 2-row groups end last);
  - kimi-k3: small.
- **If the bottom rows' extra NoC-0 traffic starves the top rows' gate reads further:** the top groups don't
  improve and the bottom ones speed up. That would show as a steeper gradient in zones.py, and the next step
  would be the opposite sign or a top-only tilt.
- **Numerics:** bit-identical. The same bytes go to the same L1 addresses, and every slot is complete before its
  multicast (reader barrier + up_done).
- **Hang risk:** low. The handshake is unchanged, and h stays in [0, w], so the loop bounds are valid.
- **Kernel size:** about +50-100 B per dataflow RISC.
- **How to read the eval:** chip µs on the 7 DRAM-bound shapes, plus the per-row first-GU table above
  (zones.py), which should flatten.
