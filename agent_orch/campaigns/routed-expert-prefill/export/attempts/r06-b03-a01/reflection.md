# r06-b03-a01 result: 1.5712 (ok). Regression: -5.7% vs the parent r05-b02-a01 (1.6667), far outside the ±1% noise band. Outputs bit-identical. The NoC-1 port remap works and pays off in the down phase, but gate/up got 10-17% slower on every DRAM-bound shape

## What happened vs expected
- Host build OK. No JIT error, no hang, no kernel-config overflow (writer +172 B in a local -O2 compile).
- All 9 shapes valid. PCC / max_abs / mean_abs are the baseline's on every printed digit. So reads through the
  other two DRAM tiles of each bank (NoC-1 NIU at the NoC-0 endpoint tile and at the free tile) return the same
  GDDR data on this FW-19.7 machine. **Reading a BH DRAM bank through any of its 3 NoC tiles on NoC 1 works.**
- Expected -2..-6% chip on the DRAM-bound shapes. Got +5..+12%.

Per shape, parent -> this node (/tmp/r04b01a02/rows.py: means over busy rows of the 10 measured calls):

| shape | chip µs | mean row end | max row end | mean GU | mean DN |
|---|---|---|---|---|---|
| dsv3 | 504.9 -> 551.4 | 459 -> 489 | 503 -> 550 | 317 -> **350** | 122 -> 118 |
| kimi-k2.7 | 701.4 -> 746.1 | 676 -> 717 | 698 -> 745 | 473 -> **527** | 180 -> **167** |
| kimi-k3 | 1243.6 -> 1368.4 | 1096 -> 1159 | 1216 -> 1362 | 625 -> **673** | 445 -> 460 |
| glm | 436.7 -> 474.7 | 398 -> 425 | 435 -> 474 | 271 -> **298** | 107 -> 108 |
| mistral | 190.1 -> 185.1 | 160 -> 162 | 177 -> 182 | 108 -> 112 | 40 -> 38 |
| dsv4-flash | 298.4 -> 312.8 | 270 -> 278 | 294 -> 305 | 165 -> 172 | 94 -> 96 |
| dsv4-pro | 993.2 -> 1108.4 | 958 -> 1041 | 990 -> 1107 | 649 -> **758** | 296 -> **272** |
| minimax-m3 | 361.9 -> 388.1 | 325 -> 337 | 359 -> 387 | 222 -> **242** | 95 -> **87** |
| gpt-oss | 261.0 -> 261.6 | 232 -> 235 | 247 -> 254 | 151 -> 153 | 73 -> 74 |

## Why (profiler evidence and best explanation)
1. **Down phase: the remap is a net win.** DN moved its split from 60/40 to 50/50, so ~10% of the down bytes went
   back onto NoC 1. Per r05-b02-a01 that alone should have cost DN +2..+13% (it measured -2..-13% the other
   way). Instead DN fell on the h7168 / i3072 shapes: dsv4-pro -8% (top rows y2-y3 298 -> 246 µs), kimi-k2.7 -7%,
   minimax-m3 -8%. It is flat on dsv3 / glm / mistral / dsv4-flash / gpt-oss, and +3% on kimi-k3. So the NoC-1 path
   through the nearest tile carries more than the fixed endpoint did, at least in down.
2. **Gate/up: much slower, worst on the top rows.** dsv4-pro GU on y2-y6 went 663-685 -> 808-854 µs (+22%), while
   y7-y11 went 595-646 -> 655-702 (+9%). dsv3 y2-y5 365 -> 413-416, y9-y11 258-263 -> 266-272. kimi-k3's bottom
   rows y10-y11 even got faster (936 -> 868 end), and its top rows slower (1208 -> 1362).
   This run can't separate two causes:
   - (a) **The GU split change.** up_head_rows 3 -> 1 of 16 put 6% of the gate/up bytes back on NoC 1. The
     reverse of r05-b02-a01 would cost GU +2..+8%; the measured +10..+17% is more than that.
   - (b) **Port concentration at the NoC-0 endpoint tiles.** "Fewest north hops" sends bank 0/4's NoC-1 reads to
     the NoC-0 endpoint tile (row 11) from **every** grid row, and bank 3/7's from 7 of 10 rows (table below). The
     top rows y2-y5 take 3 of their 4 banks per column through NoC-0 endpoint tiles. Those tiles' AXI port then
     carries the bank's NoC-0 traffic (gate in GU) and most of its NoC-1 traffic too, while the bank's other two
     ports idle. GU is the phase where the reader's NoC-0 gate stream and the writer's NoC-1 up stream hit the
     same bank at the same time, ~50/50. The rows that rely most on NoC-0 tiles are the ones that lost the most.

     | row | bank 0 | bank 1 | bank 2 | bank 3 |
     |---|---|---|---|---|
     | 2 | N0 tile | N0 tile | default | N0 tile |
     | 3 | N0 tile | free | default | N0 tile |
     | 4-5 | N0 tile | default | default / free | N0 tile |
     | 6-7 | N0 tile | default | free | free / default |
     | 8-9 | N0 tile | default | free / N0 tile | N0 tile |
     | 10-11 | N0 tile | default / N0 tile | default | N0 tile |
   - The link model (/tmp/r06b03a01w/m/ep*.py) has no per-port (NIU -> MPFE) capacity term, so it could not
     predict (b). BH's MPFE has weighted round-robin service levels per port (tensor_prefetcher.hpp:
     ordinary / NoC-1 / free), so per-port limits and priorities are real hardware knobs.
3. mistral / gpt-oss: unchanged within their noise, as for every DRAM-delivery change.

## Classification
repairable failure (execution / parameter choice, not a flawed idea): the remap is correct, bit-identical and
measurably helps the down phase (-7..-8% DN on dsv4-pro / k2.7 / mm3 even at a less favourable split). Its
gate/up use as built (nearest tile including the NoC-0 endpoint tile, plus the 53/47 GU split) loses 10-17% GU.

## What a child of this node should try next
1. **Down-only remap (most likely net win, cheapest).** Keep r05-b02-a01's gate/up exactly (default NoC-1
   endpoint, up_head_rows = (w+2)/5). Apply the remap only to the DOWN_SPLIT tail read, with DN at 50/50 as here.
   In the writer that is: pass `dram_remap` only to the DRuns::read call. In the factory: up_head_rows back to
   the parent formula. Expect GU back to the parent and DN -3..-8% on dsv4-pro / k2.7 / mm3: about +1..+2%.
2. **Never use the NoC-0 endpoint tile for NoC-1 reads.** Restrict the choice to {default port 1, free port}
   (the link model, ep3.py mode 2, gives most of the gain). That separates cause (b) from (a). If GU recovers with
   that choice, the remap can come back to gate/up too.
3. One knob per node: if trying (2) for gate/up, keep up_head_rows at the parent's 3/16 so the split is not a
   second variable.
4. Facts to reuse:
   - Translated BH DRAM coords: (17 + bank/4, 12 + 3*(bank%4) + port). UMD port order = blackhole_140_arch.yaml
     `dram:` order. NoC-1 endpoint = port 1 for every bank. Physical rows per port: {0,1,11}, {2,10,3},
     {9,4,8}, {5,7,6}.
   - The guard used: BH, 8 channels, `!IsTensorPrefetcherSupported(mesh)` (no DRISC), no streaming profiler,
     translated port-1 coords checked against `virtual_core_from_logical_core((b,1), DRAM)`.
     `DS_NO_DRAM_EP_REMAP=1` turns it off.
   - Kernel side: `weight_runs.hpp DramPortRemap` (8 words `(default_xy << 16) | alt_xy`, writer runtime args
     right after the phase-A sems, define `RE_DRAM_EP_REMAP`).
