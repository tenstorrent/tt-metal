# r07-b03-a01 result: 1.7051 (ok). Neutral: +0.7% over the parent r06-b04-a01 (1.6931), inside the ±1% noise band. Bit-identical outputs

## What happened vs expected
- **First eval: runtime_error.** It used r06-b03-a01's remap code as-is (8 runtime words, an 8-way
  coordinate compare on every weight read). kimi-k2.7's program came out at 70784 B, against a
  70656 B kernel config buffer (128 B over). The other 8 shapes ran, and their outputs were
  bit-identical. I treated this as a size/build failure and repaired it before the measured run:
  - **Zero-per-tile remap.** The host passes 4 words = (remapped - default) mod 2^32 for the
    writer's own NoC-1 row of the firmware bank -> xy table (BRISC-local, 8 x uint16).
    `dram_port_remap_toggle<arg>()` adds the words to the table and negates them in place. It is
    called right before and right after each NoC-1 weight read loop, so only those reads see the
    remapped tiles. Locally (-O2, no LTO) this is writer +108 B vs +180 B for the compare loop.
    It also takes ~8 compares per tile off the issue-bound read loop.
  - **Counters opt-in.** The profiler-only MATH perf counters (`re_hwprof.hpp`) are now behind
    `RE_HWPROF`. That is -1.0 KB text / -132 B data on TRISC1 in profiled builds.
- **Second eval: valid on every shape.** PCC / max_abs / mean_abs equal the baseline on every
  printed digit. So reads through each bank's free tile return the same data, which confirms
  r06-b03-a01's finding. The table toggle is also correct: the output drain and other NoC-1 DRAM
  traffic are unaffected.
- **Expected vs got.** Expected GU/DN -3..-7% on the DRAM-bound shapes and +2..+4% geomean. Got
  GU -1..-3% on some shapes, DN mixed, and +0.7% geomean.

Chip µs, parent -> this node:

| shape | chip µs | mean busy-row end | max row end | mean GU | mean DN |
|---|---|---|---|---|---|
| dsv3 | 494.4 -> 503.4 | 462 -> 465 | 490 -> 490 | 330 -> 330 | 112 -> 114 |
| kimi-k2.7 | 695.4 -> 681.3 | 675 -> 669 | 694 -> 679 | 484 -> 480 | 168 -> 166 |
| kimi-k3 | 1234.0 -> 1221.8 | 1076 -> 1075 | 1214 -> 1201 | 638 -> 628 | 413 -> 420 |
| glm | 442.0 -> 432.7 | 413 -> 397 | 440 -> 431 | 285 -> 276 | 109 -> 100 |
| mistral | 180.2 -> 179.7 | 154 -> 155 | 172 -> 169 | 106 -> 108 | 37 -> 37 |
| dsv4-flash | 294.7 -> 291.6 | 266 -> 262 | 284 -> 283 | 170 -> 165 | 86 -> 87 |
| dsv4-pro | 985.2 -> 980.8 | 949 -> 944 | 984 -> 979 | 665 -> 662 | 271 -> 270 |
| minimax-m3 | 347.8 -> 348.4 | 315 -> 310 | 329 -> 333 | 221 -> 218 | 88 -> 85 |
| gpt-oss | 260.3 -> 256.7 | 222 -> 222 | 247 -> 243 | 153 -> 154 | 63 -> 62 |

Source: /tmp/r04b01a02/rows.py on profile_log_device.csv, means over the 10 measured calls, then
over busy rows (/tmp/r07b03/agg.py, rows_*.txt). dsv3's chip mean rose while its max row end did
not move (490 -> 490). That is its call-to-call spread, not a slower row.

## Why (profiler evidence)
1. **The NoC-1 relief is real, but small and uneven across rows.** Per-row GU, parent -> this node
   (/tmp/r07b03 per-row diff):
   - top rows got a little faster: kimi-k2.7 y2-y6 -15..-28 µs (-3..-6%); kimi-k3 y2-y5 -12..-19;
     glm y6-y7 -28 (-10%); dsv4-flash y2-y4 -7;
   - some lower rows got slower: kimi-k2.7 y8-y10 +9..+16; mistral y6-y7 +8; minimax-m3 y2-y5
     +5..+8; gpt-oss y5-y7 +8..+10.

   So shifting 4% of the gate/up bytes onto NoC 0 (0.594 -> 0.5625) and taking the NoC-1 responses
   through the free tiles moved bandwidth between row groups more than it raised the total. DN
   moved the same way: glm -9 (bottom rows y8-y11 -9..-27), dsv4-flash top rows +6, kimi-k3
   bottom rows +13..+20.
2. **The hottest-link model over-predicts again.** It predicted the max link would fall from 54 to
   ~49-50 GB/s (-7..-10%). Measured delivered bandwidth rose ~1%:
   - dsv4-pro: 446 MB of weights in 981 µs = 455 GB/s;
   - kimi-k3: ~425 GB/s.

   Together with r06-b01 / r06-b02 / r06-b03, this is the fourth NoC-placement change whose link
   model said 7-15% and which delivered 0-2%. Two readings fit:
   - (a) The aggregate is close to a DRAM-side limit (~455-475 GB/s of the 512 GB/s peak).
   - (b) The limit is request arbitration at the DRAM tiles, not response-link load.

   With both NoC-1 endpoint ports and both free ports in use, the per-port load is ~30 GB/s
   everywhere. Port load is therefore not the limit either.
3. **No GU loss this time, unlike r06-b03-a01.** That node lost +10..+17% GU. This one has the same
   remap idea, minus the NoC-0-endpoint tiles and with a smaller split step, and GU is flat to
   slightly better. So r06-b03-a01's gate/up loss came from (b): port concentration at the NoC-0
   endpoint tiles, plus its larger split step. The free-tile path itself is harmless.
4. `re_hwprof` opt-in removes a few MATH instructions per K-block and window. On compute-bound
   gpt-oss / mistral that could be worth ~0.1%. It does not explain the gpt-oss -1.4%, which is
   inside that shape's usual ±3-5% max-core lottery.

## Classification
neutral (within noise): +0.7%, bit-identical, host+kernel. The mechanism works and is safe; it
repairs r06-b03-a01's gate/up regression. But the weight stream is no longer response-link-bound
enough for NoC-1 relief to pay much.

## What a child of this node should try next
1. **Keep this node's infrastructure. It is cheap and frees headroom.**
   - The toggle-a-table-row remap costs nothing per tile, and its pattern can retarget any
     accessor's DRAM tiles for one loop.
   - With the counters opt-in, TRISC1 has ~1.1 KB more kernel-config room on the h7168 programs.
     That room can carry the gate/up wrap fix (r06-b04-a01 next #1) or the r05-b01-a01 plan-once
     port.
2. **Stop spending attempts on byte placement between NoCs and DRAM tiles.** r05-b02 won once.
   r06-b01, r06-b03 and this node each moved bandwidth between rows without raising the total.
3. **The rows trade bandwidth, so the remaining waste is the row imbalance, not the total.**
   - kimi-k3's bottom groups still end ~270 µs early (y8-y11 end ~930 vs ~1200).
   - glm / dsv3 phase B: the top group is critical.

   Change who gets the work (planner), or cap the leading groups' in-flight reads, rather than
   which NoC carries the bytes.
4. **If anyone revisits the split.** The free-tile remap makes NoC 1 cheaper, so a split sweep is
   now one knob: `up_head_rows (w+4)/8` and `d_head (11N+10)/20` in the factory. The row-level
   data above says the split mostly moves the row gradient.
5. **Size notes for children.**
   - Local -O2 non-LTO deltas overestimate (or at least do not equal) the JIT LTO delta. The
     compare-loop version measured +180 B locally and overflowed by 128 B.
   - A noinline-lambda refactor of the writer's three multicasts made it +176 B bigger. Don't.
   - Tools: /tmp/r07b03/model.py (link model with correct x = 9 NoC-0 endpoints and free tiles),
     /tmp/r07b03/kc.sh + tr/ (TRISC size with or without RE_HWPROF), /tmp/r05b02a01/b2.sh (writer
     size, XDEF=-DRE_DRAM_EP_REMAP=1).
