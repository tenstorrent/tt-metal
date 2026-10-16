# r07-b02-a01 result: 1.7387 (ok). Win: +2.7% over the parent r06-b04-a01 (1.6931), outside the ±1% noise band; new campaign best. Bit-identical outputs. The scored run is run 2: run 1 hung, from a bug that was found and repaired (below)

## What happened vs expected
- **Run 1 (not scored): hang** on the first shape's first call (900 s timeout, device reset, no profile).
  - Cause: the reader issued block kb+1's look-ahead DRAM reads between the linked gate/up data multicasts and
    the unlinked valid-sem multicast that closes the link, on the same NoC 0.
  - The BH NoC sanitizer documents this exact deadlock (`debug_sanitize_check_linked_transactions` in
    tt_metal/hw/inc/internal/debug/sanitize.h): "Submitting a non-mcast transaction if there's a linked
    transaction on any cmd_buf will cause a deadlock." Watcher is off in the eval, so it showed up as a hang.
  - Repair: the look-ahead runs only after the valid sem, where the existing code already tried it. No other
    change (proposal addendum 2).
  - **Campaign fact:** never issue a read, unicast write or atomic on a NoC while a linked multicast is open
    on it. r01-b03-a01's "issue the next read after the multicast, before the flush/valid" advice is unsafe
    whenever the multicast is linked, and every weight / x / activated multicast here is linked.
- **Run 2 (scored):**
  - All 9 shapes valid. PCC, max_abs and mean_abs equal the parent's on every printed digit, so the reader's
    NoC-0 gate/up multicast delivers the same bytes into the same slots.
  - No JIT error, no kernel-config overflow. Local -O2 sizes: reader -44 B, writer -276 B vs the parent,
    because every multicast of one kind now goes through one out-of-line helper (see the proposal addendum).
- Expected: GU -3..-10% per mean core, +1..+3% geomean. Got +2.7% geomean. The gain did not come from where I
  expected it: mean GU moved little. The per-call spread of the group ends collapsed (point 2 below).

Chip µs (the score metric), parent -> this node. Mean core end, max row end, and mean GU / DN per busy core
come from /tmp/r07b02/ana.py on the two profile_log_device.csv files (10 measured calls):

| shape | chip µs | Δ | mean end | max row end | GU | DN |
|---|---|---|---|---|---|---|
| dsv3 | 494.4 -> 510.1 | +3.2% | 460 -> 454 | 494 -> 509 | 329 -> 324 | 112 -> 111 |
| kimi-k2.7 | 695.4 -> 700.7 | +0.8% | 673 -> 672 | 694 -> 699 | 483 -> 495 | 168 -> **154** |
| kimi-k3 | 1234.0 -> 1212.3 | -1.8% | 1075 -> 1070 | 1234 -> 1212 | 638 -> 647 | 413 -> 398 |
| glm | 442.0 -> 441.0 | -0.2% | 412 -> **394** | 441 -> 440 | 285 -> 277 | 109 -> 98 |
| mistral | 180.2 -> 167.1 | **-7.3%** | 153 -> 146 | 180 -> 167 | 106 -> 102 | 37 -> 34 |
| dsv4-flash | 294.7 -> 287.0 | -2.6% | 265 -> 263 | 294 -> 287 | 170 -> 169 | 86 -> 84 |
| dsv4-pro | 985.2 -> 997.3 | +1.2% | 947 -> 951 | 984 -> 996 | 665 -> 683 | 271 -> **256** |
| minimax-m3 | 347.8 -> 333.7 | **-4.1%** | 313 -> **295** | 347 -> 333 | 221 -> 209 | 87 -> 81 |
| gpt-oss | 260.3 -> 229.2 | **-11.9%** | 221 -> 217 | 260 -> 229 | 153 -> 150 | 63 -> 61 |

## Why (profiler evidence)
1. **The wrapped gate/up multicasts were a source of per-call randomness, not mainly a steady GU cost.**
   - Per-row kernel end, std over the 10 calls (/tmp/r04b01a02/rows.py), parent -> this node:
     - dsv3: ±14-32 -> ±4-5 µs;
     - kimi-k2.7: ±19 -> ±4-5;
     - mistral: ±10-19 -> ±1-3;
     - gpt-oss: ±7-26 -> ±2-6;
     - dsv4-pro: ±11-20 -> ±9-11.
   - The "max-core lottery" that every reflection since round 1 blamed for the 1-expert-per-group shapes'
     noise was largely this. A wrapped NoC-1 multicast holds a reserved ~12-hop path through the other groups'
     rows. Whether that path collided with another group's stream changed from call to call, so a different
     group became the tail in each call.
   - With the tail gone, the chip time follows the mean core much more closely:
     - gpt-oss: max row end 247 -> 228 on y8-11, where its 2-row pcM-4 groups used to straggle;
     - mistral: 172 -> 167 on y2-3;
     - mm3: mean end -18 µs.
2. **DN fell on the DRAM-bound shapes although the DN code did not change** (k2.7 -14, dsv4-pro -15, glm -11,
   k3 -15 µs per mean core).
   - The phases of different groups overlap in time. Down blocks now cross NoC-1 column links that no longer
     carry other groups' wrapped gate/up multicasts.
3. **On the DRAM-bound shapes, part of the gain went into GU and into a steeper row gradient.**
   - kimi-k2.7 GU +12 and dsv4-pro GU +18 per mean core: the conserved GU+DN pattern again.
   - The top group got slower and the bottom groups faster, and now deterministically:
     - dsv3: phase-B 4-row group y2-5 ends 490 -> 509; y9-11 428 -> 387;
     - kimi-k2.7: y2-6 694 -> 699; y7-11 656 -> 646;
     - dsv4-pro: y2-6 984 -> 996; y7-11 915 -> 907.
   - Those three shapes' chip time is their top group's end, which explains their +0.8..+3.2%.
   - Likely cause, not isolated by this run: top-row senders now put their gate/up multicast on NoC 0. A
     top group's NoC-0 column links also carry the NoC-0 DRAM read responses, which r06-b01-a01 found to be
     what the top groups lose to the bottom ones.
4. gpt-oss gained most (-11.9%). Its 3+3+2+2 layout has 3-row groups whose senders were on the wrap-heavy
   NoC-1 path in 6 of 11 columns. Its GU and DN per mean core barely moved (153 -> 150, 63 -> 61); its max row
   end fell 31 µs. So the gain was the tail.

## Classification
win: +2.7% geomean, bit-identical, kernel-only. It completes r05-b04-a01's wrap diagnosis (r06-b04-a01 fixed the
down half, this node the gate/up half). The main measured effect is that the group-tail variance is gone.
Run 1's hang was a repairable execution bug (a read inside an open linked multicast), fixed before the scored run.

## What a child of this node should try next
1. **Rebalance the plan toward the bottom groups. It is now cheap to get right, because the row gradient is
   stable.**
   - Per-row ends vary only ±4-5 µs per call (they varied ±15-30 before). The top group is the critical path
     on dsv3 (509 vs 387 on y9-11), kimi-k2.7 (699 vs 646) and dsv4-pro (996 vs 907).
   - Proposed moves:
     - dsv3 / glm: put phase B's 4-row group at the bottom (a {3,3,4} layout entry; the host builds the
       topology from the table), or deal the largest leftover to a lower group. Possible gain: dsv3 -5..-10%.
     - kimi-k2.7 / dsv4-pro: phase B 5+5. Give the top 5-row group fewer tile-rows (the smaller experts).
     - kimi-k3: deal the 6th experts to the lower groups.
   - This is r05-b02-a01 next #1 and r06-b01-a01 next #2, now with deterministic targets. Check the plan
     replay (/tmp/r06b01a01/pt.cpp or /tmp/r04b03a03 sim) before an eval.
2. **Combine with r07-b01-a01** (plan once + idle-row tie pass), if it wins. It touches the planner and the
   factory, this node only the two dataflow kernels; the conflicts are small. Plan-once also frees kernel-config
   space for (1).
3. **Find out why top groups lose more now** (point 3). One option is to tilt which RISC multicasts per group
   position: e.g. top senders of the TOP group only go back to the writer's NoC 1, which wraps but avoids the NoC-0
   response links. Judge it by per-row ends on dsv3 / k2.7 / dsv4-pro. Expect it to trade back some tail variance.
4. **The next-read serialization on top senders is still open.** Their look-ahead read now waits for the
   multicast data to leave the core, because a read inside an open linked multicast deadlocks.
   - To overlap the two safely, issue the look-ahead reads before the multicast, right after the in1_ready
     wait. r01-b03-a01 measured that ordering as harmful at M=1 on the root, though.
   - Or multicast unlinked and put a write barrier before the valid sem.
   - Measure first: mean GU on top-sender columns vs bottom-sender columns is identical here (lock-step rows),
     so this needs NCRISC zones around the sender's read barrier.
5. Don't retry: issuing NoC reads inside a linked multicast window (hang); moving senders between rows per
   phase (r05-b04-a01).
6. Tools: /tmp/r07b02/ana.py (mean end / max row end / GU / DN / even-odd column GU per shape, several CSVs
   side by side), /tmp/r07b02/m/gufix.py (link model with per-position gate/up NoC choice),
   /tmp/r05b02a01/b.sh + /tmp/r07b02/sz/b.sh (local reader/writer size: `RSRCP=` / `WSRCP=` override the source;
   use `XDEF=-DRE_UP_HEAD_ROWS=3 RWM=1`).
