# r05-b03-a01 result: 1.6177 (ok). Neutral: -0.6% vs the root r04-b02-a01 (1.6275), inside the ±1% noise. The x owners are now spread over the columns, but the top-row lag they were meant to remove is unchanged, so the column hotspot is not its cause

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the root and the baseline. That confirms
  the reader and compute agree on every block's owner with the per-row offset: same x data, same tilize.
- No hang, no JIT error, no kernel-config overflow on the first and only run. The reader grew +292 B in profiler
  builds (the RE-X0 zone) and the compute +16-24 B per TRISC.
- I expected the top-row groups' gate/up to shorten (mistral / mm3 / dsv4-flash -2..-6%). Nothing moved outside
  noise.

Chip µs, root -> this node, and mean core end (/tmp/r04b04/zones2.py):

| shape | chip µs | mean core end |
|---|---|---|
| dsv3 | 512.8 -> 507.7 | 482 -> 486 |
| kimi-k2.7 | 717.3 -> 719.2 | 708 -> 710 |
| kimi-k3 | 1342.8 -> 1331.3 | 1244 -> 1242 |
| glm | 449.2 -> 451.4 | 421 -> 426 |
| mistral | 183.8 -> 182.3 | 160 -> 162 |
| dsv4-flash | 308.7 -> 319.2 (+3.4%) | 287 -> 291 |
| dsv4-pro | 1056.4 -> 1061.5 | 1044 -> 1051 |
| minimax-m3 | 372.9 -> 379.7 | 343 -> 345 |
| gpt-oss | 255.8 -> 260.2 | 227 -> 231 |

The mean core end is +2..+7 µs on 8 of 9 shapes. That is small and inside run-to-run spread, but it points the
wrong way. The chip moves are the usual max-core lottery.

## Why (profiler evidence)
Scripts are in /tmp/r05b03-a01-nikola/ana; the data is reports/r05-b03-a01.

1. **The per-row gap is unchanged.** Per-row gate/up (`/tmp/r04b01a02/rows.py`):
   - mistral: y2-3 121 µs vs y8-9 96 (root: 120 vs 93);
   - mm3: y2-3 242-246 vs y8-9 215-218 (root: 242 vs 211);
   - dsv4-flash: y2-4 203 vs y10-11 168-172 (root: 190 vs 163).

   The top rows are still the slow ones, and their groups are still the chip-critical ones.
2. **New measurement, RE-X0** (first x block of each core's first unit; `rx0.py` gives the zone end, i.e. when
   x(0) landed; median over calls):

| shape | y2 | y3 | y4 | y5 | y6 | y7 | y8 | y9 | y10 | y11 |
|---|---|---|---|---|---|---|---|---|---|---|
| mistral | 74 | 60 | 57 | 39 | 40 | 25 | 26 | 20 | (idle) | |
| dsv3 | 75 | 70 | 59 | 36 | 42 | 26 | 25 | 20 | 20 | 15 |
| glm | 81 | 69 | 64 | 39 | 45 | 27 | 26 | 21 | 21 | 17 |
| gpt-oss | 45 | 36 | 37 | 26 | 28 | 23 | 23 | 15 | | |
| kimi-k3 | 62 | 64 | 66 | 47 | 49 | 44 | 46 | 35 | 38 | 32 |

   x(0) reaches the bottom row at ~15-20 µs and the top row 25-60 µs later, monotonically in grid row, on every
   shape. This holds even though each row's block-0 owner is now in a different column.
3. **It is not the prologue.** The earliest core of every row reaches its kb = 0 x wait at the same time on all
   rows (5-12 µs; 24-30 on kimi-k3 for its planner; `rx0min.py`). That core is a pure receiver with no reads
   before the wait. So counts, planning and the per-core `start` page read are not staggered. That rules out the
   220-RISC `start` read hotspot I suspected after the run.
4. **It builds up in the first block's reads, by row.** The latest core per row reaches the wait late:
   - gpt-oss (one unit, so clean): y2 44 µs, y9 25.
   - These cores have DRAM reads in front of the wait: the block-0/1 x owners' up-front stick reads and
     barrier, and the weight senders' gate read and barrier.
   - So in the start-up burst, reads issued from the top rows complete ~20 µs later than reads from the bottom
     rows. The owner spread changed which columns read x but not the rows, and the gradient is by row.
   - The NoC-0 endpoint geometry (blackhole_140_arch.yaml: worker endpoints at rows 11/2/9/5 on x=0 and
     11/3/8/6 on x=9) doesn't give an obvious monotonic per-row path-length bias, so I can't yet say whether it
     is the NoC response path, DRAM arbitration, or the multicast/ack chain of the first block.
5. **Why the spread didn't help in steady state either.** r03-b04-a02 already showed x reads are never late in
   steady state (2 blocks ahead). The per-block owner-column hotspot I computed (~50-80 GB/s on 2 columns' links)
   evidently doesn't slow those columns' weight responses enough to set the pace.

## Classification
flawed idea (measured null). The column hotspot exists geometrically (owners per column per block: 5 -> 1 on
5x2), but it isn't what delays the top rows. The code is correct, bit-identical and small. I'd revert the offset
in a child: `x_owner_offset` returning 0 restores the root behaviour exactly, because the mean core end leaned
+2..+7 µs. Keep the RE-X0 zone (profiler-only, ~0.3 KB NCRISC in eval builds).

## What a child of this node should try next
1. **Pin down the row gradient of the first block's reads. It is the biggest structured waste left on the
   1-expert-per-group shapes** (mistral's y2-3 group is critical in most calls; top-row GU is +25 µs).
   - Add profiler-only NCRISC zones around (a) the owner's up-front x read issue + barrier (before the K-loop)
     and (b) the sender's kb = 0 gate read barrier, and a BRISC zone around the kb = 0 `up` read barrier.
   - If the NoC-0 reads (x, gate) are staggered by row but the NoC-1 `up` reads are not, it is the NoC-0
     response path. Then:
     - put the top rows' block-0/1 x stick reads on the owner's idle BRISC (NoC 1; owners are never senders,
       so their writer has no reads in gate/up). That needs a reader -> writer go/done pair like UP_SPLIT's;
     - or reverse which grid rows host the first-served groups.
   - If all reads are staggered alike, it is DRAM-side arbitration, and a start-up stagger of the bottom groups
     (let the top rows' reads go first) is the cheap test.
2. Don't retry owner placement across columns (this node) or within the row (r03-b02-a02).
3. Tools:
   - `/tmp/r05b03-a01-nikola/ana/rx0.py` (x(0) arrival per row), `rx0min.py` (earliest / latest core per row at the
     x(0) wait), `owners.py` (owner-column collisions per layout for any offset formula), `uwrow.py` (r04-b03-a01
     UNPACK x-wait per row), `calls.py` (per-call group ends).
   - `/tmp/r05b03-a01-nikola/build.sh` (reader / writer local compile with a KDIR override for base vs new) and
     `kc_base` / `kc_new` (3-TRISC size check).
