# Q2 wq_b linear as a generic op (Blackhole p100a, 2026-11-12)

Shape (one chip): in0 `[1,1,160,1536]` bf16 TILE interleaved, wq_b `[1536,4096]` bfloat8_b TILE DRAM interleaved,
out `[1,1,160,4096]` bf16 TILE L1 interleaved. Op: `ops/wq_b_linear.py` (kernels `ops/kernels/wqb_*`).
Test/benchmark: `tests/test_q2_wq_b.py` (timing = median of 5 trace replays of 50 calls, us per call, includes dispatch).

## Results

| in0 memory | stock `ttnn.linear`, requested 1D config (in0_block_w=16) | stock, in0_block_w=8 (best stock found) | new generic op | speedup vs requested |
|---|---|---|---|---|
| DRAM | 29.24 us | 26.28 us | **21.86 us** | 1.34x |
| L1 | 28.81 us | 26.12 us | **21.06 us** | 1.37x |

Accuracy vs a float64 reference of the same bf16/bf8 inputs (2 seeds): PCC new 0.999895 / 0.999906, requested stock config
0.999853 / 0.999845, stock in0_block_w=8 0.999929 (seed 0); max abs error new 0.090 / 0.085, requested stock config
0.130 / 0.140. Same compute config as stock (HiFi2, 16-bit DST).
Output is bit-identical across 200 traced replays; clean under the watcher.

Breakdown (same program with parts disabled): empty launch 1.9 us; weight stream only 17.9 us; activation
distribution only 8.5 us; everything but the partial reduction 20.7 us; full 21.9 us. DRAM floor for the 6.7 MB weight
measured with a pure read probe: 16.6-17.3 us with both RISCs of 70-110 cores reading (21.6 us with one RISC per core).

## Design

- **K split over rows**: rows 0-4 ("top") take K tiles 0-23, rows 5-9 ("bottom") K tiles 24-47. In each half the
  128 x 24 weight tiles are numbered column-major and cut into 55 contiguous ranges (rank r = 5x + y mod 5), sized per
  column (COLW: 53-55 tiles on columns 0-5, 58-59 on columns 6-10, because the middle-left columns stream DRAM ~8%
  slower). Top rank r and bottom rank r cover the same output columns.
- **Weight stream**: each core's tiles (processing order) are cut into 4-tile chunks; both data-movement RISCs claim
  chunks from a shared counter (one chunk in flight per RISC), reader RISC on NOC0, writer RISC on NOC1. The unpacker
  finds each chunk through the RISCs' claim tables.
- **Activation**: each core needs only its K half of in0 (245 KB). Four multicast rects: {left cols 0-6, right cols
  7-10} x {top, bottom}. Half h is 6 blocks of 4 k-tiles; even blocks are multicast on NOC0 from the rect's min corner,
  odd blocks on NOC1 from the max corner, so packets never wrap the torus. Each block is read from DRAM once: the left
  NOC0 corner reads even blocks and forwards them to the right NOC0 corner (+x); the right NOC1 corner reads odd blocks
  and forwards them to the left NOC1 corner (-x).
- **Compute**: per segment, `matmul_block` (rt=5, ct=1) per k accumulates 5 output tiles in DST (bf16 DST, HiFi2).
- **Reduction**: the top core holding a column's last K tiles finalizes it; it adds up to 3 partials (bottom same
  segment, top and bottom HEAD of rank r-1) with dest-reuse adds and writes the 5 output tiles. Bottom partials travel
  -y on NOC1, top HEAD partials +y on NOC0; each is followed by an atomic increment on the finalizer (same VC).

Timeline (wall clock from the first kernel start, slowest/median core): activation block 0 lands at 6.1-7.0 us, the
last block at 9.3-11.0 us; weights done 17.4 us median / 18.3 us max; last outputs written by 21.1 us.

## Steps (us per call, DRAM in0)

| step | time |
|---|---|
| N split, 110 cores, 12 concurrent full-grid multicasts | 60.7 |
| 4 multicast streams ({left,right} x {NOC0,NOC1}), chained senders | 38.2 |
| K split over rows (half the activation per core) | 35.1 |
| multicast senders at the NoC's rect corner (no torus wrap) | 25.6 |
| weight reads in a 1-chunk window per RISC (activation reads not stuck behind the DRAM queue) | 23.9 |
| activation blocks read once and forwarded between column groups | 23.0 |
| dynamic chunk claiming between the two RISCs | 22.9 |
| per-column work weights | 21.9 |

Tried without gain: splitting weight tiles between RISCs by DRAM bank side; 60 activation row-fetchers with atomic
counting; delaying weight issue; full-sync DST with all partial adds at the end (and its second compute kernel costs
1.6 us of launch); activation-first on the DRAM-reading corner RISCs. Atomic-increment multicast hangs when the source
is inside the rect (documented in tests/tt_metal/.../multicast_atomics), and set-multicast counters can arrive out of
order, so each activation block has its own flag semaphore.

## Is the requested stock config optimal?

No. The same 1D config with `in0_block_w=8` is 2.96 us (10%) faster in the 50-call benchmark (26.28 vs 29.24 us) and
more accurate (PCC 0.999929 vs 0.999853). The new op is 1.20x faster than that.

**tt-matmul-codegen registry** (`results/registry/onechip.csv`, 261,157 Blackhole winner rows): no entry for this case.
It was measured on a 12x10 grid (p150, not the p100a's 11x10) and has no M=160 rows (M = 32, 64, 128, 256, ...). For
K=1536, N=4096 it only has bf16 weights with DRAM output. The raw per-candidate timings are in the Exabox bank, not the
repo. The nearest cells (M=64-512, HiFi2, 16-bit DST, packer L1 acc) are all won by 2D multicast with in0_block_w=8,
per_core_N=11 on a 12 x (M tiles) grid, never by the 1D config.

**Device sweep** on this p100a (`tests/test_q2_stock_sweep.py`), 203 configs, no failures:

- auto (no program config);
- 70 1D in0-multicast configs: per_core_N in {2,3,4,6,8}, in0_block_w in {2,4,6,8,12,16,24}, and the two largest
  subblocks with h*w <= 8;
- 132 2D configs: grid y in {1,5}, grid x 6-11, per_core_N = ceil(128/x), in0_block_w in {2,4,6,8,12,16}.

Timing is the median of 3 trace replays of 20 calls, which reads 0.5-0.9 us above the 50-call benchmark.

| config | us | PCC |
|---|---|---|
| **1D, per_core_N=2, in0_block_w=8, subblock 1x2** | **27.14** | 0.999929 |
| 1D, per_core_N=2, in0_block_w=6, subblock 1x2 | 27.90 | 0.999937 |
| 1D, per_core_N=2, in0_block_w=8, subblock 5x1 | 28.25 | 0.999929 |
| 1D, per_core_N=3, in0_block_w=8, subblock 1x3 | 28.42 | 0.999929 |
| 1D, per_core_N=2, in0_block_w=12, subblock 1x2 | 28.81 | 0.999885 |
| 1D, requested (in0_block_w=16, subblock 1x2, per_core_N=2) | 29.79 | 0.999853 |
| 1D, per_core_N=2, in0_block_w=24 | 31.72 | 0.999771 |
| 1D, per_core_N=2, in0_block_w=4 | 32.14 | 0.999956 |
| 1D, best with per_core_N=4 / 6 / 8 | 33.29 / 45.53 / 58.12 | 0.999929 |
| 1D, per_core_N=2, in0_block_w=2 | 41.92 | 0.999967 |
| 2D, best: grid 11x5, per_core_N=12, in0_block_w=12, subblock 1x4 | 35.29 | 0.999885 |
| 2D, grid 10x5, in0_block_w=8 | 37.88 | 0.999929 |
| 2D, best with grid y=1 (11x1) | 83.12 | 0.999929 |
| auto | 42.23 | 0.999967 |

Likely reasons (not profiled):

- At in0_block_w=16 each core waits for a 16-tile activation block before its first matmul.
- Below 8, per-block synchronization dominates.
- Wider per_core_N puts the weight read on fewer cores.
- 2D multicast also multicasts the weight, and with 5 M tiles it can't spread the weight reads over more cores.

Not covered: sharded inputs, other fidelity or DST settings, and 2D grids narrower than 6 columns.

## Cost of producing the op

From this Claude Code session's log: model claude-opus-5-5, one session, no subagents. Dollars are at API list prices:
$4/MTok input, $8/MTok 1-hour cache write, $0.20/MTok cache read, $20/MTok output. Actual billing depends on the plan;
`/cost` in Claude Code gives the session's own figure.

| | building the op | stock-config question |
|---|---|---|
| wall clock | 1 h 23 min (10:10-11:33 UTC) | 8.5 min (12:18-12:26 UTC), ~5 min of it waiting on GitHub login |
| model calls | 194 | 17 |
| tool calls | 213 (173 shell, 24 writes, 12 reads, 2 edits, 2 publishes) | - |
| uncached input tokens | 388 | 40 |
| cache-write tokens | 605,167 | 21,250 |
| cache-read tokens | 64,049,093 | 10,899,250 |
| output tokens (incl. thinking) | 379,656 | 11,340 |
| **cost at list price** | **$25.25** | **$2.58** |

The build's cost splits into cache reads $12.81 (re-sent context, ~330k tokens per call on average), output $7.59 and
cache writes $4.84; at the 5-minute cache-write rate it would be $23.43. Not included: the cost-question turn ($0.29),
the context-compaction summaries (not in the log) and this report update.

Approximate phases of the build:

| phase | time | what happened |
|---|---|---|
| setup | ~15 min | read the previous session's workspace notes; built this worktree offline (done 10:19); ran the first DRAM read probe |
| first version, multicast problems | ~35 min | first working op at 60.7 us. Three device hangs from multicast semaphores, each needing a board reset and a watcher run. Then the K-split rewrite, corner senders and weight window (~24 us) |
| tuning | ~20 min | kept: forwarding, dynamic chunk claiming, per-column weights. Measured and dropped: bank-side split, 60 row-fetchers, full-sync DST |
| wrap-up | ~12 min | removed instrumentation; final benchmark; watcher check; report and page |

About half the time went to work that isn't in the final op.

## Caveats

- Tuned and measured on one p100a (11 x 10 grid, worker NoC columns 1-7 and 10-13). COLW depends on the chip's
  harvesting; the op asserts the two-column-group layout.
- Timings are single-chip trace replays; the stock config was run with the model's default compute config
  (HiFi2, fp32 dest off, packer L1 acc on).
