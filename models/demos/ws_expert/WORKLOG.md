# ws_expert worklog

Weight-streaming routed expert, one expert's gate/up -> SwiGLU -> down on one Blackhole chip (generic_op), the model
of the "MoE Weight Streaming" artifact: H 3840, I 640 (20 pairs, KTD 20), bfp4 weights, LoFi, fp32 DEST. Average
M at 1M tokens = 586 rows (R 19 row tiles). Stage 1: x and weights resident in L1, h exchanged on chip.
Utilization: tile = tile products x 16 cycles / (cores x cycles) (padded rows); useful = at the real M.

## Single-core matmul rates (tests/test_mm_bench.py, resident, fp32 DEST, LoFi; peak 16 cyc / tile product)
| block | K | cyc / tile | % LoFi |
|---|---|---|---|
| 1x4 + SwiGLU (gate/up) | 120 | 17.87 | 89.5 |
| 1x4 plain | 120 | 17.28 | 92.6 |
| 2x2 / 1x2 + SwiGLU | 120 | 18.11 / 18.14 | 88.4 / 88.2 |
| 1x4 / 4x1 / 1x3 / 2x2 / 1x2 (down) | 20 | 17.41 / 17.42 / 17.55 / 17.67 / 17.85 | 91.9 - 89.6 |
| 1x1 | 20 | 27.8 | 57.6 |
| full-sync 2x4 + SwiGLU / 1x8 | 120 / 20 | 24.8 / 18.6 | 64.6 / 86.1 |
x bf16 vs bfp8: no difference. Full-sync DST loses the math / pack overlap: fp32 half-sync (4 tiles) 1x4 it is.

## Scheme B: dedicated gate/up (8 x 10 grid, stream-K rows) + 40 down cores (3 columns each, all rows) - dropped
- 120 cores. Correct (PCC 0.99998) but 509K cycles / expert (3.5%).
- Bugs on the way: a down reader credited row q + 1 only after row q's data arrived (one row in flight); the CONS
  multicast rectangle spanned the NoC x 8-9 gap (logical columns 0..7 = NoC 1..7, 10).
- Root cause: every h tile multicast to all 40 down cores. tests/test_mcast_bench.py on this box: a 4 KB multicast
  costs ~400-500 cycles whatever the sender count, aggregate into one rectangle 10-25 GB/s; 760 KB of h per expert to
  every down core cannot fit ~15 us.

## Scheme C: column clusters (ws_cluster.py, kernels wc_*.cpp) - current
- 9 clusters x 10 cores (one NoC column each) = 90 cores. Core j owns gate/up pairs {2j, 2j+1} (1x4 + SwiGLU, K 120)
  and down columns [12j, 12j + 12) (3 x 1x4, K 20). h stays in its column: each core multicasts its 2 h tiles of a
  row to the column (linked payload + flag, the mcast helper's pattern), every core then runs the row's down.
- Rows: weighted stream-K over clusters (gate/up K-tiles + each row's down paid by its owner); a split row's owner
  can be either side (partial flows toward it): exact balance for R >= ~12, lumpy below.
- No slot credits: the host simulates the program order and sizes the h ring (NS) so a slot is always consumed
  before a peer can overwrite it.
- h exchange is free: a no-sync build (no h traffic) runs at the same speed.

### Steps at M 586 (cycles / expert, tile / useful util)
| step | cyc | tile | useful |
|---|---|---|---|
| first run, owner = head holder | 31,590 | 77.0 | 74.2 |
| balanced (flexible owner) | 32,972 -> 31,512 (no matmul re-init between GU/DN) | 77.2 | 74.4 |
| fused GLU, exp_21f sigmoid | 29,726 | 81.8 | 78.9 |
| SiLU-hiding order + down lag 1 expert (NS 8) | 28,394 | 85.7 | 82.6 |
| down lag 3 rows (NS 8) | 27,348 | 88.9 | 85.7 |
| + GLU constants in programmable regs (glu 3) | 27,281 | 89.1 | 85.9 |
| ceiling: same order, no SiLU at all | 27,023 | 90.0 | 86.7 |
| fixed 4-row ring order (down(q-1) after row q), glu 3 | 28,824 | 84.4 | 81.3 |

Why the order matters: SiLU runs on the PACK thread's SFPU (~2.2-2.7K cycles a row of h) and holds one fp32 DST
half; a down block is only 1.4K cycles (1x4 x K 20; K = I / 32 is the model's), so down blocks right after a row
stall on that half. Rows back to back hide it under the next gate/up row (8.3K); down then runs in batches, which
needs a deeper h ring (NS 8 at M 586; grows with rows per cluster: R 64 -> 13-19, does not fit with resident x / W).
Down consuming freshly produced rows waits on the slowest peer (eager: 46K cycles).

### SiLU on the SFPU (tests/test_glu_bench.py, PACK thread, fp32 DEST, cycles per (gate, up) tile pair)
| variant | cyc | max rel err |
|---|---|---|
| stock silu pass + multiply pass | 1777 | 2e-7 |
| fused one pass, fp32 sigmoid | 1810-1858 | 2e-7 |
| fused, exp_21f + 1 Newton | 1327 | 1.7e-3 |
| same, constants in Prgm regs, clamp 254 (glu 3) | 1103 | 1.7e-3 |
| exp_21f + raw approx recip | 1135 | 6e-3 |
| loads + g*u + store only (floor) | 209-290 | - |
| LUT sigmoid (sigmoid_appx) | 338 | ~1 (2e-2 norm): unusable |
Fusing alone saves nothing; the exp is ~600-740 of it. Hiding a row of h (2 pairs) under one 1.4K down block needs
<= ~700 per pair: not reachable at ~1e-3 accuracy. The hardware approx_exp (SFPARECIP exp mode) is not a full-range
exp: it returns sign(x) e^|x| and saturates at ~4 (tests/test_sfpu_probe.py).

## Default: fixed 4-row h ring order (2026-11-13)
M is only known at run time, so the h ring may not grow with it: order "eager" + dn_lag 1 (down of row q - 1 after
row q) needs NS 4 for every R (checked R 1..96). M 586: 28,824 cycles, 84.4% tile / 81.3% useful.

## Stage 2: x from DRAM, tilized on the cluster cores (ws_x.py, kernels wx_*.cpp)
x: row-major bf16 [T + 1, H] in DRAM (row T zero), the expert's rows through a row index table (padding -> row T).
Per cluster the x stream (its gate/up pieces' K ranges, program order) is cut into blocks of BK K-tiles; block j of
an expert: core j % GC gathers the 32 token rows (BRISC, NOC0, 32 reads of n x 64 B), tilizes them (compute,
compute_kernel_lib::tilize = fast tilize, one batch an expert for the next expert's blocks) and multicasts them into
every column core's x ring (NCRISC, NOC1, linked payload + flag). Both data-movement RISCs run non-blocking loops
over in-order queues (a blocking wait could hold back a block a peer needs).

Measured first (why this design):
- tests/test_mcast_bench.py column: one sender at the column end 52 GB/s into all 10 (>= 16 KB a multicast), NOC0 +
  NOC1 from both ends 104, 10 senders on NOC0 33-38, 5 + 5 56-59. x needs ~24 GB/s a column at M 586.
- tests/test_tilize_bench.py: compute fast tilize 32-34 cycles a tile; + the tilize <-> matmul switch ~270 cycles;
  data-movement tilize (32 B face rows) 1,300 (local NoC reads) / 4,800 (RISC copies) cycles a tile: out.

Bugs on the way:
- per-producer x credit counter: fast consumers' later credits filled a slot's quota while a slow consumer still
  read it (PCC 0.97 at BK 8). Now each consumer writes its popped count into its own word at the producer.
- bfp8 x: gate/up's in0 (x) and down's (h) differ in format: SrcB reconfig around down (WX_X_FMT_DIFFERS).
- deadlock (BK 8, staging 4): the next expert's tilize batch did not fit beside this expert's blocks still waiting
  for a ring slot. Staging now sized by the host.

M 586, 90 cores, weights resident (cycles / expert, tile / useful util; UNPACK per expert per core: tilize / x wait):
| BK | NX | x | ring | cycles | util | tilize / x wait |
|---|---|---|---|---|---|---|
| 12 | 20 | bf16 | 480 KB | 30,558 | 79.6 / 76.7 | 1,320 / 477 |
| 12 | 15 | bf16 | 360 KB | 31,267 | 77.8 / 75.0 | 1,319 / 581 |
| 8 | 24 | bf16 | 384 KB | 35,650 | 68.2 / 65.8 | 2,038 / 1,602 |
| 8 | 20 | bf16 | 320 KB | 36,156 | 67.3 / 64.8 | 1,942 / 3,698 |
| 12 | 20 | bfp8 | 255 KB | 30,511 | 79.7 / 76.8 | 1,348 / 338 |
| 12 | 10 | bf16, tilize a piece ahead | 240 KB | 37,148 | 65.5 / 63.1 | 1,994 / 3,619 |
vs x resident 28,824. The ring must hold about an expert's x for the cluster (22 blocks of 12): shallower rings make
every core wait for the slowest consumer's credit. Tilize ~1.3K / expert / core (~0.9K tiles + ~0.4K switch).

## Stage 3: dynamic M (ws_dyn.py, kernels ws_plan.hpp + wd_*.cpp)
Every expert's schedule is computed on device from its M: each core's BRISC runs ws_plan::make (closed-form weighted
stream-K split over NA = min(9, R / 2) active clusters, split-row owner left/right by which keeps the boundary inside
the row; program; x blocks) into a 3-slot L1 plan ring, loads the expert's row index table, and the compute threads
and the NCRISC read the plans and post done words. Inputs: counts [NEXP] and index tables [NEXP, RMAX * 32] in
DRAM. Ring sizes no longer depend on M: h ring 4 rows, x ring 15 blocks (20 + staging for MMAX 1024 did not fit L1
beside the resident weights), x staging 8 blocks (MMAX 1024). plan_py mirrors make (coverage / ownership checked for
M 1..1024).
- First version multicast each consumer's popped-block count to the column after every block (no producer
  bookkeeping): ~230 small multicasts an expert a column on NOC1 beside the x data -> 112K cycles / expert, x waits
  64K. Back to one exact unicast to the producer of block c + NX, found by walking the plans' block counts.
Results (90 cores, weights resident, cycles / expert over experts 1..15, util tile / useful):
| M per expert | cycles | util | x wait / expert / core |
|---|---|---|---|
| 586 constant | 32,166 | 75.6 / 72.9 | 758 |
| Poisson(586) (537..653) | 31,384 | 78.9 / 76.2 | 565 |
| mixed 5..1000 (mean 457) | 31,543 | 59.0 / 56.9 | 5,941 |
Host-planned at the same ring (NX 15): 31,267 at constant 586.

## M sweep 32..5000 (2026-11-13; dynamic M, MMAX 5120, 90 cores, weights resident, 8 experts of constant M)
Changes for large M: plan limits raised (slot 512 words), the gather reads each row tile's 32 ids (128 B) from the
expert's index table instead of keeping tables in L1, and x is tilized in a window (own blocks within LOOK = 30 of
the consumption point, this expert and the next) instead of one batch an expert (staging would have needed ~36 blocks
at M 5000; now 7 for any M).
Hangs found with run_safe_pytest --dev + tt-triage callstacks:
- the window peeked (non-blocking) at later experts' plans: the three TRISCs took different decisions (UNPACK in a
  tilize batch, MATH in a matmul). Now blocking reads of this expert's and the next one's plan only.
- clusters with no x blocks (small M) never moved the NCRISC's credit walk, whose expert is part of its done word:
  the plan ring stopped after 3 experts and those clusters' compute waited for plan 3. The walk now advances past
  planned experts whose blocks lie before the next credit target, popped or not.
| M | R | active cores | cycles / expert | us | util tile (90 cores) | util useful | util on active cores |
|---|---|---|---|---|---|---|---|
| 32 | 1 | 10 | 22,671 | 16.8 | 5.6 | 5.6 | 51 |
| 64 | 2 | 10 | 28,778 | 21.3 | 8.9 | 8.9 | 80 |
| 96 | 3 | 10 | 42,945 | 31.8 | 8.9 | 8.9 | 80 |
| 128 | 4 | 20 | 28,622 | 21.2 | 17.9 | 17.9 | 81 |
| 192 | 6 | 30 | 28,607 | 21.2 | 26.8 | 26.8 | 80 |
| 256 | 8 | 40 | 29,117 | 21.6 | 35.2 | 35.2 | 79 |
| 320 | 10 | 50 | 30,581 | 22.7 | 41.9 | 41.9 | 75 |
| 384 | 12 | 60 | 28,609 | 21.2 | 53.7 | 53.7 | 81 |
| 448 | 14 | 70 | 34,461 | 25.5 | 52.0 | 52.0 | 67 |
| 512 | 16 | 80 | 28,613 | 21.2 | 71.6 | 71.6 | 81 |
| 586 | 19 | 90 | 31,634 | 23.4 | 76.9 | 74.1 | 77 |
| 640 | 20 | 90 | 33,746 | 25.0 | 75.9 | 75.9 | 76 |
| 768 | 24 | 90 | 39,959 | 29.6 | 76.9 | 76.9 | 77 |
| 896 | 28 | 90 | 47,378 | 35.1 | 75.6 | 75.6 | 76 |
| 1024 | 32 | 90 | 52,306 | 38.8 | 78.3 | 78.3 | 78 |
| 1280 | 40 | 90 | 66,427 | 49.2 | 77.1 | 77.1 | 77 |
| 1536 | 48 | 90 | 78,703 | 58.3 | 78.1 | 78.1 | 78 |
| 2048 | 64 | 90 | 104,105 | 77.1 | 78.7 | 78.7 | 79 |
| 2560 | 80 | 90 | 133,570 | 98.9 | 76.7 | 76.7 | 77 |
| 3072 | 96 | 90 | 158,018 | 117.1 | 77.8 | 77.8 | 78 |
| 3584 | 112 | 90 | 187,421 | 138.8 | 76.5 | 76.5 | 77 |
| 4096 | 128 | 90 | 210,383 | 155.8 | 77.9 | 77.9 | 78 |
| 4608 | 144 | 90 | 234,753 | 173.9 | 78.5 | 78.5 | 79 |
| 5000 | 157 | 90 | 258,622 | 191.6 | 77.7 | 77.3 | 78 |
Below M 576 the R / 2 active-cluster rule (no row across three clusters) leaves clusters idle: the active cores
still run at ~80%.

## Stage 4 (WIP): streamed weights, 20-core groups (ws_group.py, kernels wg_*.cpp)
Why: streaming weights into the 9 x 10 clusters needs >= 2 expert slices a core (2 x 414 KB) beside ~900 KB of x / h
rings, and the fabric lands unicast writes only (mesh API: chip multicast of NoC unicast writes / scatter / fused
atomic inc; no NoC multicast at the destination), so 9 on-chip copies would need an 8x re-multicast relay. Groups:
expert e on group e % G (4 groups of two adjacent columns = 80 cores), core j: gate/up pair j (1 x 2) + down
columns 6j..6j+5 (2 x 1 x 3), weight slice 207 KB in a 2-slot ring. Weights landed by 10 injector cores from DRAM
(stand-in for the fabric: same ring addresses, WARR semaphore after the slice, per-core credit words back). h: each
core unicasts its tile to its column's leader for the row (core (c, q % 10)), the 2 leaders multicast 10 tiles each.
Every expert has its own weights in the test; all groups' last experts PCC 0.99998 (rings / credits / slots right).
Bugs: DRAM -> L1 reads need the L1 address congruent mod 64 (watcher caught the 128 B id read): 64-aligned landing.
Perf (M 586, 32 experts, cycles / expert at chip rate, util tile):
| config | cycles | util | note |
|---|---|---|---|
| BK 12, NX 15 | 262K | 10.4% | x waits ~all; weight waits ~0; h waits small |
| G 1 / 2 / 4 groups | 478K / 349K / 262K | 22.9 / 15.7 / 10.5% | slow inside one group already |
| BK 6, NX 30 | 358K | 7.7% | more multicasts: worse |
| BK 24, NX 8 | 167K | 16.4% | |
| BK 30, NX 6 | 153K | 17.9% | |
Diagnosis (x block timeline, one group): blocks are tilized early and their slots are free, but the producer's
multicast goes out 3-15K cycles later: 20 producers' linked 24 KB multicasts into the 2-column group serialize
(~2.4K cycles a block, ~13 GB/s a group). A group needs all of its experts' x: per core 2.25x the x of a 10-core
cluster (the price of halving the weight slice), ~49 GB/s into every core at the M 586 target.

### Stage 4: back above 70% (2026-11-13)
What held it (each found with per-queue max-call timers on both RISCs + the x block timeline, G 1):
1. Multicast direction: column 0 multicast x on NOC1 (-x) and column 1 on NOC0 (+x) - both reached the other column
   only by wrapping the whole row torus (the bench's 85-89 GB/s layout is the reverse). Swapped: G 1 33 -> 41%.
2. Starvation: 10 linked x multicasts a NoC in flight at once gave aggregate rate but single calls blocked up to
   58K cycles, and every core of the group waits on the starved block. In stream order (block s is sent once block
   s - 2 has landed locally, i.e. one multicast at a time a NoC): G 1 39 -> 55% (s - 1: 44%).
3. Column 1's h leader multicast was on NOC0 too: it wrapped the row through every other group (300K-cycle calls at
   G 4). Moved to the NCRISC on NOC1: G 1 55 -> 68%, G 4 25 -> 44% (no weights).
4. bfp8 x (tilize packs bfp8; SrcB reconfig around the down): half the multicast bytes. G 1 BK 24: 80%.
5. Weights: the 10 DRAM injectors (one 576 B page read a tile, 14 KB chunks) gave ~112 GB/s; CH 120 + 20 injectors
   (columns 9, 10; one per group core) help, but at G 4 weights (~130 GB/s) + x gather (~140 GB/s) of small reads
   saturate DRAM: 64%. Injectors that land from L1 (no DRAM): 77%, weight writes themselves free (noread+nowrite
   the same). On a real chip the weights come from the fabric, so the L1 mode ("l1", P = 2 weight sets preloaded,
   expert e uses set (e // G) % 2 so a group's consecutive experts alternate slots and sets) is the default.
Lag (DN_LAG 2, 3) only moved the wait between h and x: G 4 with DRAM weights is throughput-bound, not latency.

Result (bfp8 x, BK 24, NX 15, 20 injectors, "l1" weights, 32 experts, steady state):
| G (cores) | M | cycles / expert | util tile | util useful | pcc |
|---|---|---|---|---|---|
| 1 (20) | 586 | 135,448 | 80.8% | 77.9% | 0.99994 |
| 2 (40) | 586 | 69,885 | 78.3% | 75.5% | 0.99994 |
| 3 (60) | 586 | 45,870 | 79.5% | 76.7% | 0.99994 |
| 4 (80) | 64 | 7,258 | 39.7% | 39.7% | 0.99994 |
| 4 (80) | 256 | 14,062 | 81.9% | 81.9% | 0.99994 |
| 4 (80) | 586 | 35,148 | 77.8% | 75.0% | 0.99994 |
| 4 (80) | 1024 | 57,974 | 79.5% | 79.5% | 0.99994 |
| 4 (80) | 2048 | 117,922 | 78.2% | 78.2% | 0.99994 |
M 64 (2 rows an expert) is bound by the weight landing (207 KB a core every ~7K cycles).
Other modes at M 586: DRAM weights G 1/2/3/4 81.6 / 81.2 / 79.2 / 63.9%; bf16 x G 4 (BK 12 NX 15 / BK 24 NX 8)
63.8 / 67.5% (PCC 0.99998; multicast-bound; BK 24 NX 15 bf16 does not fit L1).

## Stage 5: weights over the fabric (ws_fab.py, kernels wf_*.cpp; LoudBox 2 x 4, FABRIC_2D, 15232 B payload)
Step 1: one receiving chip R = (0, 1) (3 neighbours, 2 links each = 6 connections) runs the 80-core group program
without injectors; the neighbours only send. A sender core a link: BRISC reads the target cores' slices from its
chip's DRAM, NCRISC sends 24-tile (13.8 KB) packets straight into R's 2-slot rings, the last packet of a slice fused
with an atomic inc of the core's warr word (the arrival counter became an L1 tensor word so other chips know its
address). R's cores write freed-slot counts to a relay core on R, which forwards each change to the sender core's
credit word by fabric inline write. Targets dealt round-robin over the 6 connections.
- First version: G 1 82%, G 4 M 2048 81%, but G 4 M 586 30% at 61 GB/s of weights: senders waited on their DRAM
  reader 55% of the time; the reader was issue-bound (~58 cycles a 576 B page read, ~10 GB/s a sender). 4 chunks in
  flight (transaction ids): 76 GB/s.
- Senders' DRAM layout made chunk-contiguous (slice_layout: chunk q's 24 tiles are pages (q // 8) 24 * 8 + q % 8 + 8 k,
  i.e. consecutive in one bank): one 13.8 KB read a chunk. 159 GB/s into R; senders now ~85% busy sending.
| G (cores) | M | cycles / expert | util tile | util useful | weights in | pcc |
|---|---|---|---|---|---|---|
| 1 (20) | 586 | 136,146 | 80.4% | 77.5% | 41 GB/s | 0.99994 |
| 4 (80) | 586 | 35,574 | 76.9% | 74.1% | 157 GB/s | 0.99994 |
| 4 (80) | 2048 | 117,068 | 78.7% | 78.7% | 48 GB/s | 0.99994 |
Same as the on-chip L1-landing numbers: the fabric is not the limit for one receiver with 6 incoming links
(~26 GB/s a connection used).

### Step 2: every chip runs experts (build_all, kernels wf_bsend / wf_relay_all)
All 8 chips run the 80-core groups on their own tokens; expert e is owned by chip e % 8, whose senders read its
slices from DRAM (chunk-contiguous) and land them everywhere: one 2D multicast a branch (E-only, W-only, N/S with E/W
teeth; the source chip is not part of a fabric multicast) plus a local NoC lander. Arrival counters became one word
a ring slot (consecutive experts of a group come from different owners and can land out of order).
Credits: a freed count u of core c (group g) frees exactly one later slot, expert g + (u + 1) G, so it goes only to
that expert's owner: the relay queues {owner, c, core, u} into this chip's link-0 branch sender of the owner's
first-hop direction, which sends it as a fabric inline write between its packets (the relay holds no fabric
connection, so link 1 carries data too: a second sender a branch, targets split by core parity).
| version | M 586 (util tile, all chips) | M 2048 | pcc |
|---|---|---|---|
| link 0 data, relay on link 1 | 23-25% | 70-72% | 0.99994 |
| both links data, credits via link-0 senders | 37-40% | 78-81% | 0.99994 |
M 586 is limited by the tree: with N/S trunk + E/W teeth the horizontal link into an end-of-row chip carries 6/8 of
all weights (its own row's 3 owners + the other row's 3 teeth) while its vertical link carries 1/8. At 70% (M 586)
that link pair would need ~107 GB/s (~54 GB/s a link). Measured now: ~30 GB/s a hot link, ~69 GB/s into a chip.
Next: a row trunk with vertical teeth (owner multicasts E/W in its row; each chip of that row forwards to its
vertical neighbour) puts at most 4/8 on any link (ideal 3.5/8).
