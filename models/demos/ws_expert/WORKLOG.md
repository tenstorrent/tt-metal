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

## Stage 6: store-and-forward (ws_sf.py, kernels wr_*.cpp; branch mstaletovic/ws-sf)
Every packet goes one fabric hop: a Hamiltonian cycle through all chips, per direction and link, is a directed ring
with one forwarder core per chip (cores (9..10, 0..3), outside the 80 group cores). Lanes = (cycle, link); the 20
group cores of an expert are dealt to the lanes (j % NL) and each slice goes both ways round its lane from its owner,
stopping half way (the half-way chip alternates slice by slice).
- Forwarder BRISC (wr_land): loads the ring's order table from DRAM, reads owned chunks from DRAM into CB 0 (RD 4 in
  flight, SCB 6), lands passing packets from its S = 12 staging slots (CB 1, 13.8 KB each) into the group cores'
  2-slot rings (warr inc fused after a slice's last packet).
- Forwarder NCRISC (wr_fwd): walks the same table, sends INJ / FWD packets one hop (fused write + atomic inc into the
  next chip's staging slot and arrival word) against per-hop staging credits; the freed count goes to the opposite-
  direction forwarder on the same chip, which sends it upstream as a fabric inline write (CRB 2 batching).
- Local lander (wf_sreader + wf_bsend WF_LOCAL, WS_ITEMS list); relay core (wr_relay) fans the group cores' freed-slot
  counts out to the chip's landers. No owner-wide credits; nothing travels more than one hop.
- Order without per-packet metadata: a synchronous conveyor model of all rings (hop time D = 3 packet times) gives
  per chip and ring the time order of INJ / FWD / TERM entries (wr_seq.hpp); both RISCs follow it.

Model (conveyor; link busy fraction = max link load / model steps):
- expert-major order (all of an expert before the next): ~56%, independent of S (owners serialize round the ring);
- conveyor order: tray 98% (async replay with S >= 8, hop latency <= 2, credit latency <= 6 packet times: 98.5-99%).
| window (expert n reaches a chip after experts <= n - W of its group) | owners | tray 32 exp | 8 x 4 torus 128 exp |
|---|---|---|---|
| n - 2 (deadlocks on the device, see hang 1) | e % N | 98.4% | 46.8% |
| n - 1 (safe) | e % N | 53.0% | 49.5% |
| n - 1 (safe, current) | per slice, spread round the cycle | 98.3% | 77.3% |
| n - 2 / n - 3 (would need a 3-slot group ring) | per slice | 99.4% / 99.4% | 86.8% / 89.4% |
Owner queues split by n mod 3 change none of these: the window rule is monotone in n, so expert n + 1 is never ready
while n is held, and an in-order owner queue never limits. The torus bubble is spatial (with whole-expert owners the
experts in flight sit on a few adjacent chips); per-slice owners remove most of it, the rest is the window latency
(16-hop paths at D = 3).
Host checks: test_sf_schedule (stream consistency, slice coverage, local lists) and test_sf_replay (random-timing
replay of the device semantics: forwarder RISCs, staging credits, slot credits with the down lag, local lander);
the replay completes on the current tables and deadlocks on the n - 2 tables.

Runs (hold 213192, bh-glx-120-b06u02, tray = chips 0-7, sub_2x4 descriptor, FABRIC_2D, payload 15232):
- ws_sf_a (expert-major commit 9cf40e7fffc): host TypeError composing RT args (list += VectorUInt32); no device run.
- ws_sf_a2 (tables, window n - 2), hang 1, first launch tray M 586: all forwarders polling (landers on arrival /
  slot credit, local lander on slot credit), group cores waiting for weights. Cause: wg_compute's down rows lag
  DN_LAG rows into the next expert, so a core frees expert n - 2's slot only once n - 1's weights are in; expert n's
  first packet waited for that slot while n - 1's packets queued behind it. Reset exit 0, 32 boards re-initialised.
  Fix 2ccf341f4a9: window n - 1, per-slice owners, per-group owner queues, local lander in model-time order.
- ws_sf_a3 (2ccf341f4a9): sf_quick PASS (0 warm-up + 1 measured launch a case, cold):
| M | util tile per chip (median, min-max) | weights in | ring out (send span) | pcc |
|---|---|---|---|---|
| 586 | 33.6% (33.3-33.8) | 68-69 GB/s | 14.1-14.6 GB/s | 0.99994 |
| 2048 | 77.5% (77.2-78.2) | 47 GB/s | 11.0-11.3 GB/s | 0.99994 |
  Fabric slots per connection (nbuf) 7. M 586 forwarder NCRISC (2.7 M cycles a ring): send 54%, arrival wait 14%,
  downstream credit 2.5%, the rest bookkeeping / credit relays; BRISC landing 48%, arrival wait 33%. The rings run
  at ~60% of a link: forwarder-issue-bound (each send waits for its 13.8 KB payload to leave L1, and the L1 inline
  writes flush it too), not link-bound. WS_DEFER (860c7252813, not run yet) overlaps that drain with the next packet.
  Hang 2 (sf_bench, M 586 warm-up, launch unknown): all cores alive and polling; forwarders 40-50% through
  (240-303 of 600 INJ a ring and chip), local landers 38-48 of 80 slices waiting on slot credits, group cores at
  expert 3-4 of 8 waiting for weights. On chips 0, 2, 5 and 7 the triage CB counters do not match the model's
  expectation (the local lander's next slice's core shows enough freed slots, and a slice the lander should already
  have landed is missing at its core); unexplained. WS_DIAG (e6a02c27c52) puts table positions, counts and the
  credits seen into spare CB counters that the triage dump lists, for the next hang. Reset exit 0.
- ws_sf_a4: device open failed ("Querying size for a host channel that does not exist"): chip 3 of the Galaxy lost
  its host channel (IOMMU domain changed during another session's resets), not this code. ws_sf_a5 (pre-reset, diag
  bench) stopped before its reset ran; no further Galaxy runs.
Next: on a healthy Galaxy, ws_sf_a5 (cmds/ws_sf_a5.sh without its pre-reset): the WS_DIAG bench to pin down hang 2,
then 20 + 5 plain and WS_DEFER; stage B (cmds/ws_sf_b.sh) once the tray is clean.
- Hang 2, second look: the triage device ids are not mesh positions (on that tray device d was mesh chip (d + 4) % 8);
  mapped right, the CB counters are consistent everywhere (no slice missing that a lander claims to have landed).
  The stall is a NOC0 wedge on one chip, inside group 3 (logical columns 7-8): check_noc_status shows the column-7
  writers with x multicasts issued but not sent (3 writes, 60 acks missing) and further cores with a write / atomic
  ack missing; their BRISCs sit in noc_async_writes_flushed / noc_async_write_barrier, the group's TRISCs wait for an
  h row, and everything else waits on that group's slot credits. Hang 1 had no worker NoC mismatch (a logical
  deadlock, as the replay shows). Candidates: the forwarders' NOC0 landing writes (from columns 9-10, wrapping round
  the row) crossing the group's linked NOC0 multicasts. RCfg.swap_noc moves the landing writes to NOC1 for an A/B.
- ws_sf_c2 (hold 213479, bh-glx-110-c10u14, compute grid 12 x 10; tray = chips 0-7): five variants, each 20 warm-up
  + 5 measured launches a case, all PASS (250 launches, no hang), PCC 0.99994 everywhere:
| variant | M 586 util tile median (min-max) | M 2048 |
|---|---|---|
| plain | 33.3% (32.7-34.0) | 77.9% (75.9-79.1) |
| WS_DEFER | 33.2% (32.7-33.9) | 78.0% (76.3-79.2) |
| WS_DIAG | 28.7% (28.2-29.2) | 78.2% (76.7-79.7) |
| WS_DIAG + swap_noc | 28.3% (27.7-28.7) | 78.9% (77.8-79.9) |
| WS_DEFER + swap_noc | 32.0% (31.5-32.8) | 78.8% (77.8-80.3) |
  M 586 weights in 68 GB/s a chip, every ring 14.0-14.6 GB/s over its send span in all variants. WS_DEFER cuts the
  NCRISC send time (1.49 M -> ~1.0 M cycles a ring) but the arrival wait grows by the same amount: the forwarder RISC
  is not the limit. Suspect: every packet is a fused write + atomic increment with flush, so the downstream router
  waits for each write's ack before the increment (step 1 fused only a slice's last packet and reached ~26 GB/s a
  connection). RCfg.arr_noflush drops that wait (ordering then rests on the router's write and increment reaching the
  same core in order; PCC guards it).
- ws_sf_c3: tray arr_noflush 20 + 5: M 586 34.1% (33.4-35.0), M 2048 78.3% (77.1-79.4), PCC 0.99994, rings 14.7
  GB/s: the router's ack wait is not the limit either. Every tray variant lands at ~14.5 GB/s a ring direction with
  the forwarder RISCs partly idle, so the limit looks like the fabric link / router carrying a ring each way (one
  direction alone reached ~26 GB/s a connection in step 1).
  Stage B (8 x 4 torus XY, 128 experts, T 8192, per-slice owners, 8 rings, fabric slots 6), one cold launch with
  WS_DIAG: M 586 34.0% (33.3-35.2), M 2048 78.2% (77.2-79.0), PCC 0.99994 on all 32 chips.
  Hang 3 (stage B 20 + 5, M 586, launch 18): everything polling except one group core (0,7) on triage device 25,
  whose writer sits in noc_async_read_barrier on its first DRAM read of the launch (the counts page); check_noc_status:
  one NOC0 read response missing there, DRAM health clean. Reset exit 0, 32 boards re-initialised.
- ws_sf_c4: stage B 20 + 5 a case: M 586 PASS, util tile median 34.4% (33.4-35.3) over 32 chips x 5 launches, PCC
  0.99994, weights in ~70 GB/s a chip, rings 8.1-8.3 GB/s each (8 rings: the same ~70 GB/s into a chip as the tray's
  4 rings; the window-latency model gives 77% for these tables). M 2048 stopped at warm-up launch 20 of 25 when hold
  213479 ended (completion queue error, no device timeout); no measured M 2048 launch.
| | M 586 (20 + 5) | M 2048 |
|---|---|---|
| tree multicast, tray (ws_lb) | 30.6% | 77.6% |
| store-and-forward, tray | 33.3% (34.1% arr_noflush) | 77.9% (78.3%) |
| tree multicast, torus 640 experts | 28.7% | - |
| store-and-forward, torus 128 experts | 34.4% | 78.2% (one cold launch) |
- 640 experts (hold 213581, c10u14 after its reboot; T 31250, M 586, 8 x 4 torus XY, per-slice owners, arr_noflush;
  the local lander's 400-slice list moved into the order table's last page, tables 24912 entries a ring):
  ws_sf_d1 hang 4 at warm-up launch 4: device 19's group-2 column-0 writers (4,0) and (4,7) in noc_async_writes_flushed
  after an x multicast, 3 multicast writes issued but not sent (60 acks missing) - hang 2's signature again. The
  landers' NoC atomics into the group cores (warr increments) are suspects (on this architecture an atomic or inline
  write to a core whose memory ports are back-pressured can hang; inline writes are emulated, atomics are not), so
  the landers now write the slot's absolute arrival count instead (9c93c1bafdc). Reset exit 0.
  ws_sf_d2 (no lander atomics): 640 experts 20 + 5 PASS, util tile median 34.6% (33.4-35.9), PCC 0.99994 on all 32
  chips; multicast tree on the same Galaxy and shape: 28.5%. Weights into a chip ~70 GB/s; each of the 8 rings
  8.6-8.7 GB/s. Chip 0 ring 0+ (50 M cycles): NCRISC arrival wait 38%, send 40% (847 cycles a packet), downstream
  credit 7.5%, relays 16.9 K for 23.4 K packets; BRISC arrival wait 47%, land 29% (627 cycles a packet), group slot
  wait 10%. Group cores: own weight wait 9% but h wait 47% - a group starts an expert only when all 20 slices are in,
  so the spread of arrival times across rings shows up as h waits.
  ws_sf_d3 (forwarder variants at 640 experts): CRB 6 hang 5 at warm-up launch 11 - the same NoC0 wedge (device 23,
  group 3's column-0 leaders (7,1) and (7,4) in noc_async_writes_flushed after an h / x multicast, 3 multicast writes
  issued but not sent): removing the lander atomics did not stop it. Reset exit 0. The next step (S 16) lost the
  chips at warm-up launch 11: completion queue error, then host MMIO reads timing out (4 B load 220 ms against a 2 ms
  budget), and the node rebooted (hold 213581 gone at ~20:52). The earlier unexplained reboot (~19:58, hold 213479)
  had the same sequence during ws_sf_c4 (M 2048, launch 20, MMIO per-op timeout on device 5). Likely chain: the
  intermittent NoC0 wedge in a group's multicast stalls the chip's NoC, host MMIO through it times out, the host
  goes down. Hang rate at 640 experts: 3 of 4 runs (launches 4, 11, 11), 1 clean 25-launch run.
- Table spacing (async replay of one torus ring, 128 experts, hop latency L in packet times): tables built with
  D 3 / 4 / 5 / 6 / 8 run at 76.8 / 78.4 / 80.8 / 80.7 / 78.0% for L 2 and 63.4 / 73.1 / 75.5 / 73.7 / 68.9% for L 3:
  sparser tables (D 5) absorb hop latency better than the synchronous model suggests.
- Third in-core weight slot (test_ws_slot3, single chip of hold 213684, c10u20: group program with injector-L1
  weights, P 2, G 4, M 586, 32 experts, 20 warm-up + 5 measured; GCfg.spare_slot reserves the 207 KB slot):
| token-row ring (bfp8) | weight slots | util tile median (min-max) | x wait a core (cycles) |
|---|---|---|---|
| BK 24, NX 15 (current) | 2 | 78.7% (78.1-79.9) | 10K |
| BK 24, NX 15 | 3 | does not fit | |
| BK 20, NX 9, LOOK 18 | 3 | 76.0% (75.2-76.1) | 96K |
| BK 12, NX 15, LOOK 30 | 3 | 74.0% (73.1-74.5) | 107K |
| BK 24, NX 7, LOOK 14 | 3 | 67.9% (67.7-68.2) | 216K |
| BK 12, NX 20 / 24 | 3 | does not fit beside this test's injector staging (no injectors on a fabric chip) | |
  PCC 0.99994 throughout. Down outputs from the NCRISC on NOC1 (GCfg.y_noc1): 79.5% vs 77.7% from the BRISC on NOC0.
- ws_sf_f1, 640 experts torus, one lane a group, D 5, CRB 6, arr_noflush, down outputs on NOC1: first measured launch
  35.5-36.4% a chip (warm-up 20 done), then launch 22 hung (device timeout at 21:59:39) and host MMIO reads to device
  27 timed out at 22:01:27; the watcher cancelled the step and the node stopped answering new job steps.

### Design B: lanes with in-core forwarding (ws_lane.py, ws_lane_sched.py, kernels wl_*.cpp)
- Group g's experts travel lane g (one Hamiltonian cycle x one link, both directions); no forwarder cores. Group core
  j sends its slice of an expert one hop, 15 packets of 24 bfp4 tiles, into core j of the next chip through the
  group's mux core of that direction (fabric_mux, 20 full-size channels, 2 buffers each), straight into a weight
  slot (fused write + arrival increment). 3 slots a core: 0 takes experts arriving on direction 0, 1 on direction 1,
  2 any, including the chip's own experts (read from DRAM by the core's BRISC). An expert's down rows drain before
  the next expert starts, so a slot frees once its expert is computed and forwarded.
- Static schedule (ws_lane_sched.lane_sched): a lock-step model of one lane with typed slots and owner hold-back
  (keep 1) records per chip and slot the experts it holds in turn and per chip the compute order; the device follows
  only these orders: occupant k of a slot is sent (or read) once the receiving slot freed occupant k - 1 (credit:
  inline write one hop back through the other direction's mux, or a local word for own experts).
- Layout: groups in columns (1,2), (4,5), (7,8), (10,11), their two mux cores at rows 4 / 5 of the column just west
  (0, 3, 6, 9): core -> mux traffic (NOC1: north, then west) stays out of other groups' columns. The token-row ring
  shrinks to BK 20, NX 9, LOOK 18 so that the third slot fits; MMAX = T (no multi-pass for any M).
- Host checks (test_lane_tables): the schedule completes under random timing (model util 0.563 / 0.597 / 0.557 for
  tray E 16 / tray E 32 / torus E 160; random timing 0.516 / 0.556 / 0.482), and the tables drive a model of the
  device queues (own reads, weight pushes, compute, forwards with fabric delays, releases) to completion, never
  landing a packet in a slot that still holds another expert.
- Tray (hold 213824, c10u20, chips 0-7, sub_2x4, FABRIC_2D; row snake x 2 links = 2 lanes, G 2, 32 experts, T 8192):
  first hardware run correct: PCC 0.99994 on all 8 chips for each group's last expert and for every expert
  (out_all), M 0 and M 5 pass, dynamic M (per chip counts 0 / 1 / 31 .. 8192 = T, every expert of chips 1 and 0)
  min PCC 0.99993. Performance (20 warm-up + 5 measured, util tile median over chips and launches):
| reader / writer | M 586 | M 2048 | M 5 (weights only) |
|---|---|---|---|
| first version (blocking fabric sends, flush after each send, own-read barriers) | 34.2% | 52.4% | 3.1% (1 launch) |
| non-blocking sends (mux space checked), flush before the next send, own reads retired without barriers | 41.6% | 59.2% | 3.2% |
| + slots polled from registers | 44.0% | 64.8% | |
| + weight queues every 2nd loop turn (wpoll 2, now the default) | 44.1% | 66.7% | |
| + weight queues every 4th loop turn (wpoll 4) | 43.5% | 68.0% | |
  The store-and-forward tray best was 34.1% / 78.3%. M 5 is weight-bound: 82.6 K cycles a chip-expert = 122 us an
  expert a lane, i.e. 11.3 model steps of 276 KB at 24 GB/s - the lane model's transfer bound (11.3 steps an expert
  at C 1) at the mux bench rate. At M 586 the remaining waits a core are x (0.87 M cycles of 3.5 M) and weights
  (0.55 M); 4 mux buffers a channel instead of 2: no change. The reader's loop spends most of its time in the
  weight queues (QW / QF / QR), which delays the x / h queues on the same RISC; serving them less often helps
  M 2048, not M 586 (chip 0 47%, the tray's slowest chips 41-43%).
- Lane model vs tray at M 586: lock-step 0.563, random timing 0.516; measured 44%. Torus (E 160 a lane): lock-step
  0.557, random timing 0.482.
- Whole Galaxy, design B with 3 slots (ws_lane_b, hold 213824, c10u20, 8 x 4 torus XY, 2 cycles x 2 links = 4 lanes,
  G 4, wpoll 2): 128 experts M 586, 1 launch: PCC 0.99994 on all 32 chips (util 31.4%, a cold launch of 32 experts a
  lane). 640 experts, T 31250, M 586, 20 warm-up + 5 measured: util tile median 36.2% (35.1-37.2), PCC 0.99994 on
  all 32 chips, no hang (store-and-forward 34.6%, multicast tree 28.5%). Chip 0: 74.9 K cycles a chip-expert, weights
  in 75 GB/s; waits a core x 11.1 M / weights 13.7 M cycles of 47.6 M. Dynamic M (128 experts, per-chip counts with
  581 zeros, up to M = T = 31250, every expert of chips 1 and 0): min PCC 0.99991.
- 4 weight slots (types d0, d1, d0|d1, any; owner keep 2) in the lane model: lock-step 0.767 / 0.777, random timing
  0.664 / 0.698 (tray E 16 / torus E 160) against 0.563 / 0.557 and 0.516 / 0.482 with 3 slots - the model becomes
  compute-bound (C 9 of 7 useful). The 4th slot (207 KB) needs the remaining L1 beside BK 20 / NX 9.
- 4 weight slots on the tray (ws_lane_t5; LCfg slots d0,d1,d0d1,any, keep 2): the 4th 207 KB slot fits beside the
  BK 20 / NX 9 token-row ring without shrinking anything. Every expert PCC 0.99994 on all 8 chips; 20 warm-up + 5
  measured: M 586 51.2% (46.4-54.8; chip 0 53.5%), M 2048 70.1%; dynamic M (every expert of chips 1 and 0) min PCC
  0.99992. Weight waits a core at M 586 fell from 0.55 M to 0.30 M cycles; x waits 0.76 M remain.
- Whole Galaxy, 4 slots (ws_lane_c): 128 experts M 586, 1 launch: PCC 0.99994 on all 32 chips, util 47.2% (cold, 32
  experts a lane; 31.4% with 3 slots). 640 experts: hang 6 at warm-up launch 4 - device 18, group 2's column-0 core
  (7,9) BRISC in noc_async_writes_flushed after an x multicast (wl_writer QX), NOC0: 3 multicast writes issued but not
  sent, 60 acks missing - the signature of store-and-forward hangs 2, 4 and 5, here with no NOC0 landing writes at all
  (routers land on NOC1). The group program's multicasts were linked, with the flag multicast issued right after the
  payload; on this architecture the two go through different command buffers that are not sent in issue order (the
  matmul readers flush between payload and flag multicasts for this reason), so a linked payload can be left as a
  source's last multicast with its path reserved. Reset exit 0, no MMIO timeout. Fix, at all four group multicast
  sites: the flag as a 4-byte multicast write through the payload's own command buffer, so it always follows (and
  closes) the linked payload. Variants on the tray (4 slots, 20 + 5, util median M 586 / M 2048):
| group multicast | M 586 | M 2048 |
|---|---|---|
| linked payload, flag via noc_semaphore_set_multicast (before) | 51.2% | 70.1% |
| unlinked payload, flush, then the flag (8cfb3bed18f) | 48.7% | 61.6% |
| unlinked payload, flag from the same command buffer | 49.1% | 62.3% |
| linked payload, flag from the same command buffer (20bf7889aa5, kept) | 50.8% | 70.1% |
  All PCC 0.99994, dynamic M min PCC 0.99992. The cost of the first two is the unlinked payload, not the flush.
- Whole Galaxy, 4 slots with the same-buffer flags (ws_lane_c2): 128 experts 1 launch PCC 0.99994 on all 32 chips,
  util 46.6% (cold). 640 experts: warm-up launches 1-20 clean, then hang 7 in launch 21 (the first measured one,
  device timeout at 00:10:09), host MMIO reads to device 29 timed out (4 B load 222 ms against 2 ms), the watcher
  cancelled the step and killed the container, and the node stopped answering (NOT_RESPONDING, ssh refused) - the
  same chain as the store-and-forward reboots. No measured 640 number with 4 slots. The flag reordering fix moved
  the hang from launch 4 to launch 21 but did not remove it; no triage was possible after the node went down.
- Group multicasts unlinked by default since 66c08bbde38 (LCfg.mcast_linked / WL_MCAST_LINKED restores linked
  payloads; flags from the payload's command buffer either way). Tray soak at the 640-expert lane load (ws_lane_s1,
  hold 214106 after the reboot; 160 experts a lane, T 31250, M 586, 4 slots, G 2; about 60 ms a launch, util tile
  55% a chip): with linked payloads the soak hung after about 600 launches (37 s; device timeout at 00:23:22), host
  MMIO reads to device 2 timed out (4 B load 224 ms against 2 ms), and the watcher cancelled the test step and
  killed the container; the node stayed up. The unlinked soak did not run (stopped at the MMIO timeout).
- Unlinked tray soak (ws_lane_s2, same load; util tile 51.5% a chip against 55% linked): hung after about 170
  launches (12 s; device timeout at 00:31:09), no MMIO timeout; harness triage, reset exit 0. check_noc_status: device
  5, group 0's column-0 core (1,7) BRISC NOC0, every write sent but the 20 acks of its last multicast missing (the
  linked hangs: writes issued but not sent). Unlinked multicasts do not remove the wedge; the hang is somewhere in the
  group program's NOC0 traffic under fabric load, not in the multicast linking. No further runs.
- Isolation soaks (hold 214106): (a) the single-chip group program (ws_group, on-chip injectors with NoC atomic
  arrival increments, linked multicasts, 160 experts a group, G 4, T 31250, M 586, BK 20 / NX 9): 2267 launches in
  180 s, no hang, util tile 76.7%, PCC 0.99994 - the wedge needs the lane design's fabric traffic. (b) the lane design
  on the tray with the arrival counts as plain writes after the payloads (no router atomics; unlinked multicasts):
  util tile 42.5%, hung after about 150-200 launches (device timeout at 00:47:36), host MMIO reads to device 5 timed
  out (4 B load 56 ms), the watcher cancelled the test step, and the node went down and rebooted (boot 00:49:23).
  Router atomics are not the cause either. Not run: (c) column 0's multicasts on NOC1 (WL_MC_NOC1, built, untested).
- Wedge isolation, continued (hold 214691 on bh-glx-110-c10u20, tray soak at the 640-expert lane load, 4 slots):
  - (c) column 0's multicasts on NOC1 (WL_MC_NOC1, ws_lane_s3): hung in under 50 launches. Device 2, one group's
    cores (4,8) and (5,8), both NCRISCs stuck inside their own group multicasts on NOC1 (an h-row flag write not
    accepted by the command buffer, an x multicast not flushed). The wedge follows the multicasts to the other NoC.
  - (d) one multicaster a group a NoC (LCfg.bcast, ed5d86fc17f): every core's x blocks from its BRISC on NOC0 in
    stream order (one x multicast in flight), h tiles gathered on core 0 and each row multicast from its NCRISC on
    NOC1 (one h multicast in flight). Tray PCC 0.99994, M 0 / 5 pass, dynamic M min 0.99992, util 24.6% (x waits:
    one serialized x stream from 20 sources). Soak (ws_lane_s4): hung in under 50 launches. Device 0 core (1,4)
    BRISC NOC0: 2 writes of one x multicast issued but not sent, 40 acks missing; no other multicast in its group.
  - (e) the same with each multicaster at its NoC's multicast start corner (237feae6653): x blocks funneled on NOC1
    into core 0's 2 funnel slots and multicast from core 0 on NOC0; h rows multicast from core 19 on NOC1; x staging
    2 blocks smaller (L1 unchanged). Tray PCC 0.99994, M 0 / 5 pass, dynamic M min 0.99992, util 44.9% (20 + 5).
    Soak: launch 50 at 44-45% (7 s), hung before launch 100. Device 4 core 0 (logical (1,0), NOC0's start corner)
    BRISC NOC0: its x multicast (payload + flag) issued but not sent, 40 acks missing; every other RISC polling.
  - So the wedge is not concurrent multicasts into one rectangle, not the multicast's path around the chip to its
    start corner, not NOC0 or NOC1 in particular, not linking and not router atomics: a single multicast from the
    group's corner, alone on its NoC in that group, cannot leave its core once the lane design's fabric traffic is
    running. Not tested: the routers' landing writes into the group cores, the cores' NOC1 writes into the mux
    cores, DRAM read responses into the group, the BRISC's DRAM reads with transaction ids while it multicasts.

- Single-chip data-movement repro of the NOC0 group multicast wedge (tests/test_mcast_wedge.py, kernels/mw_mcast.cpp,
  mw_recv.cpp, mw_bg.cpp; no compute, no fabric). The lane group's x multicast exactly (21,760 B payload + 4-byte flag,
  loopback, unlinked, 20 destinations in the 2 x 10 group, flag from the payload's command buffer, flush only) from the
  group's NOC0 start corner (BRISC), NX 9 ring slots acknowledged by every receiver's NCRISC on NOC1; switchable NOC0
  background (MW_BG gather / own / h / stream, MW_SELF, MW_MC=0 control, MW_BG_NOC1, MW_GQ, MW_MC_BARRIER).
  bh-glx-110-c10u20 chip 5 (TT_VISIBLE_DEVICES=5), cmds/mc_wedge_1..4.sh:

  | Step | Traffic | Result |
  |---|---|---|
  | m0_alone | multicast only, 180 s | 208.4M multicasts clean, 25.2 GB/s |
  | c1 | gather + own DRAM reads on 19 receivers and the multicaster, no multicast, 60 s | clean |
  | c2 | multicast + the multicaster's own DRAM reads, 120 s | 61.2M multicasts clean |
  | m1_A | multicast + gather + own reads on the 19 receivers (NOC0) + the multicaster's | dead in about 3 s: MMIO timeouts, host rebooted |
  | c3 | multicast + gather reads on the 19 receivers (NOC0) | dead at once: MMIO timeouts, host rebooted |
  | d2 | as c3, 4 reads in flight a core | hang in seconds: multicaster (1,0) BRISC NOC0 2 writes issued not sent, 40 acks missing (the lane signature); 16 receivers' BRISCs in noc_async_read_barrier; chips reset, host up |
  | d3 | as c3, write barrier after each multicast | dead in seconds: MMIO timeouts, host rebooted |
  | d1 | as c3, the receivers' reads on NOC1 (NCRISC) | 124.3M multicasts in 120 s clean, 22.5 GB/s |

  Minimal trigger: a NOC0 multicast into the group while DRAM read responses stream into the receiving cores on NOC0
  (even 4 x 1280 B reads in flight a core). Reads into the multicaster alone, and the reads without the multicast, run
  clean. Capping the reads or a barrier after each multicast does not help; the same reads on NOC1 do. Not run: stream
  (mux-like) and h traffic alone, own-expert reads alone, the NOC1 fix with own reads.

## Stage 7: NoC split (hold 215516, bh-glx-110-c10u20)

Rule from the repro: on a group's multicast NoC nothing may enter the group rectangle from outside.
- v1 (3eb2f7e0646, LCfg.noc_split / WL_NOC_SPLIT): every DRAM read of a group core (counts, slot table, token-row
  gather, own experts' weights) moves to its NCRISC on NOC1, the muxes to NOC1, and every group multicast to the BRISCs
  on NOC0 (both columns, x in stream order). Tray correctness (ws_lane_n1): M 586 every expert PCC 0.99994, M 0, M 5,
  dynamic M min 0.99992. Tray soak at the 640-expert lane load: hung before launch 50, two groups' BRISCs (device 1,
  (2,3) and (5,5)) inside their NOC0 multicasts (2 writes not sent, 40 acks missing each). A multicast from a core that
  is not at the rectangle's NOC0 start corner first travels east / south round the chip to that corner, through the
  other groups' rectangles: the same class of fault, multicast on multicast.
- v2 (52a5d8ba245, bcast + noc_split): core 0's BRISC, at NOC0's start corner, is the group's only multicaster, of the
  x blocks and of the h rows (new queue QM); every producer sends its x blocks (funnel slots) and its h tiles (new
  NCRISC queue Q2: one transaction id per tile, the slot's count raised at core 0 once acknowledged) to core 0 on NOC1.
  NOC0 carries only core 0's multicasts, which never leave the rectangle; NOC1 carries only unicasts (DRAM reads, the
  funnel, h tiles, credits, muxes, router landings, outputs).
  - Tray (ws_lane_n4): M 586 every expert PCC 0.99994 (42.7% one cold launch), M 0, M 5, dynamic M min 0.99992. Soak:
    12,393 launches in 900 s without a hang, util tile median 47.1%, PCC 0.99994.
  - Whole Galaxy, 8x4 XY torus, G 4 (ws_lane_n5): 128 experts one launch PCC 0.99994 (34.4%); the real layer (640
    experts, T 31250, M 586), 20 warm-up + 5 measured launches:

| slots | util tile median (min-max) over chips and launches | per expert (chip 0 / chip 31) | weights in a chip | min PCC |
|---|---|---|---|---|
| 4 (d0, d1, d0d1, any; keep 2) | 33.8% (32.6-34.9) | 60.8 / 60.6 us | 68 GB/s | 0.99994 |
| 3 (d0, d1, any; keep 1) | 29.2% (28.1-29.8) | 69.3 / 69.0 us | 60 GB/s | 0.99994 |

    Dynamic M on all 32 chips: min PCC 0.99992. No hang and no reboot in any v2 run. Against the old 3-slot torus
    result (36.2%, linked multicasts from every leader, which hung with 4 slots), the hang-free layout gives 33.8% with
    4 slots. The cores wait mostly on x (x waits about 20M cycles a launch on chip 0, weight waits 1.3-4M with 4 slots):
    core 0 multicasts every x block and h row of its group (about 3.3 MB an expert at M 586) and takes in the group's
    funnel and h tiles on NOC1, so its two NoC ports pace the group.
- Single-chip proxy of the lane design (dd87aac97fb: LCfg.nofab / WL_NO_FABRIC, test_ws_lane_chip; hold 215516,
  bh-glx-110-c10u20, chip 5 via TT_VISIBLE_DEVICES, no fabric): the no-hang layout (bcast + noc_split), every expert
  the chip's own and read from DRAM into 4 slots of any kind (keep 0), nothing forwarded, G groups on the chip's
  tokens at the Galaxy per-group load (160 experts a group, T 31250). Correctness: 64 experts G 4, M 586 (every
  expert) / M 0 / M 5 PASS, PCC 0.99994. 20 warm-up + 5 measured, util tile median (min-max), cyc/expert of the chip,
  UNPACK waits (K) tilize/x/weight/h:
| case | proxy (one chip) | waits | fabric run |
|---|---|---|---|
| G 4, M 586 | 46.1% (45.8-46.3), 59.3K | [1107, 14334, 49, 1372] | torus 640: 33.8% (33.0-35.0), 82.6K, [1091, 20692, 1727, 1192] |
| G 2, M 586 | 53.0% (53.0-53.0), 103.3K | [1092, 11256, 49, 633] | tray soak 320: 47.1% |
| G 4, M 2048 | 53.5% (53.4-53.6), 172.1K | [3599, 38090, 159, 1205] | |
| G 2, M 2048 | 55.0% (55.0-55.1), 335.0K | [3596, 35615, 159, 770] | |
  AICLK: the tests read /sys/class/tenstorrent/tenstorrent!0/tt_aiclk, an idle chip (800 MHz) when only chip 5 is
  open; the lane tests now read the first opened chip at the start and the end. Sampled at 4 Hz (ws/power/clog.py):
  chip 5 starts at 1350 MHz and the firmware lowers it under load (measured windows 1087-1350, G 4 M 2048 the
  lowest); the Galaxy run's 32 chips stay at 1350 for about 90% of the samples (min 1206). The first proxy pass
  (reported as 800 MHz) gave the same numbers within 0.1 point.
  Reading: G 2 -> G 4 on one chip costs 13% at M 586 (3% at M 2048); the fabric runs lose a further 11% (tray, G 2) and
  27% (torus, G 4) against the proxy, with larger x waits and weight waits appearing on the torus. x waits dominate
  everywhere. The proxy's slots are untyped (any x 4, keep 0) while the fabric runs use d0,d1,d0d1,any with keep 2, so
  part of the fabric gap may be the typed-slot schedule rather than the links.

## Stage 9: every expert's outputs written by default (out_all)
- Until now the lane runs wrote only each group's last expert's down rows (the rest were computed and dropped), so the
  perf numbers left out almost all output traffic. LCfg.out_all is now True by default (3ea09c9a45f): QY on the NCRISC
  (NOC1) writes every expert's rows, packed in compute order per group; the tests size the output block for every
  expert (soak included), allocate it on the device without a host copy, and check a sample (WS_PCC_EXPERTS: each
  group's last expert plus k evenly spaced; WS_PCC_CHIPS) read back row block by row block. out_all=0 restores the old
  behaviour.
- No-hang layout (bcast + noc_split), 20 warm-up + 5 measured, hold 215516 on bh-glx-110-c10u20, out_all on vs off:
| case | out_all on | out_all off | output writes a chip |
|---|---|---|---|
| tray, 32 experts, M 586 | 42.7% (40.5-43.5) | 45.4% (42.0-48.4) | 49 GB/s |
| tray, 32 experts, M 2048 | 48.9% (48.6-49.7) | 54.3% (53.8-54.5) | 57 GB/s |
| tray soak load (160 experts a lane, T 31250, M 586), 180 s | 40.9% | 47.1% | |
| one chip, G 4, M 586 | 36.4% | 46.1% | 84 GB/s |
| one chip, G 2, M 586 | 46.0% | 53.0% | 53 GB/s |
| one chip, G 4, M 2048 | 44.5% | 53.5% | 103 GB/s |
| one chip, G 2, M 2048 | 49.9% | 55.0% | 57 GB/s |
| Galaxy torus, 640 experts, M 586 | 27.2% (26.4-28.1) | 33.8% | 62 GB/s |
| Galaxy torus, 640 experts, M 2048 | 43.3% (42.2-43.9) | not run | about 99 GB/s |
  Correctness with out_all: tray M 586 every expert PCC 0.99994, M 0, M 5, dynamic M min 0.99992; one chip 64 experts
  every expert 0.99994; torus PCC 0.99994 on the sampled experts of chips 0, 13, 31. About 48 ms a layer on the torus
  at M 586 (640 x 75.3 us), 103 ms at M 2048 (640 x 161 us).
- The output writes cost 3-10 points; they go out from the NCRISC on NOC1, the RISC and NoC that also carry the x
  funnel, the token-row gather and the fabric forwarding: on the torus at M 586 the compute cores' x waits grow from
  20.7M to 31.2M cycles (chip 0) and the NCRISC weight-queue loop time from 27.5M to 39.4M. QY retires each output row
  with a write barrier inside the NCRISC loop.

## Stage 10: the MoE layer around the stream (ws_layer.py, tests/test_ws_layer.py)
- Every phase runs on one chip, on that chip's own T tokens. The router and shared-expert weights are replicated, and
  only the routed stream crosses chips (its weight streaming). The phases in order:
  - head (ttnn ops):
    - router matmul (fp32 out, HiFi4), softmax (or sigmoid), top-12, gate weights renormalized over the 12;
    - the stream's inputs, built on device: the one-hot routing mask (scatter), its running count over the tokens
      (cumsum: a routed row's position in its expert's list), the counts (the count's last row), the per-expert token
      lists (t scattered to its position in a list of T; non-routed pairs index past the list and scatter skips them),
      and each routed row's output row.
  - shared expert: the same lane kernels.
  - routed stream: build_lane with the new LCfg.out_eord.
  - combine (ttnn ops).
- Expert-id-ordered outputs (LCfg.out_eord, WL_OUT_EORD; needs noc_split):
  - Expert e's down rows start at tile row out_rows + the sum of ceil(M / 32) over e' < e, so a routed row's output row
    is base(e) 32 + position whatever the arrival order. The head computes the same base with a cumsum.
  - Each NCRISC computes the prefix once from its counts read, for its group's experts only (e = i G + g, nexp / G
    words past the counts). A full nexp-word prefix (+2.7 KB) pushed the 640-expert stream's L1 tensors 1.5 KB into
    its static CBs (program build error).
  - Output capacity: ceil(T K / 32) + nexp tile rows bounds the sum for any routing.
- nofab on a mesh: every chip runs the one-chip program on its own inputs (the one-chip tables on each). The shared
  expert runs this way on the Galaxy.
- Shared expert on the expert machinery:
  - The inner dim 2944 is split into 5 slices of 640; the last holds 384 real columns plus 256 zero columns. The token
    chunk is TC = ceil(T / 128) 32 (7840), so there are 20 units u = i 4 + g (slice i, chunk g) on the 4 groups, 5 each.
    Each unit is one expert with M = TC (the padding tokens read x's zero row).
  - With expert-id order the units' outputs are [5 slices][4 chunks][TC] rows, so the shared output is the sum of the
    5 slice blocks.
  - A 384 slice cannot run unpadded: build_lane needs PAIRS = I / 32 = GC = 20 (each of a group's 20 cores owns one
    gate/up pair). The padding is 8.7% more work.
  - M limits do not bind: GCfg MMAX = T (RMAX = ceil(T / 32)), so one unit could even take M = T; the 4 chunks are
    there for the 4 groups.
- Combine:
  - The routed rows are untilized (the stream writes tiles).
  - Then per 32-token block: an embedding gather of its K 32 rows (block, k, token order), times the block's gate
    matrix [32, 32 K] (w (t, k) at column 32 k + t) as one batched matmul, plus the shared output.
  - The eltwise weigh-and-sum over k was slower (64.0 vs 47.8 ms on one chip).
- Results, hold 215516 on bh-glx-110-c10u20, 1350 MHz. Reference: torch on the device's routing, with operands
  quantized as the device sees them.
  - (a) One chip, T 2048, 64 experts, every token: y PCC 0.99994 (max error 1.3% of max |y|). Head exact: counts, token
    lists, output rows. Logits PCC 0.99999995, gate weights 0.999999. Top-12 is the same set as torch's on
    2024 / 2048 tokens; the rest are near ties (worst pick 0.15% below torch's 12th score).
  - (b) One chip, T 31250, 640 experts, 256 sampled tokens: y PCC 0.99994; head exact over every token. M per expert
    min 461 / median 584 / max 746.
  - (c) Galaxy torus, T 31250 a chip, 640 experts, 64 sampled tokens on chips 0 and 31: y PCC 0.99994 on both; head
    exact on both. M over 32 chips x 640 experts: min 425 / median 585 / max 785 (mean 585.9).
- Time, 20 warm-up + 5 measured layers, medians in ms. Wall time is synchronize to synchronize; the lane programs are
  built before their phase starts. Device time is the group cores' compute start to end.
| phase | one chip (b) wall | one chip device | Galaxy (c) wall | Galaxy device |
|---|---|---|---|---|
| head | 15.47 | | 15.60 | |
| shared expert | 11.16 | 10.29 | 14.09 | 10.32 |
| routed stream | 37.40 | 34.70 | 56.50 | 48.38 |
| combine | 47.89 | | 47.76 | |
| total | 111.92 | | 133.94 | |
  - On one chip the stream is the proxy (every expert local).
  - Head steps on the Galaxy (synchronized apart): router 5.2, counts + output rows 3.2, token lists 6.0, combine
    indices / gate matrices 2.0. Token lists by a sort of mask ? t : T took 164 ms (the scatter replaced it).
  - Combine steps: shared sum 4.8, untilize 15.1, gather 16.4, batched matmul 12.0, add shared 4.8, concat 1.4.
  - Shared expert: 2 T 3840 2944 3 = 2.12 TFLOP a chip in 10.3 ms is 33.8% of the 110-core SOL (608 TFLOP/s; floor
    3.48 ms), or 50.7% of the 80 group cores counting the padded work (floor 4.79 ms at 80 cores).
  - The stream's device time with expert-id-ordered outputs (48.4 ms) matches the out_all runs (48.2 ms).
- Open:
  - The combine (48 ms) is as long as the stream: three DRAM round trips of the 3 GB routed output (untilize, gather,
    matmul). A fused gather + weigh + sum kernel, and row-major outputs from the stream (no untilize), would remove
    most of it.
  - Lane wall time is 4-8 ms over device time (Galaxy). The gap is the dispatch of a newly built mesh program each
    launch (fresh arrival words), plus each kernel's time before and after the compute stamps.
  - The phases run one after another. The shared expert (20 units, M 7840) could go into the stream's program as more
    own experts.
  - The MoE block's moe_head and routed_experts_combine are not reused: they are built around dispatch (TP all-gather,
    routing exchange, fabric combine), not a chip's own tokens against streamed weights.
  - Only softmax scoring has been tested; sigmoid is selectable but untested.

## Stage 11: custom end pass and row-major stream outputs (hold 215516, bh-glx-110-c10u20)
The phases now meet in a row-major result buffer [T S, H] bf16, S = K + 1: row t S + k holds token t's k-th routed
expert's output, row t S + K its shared expert's. The ttnn combine (untilize, embedding gather, batched matmul) and
the expert-id-ordered outputs (out_eord) are gone.

### End pass (ws_end.py, kernels we_*.cpp)
- What it computes: y[t] = sum over k of w[t, k] row t S + k, plus row t S + K. Every core takes a contiguous range of
  tokens.
- Reader:
  - Reads a token's S rows: S page reads of 7680 B, each row into 4 tile slots of contiguous elements.
  - The 4th slot's 512-B tail is zeroed once and never written, so it stays finite.
  - Up to 4 tokens are in flight, each on its own read transaction id.
  - Writes the gate weights as bf16 at element 0 of scalar tiles, plus 1.0 for the shared row.
- Compute: SCALAR-broadcast ELWMULs at HiFi4 accumulate into 4 fp32 DEST tiles. On Blackhole ELWMUL adds its product
  into DEST, and the standard binary path never clears DEST, so the plain broadcast multiply is the MAC.
- Writer: writes y rows (row-major [T, H], what the stream's x reader takes).
- One chip, T 31250, 20 + 5 launches; PCC 0.999999 on every token in all cases:
| reads | device ms | GB/s |
|---|---|---|
| every core on NOC1 (writes NOC0) | 12.92 | 260 |
| half the cores on each NoC | 7.54 | 446 |
| every core on NOC0 (writes NOC1): the default | 7.85 | 428 |
  - With all reads on NOC1 the pass is NoC-bound: compute waited for input 58% of the time, LoFi vs HiFi4 made no
    difference, and 2, 4 or 6 tokens in flight made no difference.
  - The half-and-half split is faster, but it puts DRAM reads and acknowledged DRAM writes on both NoCs. That mix can
    hang the chip (below), so NOC0 reads / NOC1 writes is the default.

### Stream: row-major outputs (LCfg.y_rm)
- The down row tile is pack-untilized (pack_untilize_dest<3, 6>). DEST's untilize read remap is enabled once at the
  compute's start.
- QY (NCRISC, NOC1):
  - Reads the row tile's 32 result rows from the expert's destination list (same layout as the token lists), on its
    own transaction id, before the data is ready.
  - Writes each row's 384 B (this core's 6 columns) to its result row as posted writes.
  - Flushes them before the kernel exits.
- The head (still ttnn ops) builds the destination lists t S + k with the same scatter as the token lists.
- The hang:
  - With acknowledged (non-posted) row writes on NOC1, next to the NCRISC's NOC1 DRAM reads (weights, token rows,
    lists), one chip at T 31250 hung in the first launch, every time. The chip stopped answering MMIO, so triage could
    not attach.
  - The same code completed at T 2048 / 64 experts and at T 8192 / 640 experts (about 0.75 GB of rows, against
    2.88 GB and about 7.5 M writes a chip at T 31250).
  - At T 31250 with the writes compiled out (WL_Y_NOWRITE) the stream completed, and the head was exact.
  - Posted writes ran 25 layers clean.
  - The cause is still unknown. The trigger scales with the volume of acknowledged DRAM writes on a NoC that also
    carries DRAM reads, like the ring all-gather's NOC1 read + write hang.
  - Rules adopted for every op of the layer:
    - DRAM reads and acknowledged DRAM writes never share a NoC (posted writes are exempt).
    - Nothing enters a multicast rectangle from outside on its NoC (read responses, write acks).
    - No DRAM data written on one NoC is read back on the other within an op.
- Results, 20 + 5 layers; the end pass ran split NoCs here; the shared rows are not written yet, so y is the routed
  sum:
| case | stream device ms (tile outputs) | output writes | rows PCC | y PCC | end pass device ms |
|---|---|---|---|---|---|
| one chip, T 31250 (proxy) | 36.68 (34.70) | 79 GB/s | 0.99994 | 0.99994 | 7.56 |
| Galaxy torus, chips 0 / 31 | 49.26 (48.38) | 58 GB/s | 0.99994 | 0.99994 | 7.63 |
- Head exactness: exact in every run (counts, token lists, destination lists).
- Every token, one chip at T 31250: the result buffer read twice around a device sync is identical. y against the
  host's sum of the rows as read: worst token 0.0038 relative, none over 2e-2.

## Stage 12: the shared expert as one single-chip op (ws_shared.py, kernels wsh_*.cpp)
- What it computes: y = (silu(x W1) * (x W3)) W2 for every one of a chip's T tokens (H 3840, I 2944), written
  row-major into the result buffer's row t S + K (the end pass adds it with weight 1). Weights are replicated and
  resident in L1 (bfp4, LoFi); x is the stream's row-major bf16 x.
- Layout, from the column clusters of ws_cluster / ws_dyn:
  - 2 clusters, each a rectangle of 5 logical columns x 10 rows: 46 compute cores plus 4 without weights.
  - Core j owns gate/up pairs 2j, 2j + 1 (92 = 46 x 2) and 2 or 3 down columns (28 x 3 + 18 x 2 = 120).
  - Row tiles go round-robin to the clusters, so no row is split and no partials cross clusters.
  - Per row tile a core runs its pairs' gate/up over the whole x row, then the previous row's down over the whole h
    row (K 92), pack-untilized; its 32 rows x DC columns go out as posted writes, fenced at the end.
  - The hub (the rectangle's NOC1 multicast start corner, no weights) is the cluster's only multicaster:
    - It reads each row tile's 32 whole token rows (NOC0) and tilizes the 120 tiles to bfp8, double-buffered.
    - It multicasts them in BK-tile blocks into every core's x ring in stream order, credit-gated by every core's
      popped count.
    - The cores send their h tiles into the hub's h ring slot (acknowledged write, then the slot's arrival count +1).
      Once a row's tiles are all in, the hub multicasts the whole h row.
    - The h ring needs no slot credits: with NX <= 2 KT / BK - 1, a core producing h row i means every core is past
      the down of row i - 3 (NS 3).
  - NoCs: NOC0 carries only DRAM reads (weight preload, x). NOC1 carries the multicasts, in-cluster unicasts and the
    posted row writes, so nothing from outside a cluster enters its multicast NoC.
- M limits: none. The schedule is static (no on-device planner; ws_dyn's MAXOWN 40 rows an expert does not apply),
  489 row tiles a cluster at T 31250.
- bfp8 weights do not fit: 522 KB gate/up + 300 KB down a core next to the rings; static CBs end at 1,056,192 B.
- One chip, T 31250, bfp4 LoFi, 20 warm-up + 5 measured launches, device time (compute start to end stamps). Floor:
  2 T 3840 2944 3 = 2.12 TFLOP is 3.48 ms at the 110-core LoFi SOL, 4.17 ms on 92 cores.
| version | device ms | % 110-core SOL | % of the 92 cores | pcc |
|---|---|---|---|---|
| lane units (5 slices x 4 chunks, stage 10) | 10.3 | 33.9 | 50.8 (of 80) | 0.99994 |
| v1: every core multicasts its own h tiles, x blocks from 10 cores | 55.9 | 6.2 | 7.5 | 0.99994 |
| v2: one hub: x gathered as 768-B row pieces (320 reads a row tile), tilized per block | 7.32 | 47.6 | 56.9 | 0.99989 |
| v2, the hub's tilize with fp32 DEST (back to the reference's bfp8 rounding) | 7.48 | 46.6 | 55.7 | 0.99994 |
| v3: 4 x feeders a cluster (whole rows, every 4th row tile each), own multicasts | 8.11 | 42.9 | 51.4 | 0.99994 |
| v4: v3 with the x multicasts in stream order (block s once s - 2 landed) | 7.83 | 44.5 | 53.2 | 0.99994 |
| one hub, whole-row reads (32 a row tile), one tilized row staged | 8.85 | 39.4 | 47.1 | 0.99994 |
| same, tilized staging double-buffered | 7.54 | 46.2 | 55.3 | 0.99994 |
| + BK 24 / NX 9 (bf16 h) | 6.38 | 54.6 | 65.3 | 0.99994 |
| + bfp8 h (default) | 4.97 | 70.2 | 83.9 | 0.99988 |
| BK 40 / NX 5, bfp8 h | 4.96 | 70.3 | 84.0 | 0.99988 |
| BK 40 / NX 5, bf16 h | 6.08 | 57.3 | 68.5 | 0.99994 |
- What each step found (per-role counters on the hub):
  - v1's 46 multicast sources into one rectangle serialize: about 112 multicasts a row, ~114 us a row.
  - v2's hub issued 320 reads of 768 B a row tile (busy 62%), and its tilize waited for input.
  - v3 / v4's extra feeders put more multicast sources into the rectangle (credit waits 58-68%).
  - Whole-row reads fixed the gather (busy 5%). With one tilized row staged, though, the next row's tilize waited
    until all of the current row's multicasts had left; double-buffering the staging fixed that.
  - Larger x blocks (BK 24) halve the x multicast count, cutting the cluster cores' x waits from 3.7 M to 0.3 M
    cycles.
  - bfp8 h halves the h multicast: h waits 1.75 M -> 0.35 M cycles.
  - Every case, T 2048: every token checked, all PCC >= 0.99988.
- Bug on the way: pack_untilize_uninit leaves the packer at the output's format. With bfp8 h, the next GLU pack wrote
  h in bf16 format (PCC 0). The down row now reconfigures the packer to h's format after it.
- The deepseek_prefill unified routed expert op as the shared expert, for comparison (tests/test_ws_unified.py: one
  expert, count 31250, x row-major, grid 11 x 8, wall time):
  - bfp4 LoFi: 9.34 ms with ND-sharded weights, 9.42 ms interleaved (37% of SOL). bfp8 LoFi: 10.79 ms.
  - Output bfp8 TILE (PCC 0.99964 / 0.99944). It runs LoFi only and needs math_approx_mode False.
  - Its reader reads DRAM and multicasts on NoC 0, and its writer reads the up weights and writes the output on
    NoC 1, which breaks the layer's NoC rules.

## Stage 13: the layer with every custom phase but the head (hold 215516)
- Phases: head (ttnn ops) -> shared expert op (rows t S + K) -> stream (row-major result rows, posted, fenced) -> end
  pass (y = sum of w rows + the shared row; NOC0 reads / NOC1 writes).
- Results, 20 warm-up + 5 measured layers, 1350 MHz. Wall time is synchronize to synchronize (programs built before
  their phase); device time is compute start to end over the cores (max over chips 0 / 31 on the Galaxy):
| phase | one chip (proxy stream) wall / device ms | Galaxy torus wall / device ms |
|---|---|---|
| head (ttnn ops) | 19.00 / - | 19.11 / - |
| shared expert | 5.68 / 4.97 | 7.93 / 4.97 |
| routed stream | 39.48 / 36.74 | 57.23 / 49.25 |
| end pass | 8.24 / 7.88 | 11.37 / 8.41 |
| total | 72.40 | 95.64 |
- Accuracy (sampled tokens; every token at T 2048: y 0.99988):
  - Layer y PCC: 0.99989 on one chip; 0.99988 on the Galaxy, chips 0 and 31.
  - Routed rows 0.99994, shared rows 0.99988 (bfp8 h).
  - Head exact (counts, token lists, destination lists).
  - One chip, every token: re-read identical, y against the rows as read worst 0.0039 relative.
- The Galaxy's wall times run 2-8 ms over device time per program: the dispatch of a 32-chip mesh program.

## Stage 14: the head as one program in the MoE block router's structure (ws_router.py, kernels wsr2_*.cpp)
- Two custom heads on the way:
  - ws_head.py (kept for comparison): ttnn router matmul (fp32 logits to DRAM) + one routing program (BRISC top-K by
    integer compares of the fp32 bits, exp on the compute, the L1 exchange). One chip, device ms (wall-clock stamps
    between its programs): matmul 3.57, x to row-major 1.14, routing 5.66 (its top-K 5.27), head 10.37 (Galaxy 10.47).
  - ws_router.py (the head): moe_head's structure - the router on the cores' own compute, the top-K on the SFPU over
    the logits in DEST, transposed (a token a column), the logits gathered as top-K runs to the token's owner.
- ws_router design, 120 cores, every chip on its own tokens:
  - Router groups: the 12 grid columns, 10 cores each (one NOC1 multicast rectangle). Core c holds W_r^T's expert
    tiles 2c, 2c + 1 (all 120 K tiles, 480 KB bf16), loaded a K piece at a time each launch. A group takes a
    contiguous range of ~81 row tiles; core c owns its group-local row tiles c, c + 10, ..
  - x arrives row-major (the previous layer's end pass writes y row-major). Piece p (12 K tiles) of every row tile
    is core p's: it reads the 32 row pieces (NOC0), tilizes them two row tiles ahead, and writes them posted into the
    relay ring of the column's hub (the NOC1 multicast start end; NOC0 runs toward it, no wrap), the arrival flag
    after them through the same command buffer. The hub alone multicasts the pieces in stream order into every
    group core's x ring (20 pieces), credit-gated, at most two in flight, then returns the relay slot's grant
    (NOC1, downward).
  - Every core, every row tile: logits^T = W_r^T x^T for its 2 expert tiles (matmul_block, transposed in1, HiFi4,
    fp32 DEST); still in DEST: the id tiles into DEST 2, 3, moe_head's router_topk_key (the expert id in the low
    10 mantissa bits), the bitonic network over the pair (sort_run 16) -> its 64 experts' best 16 keys, one tile, to
    the owner's gather ring. No logits are packed; nothing goes through DRAM.
  - The owner: moe_head's loop over the 10 run tiles -> the token's top 12; DEST 1 = their logits (ids cleared),
    exp; the NCRISC: w = e / sum e (soft float, a few tokens a pass) and the records.
  - A barrier over the chip, then the L1 exchange (now wsr_exchange.hpp, shared with ws_head): counts, offsets,
    runs, per-expert lists, fence. The exchange's record buffers reuse W_r^T's L1. An expert's list holds its
    (token, slot) entries as a set (the sources hold interleaved row tiles).
  - NoC: DRAM reads and the relay writes (posted) on NOC0; multicasts, run tiles, credits, exchange and the posted
    DRAM writes on NOC1 (fenced); multicasts only from the column's NOC1 start end; nothing crosses a group's
    rectangle before the barrier.
- The keys: the network compares sign-magnitude (SFPSWAP), so clearing low mantissa bits keeps the order of
  distinct logits of either sign; ties at the key's resolution (2^-13 relative) go by expert id. Checked against the
  host's top-12 of the same fused keys of the device's own logits (WSY_LOGITS writes them): every token, logits as
  drawn and shifted by -50 (all negative).
- Limits: logits within (-87, 88) (exp in fp32); an owner core takes 16,384 records (6 experts' routed rows plus run
  padding; 3,516 at balanced routing). Past it the owner writes count 0xFFFFFFFF and the layer raises on the host
  before the stream (test_ws_head_overflow: T 8192, every token on owner 0's experts, 49,152 records).
- Results, 20 warm-up + 5 measured layers (+ 5 layers for the head's steps), 1350 MHz, device ms (head: its
  program's earliest start to latest end over the cores; Galaxy: max over chips 0, 31):
| head | one chip | Galaxy |
|---|---|---|
| ttnn ops (stage 13) | 19.00 wall | 19.11 wall |
| ws_head: ttnn matmul 3.57 + x to row-major 1.14 + routing program 5.66 | 10.37 | 10.47 |
| ws_router, each row tile's owner tilizes and multicasts it | 4.01 | 4.04 |
| x pieces from every core through the column hub's relay | 3.37 | 3.41 |
| + weights a piece at a time, x tilized two row tiles ahead, records a few tokens a pass, 6 relay slots | 2.65 | 2.66 |
- Where the head's cycles go (last row, one chip; per core, mean over the cores, max in brackets):
  - compute: router matmuls 0.87 net, waiting for x 0.88 (1.09), keys and the pair's network 0.22, the owner's
    merges 0.10, tilize 0.30; the loop ends at 2.31 ms;
  - the exchange after the loop: B 0.19, C 0.15;
  - the weights are all in at 0.19 ms (the first row tile starts on the first piece);
  - NCRISC records 0.50; BRISC waiting for relay slots 1.58 (the relay ring is the throttle); the hub's multicasts
    0.43 issue to sent, waiting for credits up to 0.83.
- Earlier versions' x waits: 2.22 ms (owners multicasting: a row tile's owner tilized it right before its own router,
  so the next blocks waited on its credits; and owners off the column's NOC1 start end wrapped), then 1.55 (the
  NCRISC's records a whole row tile at a time held up its ring pushes and credits, and the hub's multicasts).
- Layer, 20 + 5: one chip 56.46 ms wall (head 3.10, shared 5.65, stream 39.49, end 8.22); Galaxy 83.19 ms wall
  (head 6.74, shared 7.92, stream 57.11, end 11.42; device head 2.66, shared 4.97, stream 49.26, end 8.42).
- Accuracy:
  - T 2048 every token: the picks are the top 12 of the fused keys of the device's logits on 2048/2048 tokens; w
    PCC 0.9999997; y 0.99988; lists exact.
  - Logits shifted by -50: 2048/2048 as well; w PCC 0.99877 (the keys' resolution at |logit| 50 is ~6e-3, the
    matmul's own error there ~0.02).
  - T 31250: routing exact, every-token re-read identical, y 0.99989; Galaxy y 0.99988 on chips 0 and 31.

## Stage 15: the layer in a trace (tests/test_ws_layer.py run_trace, run_soak; bench_util device zeros)
- A layer's programs back to back with no host IO (layer_programs): each phase builds its program, runs it and frees
  its L1 tensors, so the phases alias L1 as in the eager layer; the whole layer captured once and replayed.
- What had to change:
  - Host writes: sharded_l1 / stamps_tensor wrote zeros from the host (from_torch). Under bench_util.DEVICE_ZERO they
    allocate without a write; the uint32 tensors (state words) of the routed stream are zeroed by one small program
    before it (9.5 KB a core). The head, shared expert and fold clear their own state words at the start of each
    launch, then a barrier over their cores on program semaphores (reset by the dispatch every launch) before
    anything is sent: their L1 is another program's between launches.
  - Nothing may stay in L1 across phases: the routed stream's L1 tensors sit ~3 KB above its circular buffers, so
    any persistent L1 tensor pushes them into the CBs (a pool of stamps did). Device times come from the eager
    layers; the trace's DRAM clocks (one-core programs writing the wall clock to DRAM between the phases) time a
    replay's programs.
  - The zero-fill program's range count is a compile-time arg (one cached program a count, runtime args always the
    same length).
- One chip, T 31250, 20 warm-up + 5 measured replays: outputs (y, counts, token lists, destination lists, w)
  bit-identical to the eager layer; every-token re-read equal. Wall per replay 62.3 ms; eager wall 56.5 (head 3.1,
  shared 5.7, stream 39.5, end 8.3), device sum 52.3. Inside a replay the programs take 52.1 ms (head 2.71, shared
  4.98, stream 36.60, end 7.81: the device sum); the other ~10 ms falls between replays.
- The Galaxy: the whole-layer trace hung in replay (twice; the stream's lane kernels and fabric muxes waiting): each
  chip ran the stream's zeros and then its stream with no sync between chips, so a neighbour's fabric writes (weight
  slots, arrival counts, credits) could land before a chip's zeros and be wiped. Confirmed: the layer as two traces
  split before the stream, a host synchronize after each, ran clean for 30 layers (76.9 ms a layer, bit-identical to
  eager). The fix: one trace with a device barrier between the stream's zeros and the stream (two all-gathers of a
  tiny DRAM tensor, one a torus axis, the second on the first's output), the stream's state words held allocated
  across it.
- The Galaxy, the folded layer (stage 16) in one trace with the barrier, 20 warm-up + 5 measured replays: bit-identical
  to eager; 74.06 ms a layer (73.14 back to back) against eager 76.91 wall, device sum 61.10. Inside a replay (chip
  0): head 2.70, stream 52.81 (its zeros and the barrier included; the stream alone 49.4), fold 8.99: 64.50 first
  clock to last; ~9.5 ms between replays, as on one chip.
- Without the trace, the layer's host work costs the Galaxy most: the same programs untraced (builds included)
  130.9 ms back to back, 172.8 synchronized; with the zeros written from the host 781 ms.

## Stage 16: the shared expert folded into the end pass (ws_fold.py, kernels wf_*.cpp, wsh_reader WSH_FOLD)
- Where the end pass can run beside the shared expert: alone on the cores the column clusters leave (T 31250, 4 tokens
  in flight), it is read-bound by the columns' vertical links:
| end pass cores | HiFi4 | HiFi2 | LoFi |
|---|---|---|---|
| 120 (all) | 7.87 | - | - |
| 20 (2 free columns) | 16.06 | - | - |
| 26 (+ the rectangles' idle cores) | 12.83 | 12.45 | 12.61 |
| 24, the 2 last rows | 9.37 | 8.76 | 8.60 |
| 24, the 2 first rows | - | 8.51 | - |
- So the shared expert's clusters take rectangles of 4 whole rows (rows 2-9; ws_shared layout "rows"), the end pass the
  2 first rows. A 12-wide rectangle spans two NoC sub-rectangles (Blackhole's Tensix columns jump over the DRAM
  columns): the hub multicasts into each, payload then a 4-B flag through the same command buffer. The shared expert
  alone in that shape: 6.36 ms (4.97 in columns: the hub multicasts every block twice).
- The fold, one program: each down core pack-untilizes its columns of a row tile's rows and writes them, posted on
  NOC1, half a row tile at a time into the owning end core's L1 ring (2 slots of 16 rows; the row tile's end core
  in the order the cluster produces them), then its flag word through the same command buffer; it waits for the end
  core's credit (NOC0) before reusing a slot. The end core reads each token's 12 routed rows (NOC0), copies the
  shared row from its ring, sums them in DEST (HiFi2; weight 1.0 for the shared row), writes y posted on NOC1 with a
  fence. The result buffer holds 12 rows a token.
- L1: a sharded tensor takes its address range on every core, so the end cores' rows, y rows and weights are plain
  circular buffers (on the end cores only; the rows first, at the 64-B aligned base); only the ring and small state
  are tensors.
- Every launch starts with each writer's credits and end core's flags (and the shared expert's own flags) zeroed,
  then a barrier over them: the first version, zeroed once at allocation, read the previous launch's flags from the
  second launch on.
- One chip, 20 + 5 launches: T 2048 and T 31250 every token exact against the routed sum plus the standalone shared
  expert's rows (PCC 0.999995). Device 8.95 ms against shared 4.97 + end 7.90 apart; the end cores still wait 1.8 ms
  for slot flags (each cluster sends its row tiles in order, an end core holds one row tile: one slow end core stalls
  the cluster).
- In the layer (head, stream, fold), 20 + 5: one chip 52.65 ms wall (56.46 unfolded), fold 9.06 ms device; Galaxy
  77.22 ms wall (83.19 unfolded), fold 9.05 ms device (shared + end 13.37 apart); y PCC 0.99988, routing exact.

## Stage 17: soak (tests/test_ws_layer.py run_soak)
- 10,000 layers on the Galaxy, T 31250 tokens a chip, the shared expert folded into the end pass, the whole layer one
  trace with the device barrier before the routed stream (WSY_SOAK_MODE barrier). Every 500 layers y and the head's
  outputs (counts, token lists, destination lists, w) on chips 0 and 31 against the first (eager) layer's.
- All 20 checks bit-identical (y PCC 1.000000); no hang; wall per layer 73.8 ms over the first 500, 75.0-75.2 from
  3,000 on. Total 16 minutes with setup.
- A capture takes no host writes: the warm-ups run the very phases a trace captures (the barrier's all-gathers do
  host writes the first time they are built).
- Open:
  - ~10-12 ms a layer between replays (one chip 62.3 ms a replay against 52.1 ms of programs inside it; the Galaxy
    74.1 against 64.5). Resolved in stage 18: not a gap - the chips' 75 W power limit throttles AICLK.
  - The barrier: the stream's segment in a replay 52.8 ms against the stream alone 49.4 (its zeros, the two
    all-gathers and the dispatch: ~3.4 ms).
  - The fold: 9.05 ms against the 7.5-8 target; its end cores wait ~1.8 ms for slot flags (each cluster sends its row
    tiles in order, an end core holds one row tile of slots: a slow end core stalls its cluster).

## Stage 18: the traced layer's missing time is AICLK throttling (tests/test_ws_layer.py run_trace_bisect; hold 217360)
- run_trace_bisect: traces of the DRAM clocks alone, each phase alone, the layer, the layer twice; wall per replay
  back to back against the clocks inside; then 20 replays back to back between two clocks outside the trace (the
  wall-clock counter against host time) with AICLK / power / temperature / current sampled from sysfs (every chip on
  a mesh). One chip T 31250 folded, 20 warm-up + 5 measured replays, then the 20 bracketed:
| trace | host ms a replay | counter ms (cycles / 1350) | counter MHz | AICLK sampled under load |
|---|---|---|---|---|
| clocks only | 0.008 | 0.001 | - | 1350 |
| head | 3.07 | 2.69 | 1184 | 1150-1350 |
| stream | 42.84 | 36.64 | 1155 | 1137-1181 mostly |
| fold | 11.46 | 9.04 | 1065 | down to 975 |
| layer | 58.79 | 48.22 | 1107 | 1081-1118 mostly |
- The counter (RISCV_DEBUG_REG_WALL_CLOCK) counts AICLK cycles: 1349.6 MHz against host time at idle, and under load
  it tracks the sampled AICLK within ~1%. The unaccounted time scales with the work (the layer twice in one trace:
  twice; between its two layers 1 us), a trace boundary costs ~7 us (one chip) / 68 us (Galaxy).
- tt-smi limits: every chip of bh-glx-110-c10u20 (and 120-c10u02) has tdp_limit 75 W (tdc 500 A, fmax 1350, fw
  19.16.0.0); 120-b08u08 / b09u08 / b09u02 have 130 W. Under the layer a chip draws ~69 W median (max ~110) with
  AICLK held at ~1100-1200: the TDP throttler, not temperature (37 C) or current.
- The Galaxy (c10u20, folded layer, device barrier): 72.25 ms a replay host time, 64.03 counter (chip 0 at 1196 MHz
  effective); AICLK medians per chip 1093-1212 under load, chip 8 the slowest (1093, min 1043) - the coupled stream
  runs at the slowest chip's pace.
- So every device time in stages 11-17 (cycles / 1350) is the time at 1350 MHz; on c10 under sustained load the real
  time is 10-25% longer. The eager layer looks faster than the traced one because its host gaps let AICLK recover.
- The Tracy device trace profiler (--device-trace-profiler) failed to post-process these programs (TT_FATAL: start and
  end marker names do not match); its host-device sync fit showed the same clock loss (1.3500 -> 1.3466 GHz over
  the run).
- The same on a 130 W Galaxy (bh-glx-120-b08u08, hold 217516; tdp_limit 130 W on all 32 chips, DRAM 14G against
  c10u20's 16G), after a Galaxy reset, 20 warm-up + 5 measured replays, then 20 bracketed:
| | c10u20 (75 W) | b08u08 (130 W) |
|---|---|---|
| one chip: head / stream / fold / layer, host ms a replay | 3.07 / 42.84 / 11.46 / 58.79 | 2.81 / 37.56 / 9.05 / 49.26 |
| one chip: layer counter MHz, power median / max | 1107, 69 / 107 W | 1334, 100 / 117 W |
| Galaxy traced layer back to back | 73.1 | 62.5 |
| Galaxy inside a replay (DRAM clocks) | 64.5 | 61.9 |
| Galaxy eager wall / eager device sum | 76.9 / 61.1 | 74.4 / 61.5 |
| Galaxy chips' AICLK median under load | 1093-1212 | 1337-1350 |
- b08u08: the traced Galaxy layer bit-identical to eager (y, counts, token lists, destinations, w); 64-token checks
  on chips 0 and 31 pass (y PCC 0.99988). Unaccounted time ~0.5 ms a layer (one chip and Galaxy). Inside a replay:
  head 2.70, stream 49.72 (its zeros and the barrier included), fold 9.44. Trace command stream 320 KB.
- The stream's cycles are ~2% higher on b08u08 (one chip 37.4 against 36.6 ms of counter): its slower DRAM.
- Open: the routed stream (49.7 ms on the Galaxy against the 14.8 target); power per phase on 75 W systems.

### No group multicasts: unicast fan-out, then unicast chains (LoudBox 2 x 4 p150b, 2026-11-16)
The 4-slot tray soak (160 experts a lane, T 31250, M 586, G 2) reproduces the wedge on this LoudBox in 7-30 s (launch
50 clean at 52-53%, hung before launch 100 / 250); the wedged chip drops off PCIe within seconds (host reads
0xffffffff, MMIO per-op timeout), so a timeout-triggered tt-triage only sees the other 7 chips. Node stayed up; tt-smi
-r recovers. The fabric routers land on NOC1 (receiver_channel_local_write_noc_ids = 1), so reads off the multicast
NoC are not enough while column 1 multicasts on NOC1. Instead every group multicast is replaced by unicasts:
- LCfg.ucast (WL_UCAST, default): each x block / h row part as 20 unicast writes (payloads, then flags, one command
  buffer). Correct (dynamic M min PCC 0.99992), soak 2300 launches / 300 s clean, but M 586 18.4%, M 2048 19.6%: a
  column's links carry each block ~4-5 times (x waits 6.6 M cycles of 9.4 M on chip 0).
- LCfg.xchain (WL_XCHAIN, default): x blocks as a chain - origin (c, r) copies into its slot and sends to its partner
  (1 - c, r) (column 0 BRISC NOC0 east, column 1 NCRISC NOC1 west); in both columns rows >= r pass it one row down
  (BRISC, NOC0), rows <= r one row up (NCRISC, NOC1), from the slot with its flag word (s + 1) as the flag source. Each
  link carries a block once. A block's slot credit waits until both forwarders are past it (BRISC progress in
  semaphore 0's low half). Stream-order gate (WG_XORDER) off with the chain (2: -3-4 points).
- LCfg.hchain (WL_HCHAIN, default): the h row parts the same way from their leader (flagged locally, partner hop,
  then down / up); the NCRISC gives row q to compute only once both forwarders are past both parts (semaphore 0's
  high half; no semaphore left beside the mux connections).
| group traffic (4 slots, keep 2, 5 warm-up + 5 measured) | M 586 util tile | M 2048 | soak 300 s |
|---|---|---|---|
| multicasts (ucast 0) | 50.4% | 70.1% (tray worklog) | hangs in 7-30 s |
| unicast fan-out | 18.4% | 19.6% | 2300 launches clean |
| x chain, h fan-out | 31.1% | 33.7% | 2872 launches clean |
| x chain + h chain (default) | 38.1% | 41.1% | 3243 launches clean, 38.6% |
PCC 0.99994 everywhere (x + h chain also with out_all: every expert checked; dynamic M min 0.99992). No help: hops
without a flush (WL_XNOFLUSH, +0.8 points), a deeper x ring (BK 12 / NX 15, BK 10 / NX 18). Chip 0 at M 586 still
waits 1.9 M cycles on x (h 0.3 M) of 4.5 M. Caveat: the h ring has no slot credits (as before: NS sized by the host);
the chain skews arrival by up to 10 hops; covered by PCC so far, not by a proof.
- Hop-by-hop x credits (LCfg.xhop, WL_XHOP; the flat routed expert's scheme: each writer checks only its target's
  freed count, pushed to the row above / below / partner): correct (out_all, dynamic M) but 37.9% vs 38.1%, and it
  deadlocked within 50 soak launches (all chips reachable; compute waiting on weights / x; likely the chain's cores
  drifting apart). Off by default.
- x block timeline (test_ws_lane_tray_trace, WL_DEFS=WG_TRACE0=912; chip 0 group 0, 40 blocks): a hop ~0.9-1 K
  cycles, the last core gets a block ~11.8 K cycles after its send (median); origins send ~1.2 K after the block is
  tilized, but blocks become ready in bursts, one x row (6 blocks) every ~14 K cycles.
- Shorter chains (LCfg.jump, WL_JUMP: the origin also sends to row (r + 5) % 10 of both columns; each column then has
  two entries, every row reached once, at most 4 hops, checked on the host): last-core delivery 9.6 K, h waits 0.3 ->
  0.17 M, but 38.2% (x waits unchanged 2.07 M); with BK 12 / NX 15 / LOOK 60 35.5%. Off by default.
- The limit is x ring capacity, not hop latency: NX 9 x BK 20 = 180 K-tiles = 1.5 rows of x a core, and block s waits
  until every core has consumed s - NX; the chain's far cores get blocks ~10 K later than the near ones, which 1.5 rows
  cannot cover. (The earlier "deeper ring" runs, BK 12 / NX 15 and BK 10 / NX 18, kept 180 K-tiles.) BK 20 with a
  longer LOOK (30, 42) does not fit L1 beside 4 slots. With 3 weight slots (d0,d1,any, keep 1) the L1 for NX 18:
| 3 slots, chains | M 586 | M 2048 | chip 0 waits x / weight / h (K, M 586) |
|---|---|---|---|
| NX 9 | 33.1% | 40.1% | 2032 / 364 / 302 |
| NX 18 | 36.7% | 54.4% | 62 / 971 / 1294 |
| NX 18 + jump | 38.1% | 51.8% | 118 / 743 / 1389 |
  x waits are gone with NX 18; M 586 then needs the 4th weight slot (weights) and a deeper h ring (NS 4 rows, the same
  capacity-vs-skew problem for h). L1 is the constraint: 4 slots + an x ring of ~3 rows + a deeper h ring.
- L1 audit (a group core, 4 slots, BK 20 / NX 9): weight slots 829,440 B (4 x 207,360), x ring 195,840, h ring
  163,840 (bf16), CB 7 row-major gather 122,880 (XRM 3), CB 8 tilize staging 108,800 (5 blocks), CB 16 / 17 28,672;
  ~1,449 KB of the 1,460,288 B bank. Freed for the x ring: bfp8 h (-76.8 KB; the packer now switches format around the
  down outputs, WX_H_PACK_DIFFERS - it packed bf16 outputs as bfp8 before: PCC nan), XRM 2 (-41 KB), staging slack 2
  blocks instead of 4 (LCfg.xst, -43.5 KB). NX 18 misses by ~26 KB.
| chains, 4 slots, 5 + 5 launches | M 586 | M 2048 | chip 0 waits x / weight / h (K, M 586) | PCC |
|---|---|---|---|---|
| NX 9, bf16 h (previous default) | 38.1% | 41.1% | 1967 / 5 / 308 | 0.99994 |
| NX 9, bfp8 h | 39.5% | 42.5% | 1872 / 5 / 227 | 0.99988 |
| NX 15, bfp8 h, XRM 2, xst 2 | 48.0% | 57.8% | 372 / 73 / 836 | 0.99988 |
| NX 16, bfp8 h, XRM 2, xst 2 (new chain default) | 47.9% | 58.7% | 291 / 93 / 899 | 0.99988 |
| NX 15 + NS 5 / NX 14 + NS 6 | 47.6% / 46.9% | 57.6% / 55.7% | h 830 / 754 | 0.99988 |
| NX 16 + jump | 46.2% | 57.9% | 243 / 99 / 1085 | 0.99988 |
  A deeper h ring or the jump does not help: the h waits that remain are not ring capacity. New default (chains):
  out_all + dynamic M pass (min PCC 0.99986), soak 300 s: 3406 launches clean, util tile median 50.1% at the 640-expert
  lane load (multicasts: 52-53% until they wedged). The multicast build (ucast=0) keeps NX 9, bf16 h, XRM 3: 50.6%.
  (ucast=0 had failed the "xchain needs ucast" assert since the chains became the default; the chain flags now just
  do not apply without ucast.)

### Unicast benches for the exchange roofline (tests/test_uc_bench.py, kernels/uc_bench.cpp; one chip, column 1)
| | 4,352 B | 10,880 B | 21,760 B |
|---|---|---|---|
| one hop, streamed (B/cyc) | 59.6 | 59.9 | 59.9 |
| one hop, flushed after each block | 37.8 | 48.7 | 53.8 |
| all 9 hops of a column at once, streamed / flushed (a stream) | 55.3 / 36.4 | 59.2 / 48.2 | 59.5 / 53.2 |
| 10-core chain, dedicated RISC a hop: latency a hop | 141 cyc | 236 cyc | 396 cyc |
| chain throughput (S slots) | 28.8 (S 16) | 36.0 (S 8) | 21.7 (S 4) / 42.6 (S 8) |
NOC0 down and NOC1 up the same within 2%. A link carries ~60 B/cyc (94% of 64); links of one column are independent.
Chain throughput = S x B / round trip (Little). The lane chain's hops measured ~0.9-1 K cycles (shared RISC loops),
2.5x the dedicated 396.
Roofline at M 586 (a core takes 26.4 B/cyc of x + h at 100% util): the busiest NOC0 column link needs ~45 B/cyc at
100% (90% of the chained x + h, plus the x gather's DRAM responses) -> link bandwidth caps util at ~130%; not binding.
The binding one is Little's law: ring bytes >= consume rate x delivery skew + the row in use. With dedicated-RISC hops
(3.6 K for 9) NX ~10 would cover 80%; with the lane's ~10 K skew it takes NX ~16 (what the L1 rework found).
NIU request FIFOs are off under tt-metal (NIU_CFG_0 0x6005, bit 16 clear; CMD_BUF_OVFL 0), so tt-regen-spec contract
D8 (a command dropped when the FIFO is full) is not the wedge. D6b (at most one path-reserving multicast tree in a
router per NoC at a time, else a request is lost: counted sent, acks missing) matches the lane design's many concurrent
group multicasts; it does not explain the single-multicaster repro (multicast + DRAM read responses), whose DRAM reads
already use static VC 1 (D31).
- ttsim's NoC model (src/noc_perf.h) says that with the request FIFOs off a command-buffer trigger before the last
  request has left takes its registers (that request lost, the counters counting both). Tested: the multicast build
  (ucast=0, 4 slots) with every multicast packet (payload split at 16 KB, then the flag) flushed before the command
  buffer is programmed again (WL_MC_FLUSHPKT): tray soak hung after ~200 launches (18 s; util 52%), as without it.
  Not the wedge.

## Chain hops in pieces, non-blocking forwarders (2026-11-16)
- WL_XP / WL_HP (LCfg.xp / hp): an x block / h row part hops in P pieces on tile boundaries, piece p's flag at word p
  of the slot's 16 B (sent from word p of the source: a NoC write keeps the 16 B offset, so origins fill all four
  scratch words; first try sent them from word 0 and hung). A slot is complete at word P - 1. Each forwarder call
  passes on every landed piece up to the end of its block (one piece a call: xp 4 fell to 37%).
- WL_FNB (LCfg.fnb): no blocking flush after a hop; walked counts published once the RISC's writes have left.
- WL_FPRIO (forwarders polled between every queue): 35-38% at M 586, h waits double. Off.
- WL_HARRW (h tile arrival as a per-producer word, no barrier + atomic in Q2): neutral. Off.
- Tray, same session (M 586 / M 2048): xp 1 no fnb 46.1 / 54.2; xp 2 + fnb 48.7 / 58.2; xp 2 alone 47.6 / 54.5;
  fnb alone 46.5 / 55.4; xp 4 hp 2 + fnb 46.8 / 51.5 (x waits fall ~45%, h waits rise more). New default xp 2, hp 1,
  fnb. Soak 300 s: 3361 launches clean, 50.1%. All tray tests pass (dynamic M min pcc 0.99987).
- Per-chip spread at M 586 is wide (44-52%); M 2048 is stable to +-0.3.

## Where the h waits come from (2026-11-16)
- Compute h waits split (stamp 44, waits list's 5th): the end-of-expert drains are only ~85K of ~0.95M at M 586; the
  rest is in-loop. DN_LAG 2 (WS_GCFG "h_dt=bfp8;XRM=2;dn_lag=2"): h waits 1050K -> 382K but x waits 205K -> 727K,
  util unchanged (48.4% / 58.6%): the waits move between x and h, the total holds -> a throughput limit of the data
  movement, not h latency or row order. Arrival-order h rows (rows are produced in the same order by all 20 cores and
  arrive near in order) would not help: not built.
- Busy-only loop time (calls that did something), M 586 chip 0, K cycles: BRISC forwarders 648 (st[60]), own weights 7
  (st[61]); NCRISC up-forwarders 426 (st[478]), weight queues ~910 (loop time). The chain hops cost the RISCs about the
  NoC injection time of the bytes they pass on (~55 B/cyc), on both RISCs.
- WL_FASYNC (forwarders issue only on a free write command buffer, yield otherwise): 40.3% / 43.0% (xp 4: 32.9%). Like
  WL_FPRIO: any yield inside a hop costs the hop a loop turn. Off.

## RISC busy time (2026-11-16)
- Busy-only timers for every queue (calls that did something; BRISC st[490..495], NCRISC st[480..484], riscs log
  line). Tray M 586 chip 0, K cycles, loop span ~3470K a core (steady state 16 x 108K = 1730K):
  BRISC forwarders 645, x origin 70, gather 164, h out 117, h leader 5, own 7 -> ~1010 busy;
  NCRISC weights / outputs / h leader 347, x origin 63, x ring + credits 506, h ring 43, forwarders 401 -> ~1360.
- Neither RISC is saturated (29% / 39% of the span; at most ~58% / ~79% even if all of it fell in the steady state).
  Offloading forwarding (idle cores, BRISC on NOC1) would not raise a throughput cap that is not there: the limit is
  the latency of the x / h exchange loops against their buffering (x ring slots, h ring rows; L1-bound).
- (Busy shares are of the whole ~3.47M span; see the timeline below: the span is all 16 experts, not 2x the window.)

## Start-up / first expert (2026-11-16)
- Timeline log (test `timeline`: loop starts and each expert's end, median / max over the 40 cores, K cycles from the
  compute start). Tray M 586 chip 0: BRISC loop starts at 0.9, NCRISC at 7.9 (fabric mux connect), compute at 0.9;
  experts (19 rows each) end at 272, 483, 673, 880, 1076 (max 1108), ... 3418 (max 3499): ~200K an expert after the
  first's 272K. Chip 7 similar, with one slow expert (321K, its second) and a last-expert spread median -> max of 127K.
- So the first expert costs ~70K over steady state (~2% of the launch), not ~1.7M: my earlier "span is 2x the window"
  read "cyc/expert" (span / (G x (E - 1)), per chip-expert) as per lane-expert. End to end (launch start to the last
  core's end) the util is ~8% below the steady-state metric: the first expert's excess plus the tail skew (last
  expert's median -> max 80-130K).

## K-split gate/up benches (2026-11-16; toward moe_fused_swiglu's K x N layout in a group)
Idea: a group's 10 rows split gate/up K (12 K tiles each), its 2 columns split N (10 pairs each); core (c, r) still
holds 1/20 of the weights (12 x 20 = 240 tiles, as 120 x 2 today), x[:, K_r] is shared only with the partner (one
hop, no chain), partials are reduce-scattered down the column (core (c, r) owns pair 10c + r: the same h tile it
produces today, so the h exchange is unchanged). All fp32 DEST, LoFi, one core, Blackhole aiclk 1350.
- Matmul (test_mm_bench ksplit, x bfp8, W bfp4, partials packed plain): 1x4 K 12 / 24 / 30 / 120: 17.52 / 17.39 /
  17.36 / 17.28 cyc a tile (91.3 / 92.0 / 92.2 / 92.6%); bf16 and bfp8 partial packing the same; 1x2 K 12 18.08.
- Fold + SwiGLU (test_fold_bench; moe_fused_swiglu's add_tiles fold, two contributions a call into DEST, one init
  a window): fold only C 2 / 4 / 5 / 10: 71 / 178 / 276 / 305 cyc an h tile (gate + up), bf16 = bfp8; fold +
  SwiGLU 1141 for every C = copy + SwiGLU alone (the PACK SFPU bounds it). One init a tile instead: 421 at C 10.
  h error against fp32 partials (h packed bf16): bf16 partials rel 2.4e-3 (pcc 0.999996), bfp8 1.0e-2 (0.99995).
- A whole row on one core (test_fold_bench row; resident operands; gate/up + epilogue + down 6 columns K 20; 360
  tile products = 5760 cyc at 100%): today 1 x 20 (1x2 K 120 + SwiGLU) 6481 (88.9%), without SwiGLU 6420; 10 x 2
  matmuls only 6414 (89.8%); 10 x 2 + fold C 10 + SwiGLU 6952 (82.9%, bf16 = bfp8 partials); C 5 6870 (83.8%),
  C 4 6802 (84.7%). Today's SwiGLU hides under the matmuls (+60); the K split's fold + SwiGLU costs ~540 a row.
- Column reduce-scatter (test_rs_bench, kernels/rs_bench.cpp; 10 cores of column 1, dedicated RISCs, every core
  sends each other core its pair's (gate, up) tiles a row: payload + flag from one command buffer, flush a row,
  slot credits as inline writes; R 400 rows):
| split | partials | D | capacity (cyc a row) | at T 6950: lag median / max |
|---|---|---|---|---|
| direction (BRISC NOC0 down, NCRISC NOC1 up) | bf16 | 2 / 4 | 3095 / 3072 | 2519 / 3049 |
| direction | bfp8 | 2 / 4 | 1910 / 1956 | 1579 / 1851 |
| gate / up (moe: gate NOC0, up NOC1, half wrapping) | bf16 | 2 / 4 | 3081 / 3076 | 2946 / 3511 |
| gate / up | bfp8 | 2 / 4 | 2145 / 2169 | 1846 / 2277 |
  Keeps up at the compute's row rate with D 2 (landing 2 x 10 x 2 bf16 tiles = 80 KB); lag ~2.5 K cycles (today's
  x chain: ~10-12 K to the last core, which needed NX ~16 x BK 20 of x ring). The direction split beats the gate /
  up split here (its wrap-around paths). Busiest link (the column's middle, all-to-all): 25 pairs x 4 KB a row =
  ~14 B/cyc at T 6950 vs today's x chain ~20 B/cyc on every link (+ h ~3 on both).
- Per core per row, bytes in: today x 130 KB (chained, forwarded by every core) + h 20 KB; 10 x 2 x 6.5 KB (half from
  the partner) + bf16 partials 36 KB + h 20 KB.

## K-split groups, single-chip prototype (ws_ks.py, kernels ks_*.cpp; 2026-11-17)
Same harness as ws_group (one chip, injector-landed weights "l1", x gathered from DRAM by index and tilized on the
group cores, each group's last expert checked), only gate/up changed: core (c, r) holds W_gu[K tiles 12 r .. 12 r +
11, pairs 10 c .. 10 c + 9] (the injector's WG_INJ_KS slice), computes 20 bf16 partial tiles a row (5 x 1x4, K 12);
partials of pair 10 c + r go to core (c, r) (rows below: BRISC NOC0, rows above: NCRISC NOC1, itself: local NoC
write; payload + flag from one command buffer, slot credits as inline writes), which folds the 10 contributions in
fp32 DEST (add_tiles, acc_to_dest), applies SwiGLU and produces h tile j as before. h (column leaders + multicasts),
down, output, weights: unchanged. No x chain: x[:, K_r] goes partner to partner, one hop.
- x gather: halves of every row (384 B reads, 4x the 1 x 20 groups' read count) slowed G 4 (59.7% M 586): the
  partners now gather / tilize whole 12-tile rows alternately (KS_XALT, 768 B reads): G 4 = G 1 (71%).
- Compute (MATH time a row, G 4 M 586): gate/up 4.4 K, fold 0.75-0.8 K, down 2.2 K, tilize 0.2 K = 7.6 K vs 5.76 K
  at 100% (today's 1 x 20 row: ~7.2 K). One full matmul init a row (after the fold); gate/up <-> down switch only
  srcB. Folding two rows at a time (KS_FOLD2) saves fold time but the down loses it (SwiGLU x 2 on PACK): no gain.
  Folding before the row's gate/up: worse.
- Results (G 4, 32 experts, util tile; PCC 0.99994 with bf16 h, 0.99988 bfp8 h):
| config | M 586 | M 2048 |
|---|---|---|
| ws_group 1 x 20 multicasts (2 slots, same session) | 79.3-80.9% | 82.1% |
| ws_ks first version (halves of x rows, 3 inits a row) | 59.7% | 64.5% |
| + whole x rows alternately, one init a row | 71.0% | 71.1% |
| + NXR 8, TL 3, DN_LAG 2 (fold of 1 row, D 3) | 74.7% | 75.3% |
| lane-like L1: 4 weight slots reserved, bfp8 h, XRM 2, D 3, NXR 8, TL 3, DN_LAG 2 | 75.5% | 75.8% |
  The lane-like L1 fits: weights 4 x 207 KB + x ring 8 x 12 bfp8 tiles (104 KB) + landing 3 x 20 bf16 (123 KB) +
  partial ring 2 x 20 bf16 (82 KB) + h ring (82 KB bfp8) + staging. Waits a core at M 586 (K cycles of ~1.0 M):
  tilize 36, x 20, weights 3, h 4, landing 8 - it is compute-bound (5.76 / 7.6 = 76%). Dynamic M (0 / 1-row
  experts) passes. Unicast only (no group multicast except the h leaders', which a lane port replaces with the h chain).

## K split on the lane (tray, 2026-11-17): every output written, bf16 partials
Port of ws_ks to the lane design (LCfg.ks / WL_KS in wl_writer / wl_reader, kernels/kl_compute.cpp): the x chain is
replaced by whole x rows partner to partner and the bf16 partial reduce-scatter; weights, fabric, h chain unchanged.
own slices in the K-split layout (ws_fab.slice_layout ks=1). out_all (every expert's down rows to DRAM, every expert
PCC-checked) is now the LCfg default.
Fixes found on the way:
- h ring slot credits (LCfg.hcred / WL_HCRED, default on): each core broadcasts its consumed h rows to its column and
  partner; every write into a core's h slot (tile to the leader, chain start, hops) waits for it. Without them the
  K split showed intermittent PCC 0.9977-0.9986 dips (one core's columns, an expert's first row: a fast core
  overwrote a slow core's h slot); the chain build had the same latent gap (NS sized by the host). ~0.6 point.
- KS_OVL final drain: the last expert's drain ran downs of the previous expert's rows with the last expert's down
  weights (fold lag 2, 1-row experts: PCC ~0 for the last expert). Found by M 5 / dynamic M, fixed.
- Slot table fields widened to 3 bits (5 weight slots); NS >= DN_LAG asserted (NS 2 with DN_LAG 3 deadlocked).
Steps (tray M 586, util tile median over 8 chips x 5 launches after 20 warm-up; all outputs written):
| change | M 586 | note |
|---|---|---|
| chains (previous default), out_all | 47.1% | M 2048 53.4% |
| ks=1 | 52.0% | M 2048 67.3% |
| + h credits, wpoll 1 | 51.3% | correct (was dipping) |
| + KS_OVL: an expert's last folds / downs overlap the next one's first rows (drain only if it reuses the slot) | 52.4% | |
| + sched_c 7 (schedule planned with the faster compute) | 53.2% | |
| + ks_bal 5: partials by distance (BRISC itself + 4 down, NCRISC 5 up, wrapping the column ring) | 54.1% | row 0's BRISC had 822K busy vs row 9's 109K |
| + DN_LAG 3, fold lag 2 | 54.6-54.9% | waits now: weight skew a core -> landing / h |
| + 5th weight slot (d0,d1,d0d1,any,any; model 0.664 -> 0.762), PD 1, ydep 1, NXR 5, TL 2, NS 3 | 56.8-57.3% | M 2048 67.0% |
| + KS_XDIRECT: compute tilizes x straight into its ring slot (no CB 8 staging), XRM 3, NXR 6, TL 5 | 57.8% | M 2048 68.8% |
No gain: h all-to-all instead of the chain (ha2a; h waits fall, landing waits rise), all partials on the BRISC
(47.4%), weight queues twice a turn, output flush instead of barrier, deeper landing / x rings at 4 slots, 4-slot
typings other than d0,d1,d0d1,any. The pipelined x gather (8 rows of ids a read, several rows of reads in flight) is
in but neutral.
Final configs (WS_LANE / WS_GCFG):
- bfp8 h: "slots=d0,d1,d0d1,any,any;keep=2;ks=1;wpoll=1;ks_ovl=1;sched_c=7;ks_bal=5;ks_pd=1;ydep=1;ks_xdirect=1;
  ks_fl=2;ks_d=3;ks_nxr=6;ks_tl=5" / "dn_lag=3;NS=3;XRM=3": M 586 57.8% (54.4-61.8), M 2048 68.8%, PCC 0.99988;
  dynamic M min 0.99986, M 0 / 5 pass; soak 300 s: 2098 launches, 56.1%, PCC 0.99988.
- bf16 h: the same with ks_nxr=4;ks_tl=3 / "dn_lag=3;NS=3;XRM=2;h_dt=bf16": M 586 55.4-55.8%, M 2048 66.0%,
  PCC 0.99994; dynamic M min 0.99992, M 0 / 5 pass.
- Galaxy not run: the lane model says 5 slots are worse on the 8 x 4 torus (0.698 -> 0.652, random timing).

## K split on the lane, part 2 (tray, 2026-11-17): weights without the router's ack wait, group layout
Same harness (all outputs written, bf16 partials, util tile median over 8 chips x 5 launches after 20 warm-up).
- Weight transport: test_mux_bench gives 26.2 GB/s a link with the fused write + increment flushed (the router waits
  for the write's ack before the increment) and 43.7 GB/s without. LCfg.arr_nf (WL_ARR_NF) drops the flush on the
  weight packets: M 5 (weights only) 72 K -> 56 K cycles a chip-expert, M 586 57.2 -> 59.5%, weight waits ~0. PCC
  unchanged (every expert checked); dynamic M identical with and without; soak 300 s: 3172 launches, no hang.
  arr_write (plain arrival writes): 56.1%.
- With the faster weights 4 slots match 5 (59.7 vs 59.5%), freeing 207 KB for deeper rings.
- Position: the group cores' gate/up MATH time a row depends on the columns (per-core stamps 64-67 added: pack
  waits on the partial / h / y CBs and MATH DEST waits are small; the slow cores are in columns 1-2 / 7-8 and row 9,
  and swapping the groups' columns moves the slowness with the columns). ws_lane.group_layout: G <= 2 now uses
  columns (4, 5) / (9, 10), muxes in 3 / 8 (WS_GLAYOUT overrides; WS_MUXROWS for the mux rows):
| layout (a, b, mux) | M 586 |
|---|---|
| (1,2,0) (4,5,3) (old) | 59.5% |
| (1,2,0) (7,8,6) | 58.7-59.1% |
| (4,5,3) (7,8,6) | 58.7% |
| (1,2,0) (9,10,8) | 60.6% |
| (3,4,2) or (5,6,4) with (9,10,8) | 61.2-61.3% |
| (4,5,3) (9,10,8) | 62.4-62.8% |
  Mux rows (0,1) / (8,9) / (0,9): within noise.
- Deeper buffers in the freed L1: XRM 4, landing D 4, x ring 8 rows, tilize look-ahead 7, NS 4: 63.8%.
- No gain: fold with 2 inits instead of 6 (MATH fold 0.74 -> 0.31 K a row, but the down then waits that long for the
  SwiGLU on PACK; kept), fold before the gate/up (ks_ff; with fold lag 2 it hangs) or after the down (ks_fz), idle
  back-off of the data-movement loops (idle), sched_c 5 / 10 / 14, ks_pd 2.
Final configs (WS_LANE / WS_GCFG), every output written, bf16 partials:
  "slots=d0,d1,d0d1,any;keep=2;ks=1;wpoll=1;ks_ovl=1;sched_c=7;ks_bal=5;ydep=1;ks_xdirect=1;arr_nf=1;ks_fl=2;ks_pd=1;
  ks_d=4;ks_nxr=8;ks_tl=7"
| h | WS_GCFG | M 586 | M 2048 | PCC | dynamic M min PCC |
|---|---|---|---|---|---|
| bfp8 | dn_lag=3;NS=4;XRM=4 | 63.8% | 68.9% | 0.99988 | 0.99983 |
| bf16 | dn_lag=3;NS=4;XRM=4;h_dt=bf16 | 61.3% | 67.3% | 0.99994 | 0.99993 |
  M 0 / M 5 pass for both. (Part 1's best: 57.8% / 55.5%.)

## M sweep of the final K-split lane configs (tray, 2026-11-17)
Reproduced first on bh-lb-19 (2 x 4 p150b, 200G eth firmware 19.12.0+eth200, aiclk 1350): M 586 / 2048 bfp8 h 63.7% /
69.0%, bf16 h 60.8% / 67.1%; M 0 / 5 and dynamic M (min PCC 0.99983) pass. test_ws_lane_tray_msweep runs the tray test
over constant M; chip_time now also returns the end-to-end device time of a launch (earliest group-core loop start,
compute / BRISC / NCRISC, to the latest compute end, per chip; the summary takes the slowest chip). 32 experts a chip
(16 a lane), T 8192, every expert's output written and checked, 20 warm-up + 5 measured. util = steady-state tile
util (40 group cores); e2e util = 32 x R x 7200 tile products x 16 cycles / 40 cores over the end-to-end time.
| M | bfp8 h util (min-max) | e2e us | e2e util | us a lane-expert (steady) | bf16 h util (min-max) | e2e us | e2e util |
|---|---|---|---|---|---|---|---|
| 32 | 5.4% (5.2-5.7) | 1,268 | 5.4% | 39.4 | 5.3% (5.1-5.7) | 1,292 | 5.3% |
| 64 | 10.0% (9.4-10.6) | 1,380 | 9.9% | 42.7 | 10.1% (9.5-10.4) | 1,377 | 9.9% |
| 128 | 20.3% (19.0-21.2) | 1,398 | 19.5% | 42.2 | 20.1% (18.9-21.1) | 1,410 | 19.4% |
| 256 | 37.7% (34.8-40.0) | 1,549 | 35.3% | 45.4 | 37.2% (33.8-40.3) | 1,599 | 34.2% |
| 384 | 51.5% (47.0-56.6) | 1,726 | 47.5% | 49.9 | 49.7% (45.4-54.1) | 1,808 | 45.3% |
| 512 | 58.9% (55.0-62.2) | 1,991 | 54.9% | 57.7 | 57.7% (52.7-59.6) | 2,071 | 52.7% |
| 586 | 63.7% (59.9-65.1) | 2,168 | 59.8% | 63.8 | 60.9% (57.0-62.2) | 2,293 | 56.6% |
| 768 | 64.3% (62.4-65.8) | 2,675 | 61.3% | 79.3 | 62.0% (58.9-63.3) | 2,789 | 58.7% |
| 1024 | 66.1% (64.2-67.6) | 3,435 | 63.6% | 103.4 | 64.0% (61.7-65.5) | 3,553 | 61.5% |
| 1536 | 67.9% (66.6-69.4) | 4,944 | 66.3% | 150.7 | 66.0% (64.5-67.2) | 5,131 | 63.9% |
| 2048 | 68.9% (67.9-69.8) | 6,479 | 67.4% | 198.1 | 67.1% (65.9-67.9) | 6,663 | 65.6% |
| 3072 | 69.9% (69.1-70.6) | 9,526 | 68.8% | 293.0 | 68.3% (67.2-68.8) | 9,777 | 67.0% |
| 4096 | 70.3% (69.4-70.9) | 12,569 | 69.5% | 388.5 | 68.8% (68.0-69.4) | 12,840 | 68.1% |
| 6144 | 70.8% (70.2-71.2) | 18,647 | 70.3% | 578.7 | 69.3% (68.9-69.9) | 19,065 | 68.8% |
| 8192 | 71.0% (70.8-71.4) | 24,759 | 70.6% | 768.9 | 69.7% (69.4-70.1) | 25,250 | 69.2% |
PCC 0.99987-0.99988 (bfp8 h) / 0.99993-0.99994 (bf16 h). Up to M ~128 a launch takes ~1.3-1.4 ms whatever M: weight
bound (32 x 4.15 MB of bfp4 weights into a chip, ~100 GB/s; ~40 us a lane-expert). The knee is M ~500-600; above it
util creeps to ~71%, the compute-bound ceiling of the K split (5.76 / 7.6 K a row ~76%, single-chip 75.5%). End to
end is 1-4 points below the steady state (first expert + tail skew), less at large M.

## Weights only: where the lane's weight delivery goes (tray, 2026-11-17)
bh-lb-19's links train at 400G (eth_status train_speed, all 8 chips; 50 GB/s a direction raw), not 200G.
Per lane and launch (16 experts x 7 hops x 4.147 MB over the cycle's 16 link directions) each link direction carries
29.0 MB. Weights-only lane time = the group's last NCRISC loop end at M 0 (chip_time nc_end; compute has no rows):
| lane config (M 0) | lane 0 (link 0) | lane 1 (link 1) |
|---|---|---|
| 4 slots, arr_nf | 1,118 us = 26.0 GB/s a link direction | 1,055 us = 27.5 GB/s |
| 5 slots (smaller rings), arr_nf | 1,073 us = 27.0 GB/s | 1,011 us = 28.7 GB/s |
| 4 slots, flushed incs (arr_nf 0) | 1,623 us = 17.9 GB/s | 1,636 us = 17.7 GB/s |
Fabric alone on the same pattern (tests/test_lane_fab_bench.py, kernels lane_fab_client / lane_fab_recv: the lane's
group cores, mux cores, cycle and links, every chip, both lanes and directions, each core streaming 300 packets a
direction with the lane's non-blocking send; no schedule, slots, credits or cut-through), 5 + 5 launches:
| pattern | GB/s a link direction median (min) |
|---|---|
| one link one way (test_mux_bench) | 43.6 (26.2 with the router's ack wait) |
| one lane, direction 0 only | 43.7 (41.7) |
| two lanes, direction 0 only | 43.7 (33.2) |
| one lane, both directions | 40.0 (37.2) |
| two lanes, both directions (the lane's load) | 39.6 (32.2); 155 GB/s into a chip |
| the same with the router's ack wait (flush) | 22.9 (21.4) |
- Link 0 of the two vertical edges (chips 0-4, 3-7) runs at 32-33 GB/s in both directions under load; link 1 of the
  same edges and every other link 37-40. Lane 0 contains both, so it is paced at ~32.
- Packet size / mux buffering / atomics (test_lane_fab_sweep, lane load, same bytes; router payload = packet):
| packet | mux buffers a channel | inc every | median | lane 0 / lane 1 slowest link |
|---|---|---|---|---|
| 13,824 B (payload 15,232) | 2 / 4 | 1 | 39.6 / 39.5 | 32.1 / 37.4 |
| 13,824 B | 2 / 4 | 15 (one a slice) | 41.4 / 41.3 | 33.6 / 39.6 |
| 6,912 B | 2 / 4 / 8 | 1 | 32.2 / 32.2 / 32.4 | 29.8 / 32.1 |
| 6,912 B | 8 | 30 | 34.0 | 31.5 / 33.9 |
| 4,608 B | 4 / 8 / 12 | 1 | 25.0 / 25.1 / 24.8 | 24.3 / 24.9 |
| 4,608 B | 12 | 45 | 26.1 | 25.4 / 26.1 |
  Mux buffers change nothing; smaller packets lose: time a packet ~100 ns + bytes / ~56 GB/s (fit over the three
  sizes), so 13.8 KB packets reach ~80% of the wire and bigger ones would help. A per-packet atomic inc costs ~4.5%.
- Lane model (ws_lane_sched, lock-step, weights only C 0): 4 slots 118 packet steps against the 105-step link floor
  (87% busy), 5 slots 104 (98%), 3 slots 181 (57%). Measured 5 vs 4 slots: +4% only.
- Budget, lane 0 at 4 slots: 50 raw -> 39.6 (both directions, 13.8 KB packets, per-packet incs) -> 32 (its slow
  link) -> x 0.87 schedule = 27.9 predicted vs 26.0 measured; lane 1: 37.4 x 0.87 = 32.5 vs 27.5. The rest (7-15%) is
  protocol latency the lock-step model does not see (cut-through hops, credit round trips).
- Reference: high_bw_all_gather on this box (realtime profiler, effective receive GB/s a chip; 14 KB payload, the
  op's perf helper insists on it). 1 x 4 row line, 2 links (test_high_bw_all_gather_512k[fabric_2d], 65536 rows a
  chip): 79.3-80.7 GB/s for all four formats = 40 GB/s a link direction on the end edges (1.5 shards a link
  direction). Whole tray, cluster_axis None = the lanes' 8-chip cycle x 2 links (tests/test_hbag_tray.py):
| rows a chip (bf16 / bfp8 tiles) | shard | time | receive a chip | a link direction (1.75 shards each) |
|---|---|---|---|---|
| 8192 | 9.4 MB | 0.487 / 0.281 ms | 135.7 / 124.9 GB/s | 33.9 / 31.2 |
| 14400 (= one lane launch: 4 own experts) | 16.6 MB | 0.811 / 0.447 ms | 143.2 / 137.9 GB/s | 35.8 / 34.5 |
| 65536 | 75.5 MB | 3.603 / 1.914 ms | 146.7 / 146.7 GB/s | 36.7 |
  The lanes move the same 116 MB into a chip in 1.06-1.12 ms (104-110 GB/s): hbag is 1.38x faster on the same cycle
  and links. Its differences: whole-shard store-and-forward into a DRAM output (no slot limit, so no schedule bubbles
  or slot credits), one inc a shard instead of a packet, one dedicated worker a (ring, direction, link) beside its
  ethernet core, no mux. Mux placement is not it for us: the lane fabric bench with muxes at rows 0-1 or 8-9 gives
  the same 39.6 median / 32-33 on the slow link-0 vertical edges.

## More L1 buffering for the weights (tray, 2026-11-17)
- Lane model (C 0 weights only / C 7; lock-step and random timing): slots 3 / 4 / 5 / 6 / 7 / 8 -> lock-step 0.58 /
  0.89 / 1.0 / 1.0 / 1.0 / 1.0 of the link floor, random 0.51 / 0.72 / 0.82 / 0.76 / 0.74 / 0.74. Beyond 5 whole-slice
  slots nothing to gain (random timing even loses: the static order is planned lock-step).
- Device, weights only (M 0, compute rings shrunk: ks_nxr 2, ks_d 1, NS 1, XRM 1, dn_lag 1), NCRISC end lane 0 /
  lane 1: 4 slots 1112 / 1055 us; 5 slots 1071 / 1000; 6 slots 1069 / 998; 4 slots + mux nbuf 4 1091 / 1015;
  5 slots + nbuf 4 1060 / 951 (best: lane 1 30.5 GB/s a link direction, -10%); 5 slots + nbuf 6 does not fit the
  mux core (20 channels x 6 x 15.3 KB).
- With compute (20 + 5, util / end to end): final 4 slots nbuf 2 M 586 63.7% / 2190 us, M 2048 69.0% / 6448 us;
  nbuf 4 61.5% / 2220, 67.8% / 6565; 5 slots (ks_d 3, nxr 6, tl 5, NS 3, XRM 3) nbuf 4 61.3% / 2243, 68.2% / 6548;
  nbuf 2 62.6% / 2204, 69.0% / 6480. No gain: at M >= ~500 the weights (1.1 ms) arrive faster than compute needs
  them (2.2 ms); buffering only matters in the weight-bound range (M < ~400).
- More muxes: two muxes a (group, direction) on the same link (10 clients each, test_lane_fab_nmux) hang - every mux
  in forward_data waiting for router space, clients waiting on their mux: the router takes one worker connection a
  direction. Extra muxes need extra links; on the tray only the two middle vertical edges (1-5, 2-6) are unused.

## 5 x 5 K-split groups, single-chip prototype (ws_g5.py, kernels g5_*.cpp; 2026-11-17)
Why: on the Galaxy (640 experts, M 586, 200G links at ~23.5 GB/s a link direction) the weight floor is 13.7 ms a
launch for any group shape, compute with 4 x 20 cores at ~75% is ~17 ms; 4 x 25 cores would bring compute to the
weight floor. Group = 5 columns c x 5 rows r (j = 5 c + r):
- K split uneven so that every core does 320 tile products a row: rows 0-3 hold 25 K tiles, row 4 holds 20; column c
  holds pairs 4 c .. 4 c + 3 (8 partial tiles a row, 2 blocks of 1 x 4). Owners (c, o), o < 4, fold the 5
  contributions of pair 4 c + o and run 4 down columns; row 4 owns no pair and runs 8 down columns. Measured MATH a
  core (M 586): owners gate/up 2061K + fold 364K + down 929K, row 4 gate/up 1695K + down 1742K: balanced.
- x: row r's K slice is needed by its 5 cores; core (c, r) gathers / tilizes rows q % 5 == c and the BRISC unicasts
  each row to the 5 x rings (one flag a slot, credits a consumer column). h: owners -> column leader (c, q % 4) ->
  multicast of the column's 4 tiles (even columns BRISC / NOC0, odd NCRISC / NOC1), 5 flags a slot. Partials: rows
  below / itself from the BRISC, rows above from the NCRISC (ks scheme).
- On this 11 x 10 grid only logical columns 0-6 are NoC contiguous: G <= 2 (columns 0-4, rows 0-4 / 5-9); the
  Galaxy's 12 x 10 grid holds 4 (columns 0-4 and 7-11). Injectors: 25 cores, P 2 weight sets preloaded.
- First run correct (PCC 0.99994, G 1 / 2, M 5 / 64 / 586 / 2048) but 61.7%: x waits. The compute's tilize
  look-ahead refilled only when compute caught up with it; with rows dealt 1 in 5 a core's next own row was often
  tilized just when peers needed it. Continuous look-ahead (every own row q <= need + TL): x waits 440K -> 5-10K.
| G 2, 64 experts, same session | cores | M 586 util / cyc an expert | M 2048 |
|---|---|---|---|
| 10 x 2 K split (ws_ks, NXR 8, TL 3, dn_lag 2, D 3) | 40 | 75.4% / 72.6K | 75.4% / 244.3K |
| 5 x 5, TL 9 | 50 | 71.4% / 61.3K | 73.5% / 200.7K |
| 5 x 5, TL 9, D 4, FL 2 | 50 | 75.3% / 58.1K | 73.8% / 199.8K |
| 5 x 5, TL 9, dn_lag 3, D 4, FL 2 (default) | 50 | 77.3% / 56.6K | 75.0% / 196.6K |
| 5 x 5, TL 12, D 4, FL 2 | 50 | 68.7% | 66.8% (h waits) |
  Per-core efficiency equal to the 10 x 2 split, 1.25 x the cores: 1.28 x / 1.24 x faster a chip-expert. Remaining
  waits are h (owners 76K, row 4 117K at M 586). Single-launch runs vary +-3-4 points (32 experts); 64 experts above.
  Configs with NS 8 or NXR 10 + TL 12 collide with the injectors' 415 KB of preloaded weights (prototype only).
- Galaxy projection (M 586, 640 experts, 200G): compute 4 x 20 at 75% 17.2 ms, 4 x 25 at 77% 13.5 ms, weight floor
  13.7 ms: with 5 x 5 the launch is bound by weight delivery unless the lanes reach ~100% of the links.
- Lane L1 estimate (4 slots of the larger slice 184 KB = 737 KB, x ring 8 x 25 bfp8 218 KB, landing 82, partials 33,
  h ring NS 6 bfp8 131, CB 7 102, CB 8 109, CB 17 33): ~1,445 KB of 1,460 KB - fits only with bfp8 h and tight rings.

## 5 x 5 groups on the lane (ws_lane5.py, kernels l5_*.cpp; tray, 2026-11-17)
The lane's schedule, slot tables, muxes and credits unchanged; the groups are the 5 x 5 K split of ws_g5.py:
- Weight packets of CHT 20 bfp4 tiles (11.5 KB): owners' slices 200 gate/up + 80 down tiles = 10 + 4 packets, row 4
  160 + 160 = 8 + 8. One weight-ring tensor a core (K slots of the largest slice, 320 tiles), CB 1 / CB 3 mapped over
  it per core type (gate/up slots, then down slots). Own experts: slice_layout5 (chunk q = own index x 360 + the
  core's first chunk + u). 25-client muxes in column 0, groups in columns 1-5 (rows 0-4 / 5-9).
- Compute (l5_compute): g5_compute in the chip's order with the weights from slot s (use_slot), each expert finished
  on its own before the next one's weights are waited for (the schedule's slot-release rule).
- h without multicasts. First version, a fan-out (each owner writes its tile + flag into all 25 cores; credits from
  every core to the 20 owners): 41.0% / 48.1% at M 586 / 2048 - the owners' BRISC spent ~3.3 K cycles a row on 50
  transactions, partials and folds waited behind it. Now a two-stage all-gather: stage A (owner BRISC) the tile to
  the 5 cores of its column; stage B (NCRISC) each core forwards its column's 4 tiles as one block to the 4 other
  cores of its row (+ a column flag); credits from each consumer to its column's 4 owners and its row's 4 other cores.
| tray, 20 warm-up + 5 measured, every expert checked | M 586 util / end to end | M 2048 | M 0 (weights only) | PCC |
|---|---|---|---|---|
| 10 x 2 lane (final config, 40 cores) | 63.7% / 2,190 us | 69.0% / 6,448 us | 1,112 / 1,055 us | 0.99988 |
| 5 x 5 lane, h fan-out | 41.0% / 2,678 us | 48.1% / 7,484 us | | 0.99988 |
| 5 x 5 lane, two-stage h (4 slots) | 54.0% / 2,086 us | 72.9% / 4,914 us | 1,185 / 1,184 us | 0.99988 |
  M 0 / M 5 pass, dynamic M (zeros .. 4000, every expert) min PCC 0.99985. Per launch 1.05 x / 1.31 x the 10 x 2 lane.
- Sweep (10 + 5 launches; M 586 / M 2048 end to end): base 2,094 / 4,907; NS 6 2,102 / 4,914; D 4 2,113 / 4,918;
  NXR 8 2,138 / 4,912; dn_lag 2 2,136 / 4,982; TL 6 2,100 / 4,904; TL 12 2,155 / 4,912; FL 1 2,111 / 4,997 - ring depths
  do not matter; 5 slots (d0,d1,d0d1,any,any; NXR 4, D 2, TL 6, NS 4 to fit) 1,924 us (58.1%) / 4,950: -8% at M 586;
  mux nbuf 4 does not fit the mux core (25 channels).
- Galaxy layout in L5Cfg (12 x 10: groups at columns 1-5 / 7-11, rows 0-4 / 5-9, muxes in columns 0 / 6) and
  test_ws_lane5_glx (not run: no Galaxy reachable from this cluster).

## Stage 19: the K-split lane in the layer (branch ws-sf-ks: origin's 1efbae75faf merged; hold 217516, bh-glx-120-b08u08)
- Merge: origin's K-split lane (LCfg.ks, kl_compute.cpp, the unicast chains, h credits, arr_nf) and this branch's
  NoC split / corner multicaster, one-chip proxy (nofab), row-major result rows (y_rm) and the layer, every feature
  behind its own switch. The K split's x queue takes precedence over the NoC split's in the reader.
- Row-major result rows on the K split: kl_compute.cpp's down row is pack-untilized into CB 17 (WL_Y_RM; the DEST
  untilize remap on once after the hardware startup, as wl_compute and conv3d's compute: the tilize, gate/up, fold
  and SwiGLU run with it on); y_rm allowed with the unicast chains (no group multicast on NOC1, QY's NoC); in K-split
  mode QY's destination scratch sits past the BRISC's 8 rows of gathered token ids (idx_l1 + 1024).
- ws_layer: KS_LANE / KS_GCFG (the pushed commit's final tray config, bfp8 h) for proxy_lcfg / glx_lcfg(ks=True)
  (tests: WSY_KS=1); the NoC-split lane pinned to ucast=False. ws_lane allocates the chains' / all-to-all's small
  state tensors only with their features (unconditional, ~1.6 KB a core pushed the NoC-split lane's L1 tensors into
  its circular buffers on one chip).
- The stream alone on the Galaxy (test_ws_lane_glx, 640 experts, M 586, 20 warm-up + 5 measured, every expert written;
  WS_REF_LAST / WS_PCC_CHIPS against host memory on b08u08):
| lane | cyc / chip-expert (chip 0) | util tile median | min PCC |
|---|---|---|---|
| NoC split + corner multicaster (stage 7 config, c10u20) | ~100,800 | 27.4% | 0.99994 |
| origin chains, no K split | 81,542 | 33.2% | 0.99988 |
| origin K split, final config | 57,515 | 46.6% | 0.99988 |
| the same after the merge | 57,973 | 46.8% | 0.99988 (every expert, chips 0 / 13 / 31) |
- One chip, the layer with the K split, T 2048, every token against torch: counts / lists / destinations exact,
  routed rows PCC 0.99988, y 0.99988 (the bfp8 h floor).
- The Galaxy, the folded layer traced with the device barrier, 20 warm-up + 5 measured replays:
| | NoC-split lane | K split |
|---|---|---|
| traced layer back to back (ms) | 62.5 | 41.9 |
| inside a replay: head / stream / fold | 2.70 / 49.72 / 9.44 | 2.69 / 28.63 / 9.44 |
| eager device sum | 61.5 | 40.9 |
  Trace bit-identical to eager (y, counts, token lists, destinations, w); 64-token checks on chips 0 and 31: y PCC
  0.99988, routed rows 0.99988.
- One chip (T 31250, traced, 20 back to back bracketed): head 2.80, stream 29.60, fold 9.03, layer 41.36 ms (old lane
  37.56 stream, 49.26 layer). The one-chip stream (weights from its own DRAM) now matches the Galaxy's (28.6): the
  fabric delivery is no longer the stream's limit. Chip power under the stream 125 W median (139 max), AICLK 1318-1350
  at the 130 W limit: on a 75 W system this version throttles harder than the old one.
- A latent fault in the barrier layer (stage 15), found on the K split: the device barrier's all-gathers create their
  global semaphores in L1 when their programs are first built (kept with the cached programs). The first barrier
  of a run came between "stream:zero" and "stream:run", so that one rebuilt stream sat 256 B lower than the words its
  zeros cleared (every state word, both lanes) and ran on stale state: the old lane survived it, the K split hung once
  (triage: compute waiting on landing rows, two cores' firmware noc_index read back corrupt). Fixed: make_layer runs
  the barrier once (tests/test_ws_layer.py device_barrier), so the semaphores exist before any stream is built;
  layer_programs asserts that the rebuilt stream's state words are exactly the zeroed ones.
- After the fix (b08u08, 20 warm-up + 5 measured replays): the K-split Galaxy traced layer 41.96 / 41.82 ms (twice),
  the old lane 62.75; both bit-identical to eager. 3000 traced K-split layers with the barrier: all 6 checks (every
  500) bit-identical to the first layer on chips 0 and 31, no hang; wall per layer 42.06 -> 43.37 ms over the run
  (the chips' clock settling under the 130 W limit at the stream's ~125 W).
- Open: the stream 28.6 ms against the 14.8 target; the fold 9.4 against 7.5; the head 2.7; a soak of 10K+ layers on
  the K split; power on 75 W systems.

## EP-4 with weights on Y (LoudBox)
Regime for 302K-588K chunks on the Galaxy: each mesh row (4 chips, X) splits the 640 experts four ways (chip column c
holds [160 c, 160 c + 160)); token rows move only along X (dispatch, combine); the column's 160 experts stream around
its Y ring (LoudBox: 2-chip column, 80 own + 80 remote a chip; Galaxy: 8 chips, 20 own + 140 remote). Combine
pre-reduces on the expert chip: one partial row a (token, expert chip) into slot c of the home chip's P[t]; the home
pass adds the present slots and the shared expert's. Branch mstaletovic/ep4-lb (from ep4-wsy aa8fcd855d3), worktree
tt-metal-ep4 on bh-lb-19 (2 x 4 p150b, FABRIC_2D, 2 links, links train 400G). Host reference ep4_ref.py,
speed-of-light ep4_targets.py.

### Stage 0: tree up, baselines (bh-lb-19)
- bh-lb-19 reflashed (user's procedure) with stock 19.12.0 patched to tensix disable-count 1: 13 Tensix columns, a
  12 x 10 compute grid with FABRIC_2D, the Galaxy chip's. Links still train 400G on every in-box link. A software-only
  unharvest does not work: the firmware programs the NoC translation tables for its harvest mask (x 7 and 10 moved to
  the end), so a host-side mask override disagrees with the hardware on every column from x 7 on.
- sparse_dispatch_2d (tests/.../sparse_dispatch_2d/test_sparse_dispatch_2d.py, 2x4 FABRIC_2D): W3584 and W6144 PASS
  (bit-exact against the torch model, program-cache hit on new buffers).
- test_ws_fab.py step 1 (test_ws_fab_recv): G 1 M 586 80.8%, G 4 M 586 79.6% (163 GB/s of weights in), G 4 M 2048 82.4%,
  PCC 0.99994 (the stage 5 numbers above: 80.4 / 76.9 / 78.7%).
- test_ws_lane5 tray (2 x 4 snake, 2 lanes, G 2, T 8192, 5 + 3 launches, AICLK 1350): 32 experts M 586 54.4% median,
  PCC 0.99988. 640 experts (320 a lane) hung in its first launch: every core polling, no progress. 160 / 320 experts
  passed, 544 hung -> cause: l5_compute stamps expert n's end at word 256 + n of a 512-word stamps tensor, so from 256
  experts a lane it wrote past the tensor (and from 214 over the NCRISC's own stamps at 470+) into the next L1
  tensor. Fix: the expert stamps at 512 + n, the tensor sized for the lane's experts (chip_time(x0=512) for lane5).
  After it: 544 experts 53.6% PCC 0.99988; 640 experts 53.6% (53.3-54.1), PCC 0.99987, 39.0 ms a launch.
- test_ws_layer_chip e640t8192 (one chip, proxy lane, NoC split): PASS; device ms shared 1.40, stream 17.73, end 2.23;
  wall total 28.24 ms (eager, 3 + 3).

### Stage 1: host reference and readiness model (ep4_ref.py, tests/test_ep4_ref.py; no device)
- Routings: uniform (top-12 of N(0, 1) logits) and zipf (logit bias -0.25 log(rank) over a random expert order;
  expert-load CV 0.98, the target ~1; the device head gets the same bias as x column 0 = 1, W_r row 0 = bias).
- Layout per chip (route words, grp, lists, runs, O rows, ready, mask, cnt) vectorised: routing + layout + invariant
  checks of 2 chips at 302K-eq (n 9449) in 3-5 s.
- Readiness F(x) (share of a chip's runs whose last local expert is done at stream progress x; groups side by side,
  x = an expert's index in its group's order / the group's length), 302K-eq; hidden share and tail from a fluid model
  (runs ready at F(x), the combine drains at most S / C of the partials per unit x; with F = x^3 it reproduces the
  spec's 70% / 0.85 ms and 92% / 0.22 ms):
| mesh, routing, chip | experts a run | F(.25 / .5 / .75 / .9) | x^3 | hidden, tail (@50; @40) |
|---|---|---|---|---|
| LB uniform (0,0), lane order G 2 | 3.10 | .051 .173 .441 .729 | .016 .125 .422 .729 | 72% 0.79 ms; 71% 1.02 ms |
| LB zipf (0,0) (light column), lane G 2 | 2.86 | .058 .206 .523 .768 | | 76% 0.67; 75% 0.87 |
| LB zipf (0,2) (busiest column), lane G 2 | 3.26 | .035 .154 .419 .764 | | 71% 0.82; 70% 1.04 |
| LB zipf (0,2), own experts first | | .055 .168 .388 .649 | | 66% 0.96; 65% 1.23 |
| GLX uniform (0,0), lane order G 4 | 3.10 | .053 .176 .437 .727 | | 88% 0.33 ms (x^3: 92%, 0.22) |
| GLX zipf (0,2) (busiest), lane G 4 | 3.27 | .033 .155 .384 .642 | | 79% 0.58 |
  - F follows the x^c model (each run's experts at independent random positions) to +-0.01 for uniform routing in every
    order: the order does not matter for uniform routing. F is above x^3 early (runs of 1-2 experts) and equal at
    x = 0.9.
  - Own-experts-first loses 4-6 points of hidden share with zipf on the LoudBox; the lane's own order (own and remote
    interleaved by the schedule) and the G 4 split (own groups beside lane groups) are within a point of each other.
  - Galaxy: 88% hidden (uniform) against the x^3 model's 92%; the busiest zipf chip 79% (0.58 ms tail).
  - The two-partials knob: a 0.8-1.0 ms tail on the LoudBox, 0.3-0.6 ms on the Galaxy. Decide in stage 7.
- Capacity (10 seeds; LB n 9449, Galaxy n 4096): busiest chip's O rows / 12 n: uniform max 1.006 (LB) / 1.012 (GLX),
  zipf mean 1.075-1.078, max 1.138-1.140. Partials a (source, home) pair: max 0.975 n (uniform) / 0.988 n (zipf)
  against the mean p = 0.968 n. Sizing: O at 1.25 x 12 n covers both routings with margin; past it, assert / fall back.

### Stage 2: head + group exchange (ws_router EP-4 mode, ep4_layer.head_program / exchange; tests/test_ep4_layer.py)
- ws_router with WSR_EP4 (= experts a chip column): per token a 128-B topk row (gate weights fp32 bits in words 0-15,
  expert ids 16 + k, the route word 28), written as 4-KB pages of 32 tokens; per row tile its 32 route words into the
  route tensor (sparse_dispatch_2d's format: bit r C + c' for each chip of the row holding >= 1 expert, own chip
  excluded, order 0, link t & 1 << 17; tokens >= T 0); no per-expert exchange (one program a chip: RT r C, c).
- grp: stock ttnn.all_gather of the topk pages along X (cluster_axis 1).
- 302K-eq (n 9449, 2 x 4), both routings: route words and grp bit-exact against ep4_ref from the device's own top-K;
  top-K equal to torch's on 99.5% (uniform) / 98.9% (zipf) of the tokens, the rest near ties (worst miss 0.13% / 0.26%
  below torch's 12th score); gate weights PCC 0.9999997 / 0.9999986 (the head's exp, as the WS layer's).
- Traced (20 + 5, AICLK 1350): head 0.985 ms, exchange 0.088 ms, head + exchange 1.05 ms. With 128-B pages (a token a
  page) the all-gather took 0.92 ms; 4-KB pages 0.088. (WS head at T 31250: 2.70 ms on the Galaxy.)

### Stage 3: dispatch on X (sparse_dispatch_2d, masks inside the mesh row)
- sparse_dispatch_2d as it was takes N S < 2^16 slots and sizes its manifest rings in L1 for every row of a call (N S
  = 75.8K at 302K-eq; 303K on the Galaxy). Two changes to the op: slots are 21 bits (word 0's free bits 25-29 hold
  slot bits 16-20; N S < 2^21), and a row_offset argument (xbuf row of slot 0, in the program hash). EP-4 dispatches in
  KP pieces of SP tokens a chip (EP4Cfg.SP, default 1184): chip ch's token t at xbuf row (t / SP) N SP + ch SP + t %
  SP, piece p one call at row offset p N SP with the [1, 1, 1, SP] slice of the route words. The MoE block's own
  dispatch tests still pass (2x4 W3584 / W6144).
- Bit-exact at every destination for both routings (2.91 rows sent a token); pass-through chips keep some home rows
  (allowed, as the op's own test).
- Traced: 4.26 / 4.21 / 4.19 ms for 8 / 4 / 2 pieces (uniform 302K-eq) against 2.11 ms speed of light at 50 GB/s and
  2.64 at 40: 1.6x of the @40 column. Piece count does not matter. The gap is the fabric connection: one connection a
  (direction, link), 1 row (7.7 KB) a packet (2 rows need a 15.4 KB payload; Blackhole's maximum is 15232), and the
  busiest link pair's 210 MB in 4.2 ms is 25 GB/s a connection - the per-connection ceiling measured on the LoudBox
  (bench_unicast ~0.3 us a packet + bytes at 50 GB/s). On the Galaxy the 25 GB/s link is the same rate, so the LoudBox
  line is no faster than the Galaxy ring here: X phases on the LoudBox run at the Galaxy's link speed.

### Stage 4: receive-side lists (kernels e4_lists_rd.cpp / e4_lists.cpp, ep4_layer.build_lists)
- One program a chip, every core: BRISC reads the core's range of grp rows; NCRISC counts each token's local pairs,
  an exclusive prefix over the cores (core 0) gives each token's first O row (a token's O rows contiguous, k order),
  writes every token's 128-B run record (base, count, last local expert in stream order, gate weights, local experts),
  and sends 2-word records (token | expert, O row) plus one ready record a token to the expert's owner core (2
  experts an owner, 80 owners; the ws_head exchange with wider records: group token indices reach 125K at 1M-eq).
  Owners write each expert's token list (xbuf rows), O rows and ready list as whole pages. Also the expected partial
  counts a home chip gets from each row chip (atomic adds at core 0), and the O rows used.
- Exact against ep4_ref.chip_layout on every chip, both routings (counts, lists, O rows, run records, ready lists,
  expected counts); head -> exchange -> lists bit-identical over two runs. O rows used / 12 n: uniform 0.993-1.009,
  zipf 0.908-1.070.
- Traced: 0.59 ms uniform, 1.07 ms zipf (owners serial over their records; the hot experts' owners carry more).

### Stage 5: experts with weights on Y (the 5 x 5 lane on the Y columns; ep4_layer.lane_cfgs / lane_program)
- The lane on 2-chip column cycles (LCfg.subcycles, one mux a group: WL5_ONEMUX), EPC = 160 local experts, xbuf rows
  (the lists' token lists), O rows from the lists through the row collectors (yc 3, yc_n0 1), G 2 (2 links = 2 lanes).
- 302K-eq, 2 x 4: O PCC 0.99988 on every chip (6 sampled experts a chip), both routings.
- Weights only (zero counts): 5.25 ms for 40 experts a direction a lane = 31.5 GB/s a link direction - the lane's
  own ceiling (fabric ~40 a link direction; earlier tray lane runs 26-30), not the link.
- Expert phase (end to end, max over chips): 10.95-11.02 ms uniform, 11.66 ms zipf (busiest chip 1.07 x the rows),
  against 3.40 ms speed of light (DRAM) / 4.15 (stream @40). G 2 is compute bound: 1.67 TFLOP a chip on 50 cores at
  the lane's ~54% utilization. G 4 (100 cores; the 12 x 10 grid holds it) needs the own experts forwarded without
  being computed: groups 0-1 own the 2 links and compute the partner's 80, groups 2-3 compute the own 80 from DRAM, so
  the links' groups must send own experts they do not compute. Not built yet.

### Stage 6: pre-sum, combine, home pass, in sequence (kernels e4_comb_*.cpp, e4_home_rd.cpp; test_ep4_layer)
- Pre-sum: every core but the X ports an adder over a range of the group's tokens (BRISC reads run records and O rows
  on NOC0; compute sums w_j row_j in fp32 DEST, j order, packs bf16; NCRISC writes the own column's partial to local P
  (posted) or into its slot at the port of (direction of home, link g & 1), a flag after it). Ports
  (resources.port_cores, one fabric connection a (X direction, link)) send each partial into the home's P page t 5 + c;
  per home one atomic increment by the rows sent, flushed (the last router waits for every write it delivered to be
  acked), into the gate core's counter. Shared expert (ws_shared, S 5, KSH 4) into P slot 4. Home pass: core 0 waits
  for every other chip's counter to reach the expected count (from the lists), clears them, releases the cores; per
  token the present slots (mask from the topk row; absent slots zeroed in L1, never read) + the shared row, summed in
  slot order (we_compute S 5).
- y min PCC 0.99988 on every chip (32 sampled tokens a chip against torch on the device's routing), both routings; two
  layers bit-identical.
- Per-phase device ms, 302K-eq uniform (max over chips; exchange / dispatch traced; others the programs' stamps):
| phase | ms | LoudBox SOL (@50 / @40) |
|---|---|---|
| head | 0.91 | |
| exchange | 0.09 | |
| dispatch (8 pieces) | 4.24 | 2.11 / 2.64 |
| lists | 0.51 (zipf 0.99) | |
| experts (G 2) | 11.0 (zipf 11.7) | 3.40 / 4.15 |
| shared | 1.60 | 1.05 (math) |
| pre-sum + combine | 7.03 | max(1.84, 2.81 / 3.51) |
| home pass | 1.04 | 0.83 |
| sum | 26.5 | 9.2 / 11.1 |
- Combine steps: a flushed fused write + inc a packet 10.07 ms; plain writes + one flushed inc a (port, home) 8.20;
  + PIPE 3 in flight with credits retired in batches 8.06; + route-preset headers and the connection's slot API (the
  mesh API flushes before every packet and sets the route each time) 7.03. End chip ports: 13.7K rows each in 6.6 ms
  (15.9 GB/s a connection), router-space wait 3.2 ms of it; adders mostly wait for port credits (1 slot a port).
  Floor on this line: the end chips' connections carry 105 MB each, 4.2 ms at 25 GB/s (7.7 KB packets: ~18 GB/s by
  the bench_unicast fit) - the LoudBox X phases run at the per-connection rate, not 50 GB/s.

### Stage 8: the Galaxy (8 x 4 FABRIC_2D_TORUS_XY, bh-glx-120-b08u02, 130 W, AICLK 1350; tree ws_ep4)
- Changes for 32 chips:
  - route words: two a token past 16 chips (sparse_dispatch_2d: mask, order | link << 1). ws_router WSR_EP4_RW (the
    topk row's word 28 the mask, 29 the flags; route pieces [1, 1, 1, RW SP]); ep4_ref.route_words(words=2).
  - combine ports: the port of (home, link) is the fabric's first hop toward home (MoEBlockResources.first_hops(), a
    new binding of FirstHops), not the line's "home right of me: X+"; comb_ports' mesh plan takes the resources' wrap.
  - xbuf built a dispatch piece at a time and joined on device; the dispatch check reads a chip at a time (all 32
    chips' xbuf at once is 75 GB of host memory at 302K).
  - tests: EP4_MESH=glx runs every test on (8, 4) with the torus fabric.
  - The lane as on the LoudBox: G 2 on the Y columns (column_cycles, subcycles; 8-chip cycles, 2 muxes a group):
    a bidirectional lane on each of the 2 links uses all 4 Y link-planes.
- 302K (n 9449), uniform, every stage PASS (cmds/ep4_g1.sh; WS_WARM 3, WS_MEAS 5):
  - head + exchange 1.10 ms traced; route words and grp bit-exact.
  - dispatch 2.84 ms traced (8 pieces of 1184; 2.91 rows sent a token) against 2.13 ms speed of light on the X ring.
  - lists 0.62 ms traced, bit-identical over two runs.
  - experts (G 2) 11.6-12.3 ms per chip, O min PCC 0.99988 on all 32 chips; rows a chip / 12 n 0.993-1.009. Weights
    only: 6.9 ms for 140 remote experts (84 GB/s into a chip, 21 GB/s a link direction): G 2 is compute bound, as on
    the LoudBox.
  - layer in sequence: y min PCC 0.99987-0.99988 on every chip, two layers bit-identical. Per phase (ms, max over
    chips): head 0.92, exchange 0.12, dispatch 2.82, lists 0.52, experts 12.29, shared 1.61, combine 5.04, home 1.13;
    sum 24.45 (WS on this Galaxy at 302K: 22.10; EP-4 speed of light in sequence 11.6).
- Combine: the fabric splits ring ties by the source's position on the axis (#63501: even E/N, odd W/S), so each X
  link carries 2 p n partials (balanced: 70 MB a link at 302K, 2.8 ms at 25 GB/s). The limit is one port's fabric
  connection: on the side that carries its own opposite-chip partials a port injects 2 p n / 2 rows (chip 0: 9175 /
  9161 on X-, 4582 / 4588 on X+; chip 1 the other way) at about 16 GB/s (7.7-KB packets, a per-packet cost), 4.5 ms,
  waiting on router space 1.5-2.4 ms. Larger packets (two partials a packet: scatter, payload >= 15360 B) or spreading
  the injection over all four connections are the levers, not the routes.
- Next on the critical path: G 4 (experts 12.3 -> about 6 ms; the Y lane's own experts forwarded without being
  computed, or lanes split by direction), then the combine's per-connection rate (5.0 -> about 3.5 ms), then dispatch.

### Galaxy prep: one-way lanes (LCfg.oneway; ws_lane5, l5h_reader WL5_HO_BUFSZ)
- Why: in EP-4 the weights get only the Y ring (8 chips, 2 links). Bidirectional lanes give 1 cycle x 2 links = 2
  lanes, so G 2: 50 cores, compute-bound at ~11 ms (as on the LoudBox), against the 5.8 ms stream. One-way lanes give 4:
  each (cycle, link) is two lanes, one a direction (the second on the reversed cycle), every expert 7 hops one way. A
  lane's credits go back on the other direction's connection, its partner lane's (g ^ 1): header-only channels on the
  partner's mux (25 full-size + 25 header-only channels a mux, one mux a group). Every connection signals its done
  before any termination master waits (two groups' masters count each other's clients).
- Schedule model (lock-step, N 8, 40 experts a lane, P 14): one-way 98-100% of the link floor, bidirectional (80 a
  lane, same per-link load) 86%.
- LoudBox proxy (the 8-chip snake as the ring, 2 links x 2 directions, G 4 on the 12 x 10 grid, 160 experts = 40 a
  lane = a Galaxy column's load, hmc 2, muxes on NOC1: muxes on NOC0 would send the partner group's credits across a
  rectangle on NOC0), T 8192, 3 + 3 launches:
  - M 586: PCC 0.99988 every expert, 23.5% util, 11.4 ms a launch (lanes on link 0 end at 7.9 ms, link 1 at 11.4).
  - M 0 (weights only): 8.39 / 8.25 ms a lane = 17.3 / 17.6 GB/s a link direction (35 experts, 145 MB, each link
    direction). nbuf 3: 8.37 / 8.18; 5 slots (smaller rings): 8.43 / 8.26 - no change.
  - Bidirectional twin (same snake and config, G 2, 80 experts a lane: the same 35 experts a mux): 6.93 / 7.90 ms =
    21.0 / 18.4 GB/s. One-way costs ~10%; the larger gap is this lane configuration's delivery (~17-21 GB/s a link
    direction) against the Galaxy's 25 GB/s links: 145 MB at 17.5 GB/s is 8.3 ms, not the 5.8 ms stream floor.

### Galaxy prep: where the lane's weight delivery goes (2026-11-18, bh-lb-19 12 x 10, 400G, aiclk 1350 on all 8 chips)
- L5Cfg.why (WL5_WHY): the group NCRISCs charge every idle loop iteration to what blocked the weight forwards (packet
  not arrived / downstream slot credit / mux space / slot refill), stamps 477-483. One-way G 4, M 0 (8.39 ms):
  each core's NCRISC sends for only ~0.37 ms; chips 4 and 7 wait ~5.5 ms of 8 on mux space (all four groups),
  every other chip waits 2.3-4.5 ms on downstream credits (the ring backed up behind 4 / 7). Bidirectional twin:
  mux-space waits 2.3-4.2 ms on every chip. The lane protocol (arrival, credits, slots) is not the limit; the
  mux -> router path is. Links all trained 400G (eth_speed.py), no chip throttled (aiclk 1350, ~60 W).
- Ethernet cores used (mux runtime args' eth channel -> NoC x, BH order 1,16,2,15,3,14,4,13,5,12,6,11,7,10), lanes
  0 / 1 = link 0 fwd / rev, 2 / 3 = link 1 fwd / rev, chips 0-7:
  lane 0: 14 14 14 6 11 4 3 3; lane 1: 5 4 4 4 14 13 13 12; lane 2: 13 13 13 5 12 3 4 4; lane 3: 6 3 3 3 13 14 14 11.
  Muxes at NoC (1,3) (1,8) (7,3) (7,8) on NOC1 (north, then west): most paths to the ethernet row wrap at x 0 / 16,
  and the four muxes' writes share the ethernet row's links with each other (and with the clients' writes into the
  muxes and the routers' landing writes, all on NOC1).
- Single-variable runs, one-way, M 0, same per-lane load (40 experts, 145 MB a link direction):
| variant | NCRISC end | GB/s a link direction | modelled max NoC link load (lane rates) |
|---|---|---|---|
| 4 lanes, muxes NOC1 (baseline) | 8.39 ms | 17.3 | 3-4 |
| 4 lanes, links reversed (LCfg.links_rev) | 9.16 ms | 15.8 (chip 5 now the mux-bound one) | 3-4 |
| 4 lanes, muxes NOC0 (diagnostic: enters the rectangles on NOC0) | 6.01 ms | 24.2 | 2 |
| link 0 only: 2 lanes, G 2, both muxes column 0 (test_ws_lane5_tray_oneway_link0) | 4.94 ms | 29.4 (every chip mux-bound) | 2 |
  Model: clients -> mux, mux -> ethernet core, ethernet core -> group cores, each flow routed on the 17 x 12 torus,
  max over links. 4 x 17.5 = 70 GB/s on one NoC link (~86 GB/s raw) at the bottleneck chips.
- The LoudBox snake cannot be fixed by placement: chip 3's four ethernet cores are all at x 3-6 and chip 4's all at
  x 11-14, so every lane -> rectangle assignment leaves a NoC link with 4 flows there (the other six chips reach 2
  with a per-chip assignment). The 4-lane proxy stays at ~17.5 GB/s.
- Galaxy (blackhole_galaxy.yaml): every neighbour pair's two links are channels (2k, 2k + 1), i.e. link 0 at x 1-5
  (west) and link 1 at x 12-16 (east), the same on every chip. With the default order (groups 0 / 1, muxes at x 1,
  on link 0) every chip is the bad case (model 3-4 for all 20 N / S channel-pair combinations); with links_rev
  (muxes at x 1 serve the east link over the short wrap, muxes at x 7 the west link directly) the model gives 2 for
  all of them - the link-0-only case above, which ran at 29.4 GB/s, above the Galaxy's 25 GB/s links. Galaxy EP-4
  lanes: LCfg.links_rev.
- Correction: links_rev (e4a4ed363b3) leaned on blackhole_galaxy.yaml, a generic descriptor, not a real Galaxy's
  wiring; removed. Instead L5Cfg.place (default 1, hmc builds): every chip reads the ethernet cores its lanes send on
  (the fabric's channel for (neighbour, link), BH channel -> NoC x) and assigns lanes to its group rectangles with the
  fewest mux flows on the busiest westward ethernet-row link (then fewest link crossings, then identity). A lane's core
  j writes its packets / credits to the same lane's core j on the neighbour, wherever that chip placed it (l5h_reader
  RT 7 / 8: chip id | (x | y << 8) << 8). Physical columns checked (ttexalens): bh-lb-19 has NoC column 10 harvested,
  translated 10-15 = physical 11-16; the mux columns (x 1 / 7) are unaffected.
- One-way G 4 on two 4-chip rings (test_ws_lane5_tray_oneway_rings: (0,0) (0,1) (1,1) (1,0) and (0,2) (0,3) (1,3)
  (1,2); 40 experts a lane, 124 MB a link direction), M 0:
| placement | ring 0-1-5-4 lanes end | ring 2-3-7-6 lanes end |
|---|---|---|
| fixed (lane g on rectangle g) | 6.7-6.9 ms (18 GB/s) | 6.7-6.9 ms (18 GB/s) |
| placed | 6.8 ms (chip 4: all four ethernet cores at x 11-14, every mux ~4.8 ms on mux space) | 4.7-5.1 ms (~25 GB/s, +37%) |
  M 586 placed: PCC 0.99988 every expert (10.39 ms; fixed 10.34 - compute-bound there). Snake one-way M 0 placed:
  8.37 ms (chips 3 / 4 bind, as before). EP-4 experts (2-chip columns, G 2) with placement: weights 31.6 GB/s, phase
  11.05 ms, PCC 0.99988 - unchanged.
- Limit that placement cannot lift: a chip whose four ethernet cores lie on one side of both mux columns (LoudBox
  chips 3 and 4 on the tray's cycles) sends every flow through one westward ethernet-row link on NOC1, ~70 GB/s
  for its four lanes. Going below it needs either mux cores east of / between those ethernet cores (inside the group
  columns) or the mux writes on NOC0, which would enter the rectangles on their multicast NoC (the rule from the
  multicast wedge repro). Whether a Galaxy has such chips depends on its real wiring: read it with eth_x on the
  machine.
- Open: at M 586 the rectangles at x 11-15 (groups 2 / 3) finish later than x 1-7 (snake 11.4 vs 7.9 ms, rings
  10.3 vs 7.5 fixed) - not the weights; not looked at yet.

### Stage 8b: G 4 one-way lanes on the Galaxy (ep4-lb merged; lane_cfgs G 4 = LCfg.oneway, muxes NOC1, placed)
- 302K uniform, experts (max over chips) / layer sum; y min PCC 0.99987-0.99988, two layers bit-identical:
| node | G 2 | G 4 |
|---|---|---|
| 110-a10u02 (75 W, AICLK 1350 held) | 11.93 | 12.26 / 24.38 |
| 120-b08u02 (130 W) | 12.29 / 24.45 | 12.75 |
- Weights only: G 4 7.3 ms on every group (19.7 GB/s a link direction, 35 experts each), G 2 7.0 ms. G 2 is compute
  bound (11.4 ms with or without its row writes); G 4's link 1 lanes finish 40 experts in 7.7 ms.
- Placement on this Galaxy: every chip's link 0 ethernet cores at NoC x 2 / 3, link 1 at x 14 / 15 (rows 0-3 of the
  mesh one way, 4-7 the other); placed lanes -> rectangles [2, 3, 0, 1] on all 32 chips. Fixed placement (lane g on
  rectangle g): weights only 8.4-8.9 ms, experts 13.0 ms.
- Stamps rows are per rectangle. At M > 0 rectangles 0 / 1 (muxes and row collectors in logical column 0, NoC x 1)
  end at 9.2 / 12.4 ms, 2 / 3 at 7.9; their group NCRISCs wait 3.8-5.3 ms on mux space. Rectangle 0 / 1's collectors
  write the result rows on NOC1, north up x 1, the column their muxes send up to the ethernet row:
| b08u02, G 4 | experts |
|---|---|
| baseline | 12.75 |
| rectangles 0 / 1's collectors without row writes (yc_sink_mask 3) | 8.63 |
| rectangles 2 / 3's collectors without row writes (yc_sink_mask 12) | 12.53 |
- The G 4 layout breaks the multicast rule: rectangles 2 / 3's collectors (x 7, yc_n0 split) write on NOC0, east along
  their row; O pages in banks 0-3 (x 0) are reached through rectangles 2 / 3 (x 10-14) on their multicast NoC.
- Tried (EP4_OEAST=1, off by default): every collector in column 6 (2 a group), all row writes on NOC0 into banks 4-7
  (page (d >> 2) 8 + 4 + (d & 3), the combine reads the same). Experts 12.51 ms (now rectangles 0 / 2 slow, waiting on
  credits: rectangle 0's slices cross the chip west on rows 0 / 2), combine 6.24 ms (4 banks) vs 4.93, layer 25.88.
- NoC flow model (noc_flow_model.py): every experts-phase flow (ethernet -> lane cores, lane cores -> mux -> ethernet,
  x gathers, cores -> collectors, collectors -> DRAM) routed on the physical 17 x 12 NoC (NOC0 east then south, NOC1
  north then west), per harvest mask (each chip here has one Tensix column harvested, ENABLED_TENSIX_COL: 12 different
  masks over the 32 chips; logical column c = the c-th enabled one) and DRAM endpoint view (legacy / relocated). It
  reproduces the baseline and the yc_sink runs (rectangles 0 / 1 at 0.85 of a link, 2 / 3 at ~0.5; 0.46 everywhere
  with 0 / 1's row writes gone) and the oeast run's output bottleneck (the x 9 column, 0.70), not which rectangles
  stall first.
- The muxes' NOC1 path to the ethernet row and a mux column's NOC1 row writes share that column's top links in every
  layout: channels 0 and 4 take NOC1 writes at y 1 (legacy view), and wrapping writes cross it. On 18 of the 32 chips
  the harvested column is in the west half, so logical column 6 is NoC x 10: the baseline's NOC0 row writes from it
  (yc_n0) cross rectangles 2 / 3 on every write, not only towards banks 0-3.
- The model's best all-NOC1 layout (EP4_MUXTOP: muxes on rows 0 / 1, collectors below; no NOC0 row write, no multicast
  rule break on any mask), b08u02, 302K uniform:
| layout | experts | rectangles 0 / 1 end | 2 / 3 end |
|---|---|---|---|
| baseline | 12.75 | 9.2 / 12.4 | 7.9 |
| muxtop, yc 3 | 12.37 | 9.9 | 11.5 |
| muxtop, yc 2 | 11.86 | 9.3 | 11.0 |
  layer (muxtop yc 3) 24.49 ms, y min PCC 0.99988, two layers bit-identical. Rectangles 2 / 3 lose what the NOC0 writes
  gave them; the model scores every rectangle 0.75 (no asymmetry): it ranks the shared links, it does not predict times.
- G 4 on this grid: two muxes a column (40 GB/s north) and their rectangles' row writes (~54 GB/s at 8 ms) do not fit
  the column's NOC1 links; without the row writes the phase is 8.6 ms.
