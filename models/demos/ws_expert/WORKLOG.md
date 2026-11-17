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
