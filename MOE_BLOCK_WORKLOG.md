# MoE block worklog

## 2026-10-22: the block end to end on the LoudBox 2x4 (bring-up, separate programs)

`models/demos/moe_block/moe_block.py` (MoEBlock behind the proposed API), test
`models/demos/moe_block/tests/test_moe_block.py`, runs `/localdev/mstaletovic/moe_block_runs/` (run.sh).

Built (one module per program or group of programs; weights laid out at construction):
- **Router**: gate weight split along H over TP (`[H/TP, E]` a chip, replicated over SP), fp32 partial logits
  (ttnn.linear, HiFi4) -> `ttnn.reduce_scatter` by tokens over TP -> owners' `[S/N, E]` -> sigmoid + bias add (one
  ttnn.add) -> `combine_rf/router_topk.py` (top-K by the biased score, renormalized x route_scale) writing the joint
  routing rows. Only sigmoid + bias, normalized, one group (NotImplementedError otherwise).
- **x all-gather** over TP: `ttnn.all_gather` -> `[S/SP, H]`.
- **SharedExpert**, TP-split: gate / up split by columns over I_s, down by rows (bf8, HiFi3, plain ttnn.linear + the
  activation as ttnn elementwise ops) -> partial `[S/SP, H]` -> `ttnn.reduce_scatter` over TP in the head.
- **LatentProjections** (K3): down split by columns on x_full -> `[S/SP, Hr/TP]`; up split by columns after the
  latent all-gather and `ttnn.rms_norm` -> `[S/SP, H/TP]` (no reduce-scatter).
- **Owners' rows**: `ttnn.mesh_partition` of x_full (non-latent) / of the gathered latent (K3).
- **TokenExchange**: the K3 token-split block's routed-only path (`TtK3TokenSplitMoE(routed_only=True,
  joint_routing=True)`): routing gather (fabric all-gather of the joint rows) -> combine_plan planner (32 cores) ->
  dispatch prep -> sparse 2D dispatch -> the fused flat experts + combine -> owners' sums `[S/N, Hr]`. New option
  `combine_auto_cfg`: the combine sized by `combine_auto` (close-out sizing) and the arena taking the free L1 less the
  flat program's static CBs, the dispatch manifest and its CBs.
- **RoutedExperts**: the flat op with the layer's weights, close-out defaults (pure DRISC, even splits,
  `rows_hint = chunk x K / E`, K group 2) plus `x_helpers=1` (a second helper's ~104 KB a core is what the combine
  needs at the LoudBox's per-chip load).
- **Tail**: non-latent: all-gather the owners' sums over TP + `mesh_partition` of this chip's columns; K3: latent
  all-gather, norm, up. Then `+ shared`.

Substitutes (bring-up): the two all-to-alls over TP (latent to the owners, the non-latent tail back to the width
split) are an all-gather + `mesh_partition` (4x the bytes of an all-to-all); the routing exchange is the gather path.

Correctness (random weights, routed experts from a pool of 4; fp32 torch reference written to the MoEWeights contract;
bar PCC > 0.98 vs the reference run on the device's routing - the K3 block tests' bar, bfp4 routed weights against
unquantized fp32 weights):

| model (2x4, chunk 5120, 640 owned tokens a chip) | PCC (device routing) | PCC (host routing) | routing picks equal | gate weights PCC |
|---|---|---|---|---|
| GLM-5.3-Flash (E 288, 36 a chip) | 0.98704 | 0.98664 | 99.17% | 0.99571 |
| K3 dims, E 448 (56 a chip) | 0.99637 | 0.99632 | 98.75% | 0.99167 |

Per-stage device time (us, slowest chip, median of 5; profiler, stages = the programs between marks):

| stage | GLM-5.3-Flash | K3 (E 448) |
|---|---|---|
| router matmul | 62 | 115 |
| logits reduce-scatter | 71 | 108 |
| top-k (add + router_topk) | 57 | 100 |
| x all-gather | 354 | 608 |
| shared expert (matmuls + act) | 620 (5 programs) | 1838 (12) |
| shared reduce-scatter | 420 | 729 |
| latent down / latent a2a (AG + partition) | - | 412 / 340 |
| owners' rows (partition) | 28 | - |
| routing gather | 44 | 47 |
| planner | 73 | 109 |
| dispatch prep | 34 | 37 |
| arena zero | 5 | 5 |
| dispatch | 455 | 412 |
| experts + combine | 2070 | 2856 |
| tilize | 27 | 24 |
| tail: a2a (AG + partition) / latent AG + up | 382 | 317 + 377 |
| add | 38 | 64 |
| **sum** | **4739** | **8496** |
| wall, one forward (no stage syncs) | 4978 | 8623 |

Problems found, largest first:
1. The TP collectives cost 1.2 ms (GLM-Flash) / 2.4 ms (K3) of the block: ttnn all_gather / reduce_scatter run at
   ~45 GB/s a chip on the 4-chip line (x all-gather 354 us for 15.7 MB in), and the two all-to-all substitutes move
   4x the bytes. Real all-to-alls and faster collectives (the bring-up fabric ops) are the first lever.
2. The shared expert on plain ttnn ops: 620 us (GLM-Flash, ~54 us of LoFi math) / 1838 us (K3: SiTU as 7 elementwise
   ops, 12 programs). The lin2d / flat shared-expert builders exist (K3 block) - not wired yet.
3. The exchange is rebuilt on the second call (~10-17 s): L1 the first call's tail allocates moves the arena (its
   address is baked into the kernels). Needs a warm-up of the head and tail before the exchange is built. The arena's
   reserve (manifest + dispatch CBs) is an estimate (2 x 2 x group x row bytes + 8 KB).
Also: GLM-Flash's PCC (0.987) is lower than K3's (0.996) - not localized yet; Zipf routing not run (the routing comes
from the router; skew needs biased router weights).

## 2026-10-22 (pm): wiring in the tuned pieces

Same test and bar (models/demos/moe_block/tests/test_moe_block.py, chunk 5120, 640 owned tokens a chip, PCC > 0.98 vs the
fp32 reference on the device's routing); per-stage device time, slowest chip, median of 5; runs `moe_block_runs/s*_*.log`.

### Step 1: the exchange built once (arena address fixed), the arena reserve from the real footprint
- `MoEBlock._warm_up()` runs the head and the tail once on zeros (the exchange skipped) in the constructor, then
  `TokenExchange.build()`: every op's one-time L1 (collective semaphores and the like) lands before the arena is sized,
  so the second call no longer moves it (exchange builds: 2 -> 1; the 10-17 s rebuild is gone). The arena check stays
  (rebuild with a warning) for L1 allocated by anything else later.
- The arena's reserve (TtK3TokenSplitMoE, combine_auto_cfg): was flat static CBs + manifest + an estimate of the
  dispatch's CBs (2 x 2 x group x row bytes + 8 KB). Now the writers' words + max(the fused op's static CBs, the
  dispatch's real CBs (`sparse_all_gather_2d.static_cb_bytes`, the descriptor's sizes) + its manifest) + 1 KB: the two
  programs never run together. GLM-Flash and K3 frees ~37 / ~118 KB a core for the arena.
- Device times unchanged (GLM-Flash 4739 -> 4738 us, K3 8496 -> 8493; experts + combine 2070 -> 2083 / 2856 -> 2855,
  the extra arena not used by the combine's sizing at this load); PCC unchanged (0.98704 / 0.99637).

### Step 2: the router
The block's router already was the tuned path (H-split logits, reduce-scatter to the owners, one ttnn.add with the
sigmoid as its input activation, combine_rf/router_topk writing the joint rows; the GLM-Flash bring-up's "9 + 48 us").
Changed: the reduce-scatter's output and the biased scores in L1 (the top-k's 64 B reads at L1 latency, as the K3 block's
K3_L1=router): top-k GLM-Flash 57.0 -> 45.4 us, K3 100.0 -> 81.2 us; reduce-scatter unchanged (70 -> 75 / 111 -> 108,
noise). Tried: the router matmul writing L1: 62 -> 120 us (kept DRAM). Totals GLM-Flash 4738 -> 4782 (run-to-run noise
in the experts + combine, 2083 -> 2111), K3 8493 -> 8455. PCC unchanged.

### Step 3: TP collectives
Candidates measured per collective at the block's shapes (`models/demos/moe_block/tests/test_tp_collectives.py`, runs
`moe_block_runs/tpc*.log`; us, slowest chip, the 4-chip TP line, 2 links):

| collective | shard / chip | ttnn | alternative |
|---|---|---|---|
| x all-gather, width (dim 3) | 5.2 / 9.2 MB | 354 / 601 | none: high_bw / bring-up fabric_all_gather gather dim 0 / -2 only (the C++ ops also refused the preallocated output: "input and output tensors must be on the same mesh device") |
| tail all-gather, rows (K3 latent, dim 2) | 5.2 / 4.6 MB | 354 / 313 | combine_rf fabric_all_gather 338 / 296 |
| shared reduce-scatter, width (dim 3) | 21 / 37 MB | 417 / 713 | (num_links=2: same) |
| shared reduce-scatter, rows (dim 2) | 21 / 37 MB | 418 / 714 | fabric_reduce_scatter 347 / 594 |
| logits reduce-scatter, rows, fp32 | 1.5 / 2.3 MB | 70 / 97 | fabric_reduce_scatter 61 / 86, but max err 0.06 / 0.08 (rounds fp32): kept ttnn |
| K3 latent to the owners (all-to-all) | 4.6 MB | AG + partition 337 | **all_to_all_async_generic 115**, exact |
| GLM tail back to the width split (all-to-all) | 5.2 MB | AG + partition 381 | **all_to_all_async_generic 131**, exact |

(all_to_all_async_generic works on the Linear TP axis; its `in_dim` is the dimension it CONCATENATES along and `out_dim`
the one it splits, the reverse of what the docstring's names suggest.)

Wired in: both all-to-alls; the K3 tail's latent all-gather on combine_rf's fabric all-gather; for the non-latent
models the shared expert reduce-scattered BY ROWS to the owners (fabric_reduce_scatter) and added to the routed sums
in the owner layout, so the tail's all-to-all carries both (the add stays one [S/N, H] add). K3 keeps the width
reduce-scatter (its latent up lands the width split; rows + an all-to-all would cost ~594 + ~230). x all-gather stays
ttnn.

| stage, us | GLM-Flash before -> after | K3 before -> after |
|---|---|---|
| shared reduce-scatter | 419 -> 348 (rows, fabric) | 728 -> 728 |
| latent all-to-all | - | 341 -> 119 |
| tail | 383 (AG + partition) -> 130 (all-to-all) | 316 -> 301 (fabric AG) |
| **sum** | **4782 -> 4424** | **8455 -> 8204** |
PCC unchanged (0.98704 / 0.99637).

### Step 4: the shared expert
`SharedExpert`: mm2d program configs (glm53_flash_d_p mm_configs.linear_config) on every matmul; SiLU as ONE gate / up
matmul ([up slice | gate slice] columns a chip) + ttnn.swiglu (first half x silu(second half)); SiTU as two matmuls +
ttnn.situ_glu (was 7 elementwise ops). lin2d was not used: it takes M <= 32 x grid rows = 320 rows a call, and the
TP-split shared expert here runs on all 2560 rows of a mesh row (8 calls, the weight slice re-read each time); the mm2d
path reaches ~180 TFLOP/s at HiFi2 on K3 (169 GFLOP a chip), lin2d's K3 calls ~208 at M 320.
Fidelity (test_shared_expert: the shared expert alone, chunk 5120, bf8 weights, vs fp32 on the host; us, slowest chip):

| | LoFi | HiFi2 | HiFi3 |
|---|---|---|---|
| GLM-Flash | 282 us, PCC 0.99973, rel L2 5.65% | 292, 0.99992, 1.64% | 324, 0.99992, 1.29% |
| K3 (E 448) | 840, 0.99959, 5.24% | 947, 0.99990, 1.58% | 1188, 0.99991, 1.31% |

Default `MoEDtypes.shared_fidelity` HiFi3 -> **HiFi2** (+0.3 points rel L2, PCC unchanged; LoFi's 5-6% is not usable).
Block: shared expert GLM-Flash 617 -> 294 us, K3 1828 -> 949 us; sums **GLM-Flash 4424 -> 4088 us, K3 8204 -> 7353 us**;
block PCC 0.98696 / 0.99633 (was 0.98704 / 0.99637).

### Where it stands (eager device sums; step 5, the trace, skipped by request)
GLM-5.3-Flash 4739 -> **4088 us**, K3 (E 448) 8496 -> **7353 us** at chunk 5120 (640 owned tokens a chip).
- vs GLM-Flash token-split 2811 us (5120, folded_sp shared expert = 2x the shared weights a chip, eager): +1277 us. The
  layout's own TP work is ~920 of it (x all-gather 354, shared reduce-scatter 347, tail all-to-all 129, logits
  reduce-scatter 68, own rows 27 - none of which token-split runs); experts + combine 2089 vs 1784 there.
- K3's references (3315-3401 us local shared expert, 4402 TP-split, traced) were at 320 own tokens a chip (chunk 2560),
  half this test's tokens: not comparable as measured here.
- Top remaining costs: experts + combine (GLM 2089 / K3 2859), the x all-gather over TP (354 / 609: ttnn only, no
  width-gather alternative), K3's shared expert (949) and its width reduce-scatter (731), the dispatch (452 / 414),
  K3's latent down / up (409 / 377: ttnn mm, lin2d's 93 / 89 us were at 320 rows).

## 2026-10-22 (pm): the fused head - one custom program per chip (in progress)

User decisions: one fully custom fused head program (no descriptor merging of existing ops, no sub-devices); fabric
UNICAST only - all-gather-type streams store-and-forward through the head's own fabric workers (a multicast path only
as an off-by-default probe); HiFi2 for the shared expert.

### The bar for x all-gather -> shared expert gate / up
ttnn.all_gather + the block's mm2d linear (HiFi2, bf8 weights): **GLM-Flash 468 us, K3 1021 us** (test_agmm_bar).
The fused gather-over-K matmul ops (all_gather_minimal_matmul_async; the strided / all_gather_matmul_async variants share
its kernels) do not build on FABRIC_2D: their kernels use the 1D LowLatencyPacketHeader helpers
(matmul_dataflow_common.hpp: "cannot convert HybridMeshPacketHeader* to LowLatencyPacketHeader*"), and the block needs
FABRIC_2D (dispatch, combine). So the bar is the separate path.

### Building block 1: the x all-gather over K by the head's own fabric workers (models/demos/moe_block/tt/tp_gather.py)
A port per (direction, link) (combine_rf.fabric_all_gather_py.plan's placement), store-and-forward one hop at a time:
its own columns first, then each shard the upstream port landed (read back from x_full once the arrival counter covers
it). A packet is a run of tiles consecutive in one DRAM bank in x AND in x_full (row tile r, bank b, tiles b, b + 8, ..;
needs H / TP / 32 a multiple of the 8 banks), so the width concat costs nothing. Fused write + increment every 8th
packet; a fence per call (no writes into a neighbour before it started the call).
| x all-gather, us (exact vs ttnn) | GLM-Flash [2560, 1024] | K3 [2560, 1792] |
|---|---|---|
| ttnn.all_gather | 355 | 602 |
| tp_gather, store-and-forward | **336** | **585** |
| (probe, not kept: line multicast, 2 links / 1 link) | 341 / 670 | 599 / - |
**The TP links bound it**: one link halves the rate (670 vs 341 us). The links run at their ceiling, ~23.6-24 GB/s a
link a direction at 200G (link-rate probe, 24abecc2913): on the 4-chip line the END link carries G - 1 = 3 shards
(the end chip has one neighbour), 3 x 5.24 MB / (2 x 23.6 GB/s) = 333 us for GLM-Flash vs 336 measured (K3: ~585).
(An earlier note here said "15.6 GB/s, 2 shards" - wrong bookkeeping, same floor.) The x all-gather is at the line's
floor - only hidden under compute; same for the shared expert's reduce-scatter. On a Galaxy TP ring of 4 the busiest
link carries 1.5 shards (half the line's). So the fused head's TP links carry ~730 us (GLM-Flash) / ~1350 us (K3) on
the LoudBox line, against today's 1352 / 3309 us serial heads.

### Building block 2: the chunked matmul engine (models/demos/moe_block/tt/lin2c.py)
lin2d (x resident per core row, W streamed down the columns, 2D multicasts) chunked over M: chunk m = 32 GY rows, x
gated per chunk by XREADY (a producer's increments), XFREE back-pressure, a column barrier a chunk, a y notify of the
consumer cores per chunk, GLU (silu / SiTU) on the packer's SFPU. W is re-read every chunk (8 chunks of 320 rows at
chunk 5120) - acceptable: the head is TP-link-bound (~730 / ~1350 us), so the shared expert only has to fit under that.
Bug on the way: the W multicast destination took CB1's write pointer at send time (it moves with the pushes) instead of
the ring base captured at kernel start: wrong slots once the steps a chunk exceed the ring (gate / up PCC 0.2-0.6, down
fine with NS = NB).
| test_lin2c, one engine, all 12 x 10 cores (us, PCC) | GLM-Flash | K3 (E 448) |
|---|---|---|
| gate / up + GLU [2560, H] x [H, 2 I/TP] | 232 (93 TFLOP/s), 0.99993 | 808 (140), 0.99988 |
| down [2560, I/TP] x [I/TP, H] | 142, 0.99978 | 383, 0.99976 |
| **chain in one program** (gate / up -> h DRAM -> down, per-chunk notifies; test_lin2c_chain) | | |
| 8 + 4 columns | **294**, 0.99974 | **1040**, 0.99962 |
| 8 + 3 (a column left for the head's movers) | 309 | 1138 |
| 9 + 3 / 7 + 4 / 7 + 5 | 310 / 323 / 320 | 1144 / 1176 / 1222 |
vs the block's mm2d shared expert (HiFi2): 292 / 947 us - the chain is level on GLM-Flash, +10-20% on K3, and it can
now start on chunk 0's x and stream h -> down while later chunks are still arriving.

tp_gather made ring-capable (plan topology from the fabric config, balanced ring schedule: the distance-TP/2 shard split
into even / odd row tiles over the two directions, relay waits per part); line regression exact, 336 / 585 us.

### Step 3: the x chain in one program (tt/fused_head.py HeadXChain, v1: x_full in DRAM, W per chunk)
tp_gather became TpGather: chunk-major streams (per chunk: own shard, then the relays; increments counted per chunk),
each stream's chunk end sends one fabric atomic increment to the downstream chip's watcher (behind the chunk's packets
on that link), the copy cores count their chunk ends too, and a watcher kernel (first copy core's NCRISC) raises
XREADY on the gate / up engine's x senders once a chunk of x_full is complete. Chunked gather alone: exact, 338 / 586 us
(no cost for chunking). Movers in the last column (simple placement: one column cannot host the Ethernet probe).
| x chain, us (PCC vs fp32 host) | GLM-Flash | K3 |
|---|---|---|
| serial: tp_gather + chain on 8 + 3 columns | 337 + 309 = 646 | 586 + 1138 = 1724 |
| **fused, one program** (gather -> gate / up + GLU -> h DRAM -> down) | **595-603**, 0.99972 | **1543-1567**, 0.99963 |
| probe: movers alone in one column | 414 (337 with Ethernet placement) | 724 (586) |
| probe: engines alone | 307 | 1140 |
| probe: both, engines not waiting for the gather | 580 | 1426 |
| movers at the west edge instead | same | same |
| movers 8 packets in flight (was 4), copy core pipelined | 551 | 1380 |
| + W prefetch depth 2 / 1 (engines alone 304 / 333, K3 1193 / 1324) | 525 / 466 | 1358 / 1480 |
**Interference, not gating**: per-core spans (K3, ungated) - gate / up ends ~1040 us, down ~1220, the movers 1160-1412
(724 alone), even the copy cores (local DRAM -> DRAM) ~1285. The W stream readers re-read W every chunk at full DRAM
rate (8 x: ~53 MB GLM-Flash / ~280 MB K3 a call) and starve the movers' DRAM traffic; throttling W confirms it (GLM
466 us at depth 1). Matches the fabric_backpressure finding (store-and-forward relays reading DRAM slow 4-9x on a
DRAM-loaded chip).
Also: group 16 hung - the sender takes group + 1 packet headers and Blackhole's pool has 12 a RISC (the pool spins
when exhausted); asserted. Two port cores on one link (workers_per_link 2) hung: FABRIC_2D gives one worker a link
direction; removed. lin2c MT = 2 (two row tiles a core a chunk) hangs (pack / y-writer CB 4 wait from chunk 0, cause
not found); parked behind an assert - superseded below.

### Next (user): x lands in L1, the shared expert's weights read once a call
Design (being built, tt/se_engine.py):
- one engine on all compute columns, two phases on the same cores: gate / up, then down (down needs all of h anyway);
- x lands in L1 as K-blocks per source chip: core (cx, cy) owns rows R_cy (2560 / 32 / GY row tiles) x the K steps
  assigned to column cx; the copy cores write the chip's own shard there, fabric packets write the others' (runs of
  tiles contiguous in the owner's L1), relays read from that L1 - no x_full in DRAM;
- gate / up streams each W block ONCE, in the chip's arrival order (own K block first, then by distance), every core
  multiplies its rows x the block (x step multicast along the row from its owner, L1 -> L1), partials accumulated in
  fp32 L1 (packer L1 accumulation); GLU after the last block -> h, which stays in L1 exactly where down wants it
  (core (cx, cy) owns h[R_cy, column cx's slice] = down's K step);
- down: N passes (W[:, n block] once each), h re-multicast along rows from L1 a pass, y a pass at a time (the
  reduce-scatter can then pipeline by column blocks);
- DRAM a call: W GLM-Flash 6.7 MB (was ~53), K3 35 MB (was ~280); x own shard once; y.

### Step 4: the shared expert engine, x in L1, weights read once (tt/se_engine.py SeEngine)
Built as designed above (one engine on 11 x 10 cores, gate / up then down on the same cores, all buffers in one L1
arena tensor - CBs via cb_descriptor_from_sharded_tensor at offsets, the two partials CBs aliased; fp32 partials by
packer L1 accumulation, reloaded into DEST (UnpackToDestFp32) on the last step, GLU there, h packed in place).
| test_se_engine (x from DRAM into L1, 2560 rows) | GLM-Flash | K3 (E 448) |
|---|---|---|
| time, PCC | **340 us**, 0.99975 | **1086 us**, 0.99964 |
| gate / up alone, down alone | 203, 145 | 715, 382 |
| shared expert weights read from DRAM a call | **6.7 MB** (lin2c chain 53.5) | **35.1 MB** (lin2c chain 281) |
| vs lin2c chain (12 cols) / block mm2d | 294 / 292 | 1040 / 947 |
Down is pack-bound (K steps of 1-5 tiles, each an fp32 L1-accumulating pack of the whole output block): 74 / 147 TFLOP/s.

**Hang hunt (closed)**. Symptom: on a random chip, all engine cores' PACK threads in _llk_pack_dest_section_done_
(waiting on its own pack), MATH in the matmul, UNPACK in cb_wait_front, BRISCs in cb_reserve_back(W) - or wrong results
on one chip (PCC 0.94-0.98, one core row, error growing with the row sub-block). Narrowing (each one variable): L1 acc
off, reload off, GLU off, full-sync DEST, gate / up only, down only, KB 4 - all still hung; small shapes passed (MT 2),
MT 4 intermittently wrong, MT 8 hung; W column multicast -> unicasts and the x row multicasts split so each starts at
its source (east half NOC0 / west half NOC1) - still hung. Cause (codex on the evidence): the W credits were ONE
aggregate count on the reader - a fast core credits two successive slots while a slow one credits none, the reader
overwrites the slow core's occupied slot, its BRISC blocks in cb_reserve_back (and stops serving x), everything stalls
behind it (the packer included). Fix: credits per slot (WFREE[s] >= GY x round; XFREE likewise per slot on the owner),
plus a flush between each data write and its valid-semaphore write (codex's other point: separate command FIFOs).
Also kept: the W reader on the last row (the start of its column's path); x multicast halves from the owner.

### Step 5: the x chain v2 - the gather straight into the engine's L1 (tt/x_gather.py XGather + fused_head.HeadSE)
x lands as K steps in the engine owners' L1 (no x_full in DRAM): packets of up to 4 tiles within one K step, row tile R
over link (R / 2) % 2 (orthogonal to the ring's even / odd halves), relays read the owners' L1; after each shard a port
increments the downstream watcher's (direction, link) counter, and the watcher raises XREADY on all 110 engine cores per
shard in the engine's order (own shard first - loaded by the engine itself - then by distance). Gate / up streams W
once in that order. Movers on column 11 (simple placement).
| x chain (gather -> gate / up -> down), us, PCC | GLM-Flash | K3 (E 448) |
|---|---|---|
| serial: tp_gather + lin2c chain | 646 | 1724 |
| v1 fused (chunked gather via DRAM, W 8x) | 595-603 | 1543-1567 |
| **v2 fused (gather into L1, W once)** | **610**, 0.99973 | **1314**, 0.99964 |
| shared expert W from DRAM a call, v1 -> v2 | 53.5 -> 6.7 MB | 281 -> 35.1 MB |
| x_full DRAM traffic (write + relay / engine reads) v1 -> v2 | ~63 MB -> 0 | ~110 MB -> 0 |
Per chip (K3): gather ends 638-741 us (585 alone - the engine's NoC traffic), engines 1113-1314: the line's end chips
get their distance-3 shard last (~730 us), then that shard's gate / up (~180) and all of down (382) -> 1314. GLM-Flash:
gather 369-422, engines 457-610 (tail = one shard of gate / up ~50 + down 145). The floor of this structure is
gather end + one shard of gate / up + down; next: down + the reduce-scatter as one stage (matmul_reduce_scatter port).

### Parked: down with all of K in DEST (user ask), superseded by the row-chunk-major direction
Tried: h for all of I resident in L1 ("h all", each owner writes its rows into every row core once), the W_down block
[I/32 x 2] streamed once a pass, each 2 x 2 sub-block walking all of K in DEST, packing once; h-all and down's W slots
reusing gate / up's L1 behind a barrier (each core tells its row and its column's reader once its gate / up is done, its
x sends are out and the gather stopped reading its x-own). GLM-Flash: correct, 339 us (vs 340: no gain). K3: wrong
(PCC 0.993; at K3's H with I 512: 0.07), GLM at KB 4: NaN, and 0.992 with the partials moved out of the reused
region - an aliasing / ordering bug in the reuse not found before the direction changed. Not committed; the committed
engine (65eaae64c66) stays. Lesson for the next engine: no L1 reuse across phases without a proof of who still reads.

### Step 6: the x chain v3 - row-chunk-major, shared expert weights resident in L1 (user direction)
tt/se_engine3.py SeEngine3 + tt/x_gather3.py XGather3 + fused_head.HeadSE3.
- weights: each column's W blocks (gate / up [KB x pass width], down [KBD x pass width]) spread over the column's
  cores (block b on row b % GY), read from DRAM once at the call's start, streamed down the column every chunk from
  their holders (L1 -> L1 loopback multicast, per-slot credits to the slot's next holder) - hand-written: a kernel_lib
  rotation needs each family's holder count to divide its rounds a chunk, which unbalances K3's L1 beyond fit;
- x: all of it resident in its owners' L1 (the gather lands it, the engine loads its own shard); per gate / up pass
  multicast along the row by two kernel_lib rotating PerRow families (x slots; owner subsets dividing the steps a pass:
  GLM 8 + 8 columns, K3 7 + 7); h all-gathered along the row by a kernel_lib rotating family (rounds (row tile,
  column)); kernel_lib's mid-line sender is one loopback rectangle, so the east / west split was unnecessary;
- compute: all of K in DEST per pass (full-sync fp32, MT x width <= 8), GLU, pack once; no partials, no L1 reuse;
- gather: chunk-major (each chunk: own rows, then the relayed shards' rows), packets at the step's owner, a watcher
  increment a chunk per incoming stream, XREADY = m + 1 when chunk m is in.
| engine alone (x from DRAM), us, PCC | GLM-Flash | K3 |
|---|---|---|
| MT 1 (8 chunks of 320 rows), KB 8 / 4 | 436, 0.99975 | 1700 (KB 4) / 1470 (KB 8), 0.99964 |
| **MT 2 (4 chunks of 640), KB 8** | **325** | **1041** |
| (v2 engine, W once, K-shard-major) | 340 | 1086 |
| **x chain v3 fused (MT 2)** | **541**, 0.99973 | **1253**, 0.99964 |
| per chip: gather end / engine end | 449-479 / 372-543 | 835-891 / 1152-1253 |
| floors (user): gather + one chunk's compute | 341 | 643 |
Rounds matter: each x / h round is a blocking handshake (receivers ack, the sender waits all acks): MT 1 / KB 4 doubled
the rounds and cost 30-60%. Gaps to the floor: the gather runs 35-50% slower beside the engine (336 -> ~470,
585 -> ~870: GLM is gather-bound), and the engine computes at ~100 / 160 TFLOP/s (2 x 4 sub-blocks, unpack-bound).
Gather gap split (v3): standalone on the same single mover column (simple placement) 414 / 724 us vs 336 / 585 with
Ethernet placement - ~25% placement, the rest (~15-20%) the engine's NoC traffic. Next levers: ports placed by their
Ethernet cores within the mover column (a placement table instead of the probe that needs 4 cores in a row); engine
compute (2 x 4 sub-blocks at ~100-160 TFLOP/s); then down + the reduce-scatter per chunk (matmul_reduce_scatter port).
(A gather-only probe inside the fused program aborted on the host - free(): invalid next size - not chased.)

### Galaxy per-chip shapes (user steer): 640 rows a mesh row, TP ring of 4
Galaxy floors a chip (ring: busiest link 1.5 shards; the LoudBox line's end link carries 3, i.e. 2x): x all-gather
GLM ~41 us / K3 ~72 (line 84 / 146); shared expert HiFi2 compute GLM ~26 / K3 ~139; weights once 13 / 68 (overlap).
Tests now default to 640 rows a mesh row (chunk 1280 on the 2x4). v3 at 640 rows (x chain on the LoudBox line):
| | GLM-Flash | K3 |
|---|---|---|
| engine alone, MT 2 (one chunk) / MT 1 (two) | 98.7 / 126.5 us | 403 / 542 |
| fused x chain, one chunk / two | 195.8 / 193.7, PCC 0.99982 | 521.8 / 572.3, PCC 0.99973 |
Fixed: x_gather3 picked a row's link by the global row while the host counted packets per chunk-local row - chunks
not a multiple of 4 row tiles (MT 1) hung.
Engine phases (device zones, 640 rows, us from start): GLM - W load 0-30, gate / up 0-57, h rounds 55-75, down to
95; K3 - W load 0-148 (35 MB at ~237 GB/s), gate / up 0-316 (~89 TFLOP/s), h 315-334, down to 395.
Tried: (a) weights issued all at once with a transaction id a block, a block streamed as soon as it landed - kept
(GLM 97 -> 95); (b) NoC swap (weight reads + column multicasts on NOC0): K3 gate / up 316 -> 274 but down +20 us,
GLM 97 -> 113 - reverted; (c) h exchanged all at once with counters instead of 22 kernel_lib rounds: no faster
(the concurrent row multicasts serialize: 18-41 us) - reverted to the rounds. KB 16: K3 403 -> 367, GLM no change.

### Ring readiness of the x all-gather (v3)
- CPU protocol model (tests/test_head_protocol_model.py, host only): x_gather3's real per-port plan (port_plan, shared
  with the device build) and the real line / balanced-ring schedule, replayed as reader / sender / fabric channel /
  watcher / call-fence processes under a random scheduler, two calls with chips drifting apart; line and ring of 4
  and 8, 1-2 links, channel depth 1 and 4, 6 seeds each: delivery exactly once, every chunk complete at its XREADY, no
  write into a chip still in the previous call, termination with every counter re-armed, no deadlock - 14 / 14 pass.
  It catches injected bugs: no fence -> 36 / 40 seeds flag a cross-call write; the old global-row link mapping -> the
  packet-count check.
- Mock Galaxy host build (tests/test_head_mock_galaxy.py, TT_METAL_MOCK_CLUSTER_DESC_PATH=blackhole_galaxy.yaml):
  8 x 4, TP a ring of 4 (FABRIC_2D_TORUS_X), 12 x 10 grid, 640 rows a mesh row: the planner's ring schedule (own
  shard both ways, the opposite shard's even / odd halves one way each = 1.5 shards on the busiest link), every chip's
  program built and enqueued for GLM-Flash and K3 - pass.
- Device-tested: the line only (LoudBox). Ring-verified: by the model (protocol) and the host build (plans,
  descriptors); not on ring hardware.

## Toward 2x of the roofline (user target), lever by lever - the 2x4 at chunk 1280 (640 rows a mesh row)
Floors (LoudBox line: end link 3 shards over 2 links ~48 GB/s; HiFi2 ~304 TFLOP/s on 110 cores; DRAM 512 GB/s):
x chain = gather + last shard's gate / up + down: GLM-Flash 82 + 4 + 9 ~ 95 us (2x: 190), K3 143 + 23 + 46 ~ 212
(2x: 424; also >= W load 68 and compute 139). Whole head: GLM ~180 (2x 360), K3 ~330 (2x 660).

### Lever 1: gate / up overlaps the gather inside the chunk (own K steps first, then each shard as it lands)
The watcher now raises XREADY per shard (each port increments the downstream watcher after every shard it sends),
in the engine's order (x_gather3.shard_order: own, then by arrival); an x position's owner is fixed by the position
(the kernel_lib rotation still fits: rounds stay in position order), so the gather lands step j at the destination
chip's position for it (its shard-rank table) and relays read with the local one; the W stream follows the chip's
order (holders keep their blocks, a send looks up the block's offset among its holder's). The protocol model follows
(per-shard watching; 14 / 14 pass). Also fixed: GLM then K3 in one process ran out of L1 (fragmentation by the first
head's global semaphores) - the test now drops the head and collects.
| x chain, us (PCC) | GLM-Flash | K3 |
|---|---|---|
| before (v3, one chunk) | 196 | 523 |
| **lever 1** | **180** (0.99982) | **500-502** (0.99973) |
| floor / 2x target | 95 / 190 - **within 2x (1.89x)** | 212 / 424 - 2.37x |
K3 per chip: gather ends 279-307 (143 floor: slowed by the engine's traffic), engine 433-496: the engine binds.

### Lever 2: K3 engine efficiency - per-bank weights, K steps of 16 (14 in the fused head: whole steps a shard)
- weights: one contiguous W-own region per engine core in bank (core % 8) (flat_expert._bank_sharded), read in large
  bursts: GLM-Flash's reads issue in 3 us; K3's gate / up did not move (316 us) - the load was not what binds it;
- K steps of 16: K3 engine 403 -> 341 us (half the rounds); GLM-Flash neutral;
- measured where gate / up's unpacker waits (zones around its two cb waits, summed a core): K3 x median 104 us, W
  35 us of 250; the cores that wait least are the 10-tile columns (48 pairs over 11 columns): **K3's engine is compute
  bound on its heaviest columns at ~50 cycles a tile multiply (2 x 4 sub-blocks, HiFi2 - mm2d's ~180 TFLOP/s is the
  same rate)**. The 304 TFLOP/s in the floor is optimistic for this sub-block; at the practical ~190 TFLOP/s K3's
  shared expert computes in ~220 us, not 139;
- tried, not kept: the x stream hand-written with 3-4 slots and per-slot credits (owner = position % GX, no helper):
  x waits unchanged (they are the slow columns, not the handshake), GLM-Flash 93 -> 103-120; W slots 2 / 3 / 4: no
  effect beyond noise. Kept the kernel_lib x families (saved the hand-written stream in the scratchpad).
| x chain, us (PCC) | GLM-Flash | K3 |
|---|---|---|
| lever 1 | 180 | 500 |
| **lever 2** | **168** (0.99981) | **499** (0.99973) |
| floor / 2x target | 95 / 190 - 1.77x | 212 / 424 - 2.35x |
K3 fused (device 0): gate / up ends 401 - only its first pass (of 3) overlaps the gather (ends ~300 us, slowed from
the 143 floor by the engine's traffic and the one-column movers); passes 2-3 run after it, then h (20) and down (80).

### Lever 2b: gate / up shard-major (each shard through every pass as it lands, fp32 partials between shards)
Rounds per chunk ordered (shard, pass, step): the own shard's every pass first, then each remote shard's as it lands;
a pass's fp32 partial over the shards so far is packed to a small FIFO CB and unpacked into DEST (UnpackToDestFp32)
by the next shard's same pass (a dozen reloads of <= 8 tiles a call); the last shard applies the GLU and packs h. The
kernel_lib x families stay (a family's owner subset divides its rounds a (shard, pass) block); the W stream follows
the same order. h-all single-buffered when there is one chunk (K3's arena was 1.48 MB).
| x chain, us (PCC) | GLM-Flash | K3 |
|---|---|---|
| lever 2 | 168 | 499 |
| **lever 2b** | **167** (0.99981) | **449-450** (0.99973) |
| floor / 2x target | 95 / 190 - 1.76x | 212 / 424 - 2.12x |
Secondary check (chunk 5120 = 2560 rows a mesh row): GLM-Flash 524 us (was 541); K3 does not fit L1 (its x alone is
334 KB a core even spread over all 11 columns, and the shard-major families concentrate it on 4 owner columns) -
the fused head now picks the largest K step whose arena fits, and says so when none does.

### Lever 3: the K3 gather starved by the weight load - weights streamed a window ahead; h a block a column
Narrowing why K3's gather took ~310 us inside the head (182 alone on the same column; GLM-Flash unaffected), each one
variable: no x data multicast / no W data multicast / both / no matmuls / K steps of 4 (exact 4-tile packets, 7
owners a family) / the own shard sent from the engine's L1 instead of DRAM with the weights' DRAM reads after the x
load - the gather stayed at ~300-345 us every time. Per port (device 0, a line end): the ports sending its shard ran
~6 GB/s a link (23.6 is the link) and finished at ~178 us. The burst it ran into: all 110 cores issuing their whole
W-own (35 MB) at the start. Fix: **the weights' DRAM reads go in stream order a window of 3 blocks ahead of the
column's sends** (issued, tracked by transaction id, waited on at send) - spread over gate / up at the stream's rate.
Kept too: the gather's own shard from the engine's L1 (each engine core counts its x load on the ports; its W reads
start after it), and h-all as one [MT][POMAX] block a column (11 kernel_lib rounds, not 22).
| x chain, us (PCC) | GLM-Flash | K3 |
|---|---|---|
| lever 2b | 167 | 449 |
| h a block a column | 161.5 | 444 |
| **W prefetch window (+ own shard from L1)** | **163** (0.99981) | **377-380** (0.99973) |
| floor / 2x target | 95 / 190 - **1.72x** | 212 / 424 - **1.78x** |
K3 per chip: gather ends 222-253 us (was 300-334), engine 326-377. Engine alone: K3 318 us (133 TFLOP/s), GLM 99.
Both x chains are within 2x of their floors.

### Down + SE reduce-scatter: the matmul_reduce_scatter op ported (the bar)
Ported from malimpic/2026_10_15_1521_run1_matmul_reduce_scatter: ttnn/ttnn/operations/matmul_reduce_scatter (op +
design docs, no perf_experiments / agent logs), kernel_lib/perf_instrumentation.hpp (profiler zones), its acceptance
test (the codegen harness's topology helpers inlined: eval.golden_tests is not in this tree; the harness-bound perf test
dropped). Its Ethernet-port placement imports _chip_maps / _worker_nearest_noc1 from bringup/combine_rf. kernel_lib
helpers it uses (matmul_block_helpers, mcast_args, eltwise convenience) and _op_contract.py are identical to ours.
Acceptance on the LoudBox: line 10 passed (links 1 / 2, every partial exactly once, repeated calls), the 1 x 8 snake
ring under TORUS_XY 20 passed (ring_balanced + Linear, 1 / 2 links). (Line and ring must run in separate processes:
the module-scoped meshes cannot change the fabric config while one is open.)
As-is on the SE down shapes (models/demos/moe_block/tests/test_mrs_bar.py): h [640, I] bf16 @ W [I, H] bf8 per chip,
reduced over TP (the 4-chip line, 2 links), the model's fabric payload (8256 B), HiFi2:
| down + RS, us (worst PCC) | GLM-Flash (I 512, H 4096, scatter rows) | K3 (I 1536, H 7168, scatter width) |
|---|---|---|
| stock ttnn reduce_scatter alone (no matmul) | 116 | 190 |
| fabric_rs alone (bringup) | 98 | - |
| **matmul_reduce_scatter (down + RS fused)** | **118** (0.99998) | **187** (0.99998) |
| floor (end link 3 shards at 47.2 GB/s; down compute overlaps) | 83 | 146 |
| ratio | 1.42x | 1.28x |
The fused op costs what the stock reduce-scatter alone does. Whole head so far, sequential: GLM 163 + 118 = 281
(target 360), K3 377 + 187 = 564 (target 660); left for router / logits RS / top-k / routing prep / K3 latent: 79 / 96.
Stage A (the bar wired in, two programs - superseded below; kept as rs_mode "mrs" for comparison only): the engine
stops at h (down=False, h to DRAM), then the op. GLM 131 + 117 = 237 us, K3 310 + 186 = 490 (sum of the two programs'
slowest chips), PCC 0.99988. The op's zones (device 1): GLM's first send waits for entry 0's matmul with the resident
W fill (20 us), then 3 blocks at ~23.5 GB/s a link (28 us each); K3 the same with 48 us blocks. Its knobs on our
shapes: MRS_WAVES 2 / INC_PER_WAVE 0 / PORT_GROUP 8 - all within noise.

### Down + SE reduce-scatter inside the fused program (tt/se_rs.py; user: h never leaves L1)
matmul_reduce_scatter's schedule (plan_transport: block order by distance to the owner, start / relay / fin, arrival
thresholds) on the engine's own tiling - no 2D re-tiling, no h in DRAM: the row h exchange (h from the L1 gate / up
packed it to) gives each core all of K for its MT rows, the W_down stream is the engine's column stream (resident
weights, loaded during the gather).
- Tiling: K3 (scatter -1, width) - every column owns a 5-6 tile slice of every block, so every core works on every
  entry and the entries run in transport order (2 passes a block); GLM (scatter -2, rows) - a core row's 2 row tiles
  fall into one or two blocks, all passes first (GLM's down is ~10 us).
- Per core: own partials of the entries that add an arrival stay **fp32 in L1 (HOLD)**, start sub-blocks are packed
  bf16 straight into FINB (the exchanged buffer, compact in entry order); a finish (behind the next entry's passes,
  matmul_reduce_scatter's lag) adds the bf16 arrival(s) in fp32 DEST (HOLD + arrival, the second direction by DEST
  reuse) and packs bf16 into FINB (relay, then handed off) or OUTF (fin: the NCRISC writes y - the stage's only DRAM
  write).
- Ports: the gather's movers, (direction, link) - after their gather units: per unit, per core of their link, wait
  the core's hand-off on its mailbox word (an L1 word a core a port), NoC-read its FINB sub-block, one fabric packet a
  4-tile chunk into the downstream chip's same core's zone slot (written once a call: no credits), the core's last
  chunk carrying the fused increment of its arrival counter (global semaphore a direction). The receiving NCRISC
  credits zone slot i (P pages) once its counter reached the slot's threshold. The gather's call fence covers it.
- CPU protocol model (tests/test_rs_protocol_model.py): the plan's compute programs, hand-offs, ports and thresholds
  under random interleavings, symbolic sums - line / ring of 4, G 2, 1-2 links: 24 / 24, and an off-by-one threshold
  is caught.
| x chain + SE reduce-scatter, us (PCC) | GLM-Flash | K3 |
|---|---|---|
| bar: x chain v3 + matmul_reduce_scatter (2 programs, h via DRAM) | 237 (0.99988) | 490 (0.99988) |
| **one program, in-L1 down + RS** | 250.5 (0.99988) | 455.5 (0.99987) |
| + the first down pass on each column's h as it lands | **248.2** | **449.9** |
| floor (AG + last shard's gate / up + RS links) | ~170 | ~312 |
Per chip (K3): gate / up ends 220-257 us, the h exchange 14, down + RS to 435-455 (~185 us; the end link's 3 blocks
146). GLM: gate / up 96-125, h 16, RS to 212-250 (the end chips' 3 blocks 84 us from their first hand-off ~135).
The stage is link-bound once it sends; what is left above the floor is the gather (K3 ends 220-257 vs 143).

Precision (user rule: accumulate fp32, exchange bf16 unless measured): x gather bf16 (the input); h exchange bf16;
gate / up between-shard partials fp32 in L1 (not exchanged); SE reduce-scatter: own partials fp32 (HOLD), exchanged
sub-blocks bf16, adds in fp32 DEST; output bf16. No fp32 on any wire. Router logits: not built yet - bf16 vs fp32
exchange to be measured on routing-pick agreement.

### The router chain in the program, under the x chain (tt/router_stream.py)
Logits wire format measured first (host emulation of the line's chain with the block test's weights, 1280 tokens):
bf16 partials on the wire change the picks in 2.7-5.9% of the rows (max |logit error| 0.02-0.03 on logits of std
1.3-1.7) - **the logits exchange stays fp32** (the user's exception; everything else exchanged is bf16).
- Cores: the mover column's 5 free cells (ports at rows 0-3, watcher at 5) = RB = 5 router cores; core j owns row tile j
  of every block, so at its fin it holds 32 owner tokens with all E. L1 addresses are allocated in lockstep over the
  grid, so the engine's arena is also carved on the router cores and the ports (the router's buffers live there).
- Router matmul: x (own shard, DRAM) and W_r [H / TP, E] (bf16, HiFi4) streamed in K blocks, K-outer with the G row
  tiles' partials accumulated fp32 in L1 (packer L1 accumulation). GLM 76-82 us, K3 ~200 us on 5 cores.
- Logits reduce-scatter (se_rs.LogitsPlan: the SE reduce-scatter's schedule and protocol, CPU model 8 / 8 line / ring):
  a router core pushes each hand-off into its port's L1 (+1 on the port's word of it); the port's sender polls them
  between its own packets - GLM-type (logits_first): between every x-gather packet; K3: after the x gather - and
  sends them fp32 into the downstream router core's zone with a fused increment. Relays add in fp32 DEST.
- Owner: sigmoid + bias in DEST (scores), top-K by the biased score, weights = (biased - bias) / sum x route scale,
  router_topk's joint rows. The scan is scalar on 4 RISCs a core (BRISC, NCRISC, UNPACK, PACK threads: 8 tokens each).
| fused head, us (SE PCC; routing vs host fp32) | GLM-Flash | K3 |
|---|---|---|
| x chain + SE reduce-scatter | 248.2 | 449.9 |
| + router chain, top-K on one RISC a core | 405.9 | 833.8 |
| + top-K on 4 RISCs a core | **261.2** (0.99988; picks 99.53%, weights PCC 0.9955) | **474.1** (0.99987; 99.22%, 0.9912) |
Same routing quality as the block's ttnn router chain (0.9957 / 0.9916 weights PCC). The scalar scan is L1-load bound
(~48 cycles an element, ~16 us a K3 token; a face-row max prefilter changed nothing): K3's top-K ends ~400-420 us,
GLM's under the RS. K3's gather end moved 220-257 -> 252-277 us (the router's DRAM reads: W_r read by each of the 5
cores). Next: the SFPU top-K (ttnn topk's LLK) if the routing gather + planner behind it land on the critical path;
W_r multicast instead of 5 reads.
Mock Galaxy (8 x 4, TP ring): the head with the in-program reduce-scatter and the router chain builds and enqueues.

### K3's latent down + all-to-all in the program
- Latent down = extra plain engine passes after gate / up on the gathered x (x from L1; W_ld's chip columns
  [H, Hr / TP] in the column's W stream; fp32 between-shard partials like gate / up; bf16 out into LAT [MT][LWMAX]).
- All-to-all to the owners through the ports: per destination block, per core of the port's link with rows in it, one
  sub-block (its rows x its latent columns) by multi-hop fabric unicast (no multicast, nothing replicated) into the
  destination chip's same core, zone slot = the source chip, fused increment; the owner core's NCRISC writes its own
  and the G - 1 arrived slices to the owners' latent [rows / TP, Hr] (DRAM: the stage's output).
- L1: K3 + latent did not fit (1481 KB at KB 7, 1583 at 14). HOLD / FINB now overlay XIN + ACC (only this core's
  compute touches ACC; no row peer writes XIN after the last round it consumed): K3 fits at KB 7 (1289 KB).
- Hang (stopped at the first, narrowed: router alone pass / latent alone hang / latent without the all-to-all pass):
  each port phase after the gather restarted its CB chunk count at 0, but the latent phase leaves the CB unaligned -
  the next phase's reservation ran past the CB's end (triage: the ports' low L1 / firmware text overwritten). Fixed:
  one chunk count over the phases. Latent PCC 0.99987-0.99994.

## Where the head stands (2x4, 640 rows a mesh row; eager device times)
| us | GLM-Flash | K3 |
|---|---|---|
| fused program: x chain + SE (down + reduce-scatter) | 248.2 | 449.9 |
| + router chain (router, logits RS fp32, top-K) | 260.0 | 474.1 |
| + latent down + all-to-all (K3) | - | 497.8 |
| routing gather + planner + dispatch prep (block's programs, chunk 1280, not fused yet) | 50.9 + 34.0 + 11.0 | 45.6 + 46.0 + 11.4 |
| **whole head** | **355.9** | **600.8** |
| floor (TP-link bytes) / 2x target | 180 / 360 - **1.98x** | 330 / 660 - **1.82x** |
Correctness, all in one program: SE PCC 0.99988 / 0.99985, routing picks 99.53 / 99.45% and weights PCC 0.9955 /
0.9919 vs the host's fp32 routing (= the block's ttnn router chain), K3 latent PCC 0.99987. Mock Galaxy (TP ring)
builds the whole program for both models.
Next: the routing gather (needs SP-axis ports) + planner + prep in the program (GLM's margin is 4 us); the SFPU top-K
(the scalar scan is L1-load bound, ~16 us a K3 token); W_r multicast (5 DRAM reads of it slow K3's gather end by
~25 us); KB 14 for K3 back (the latent pushed it to 7).

## Toward 1.5x of the floor, fully fused (user: GLM <= 270 us, K3 <= 495 at chunk 1280)
| piece (commit) | GLM-Flash | K3 |
|---|---|---|
| start: x chain + SE RS + router + latent (routing gather / planner / prep separate: +96 / +103) | 254.8 (+96) | 479.8 (+103) |
| W_r read once, a rotating column multicast down the router cores (26d63670478) | 256.4 | 480.1 |
| top-K on the SFPU, logits computed transposed (95e78c22b13) | 254.8 | 479.8 |
| **routing exchange + combine planner in the program** (37c22bcb94d; bit-exact vs combine_plan's host model) | 295 | 534 |
| joint-row weights in integer math, 2 RISCs; router HiFi3 (same picks as HiFi4; HiFi2 changes them) | 278-287 | 515-540 |
| logits straight to their owners (no relay chain); routing exchange store-and-forward through the ports | 284-302 | 517-537 |
| planner on 10 instances (eae243c76ba) | 279.0 | 510.8 |
| top-K: pairs of score tiles to K-runs + bitonic folds, fp32 keys (top-K 35 -> 14 / 55 -> 20 us) | 273.5-274.3 | 511.0-511.4 |
| logits port units in the hand-off order (bug fix); router matmul half-sync DEST | 267.5-271.5 | 512.8-515.5 |
| **dispatch prep in the program: route words + owned rows (now included)** | 271.9-273.0 | 509.7-510.6 |
| floor (TP-link bytes) / 1.5x target | 180 / 270 | 330 / 495 |
Everything is in the one program now except dispatch prep (route words + owners' rows into the dispatch buffer).
- Routing exchange + planner (tt/routing_stream.py): routing_a2a's planner runs on the 5 router cores' BRISC after
  their joint rows (a barrier among them); its slices go into the senders' L1 packet buffers: same-row destinations
  on the gather's ports (a polled stream), other rows Y first on the watcher core's BRISC (Galaxy: the second Y
  direction on its NCRISC after the watch). combine_plan's counts / regions / token_index / table / plan bit-exact vs
  plan_reference on all 8 chips, both models; mock Galaxy builds it (TP ring, 28-destination Y senders).
- What binds now: the routing chain, not the links. Per chip (GLM): router 63-71 us, logits in 76-127 (handed off by
  ~70-80; the rest is transit through rows the x gather saturates), top-K 35, joint rows 14, planner 21 before the
  exchange, its wait, 25 after. The chips' chains are skewed by up to ~50 us (logits arrival), and the last packets
  of one mesh row land up to ~35 us after the last send (forwarding / transit through ports busy with the reduce-
  scatter): GLM's head ends at the routing chain's end (~280-300), K3's too (its logits wait for the x gather by the
  K3 priority; the exchange waits up to ~110 us). Tried without effect: the logits as a direct all-to-all instead of
  the relay chain; the routing exchange store-and-forward instead of multi-hop transit (both kept: no relay chain,
  every hop a local injection - ring-ready). The x chain + SE parts alone are within target (GLM 248, K3 ~450-470).
- Floors: the TP-link-bytes floor (180 / 330) still holds; the routing chain's own critical path (router compute +
  logits on the wire + top-K + planner + exchange latency) is ~150 us (GLM) at best - below the link floor, so the
  floor stands; the gap is the chain's serialization behind the saturated links.
- Planner on 10 instances (every router core's BRISC + NCRISC; regions overlaying the router's dead matmul / zone
  buffers; per-instance exchange semaphores and bases - routing_a2a assumed one base): its work before the exchange
  ~12 -> ~10 us, after the arrivals ~38 -> ~22 us; still bit-exact. GLM 279.0 us, K3 510.8 (one run; run-to-run
  spread ~+-10 us now that the routing chain binds).
- K3 logits-first (data check of the user's K3 priority): the routing chain ends earlier (~490) but the gather slows
  and the SE ends ~520 (528-531 us twice) - x first kept.
- Where the routing chain loses time (instrumented: per-arrival stamps at planner core 0, zones on every send /
  forward): the senders and forwarders act within a poll of their packets landing; the exchange waits are the chips'
  skew (logits arriving 76-127 us across chips under the gather's load, plus launch skew - every device's zone times
  are relative to its own start, so cross-chip gaps include it) carried into an all-to-all that waits for the last
  chip. Not tried yet: a cheaper top-K network (13 full 64-sorts: 35 / 55 us), the router matmul on more cores.

### Lever 1 (1.5x push): a cheaper top-K - K-runs and bitonic folds instead of 64-wide sorts (tt/router_stream.py)
- Was: the running best 32 against each next 32 experts, a full 64-wide bitonic sort a step (ET - 1 sorts: 8 GLM,
  13 K3), the best pair packed and re-read every step: 35 us (GLM, K = 8) / 55 (K3, K = 16).
- Now, all in DEST under one acquire, nothing packed until the end:
  - keys: fp32 score with its low 9 mantissa bits replaced by the expert id (made on the SFPU in the sigmoid / bias
    pass from the id tiles, 4 score tiles a pass) - ttnn.topk's fused-key mode, one word a datum, no id tiles
    riding the swaps; but fp32 - 9 bits instead of bf16 (bf16 keys would change picks). The joint-row RISCs take
    the id from the key's low bits (the weight's score loses 9 bits: ~6e-5 absolute, far under a bf16 weight).
  - each PAIR of score tiles (64 experts a token column): bitonic phases 0 .. log K - 1 (sorted K-runs), then
    log(64 / K) merge + rebuild rounds down to one sorted K-run. The pair's last merge keeps the max in tile 1
    (ascending run: the network's K = 8 rebuild has no direction knob - its run 0 is always descending, run 1
    ascending), the running top K (parked in DEST tile 4, descending) is copied into tile 0, one bitonic merge
    (max on top) + rebuild of K -> the new running top K. GLM's odd 9th tile: one full round against it.
  - work: ceil(ET / 2) pair reductions + folds instead of ET - 1 full sorts (+ their packs / unpacks).
- Timeline (zone SE-RTOPK, GLM): 34 -> 26 (DEST-resident running best, K-run merges instead of the full sort) ->
  22.5 (fused keys) -> 18.5 (keys made in the sigmoid pass) -> 13.5-14.7 (pairs + folds). K3: 55 -> 19-21.
- Picks: GLM 99.30% (unchanged), K3 99.45% (was 99.37 - the fp32 keys are finer than the old bf16-ish id
  tie-breaking); weights PCC 0.99540 / 0.99171; plan tables bit-exact; protocol model 33 pass, mock Galaxy 2 pass.
- Whole head: GLM 273.5-274.3 us (279.0 before), K3 511.0-511.4 (510.8: K3's top-K was not on its critical path -
  its routing chain waits for the x gather's rows). Target 270 / 495.
- Tried and dropped: the fold in a second DEST window (topk's LLK addresses DEST absolutely - window 0 only).

### Bug: the logits port sent sub-blocks before they were written (fixed, 667094a21e1)
The port sends unit u of a router core once that core's word counts u + 1 hand-offs; its units were nearest-first per
direction (`_dests`) while the cores hand off farthest-first: with two owners in one direction the port could send
the far owner's slot before the core had written it. Masked in the tests by identical repeated inputs (the slot held
the previous call's identical data). Both orders are now farthest-first.

### Lever 2 (1.5x push): the router matmul - data first, then more cores (not done)
- Measured what bounds the router matmul (GLM, 5 router cores, 56-65 us): no W DRAM reads (multicast of garbage):
  unchanged; no x reads either: unchanged; LoFi instead of HiFi3: 47-58 (only -10); K blocks of 8 instead of 4 (half
  the L1-accumulated packs): unchanged; no in1 transpose: unchanged. Per tile-matmul ~60-66 cycles at HiFi3 (48 ideal),
  ~55 at LoFi: unpack + math bound, ~75% of the FPU - only more cores would cut it.
- Kept: half-sync DEST (a block's packs overlap the next block's math): GLM 63-69 -> 57-65 us, K3 156-161 -> 138-151.
- Not built (yet): the port / watcher cores' TRISCs as helpers. The data says it does not pay now: K3's logits leave
  the ports only after their x gather (the x-first priority), so a faster K3 router changes nothing (its RLWAIT ends
  ~230-245 either way); GLM's chain would start ~25 us earlier, but GLM's end is now the reduce-scatter's tail on the
  even chips (DOWNRS max ~266-269 with prep) as much as the planner (~265-270). Cost: no NCRISC handshake on the
  helpers (their data movers are the ports' loops) -> a sender-polled multicast + TRISC-side CB signalling, and a
  second hop for the helpers' hand-offs. Revisit if the routing chain binds alone again.

### Lever 3 (1.5x push): dispatch prep in the program (tt/prep_stream.py, se_engine3 prep; RoutingSpec(prep=True))
dispatch_prep's two outputs, made inside the head (the whole-head numbers include them from here on):
- Route words: every owned token's route word (sparse_all_gather_2d's format: destination chips, X/Y-first order,
  link; one word a token up to 16 chips, two beyond) computed on the router cores' compute UNPACK thread (idle after
  the top-K) from the joint rows in L1, written by the router BRISC after its planner (128 B a core). Bit-exact vs
  combine_rf/dispatch_prep's route kernel run on the head's joint rows (both models). Mock Galaxy: both axes wrapped,
  two words a token at 32 chips.
- Owned rows: the dispatch buffer [N S, W] row-major (W: H for GLM - x; the latent for K3), this chip's slots me S ..:
  - GLM: x own (resident, never overlaid) - each owned row tile's K steps untilized by the holding core's compute
    (pack_untilize from a CB view over x own, read pointer moved per unit), written row-major by its NCRISC (32 rows
    a unit, 1 KB each). Bit-exact.
  - K3: the latent blocks (mine in LAT, the arrivals in LATZ) untilized the same way (the arrivals' units after the
    NCRISC's latent-ready push); the tile-layout latent output is no longer written. PCC 0.99994 vs the host.
  - Placement: one prep op right after the core's last down pass. Spread over the passes it delayed them (GLM
    282-286 us); at the end of the op list it delayed the reduce-scatter's tail fins (283). GLM's x own sits on 2
    columns (the x families' owner subsets): 6 cores carry all its units (~10 us each).
- Whole head with prep: GLM 271.9 / 272.9 us (268.5 before the owned rows), K3 509.7-510.6 (511-515 before: the
  latent's tile writes are gone). Targets 270 / 495.

### What binds after the prep: the RS on the line's end links; on the Galaxy, the routing chain
- RS probe (SE_PROBE=RSSTAT: the port reader's waits on hand-offs / CB room, the sender's on the reader / fabric slots,
  the down passes as zones): the end chips' ports send at link rate (K3 23.5 GB/s, fabric-slot waits 18-21 us, no
  waits on hand-offs) - on a line every byte of an end chip's partials leaves through its one link: 3 blocks (GLM 84,
  K3 146 us); the middle chips' ports wait on relays paced by them. GLM ~19 -> ~21 GB/s with the logits / routing
  polled every 4th RS packet (491f0f05a7b). The down passes end 16-20 us after gate / up.
- 2 row chunks (chunk k's down + RS under chunk k + 1's gather) would turn the line's 6u (gather 3 inbound + RS 3
  outbound on an end link, sequential) into 5u - a LINE artifact: on the Galaxy's TP ring both balance to 1.5u a
  direction (3u with or without overlap), and two chunks at 640 rows need MT 1 (K3 engine 403 -> 542 us). Not built.
- Engine alone (MT 2, x from DRAM): GLM gate / up ends ~60 us (100.4 total), K3 ~237 (318.4; + the latent in the head).
- Galaxy projection (tests/galaxy_projection.py; rules in its docstring: the ring halves the link-bound gather / RS,
  gate / up never before its compute, K3's logits shift with the gather, the routing chain + 20 us at N = 32):
  GLM LoudBox 277.6 -> Galaxy 296 (SE 176, chain 296); K3 509.4 -> 505 (SE 399, chain 505). The routing chain binds
  both on the Galaxy; its largest piece is the exchange wait (the slowest chip's chain end; paths are short).
- Routing exchange at N = 32 (mock 8x4 torus, the real RoutingStream): 992 packets, 288 direct / 480 one store-and-
  forward hop / 224 two; Y legs in transit (<= 4 hops), X legs through the TP ports (29 forwards a chip, <= 15 a
  port); header 384 B + ~40 (GLM) / ~80 (K3) 4-byte entries. Planner: pre-exchange +4 us (3 N NoC writes), post-arrival
  22 -> ~29 (2 N DRAM reads, N x M source loops), the slowest of 8 TP groups +9: ~+20 us in all.
- Tried: the L1 data cache on the planner's post-arrival passes: phases unchanged (reverted).

### Lever 2 again (Galaxy view): the router matmul on helper cores - built, measured, reverted
- Built: each router core's G token tiles split NR = 2 here / NH = 2 on a helper (the 4 ports' and the watcher's idle
  TRISCs): the helper's x forwarded by the router core's BRISC (staged by its NCRISC), W_r^T multicast to the helpers'
  column from each K block's sender (prefetched from DRAM up front), helper flow control by polled consumed counts (3
  slots), the helper's fp32 partials pulled into the router core's HOLD before their hand-offs. Bit-exact routing /
  plans / prep, picks unchanged.
- Router matmul 57-65 -> 45 us, but the x gather slowed 10-25 us (the helpers' L1 / NoC traffic on the port cores
  under the gather) and the SE path with it: GLM 300 us (272-278 without), Galaxy-projected 312 (299). Reverted; the
  W prefetch alone (each sender's blocks read up front): router matmul unchanged (60-63) - also reverted.
- The router matmul's floor here is its own cores' compute + the column multicast; not the binding piece once the
  ports' cost is counted.

### Logits-vs-gather priority: the head's choice (RouterSpec.logits_first=None)
- HeadSE3.logits_priority: logits first on a TP ring (Galaxy: the gather is 1.5 shards a direction with slack under
  gate / up; the routing chain binds - K3 Galaxy-projected 466 us logits-first vs 505 x-first, from the LoudBox
  profiles of both priorities); on a line logits first unless a latent all-to-all shares the ports during the gather
  (K3 on the LoudBox: x first, 509 vs 527 measured). Explicit True / False still override.
- Galaxy floors in tests/galaxy_projection.py (ring link loads per direction, HiFi2 compute at peak, the routing
  chain's per-step minimums): GLM 91 us (links), K3 170 (links ~ compute 166). Projected / floor: GLM ~3.3x (chain),
  K3 2.74x with logits first on the ring.

### The router's matmul on the engine's cores at the start (RouterSpec(on_engine=True); built, default off)
- se_engine3 SE_RTR: before gate / up, every engine core computes its expert tiles (ETC = ceil(ET / GX), the same on
  every core) x its MT rows over the chip's K slice at HiFi3 (matmul's LLK calls with the fidelity fixed); x rows by
  the row's first core (DRAM -> row multicast), W_r^T expert tiles by the column's first core (DRAM -> column
  multicast), in K chunks of <= 8 tiles with acks (one buffer each: fits K3's arena, in the tail past the words);
  the fp32 partials written into the router cores' HOLD with a count (write barrier, then the increment); the router
  cores skip their matmul (RT_ENG: wait the count). Routing / plans / route words / dispatch rows bit-exact; mock
  Galaxy builds.
- Timing: the phase takes 18 us (GLM) / 44 us (K3) on the engine - gate / up starts that much later - and the router
  cores have the partials at ~42-47 / ~75-78 us (the writes into the mover column: 0.7 / 1.1 MB a chip, the slowest
  engine cores ~20 us after the median), vs the router cores' own matmul ending at 57-65 / 138-151.
  GLM: LoudBox 291-295 us (272-278 off), Galaxy-projected 265-273 (296-299 off) - the chain ~25 us earlier, the SE
  path ~15 us later. K3: LoudBox 519-524 (509 off); its Galaxy chain needs a logits-first profile.
- Picks: GLM unchanged (99.30%); K3 99.10% vs 99.45% with the router cores' matmul - not fidelity (HiFi4 99.14, LoFi
  97.54), not the K chunking or accumulating in DEST vs the packer's L1 (both 99.10), not a race (write barrier
  before the count). Ideal bf16-input routing (host, bf16 x and W, fp32 sums) agrees 99.65% (K3) / 99.69% (GLM)
  with the fp32 host. Unexplained: default off until it is.
- Plan choice (RouterSpec.on_engine=None -> HeadSE3.router_on_engine): on unless a latent (the K3 pick drop).
  GLM LoudBox 291.3 us, Galaxy-projected 261 (2.86x its 91 us floor); K3 (router cores) LoudBox 510.2, Galaxy 492
  x-first-profile projection (2.89x of 170). Pick-drop checks: the engine's fp32 partials are full fp32 (low 16 bits
  set: not a bf16 pack); one partial vs the host's exact bf16-input dot product: 0.268743 vs 0.268982 (~1e-3
  relative, ~1/4 bf16 ulp) - far above fp32 accumulation; host models: bf16-rounded or chopped per-chip partials
  99.57%, bf16 accumulation per 8 K tiles 99.22% (K3) - the device's 99.10% is closer to a low-precision accumulation /
  product than to a partial-format issue. A second hw startup for the router's operands hung (reverted).

### Port priority, step 1: the forwarders' landed count, the logits / routing polled before every RS packet
- RAFwd: one shared landed count a forwarding port (+1 from a header-only atomic packet behind each forwarded packet's
  data + flag); a poll reads that word and scans the 16 flags only when it moved (was: an invalidate + 16 flag reads
  every call - why polling every RS packet cost ~2 GB/s of the end links). rs_send polls before every packet again.
- GLM 287.8 us LoudBox, Galaxy-projected 260 (2.86x); K3 508.8, Galaxy 496 (2.92x) - within noise of before: the
  routing packets' wait at the ports is not what binds (the slowest chip's chain end is).

### Launch skew inside the eager numbers (SE-RDYWAIT: the ports' ready handshake, timed)
- Each port facing a neighbour waits for that port's "started" increment; the wait's end, from the chip's own start,
  is ~3 us when the neighbour launched first, else the start difference + ~3 us. galaxy_projection.py skew <rows>
  turns that into start offsets: GLM run, row 0 (devs 7,6,4,5 by TP position) 52 / 37 / 21 / 0 us, row 1 (3,2,0,1)
  48 / 32 / 0 / 17 - the eager dispatch launches the chips ~16 us apart (dev 0, 1, 2, 3 ... in id order within a
  row; the rows' offset to each other not measured: no X-free port pair).
- The eager metric (each chip's own end - own start, the slowest) carries that: here 281-292 us; from the row's
  last launch to its last end 266-269; the logits arrival skew and the exchange waits seen before are mostly it.
  In a traced / back-to-back forward the chips start when their previous op ends - this skew is the eager harness's.

### The rows' launch skew too (SE-YHELLO): numbers from the mesh's last launch
- The watcher's Y sender sends one header-only "started" to the other row's watcher at its start and stamps (a
  timestamp, no wait before its sends are done) when the other row's lands; with the in-row offsets
  (SE-RDYWAIT) that aligns the rows: galaxy_projection.py's first line is now "from the mesh's last launch".
- GLM run: launch offsets from the last launch: devs 0-3 (row 1) -100 / -83 / -69 / -54 us, row 0 (4-7) -28 / -42 /
  -15 / 0; K3 run -154 / -120 / -103 / -85, -39 / -61 / -20 / 0 - the eager harness launches the 8 chips over
  100-150 us.
| (us) | LoudBox from the last launch | Galaxy-projected | / Galaxy floor | eager LoudBox / Galaxy |
|---|---|---|---|---|
| GLM (router on the engine) | 268 | 168 (SE 168, chain 168) | 1.84x of 91 | 287 / 260 |
| K3 (router cores, x first) | 488 | 381 (SE 380, chain 381) | 2.24x of 170 | 520 / 501 |
- With the launch skew out, the routing chain no longer stands above the SE path: the two meet (GLM 168 / 168, K3
  380 / 381 Galaxy-projected).

### Router on the engine: the K3 pick drop solved (the hand-offs ran ahead of the engine's partials)
- Localized with an in-call capture (debug, not kept): the router NCRISC copied HOLD and the logits arrival zone to
  DRAM at the top-K's done wait (before the planner overlays them). Every chip's HOLD was right (1e-3); p0's
  arrivals from all three sources were wrong in owner 0's expert tile 3 (rows 16-17 and 25 + j, tokens 0-15) -
  small integers (planner counts). Owner 0 is the lowest HOLD entry, inside the planner's second overlay region.
- Cause: in engine mode the compute skips its matmul and waits RDY (cb_wait_front: the UNPACK thread only); the
  hand-off pushes (FINC) are the PACK thread's - nothing held it, so the BRISC handed HOLD to the ports at the start,
  before the engine's partials landed. HOLD then held the last call's partials (the same input every call: right by
  luck) except where the last call's planner overlay wrote - K3's tile 3 of owner 0; GLM's overlay misses its HOLD.
- Fix: the NCRISC sets word 112 after the engine's count, PACK waits on it (router_stream). K3: max abs score error
  1.6e-3 (was 0.70), picks 99.45% (= the router cores' path). GLM picks unchanged (99.30%).
- Not the cause (tested): a cross-call race on the arrival zone - a parity double-buffered zone (per-RISC call
  counts) changed nothing; reverted: a source cannot start its next call before every receiver summed this call's
  arrivals (its planner needs their routing packets, sent after the sum). Kept: the zone above the planner's overlay
  (a source's next-call partials may land while this chip's planner still runs).
- The cost: GLM's earlier 168 us (Galaxy-projected from the mesh's last launch) came from the early hand-offs; A/B
  with the PACK wait off: 168 us, on: 193-196 - the chain binds at 193-196 (SE path 162-166).
- K3 (logits-first profile measured): from the mesh's last launch / each row's / eager, Galaxy-projected:
| K3 (us) | mesh last launch | row last launch | eager |
|---|---|---|---|
| router cores, x first (was the default) | 385 (SE 379, chain 385) | 464 | 499 |
| router cores, logits first | 365 (SE 365, chain 359) | 443 | 507 |
| engine, x first | 380 (SE 380, chain 358) | 423 | 461 |
| engine, logits first (new default) | 366-367 (SE 366, chain 290-294) | 367-376 | 397-434 |
- Plan choices: router_on_engine True for every plan; logits_priority True with the router on the engine (also on
  the line). K3 LoudBox 504 us from the mesh's last launch (524 eager), Galaxy-projected 366 us = 2.15x of 170: the
  SE path binds by ~75 us over the chain. GLM 193-196 us = 2.12-2.15x of 91: the chain binds (SE 162).

### Every multi-call test: its own inputs a call, each call checked against its own reference
- Two bugs hid behind identical inputs every call (the logits port's send order; the engine-mode hand-offs above).
  tests/test_tp_collectives.py: _calls(mesh, run(k), check(k, out)) + gen(seed, k) / ncalls() (1 + ITERS with the
  profiler, call 0 untimed; else 3) replace _time. Converted: test_head_se3 (x per call - its own routing; every check
  per call: SE / RS output, joint rows vs the host's routing, latent dispatch rows, planner tables, route words),
  test_se_engine3, test_lin2c, test_lin2c_chain, test_head_xchain, test_se_engine, test_head_se, test_mrs_bar,
  test_tp_collectives / test_tp_gather / test_agmm_bar (vs host collectives of each call's x, not vs ttnn's output on
  the same x), test_moe_block (3 checked calls, each its own x; the timing calls on further distinct inputs).
- The check catches the hand-off bug: the PACK wait off -> GLM picks 6.95% on call 0, assert.
- Current defaults, 6 calls each its own x, all pass: GLM picks 99.14-99.45% a call, SE+RS PCC 0.99988, dispatch rows
  / planner / route words bit-exact; LoudBox 262 us from the mesh's last launch (285-290 eager), Galaxy-projected 198
  (SE 162, chain 198) = 2.17x of 91. K3 picks 99.14-99.49%, PCC 0.99985, latent rows >= 0.99992, plans / route words
  bit-exact; 494 us (514-522 eager), Galaxy 366 (SE 366, chain 293) = 2.15x of 170.
- Pass (3 calls each): se_engine3, mrs_bar, lin2c, lin2c_chain, tp_collectives, tp_gather (bit-exact), agmm_bar's
  ag+mm2d. Failing on HEAD too (legacy, superseded by v3): test_head_xchain (NaN), test_se_engine (0.985),
  test_head_se (0.85); the fused agmm variants and a2a_generic throw as before.

### Planner post-arrival (routing_a2a's planner, ttnn/ttnn/bringup/combine_rf/routing_a2a.py)
- RA_DBG phases (GLM chip 7, core 0; us after the last packet): reads + totals + ranges + zeroing 5.7, count +
  prefix 6.6, fill 10.8, core 0's tail (the slowest instance, then 3 DRAM writes serialized) 5.4 = 28.5. Per instance
  (RA_DBG words 24 / 25): experts 2-5, entries 80-200 - the range rule rounded every boundary up to the next expert.
- Ranges: instance j's first expert = the expert boundary nearest j T / M entries (was: the next one) -> entries a
  instance 110-164. Counts / regions written by core 0 as soon as the ranges are known (under the fill; own buffers),
  the tail writes the plan words only. Expert totals as packed 16-bit lanes (N S < 65536), the (expert, token)
  counters zeroed with 32-bit stores.
- Post-arrival 28.5 -> 24.4 us (core 0's tail 5.4 -> 2.0). Planner tables bit-exact every call (6 calls, own x).
  GLM Galaxy-projected 198 -> 193 us from the mesh's last launch (chain binds; SE 157); K3 367 (SE binds; chain 294).
- Left: the fill ~95 cycles an entry (eo / tidx / tab L1 traffic, the 80 source-slices' loop), count ~40.
- Seen on the way (the chain before the planner, GLM dev 7 from the mesh's last launch): the engine's partial logits
  into the router cores (SE-RTROUT) 18 -> 43 us - ~5 GB/s into a router core; not the column's vertical links (a
  row-tile remap: no change), not the DRAM columns (router tiles on the west block only: the slow cores moved to
  x 1-4) - reverted, needs a traffic-only probe; the ports' routing sends (SE-RASEND) ~25 us a chip.

### Housekeeping: the superseded v1 / v2 heads and their tests removed
- Deleted: tt/lin2c.py (the v1 chunked engine), tt/se_engine.py (v2), tt/x_gather.py (v2's gather), tt/tp_gather.py
  (v1's store-and-forward gather), fused_head's HeadLayout / HeadXChain / HeadSE; tests test_lin2c, test_lin2c_chain,
  test_lin2d_ref, test_head_xchain, test_se_engine, test_head_se, test_tp_gather (the v1 / v2 tests failed on HEAD:
  NaN / 0.985 / 0.85). The sections above about them stay as history.
- test_tp_collectives: every candidate asserted against the host's collective of each call's x (moves exact, sums
  within 2^-6 of the largest); a2a_generic is the in / out-swapped call (the other one computes the transposed
  split: it never matched); test_agmm_bar keeps the ag + mm2d bar (all_gather_minimal_matmul_async throws in its
  program build at these shapes).
- Green: test_fused_head_parts (se_engine3, head_se3 with RS / router / routing / latent), test_tp_collectives,
  test_agmm_bar, test_mrs_bar, test_moe_block (4 cases, 3 checked calls each), protocol models 47, mock Galaxy 2.

### The ports' routing sends (SE-RASEND): the header totals from the planners
- SE-RASEND ~25 us a chip (GLM): zones inside (SE-RATOT / SE-RAPKT, probes) - the per-expert totals of a packet's
  header (M slices' prefixes summed on the port RISC) 2.2 us a destination, the packet itself 0.6; the Y sender
  serves 5 destinations in a row. Packed 16-bit lanes: 1.7-2.2 us (the port RISC's ~10 cycles an element; its data
  cache: 2.2 -> 1.7).
- Now (routing_a2a's planner): in the exchange each instance also sends its per-expert counts of destination d to the
  owner instance d % M (behind the same write barrier / exchange count: no new semaphore); the owner sums the M
  slices' (packed lanes, the planner's data cache) and writes the header's totals into the sender's packet buffer
  with its slice. The senders (the head's RASend, routing_a2a's own) only sum the slice counts for the bytes and send.
  The planner's debug page sized to the 128 B it writes.
- SE-RASEND 25 -> 3-4 us a chip. GLM Galaxy-projected 193 -> 182 (packed totals) -> 178 us from the mesh's last
  launch (1.95x of 91; chain binds, SE 160). K3 chain 294 -> 271 (SE path binds: 373). Plans / route words bit-exact
  every call; mock Galaxy builds.

### The partial logits' way out: straight from the engine to the ports (no router-core hop); the W gate (option)
- SE-RTROUT (the engine's partials into the router cores) 18 -> 42-47 us. Probes (timing only): a quarter of the
  bytes -> most in by ~20; the writes looped back to the writer -> all done at 19-20 (so: delivery); the router
  cores' polls spaced out -> no change; the engine's W streams held back 30 us after a core's partials -> all in by
  21-22. The cores done first start their W streams (DRAM reads on NOC1) and crowd the rest's writes.
- Bypass (the coordinator's): each engine core writes its own owner's row tiles into the router core's HOLD and every
  other owner's straight into the port sub-block the router core used to hand it to (router engine_args(coord): per
  row tile (NoC, base, count)); the router cores wait their own block only (ET) and hand nothing off. The ports'
  LStream: one word a (unit, core) sub-block (16 a port, a new mailbox region past the routing's), "in" at need =
  ET tiles (engine) or 1 (a router core's hand-off, the router-core path).
- The W gate (RouterSpec.w_gate, default off): every engine core counts its partials written on router core 0, which
  multicasts one word to the engine (a 16-byte-aligned source word: a misaligned one never landed - probe); the
  engine's W streams wait for it. Gated: partials in at ~22-27 but the SE path +10 us (GLM 173-179); not gated with
  the bypass: RTROUT 18 -> 30, GLM 166.
- GLM Galaxy-projected 178 -> 166 us from the mesh's last launch (1.82x of 91; SE path 166, chain 166). K3 chain
  271 -> 253 (SE path 370 binds). Picks / plans / route words per call as before; the router-core path and the gate
  pass; mock Galaxy builds.
- Next in the chain (GLM, from the mesh's last launch): logits in at ~45 (dev 7) / 60-71 (the others: dev 7's
  partials at its ports ~25, ~35-45 us on the line under the x gather); then on every router core ~14 us from the last
  arrival to the top-K start (the fin), top-K ~12, rows ~12.
- The router cores' fin (last arrival -> top-K start), GLM: 16 us, of it the full-precision fp32 sigmoid 11.3 (probe:
  4.7 without it); top-K 10.4; rows (after the top-K) 12. For later: a cheaper sigmoid (or the arrivals' sums and
  sigmoid as each source lands), the top-K and rows over more RISCs.

### The SE engine's compute (K3's SE path binds; GLM's ties): what bounds it (engine alone, MT 2, 640 rows)
| probe (SE_PROBE) | K3 us | GLM us |
|---|---|---|
| none | 317 | 97 |
| NOX,NOW (no x / W data in the multicasts) | 307 | 93 |
| NOMATH | 310 | 78 |
| NOWL (no W DRAM reads) | 256 | 91 |
| NOMATH,NOWL | 222 | 64 |
| NOX,NOW,NOMATH | 209 | 62 |
| NOX,NOW,NOMATH,NOWL | 107 | 38 |
- K3 zones (NOX,NOW): gate / up + latent 231 us (52% of HiFi2 peak), down 72 (64%); LoFi: 307 -> 242 (math ~65).
- Data movement and compute each bind at ~300 (taking either away leaves the other); the W stream from DRAM (~40 MB a
  call, ~400 GB/s, the NCRISC's NOC1 reads) ~60 us on K3's path; the protocol rest ~107.
- Tried, no gain: KB 8 (303) / 4 (345) vs 16 (317); 4 x slots (298 at KB 8; 6 slots / KB 16 x 4 slots wrong results -
  an overlay); credit-based x rounds (W-style, XS in flight, no per-round handshake): 334 full, protocol-only 211 -
  the rounds' handshake latency does not bind. Reverted.
- Structural: DEST 8 fp32 tiles and MT 2 (all 640 rows over 10 core rows) -> ~1 KB of x + W a tile-MAC into each
  core; x re-multicast every pass (K3 3 passes). Options: x-stationary passes with fp32 accumulation in L1 (packer
  L1 acc: x reuse over all of a column's tiles, ~28% less per-MAC traffic, more pack traffic); the W stream on NOC0 /
  bank-local readers (NOC1 DRAM reads are the slow ones on BH); W for the next layer prefetched under other ops (a
  model-level change).

### GU_ALONE re-measured at the head's shapes; the router back on its own cores for K3
- The projection's SE front = max(gate / up end - the ring's gather shrink, GU_ALONE). GU_ALONE was 60 / 250 us
  (an older engine-alone measurement, before the router's matmul moved onto the engine). SE_PROBE=NOXWAIT (the x
  rounds do not wait for a shard: gate / up at its own pace inside the head) - the slowest chip's gate / up end from
  its start: GLM 89 (router on the engine) / 73 (router cores), K3 307 / 272. galaxy_projection keys GU_ALONE by the
  router's place (detected: the engine's SE-RTRLD zone).
- Galaxy-projected from the mesh's last launch: K3 router on the engine 424 (SE binds; was shown as 366-370 with the
  250 floor), router cores logits first 388 (SE 388, chain 377), x first 391. GLM engine 166 (both paths), router
  cores 191 (chain binds, SE 149). router_on_engine: not latent (GLM on, K3 off).
- The SE tail (K3, RSSTAT): after gate / up ~27 us to the first RS packet (h exchange ~9 + the first entry's down
  passes), the end links at link rate (~152 us = 3u on the line), ~18 after (the last arrivals and fins): ring-
  projected 27 + 73.5 + 18 ~ 119. The W DRAM stream runs at ~390 GB/s (skeleton: 40 MB in ~102 us) - not slow; its
  ~60 us in the full run is NOC1 sharing with the W multicasts.

### The router cores' joint rows over four RISCs
- After the top-K, the rows (ids + normalized weights, soft float: ~1000 cycles a token) ran on the BRISC and the
  NCRISC, 16 tokens each: ~12 us. Now 8 tokens each on the BRISC, NCRISC and the compute's MATH and PACK threads
  (PACK after its push of the BEST page; MATH waits a 16 B word PACK sets - "msync", zeroed by MATH at its start: the
  arena is unzeroed scratch). Rows ~12 -> ~6 us.
- GLM chain 166 -> 159-162 us (Galaxy-projected from the mesh's last launch; SE path 165 binds: 165); K3 chain 377 ->
  366 (SE 388-395 binds). Picks / plans / route words per call as before.
- One hang (call 0 of the first run after the change, before the msync zeroing): not reproduced in 10 runs since (4
  before the zeroing, 6 after); the unzeroed word was the one hazard found (MATH could start on a stale 1).
- The fin's sigmoid (11.3 of 16 us): approximate (fast_and_approx) breaks picks (59%); exp_21f + 2 Newton steps keeps
  picks (99.14-99.53% a call) for 16.0 -> 13.2 us - not taken (precision for ~3 us).
