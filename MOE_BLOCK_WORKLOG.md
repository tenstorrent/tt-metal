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
