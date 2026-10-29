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

### Soak (HSE3_SOAK=N: N more calls over a pool of 8 inputs, every HSE3_SOAK_CHECK-th (500) checked) - the K3 hang
- GLM defaults: 3000 calls, no hang, checks pass.
- K3 defaults: hangs within 150-750 calls (5 repros; router on the engine too). Triage (bh-lb-09, twice): row 0
  (devs 4-7) in the SE reduce-scatter - engines in the down loop waiting arrival thresholds / W rounds, ports' readers
  waiting engine hand-offs, some senders in noc_async_writes_flushed; row 1 done but its router cores in ra::plan
  waiting packets; ETH (fabric router) cores report mismatched NoC counters. Twice the latent rows on row 0 chips
  came out at PCC 0.9986-0.9988 (a correctness failure, call 0 / 1).
- Fixed on the way (cross-call hazards, kept): (1) the latent all-to-all had only the first hop fenced - a source
  could write its next call's latent into a destination still reading this call's (the source port now waits for every
  destination chip's "started" - one header-only increment per destination a call, subtract re-armed); (2) RAFwd
  re-armed its landed count with "= 0": another row a call ahead could land a next-call packet's count before that,
  the next call's poll then never scans the flag (forward lost) - now subtracted.
- A/B (3000-call budgets): without the latent: 2000 clean; without the router: 1 hang at 850 then 3000 clean (and one
  latent PCC failure); without the routing exchange: hang at ~400. The hang needs the latent; not yet explained.
- Narrowed with hang words (SE_PROBE=HW: a persistent L1 tensor, a sparse snapshot of each engine NCRISC's W stream
  every 1024 polls - per-poll stores hid the hang; a triage add-on reads them): every repro (dev 6 twice, dev 0) one
  engine column's W stream stuck at round ~60-68 of chunk 0. That column's compute waits W, its XIN stays full, the
  row x multicasts stall, the head stops. The slot's W-ready word (a set of round + 1 by the round's holder) read
  60 on all 10 cores of the column after 9 of them had taken round 62 (needs >= 63).
- Not a NoC ordering race: a holder sends round r only once every column core credited it, i.e. took round
  r - NWS, so the older set had landed everywhere. A detector (a ready word going backwards: the first old / new)
  fired on ~51 cores per device within 950 calls, all 8 devices, the same 5 of 11 columns, at round ~60: old values
  fp32 numbers (0x3FFD63B8 ~ 1.98, 0xBF4A9EEC ~ -0.79), the same down a column, then the real set (60).
- Cause: the ACC CB overrun. ACC is a FIFO of fp32 pass partials, MT x pw tiles a block; K3's latent-down passes
  make the widths differ, so a block may straddle the CB's end - pack_tile / copy_tile address a block from the
  pointer without wrapping (the push / pop wrap after), the tail lands past the end: in "words" (OUT is 0 B with the
  fused RS), where the holder's W-ready source word sits. The packer overwrote it during the data multicast and the
  set sent fp32 to the column: a W round taken before its data (silently wrong) or, when the real set came after a
  later round's, the word going back (the hang). A host model of the FIFO: K3 + latent G 4 MT 2 straddles 5 times a
  chunk by up to 2 pages; GLM never (uniform widths) - why only K3 hung. The likely source of the latent PCC
  failures too (the overrun's reach).
- Fix: ACC's region carries the straddle room (the largest block - 1 pages past the CB; CB size and protocol
  unchanged); a host plan check (_ring_pages: a CB with blocks of differing sizes, or one size not dividing it,
  must have the room) in the CB helper. Audit of every CB in the head (engine, RS, latent, router, ports, routing,
  prep): ACC the only one that can straddle; SE_OUTF (8-tile blocks, the tail last, one fin a launch) cannot - and
  room would not fix it (its NCRISC reader pops page by page, wrapping at the end). The remote-SET audit: the W
  stream the only multi-holder set; the engine router's RXR / RWR one sender a word, RT_GATE one, the x / h pipes
  the valid flag with a handshake, the rest increments.
- The W-ready check stays under SE_PROBE=HW: a ready word going backwards or past round wr + NWS records old / new
  and spins (a stray write halts the soak at once).
- After the fix: K3 4000 calls with the check, no hang, never fired; K3 300 calls checked every call: PCC min 0.99987.

### The SE tail cut (per-pass hand-offs + a narrow first pass) - tried, not taken
- Measured first (RSSTAT, a new SE-RSS-T1 stamp at the first RS packet; LoudBox, from the mesh's last launch): the
  tail is link-bound once the RS starts - the end chips' 3u streams run at link rate (GLM 82 us, K3 151). K3: the
  ports send the latent all-to-all first (~20 us after gate / up, while the h exchange and the first passes run), so
  K3's line tail is link time throughout (latent + 3u) - nothing for an earlier hand-off to cut. GLM: the link idles
  ~16 us after gate / up (the h exchange ~6 + scatter -2's three passes ~10 before a start entry's hand-off).
- Tried: the port's first unit (if a start) handed off pass by pass (the compute hands after every pass, the port
  walks piece by piece over the cores, a piece the pass's column band of the canonical [rows][w] sub-block as row
  runs; receivers unchanged), and a narrow (1-column) first pass. The CPU RS model (pieces read only after their pass,
  every piece once) passed. Device: correct (PCC 0.99987 / 0.99988). Perf, Galaxy-projected: K3 448 (was 390 - the
  first packet stayed behind the latent, the row runs of 1-3 tiles slowed its end streams 151 -> 237 us); GLM 168
  (was 163-167): the first packet at h exchange + 1 (was + 10) but the end streams 82 -> 102 us (the row-run packets,
  and the reader flushing before every blocked piece). Narrow passes without pieces: first packet + 12, streams 87.
- Reverted (kept: the SE-RSS-T1 stamp). A contiguous variant (pass-major sub-blocks, the same pass split for a block
  on every chip) could keep the GLM streams at link rate, for at most ~9 us on GLM and none on K3.
- Projection note: the latent's line-vs-ring difference (3 blocks a direction on the line's end chips, 2 on the
  ring) is not credited in galaxy_projection - K3's SE path is overestimated by about one latent block (~6 us).

### GLM: the engine's router phase (both paths) - two buffered rounds, the x lead on NOC0
- GLM's two paths bind together (SE 158-161, chain 153-163). The router phase on the engine (x rows along the rows,
  W_r^T down the columns, the partial logits) ran ~18 us before gate / up: in GU_ALONE (89 vs 73 without it) and in
  front of every chain step. Per-round stamps (SE_PROBE=RTRSTAT, not kept): 4 rounds of 8-tile chunks, each a chain
  of DRAM read + multicast + ready word + acks, lockstep over the grid; the x leads' 64 KB reads take 4-8 us at
  kernel start (DRAM contended: XLOAD and the gather's ports read the same x shard - XLOAD ends at up to 19 us).
- Now: two rounds of KTO / 2, a buffer each (no slot reused: no acks), the W_r^T lead reading the next round while
  the last one's multicast goes; the row's x lead is the BRISC (NOC0) - the NCRISC's x row and W_r^T column
  multicasts were on one NoC (row + column multicasts on one NoC stall). The x lead sets the row's ready word from
  its own source word (RXS): each core's NCRISC re-arms its ready words after the last round.
- Router load end 19 -> 16 us; GU_ALONE (GLM, router on the engine; SE_PROBE=NOXWAIT, the slowest chip's gate / up
  end) re-measured 89 -> 82 (galaxy_projection updated). Head, Galaxy-projected from the mesh's last launch, 3 runs:
  162 / 161 / 164 (SE 162 / 150 / 164 - gather-bound runs vary with the launch skew; chain 160 / 161 / 161). K3
  (router cores) unaffected: 388. Soak: GLM 1000, K3 300, clean.
- The chain (dev 6, LoudBox): logits in 60, fin + sigmoid + top-K to 96 (floor 14), rows 92, the planner's
  pre-exchange to ~110 (RA_DBG: home counts 2.4, totals exchange 3.9, staging 7-9), the exchange, post-arrival
  20-24 (headers / sources / ranges 4.5, then 6, 8.5). Not cut yet.

### The fused head in MoEBlock.forward (MoEOptions.head: auto | fused | ops)
- MoEBlock builds HeadSE3 (FusedHead) where it is built for the load (640 rows a mesh row: chunk 1280 on the 2x4,
  the Galaxy's per-chip load; "auto" falls back to the ops head elsewhere - logged): the head does the x all-gather,
  the shared expert + its reduce-scatter, the router + top-K + joint rows, the routing exchange + the combine planner
  (writing the exchange's own plan tensors: its combine baked their addresses; A must agree - FusedHead.A 12 / 16,
  asserted), the dispatch prep (the dispatch buffer's own rows, the route words) and K3's latent down + all-to-all.
  The exchange then runs from the dispatch (TtK3TokenSplitMoE.routed_prepped: arena zero, dispatch, experts +
  combine, tilize); the tail as before.
- L1: the head's arena (GLM 1.22 MB, K3 1.30 MB a core) and the exchange's (~1.13 MB) alternate in the L1 left, each
  allocated for its own call at the address its programs baked. Placement is top-down, so the build order matters:
  the tail's one-time L1 first; L1 holes above the head's arena for what the exchange keeps (its words) and for what
  the head keeps after placing its arena (twice - the gather's / router's / routing's semaphores and mailboxes; left
  below the arena they sat in the other programs' static CB range: K3's clash); the arena settled once (free L1
  right above it filled, else a later placement lands higher). The head's cached programs survive the per-call arena:
  only its arena-backed CBs are rebound (else every chip's program was rebuilt each call - 175 ms of host), and x is
  copied into one buffer (the programs bake x's address).
- Correctness (3 calls on fresh x + 300-call soaks, every 25th checked): PCC vs host with the device's routing GLM
  0.9866-0.9868, K3 0.9960-0.9963 - the same as the ops head; routing picks equal to the host's 98.2-98.6% (GLM) /
  97.1-97.6% (K3) vs the ops head's 99.0-99.6 / 98.4-98.8 (the fused router's precision: the borderline picks).
- Per stage, LoudBox device time (slowest chip, median of 5), chunk 1280:
  GLM fused: head 293.5 (x copy + the head) | arena_zero 5.1 | dispatch 140.5 | experts_combine 1312.4 | tilize 8.2
  | add 11.0 | tail_a2a 47.2 = 1818 us, 8 programs (ops head: 10 head stages 487 us, 2002 total).
  K3 fused: head 532.5 | arena_zero 5.1 | dispatch 133.7 | experts_combine 2300.3 | tilize 7.8 | tail_ag 82.6 |
  latent_up 162.6 | add 17.3 = 3242 us, 10 programs (ops head: 11 stages 1130 us, 3848 total).
  Head Galaxy-projected (tests/galaxy_projection.py on the head's own profile): GLM 161-164, K3 384-388.

### The fused router's picks: the fin in fp32
- The block test's routing agreement with the host: fused 98.2-98.6% (GLM) / 97.1-97.6 (K3) vs the ops head's 99.0-99.6 /
  98.4-98.8. HiFi4 for the router matmul (engine and router cores): no change. test_router_vs_ops (GLM: both routers on
  the same x / W_r / bias vs the host on bf16 W_r, per chip; for the tokens off, the host's gap between the K-th and
  (K + 1)-th biased scores): fused 99.61% with misses at gaps up to 8.4e-4, ops 99.88% with misses at gaps <= 1.5e-4 -
  the fused scores ~4x noisier.
- Cause: the fin summed the fp32 partials (add_tiles, add_reuse_dest) and added the bias (add_reuse_dest) through the
  FPU, whose srcA / srcB hold TF32 (10-bit mantissa): logits ~ +-5 rounded by ~4e-3, scores moved ~1e-3. Now all in
  DEST at fp32: HOLD / ARRF / BIAS unpacked straight to DEST, the adds on the SFPU (add_binary_tile).
- After: GLM fused 99.92% vs ops 99.88% (test_router_vs_ops); block test GLM 98.95-99.46% (ops 98.97-99.56), K3
  98.48-98.73 (ops 98.44-98.79); the head test's picks GLM 99.45%, K3 99.73% (worst chip). Cost: the fin + top-K
  ~+8 us - GLM's chain 160-161 -> 172 (binds; SE 158), K3 378 (SE 392 binds).

### Exchange stage 3: the dispatch inside the experts' program (opt-in, MoEOptions.experts_during_dispatch)
- sparse_all_gather_2d.dispatch_phase: the dispatch's reader / sender bodies as phase functions (namespace sa) of the
  combine's port kernels (sparse_combine_rf dispatch=: on its ports - place_ports with its cells - and its open fabric
  connection: one sender a link; CBs 24..28, kept across the arena rebind; global semaphores on those cores before the
  arena; the manifest per call below the arena - the reserve is now max(flat CBs, dispatch CBs) + manifest). After its
  reader phase a port counts in on every x relay's ready word (+1, the chip's first port also the pad to 8).
- Flat op (C++): FlatLayoutOptions.x_ready (plan key) - the relays (se11_xrd) wait for that count on their words at
  X_READY_OFF, then re-arm by subtraction. "Landed" rests on every incoming link delivering END(GMAX) to a port of the
  chip (asserted per port). CPU model test_exchange_phase_model.py (line / ring / torus, random interleavings, one FIFO a
  link with rows before the entries behind them): no deadlock, the gate sound, counts exact, combine after the dispatch on
  a link; an early gate trips it.
- Correct (block PCC as before, 300-call soaks both models). Perf at the LoudBox's experts per chip (GLM 36, K3 56):
  GLM dispatch + experts + combine 1460 us (ring 2) / 1440 (3) / 1412 (4) vs separate 141 + 1295 = 1436 at ring 4;
  K3 2438 / 2447 / 2454 vs 133 + 2309 = 2442. Zones: the gate/up rings fill by ~75 us, but x-ready lands at 170-284 us
  from a chip's start (the standalone dispatch 133-141): the dispatch stretches under the weight stream.
- Dispatch DRAM per chip (rows written at every hop and read back by the next hop's reader; CPU count over the trees,
  uniform routing, 160 tokens a chip): LoudBox ~19 MB (~37 us at 512 GB/s); Galaxy 8x4 35 MB GLM / 43 MB K3 (68 / 84 us)
  vs weights 127 / 520 MB.
- At the Galaxy's per-chip load on the 2x4 (160 tokens; GLM 64 experts = 8 a chip - the fused router needs E % 32 = 0 -
  K3 224 = 28 a chip), slowest chip: GLM separate 149 + 444 = 593 vs in-program 604 (ring 2), 604 vs 630 (ring 4);
  K3 139 + 1290 = 1429 vs 1429 (ring 2), 1455 vs 1441 (ring 4). K3 zones: x-ready at 178-246 us from a chip's start
  with the weights streaming, 127-215 with them held until the first x block (probe MIMO_FL_W_AFTER_X; whole program
  1410): the weight stream's contention stretches the dispatch ~40-50 us; after x-ready the experts take ~1150 (separate
  ~1290: the prefetch saves ~130 - eaten by the stretch). Next: an L1-relay dispatch (forwarded rows not round-tripped
  through DRAM).

### Rejected: x's address patched into the head's cached runtime args (instead of the copy into a persistent x)
- Fresh x every call moves its address, so the patch runs every call: 339 arg slots a chip x 8 chips, plus the kernel
  descriptors reassigned to each program. That costs 2.5-3.6 ms of host a call; the device copy it would replace is
  ~0.1 ms and one program. GLM wall 4.8 -> 7.2 ms; reverted. Traced, x is the trace's fixed input buffer anyway (written
  before each replay), so the copy leaves the forward altogether.

### Traced forward (test_moe_block_traced, chunk 1280)
- The forward captures as is (ttnn begin/end_trace_capture on the block's input buffer): every per-call allocation (the
  head's and the exchange's alternating L1 arenas, the manifest, the DRAM outputs) lands at the same address each call, so
  the capture's addresses are the replay's; no host reads (taps "device": the routing rows kept as a device tensor, read
  after the replay) and no host writes in the forward; the programs are built and the DRISC senders started by the eager
  calls before the capture. MoEBlock.input_buffer(): the fused head's x (allocated once; read in place - no copy).
- 300 replays a model, each on a fresh x (16-input pool, never the previous one) written into the input buffer first,
  each checked against its x's own reference (device routing; the rows must repeat the x's first routing): GLM PCC min
  0.98654, K3 0.99605. Eager soaks (150 calls) unchanged.
- Wall (LoudBox 2x4, slowest chip done): GLM replay 1731 us (1699 back to back) vs eager 3856 us on the input buffer
  (4.9-5.3 ms with the x copy from another tensor), eager device sum 1816; K3 3196 (3144) vs 5278 (6.5 ms), device sum
  3242. Traced is at the device time; the host left is the input write (host -> device x: +260 us GLM, +435 us K3), gone
  when the block's x comes from attention in the same trace.

### GLM's chain: the router fin sums the sources' partials as they land
- The chain, per router core (GLM, dev 6 - the chip that binds: its last source is the last-launched chip's
  partial): last arrival -> the fin's compute 23.6 us -> top-K 21 us. Ablations (timing only, picks wrong): sigmoid
  off -> 12.4 us (the fp32 sigmoid ~1.25 us a tile, 9 tiles), the sources' adds off -> 16.9 (3 x 9 tile adds ~6.7 us).
- Now: a word a source slot (the ports' fused increment to arr base + 16 x slot, re-armed by the router NCRISC that
  saw it: one increment a call), the NCRISC posting each slot as it lands (ARRB, CB 4 - was unused); the fin sums every
  arrival but the last into its HOLD block in place (fp32: unpack to DEST, SFPU add, packed back; a PACK -> UNPACK
  sync between arrivals) and only the last in the passes with the sigmoid. SE-RFARR: a mark at the last arrival.
- GLM: the fin after the last arrival 23.6 -> 19.2 us; head Galaxy-projected 164-168 -> 161-162 (chain 161-162 binds,
  SE 148-149). K3: chain 378 -> 354 (SE 384 binds). Picks unchanged (head test worst chip GLM 99.45-99.77%, K3
  99.65-99.69%).
- Order: the landing order changes the fp32 sum's association between runs - a near tie's bits can move: 600 traced
  GLM replays, one replay's rows differed from its x's first in a weight's last bit, no pick. MoEOptions.deterministic
  (RouterSpec.ordered: the slots in slot order, bit-identical to before) keeps the sum fixed - and the chain at 169.
  The traced test now checks such a replay against its own routing's reference (picks may differ in <= 4 tokens).
- Soak: eager 300 GLM / 100 K3, traced 600 GLM / 100 K3; mock Galaxy 2, protocol model 16. Traced replay unchanged
  within noise (GLM 1732 us, K3 3171).
- Tried, not kept: a router-local sigmoid at the keys' precision (exp without the inf / NaN guards, 2 or 1 Newton
  steps for the reciprocal): the fin after the last arrival 19.2 -> 18.1 / 17.5 us, picks unchanged - <= 1.7 us, inside
  the projection's run-to-run spread (chain 161-163). What is left after the last arrival: the fin ~19 us (sigmoid ~11)
  and the top-K ~21 (floor 14 for both).

### The fused K3 tail, step A: norm + latent up + the shared expert's add in one program (tt/tail_engine.py)
- Today's tail at chunk 1280 (LoudBox, slowest chip): latent all-gather 83 us, rms norm + latent up 163 (2 programs),
  add 17 = ~263 us, 4 programs. Step A replaces the last three with one program reading the gathered z from DRAM (the
  all-gather stays for now - step B lands it in the engine's L1).
- Folds: the norm weight g into W' = diag(g) W_up (upload); 1 / rms per row applied to the matmul's fp32 sums
  (out = diag(s) (z @ W') + se): the matmul never waits for whole rows, only s does. s from partials along Hr: each core
  squares its K share (SFPU, fp32) and sums it along the row at HiFi4 ((z * z) @ ones); a row's partials combined on
  every core, rsqrt on the SFPU.
- Layout (12 x 10 grid here, the last column left for step B's ports): GX x GY = 11 x 10; a core row owns a chunk's row
  tile (r = cy % 5) and half the output columns (h = cy // 5; column_split of 28 over 11: 3 or 2 a core); W' resident
  (each half column reads its slice once, K ranges multicast down it, NOC0); z through a 3-slot ring a core with FREE
  credits (a core frees a chunk once its output is packed), each core's K share read into a per-chunk landing region and
  row-multicast (NOC1). All program semaphores (every producer on the chip).
- Bring-up fixes: (1) CB straddle - 2-wide slices pushed into 3-page FIFOs (ACC; SQ's short last group): chunk 1 was
  garbage; every FIFO now sized to a multiple of each core's block and checked at build with se_engine3's _ring_pages.
  (2) the packer's L1-accumulate flag left on from the partial's pack into the next group's squares (prefix sums: sum of
  squares 1.3-5.5x). (3) HiFi2 truncation biased the squares' sums 0.8% low: SFPU squares + HiFi4 row sums (0.01%).
  (4) NoC stalls, most of the time: up to ~20 concurrent row multicasts a row plus cross-row ones (a column hop then the
  row) took 20-70 us each - every multicast now row-local, a row's shares one at a time in column order (the chunk's
  arrival count as the token: ~1 us a share), the 4 KB partials as unicasts. 421 -> 344 -> 186 -> 133-145 us.
- Correct: PCC vs fp32 torch 0.99992-0.99997 (8 chips), 300-call soak (fresh inputs, scales 1..300) min 0.999915.
- Where the 133 us goes (chip 0): W' from DRAM 3-47 us (1088-B pages, ~150 GB/s: chunk 0 waits for it); then a chunk
  every ~17-18 us = squares 3.4 + matmul 8.4 (the 3-wide cores at HiFi2 peak) + s 2.8 + out 2 in series on the compute,
  plus the partials' wait. Next: (a) squares accumulated in DEST, one 11-tile matmul for the combine; (b) W' in large
  per-core runs, chunk 0's matmul per K range as it lands; (c) step B, the gather straight into L1.
- (a) the squares accumulated elementwise in DEST (one fp32 E tile a core, no per-tile matmul) and the row's E tiles
  combined by one HiFi4 matmul with ones accumulated over them (no SFPU copy + adds): squares 3.4 -> 0.5 us a chunk, s
  2.8 -> 0.6. (b) W' bank-sharded: a core's K range of its slice one contiguous region of one bank, one burst (median
  load 14 -> 6.6 us); W' pushed to the compute range by range in K order, chunk 0's matmul waiting per range.
  End to end about unchanged (chips end 122-132 us, device time ~146 with the chips' skew): what paces a chunk now is
  the matmul (11.2 us: unpack-bound - a k step unpacks 1 A + 3 bf8 W tiles for 3 tile matmuls; HiFi2 math alone 8.4)
  and chunk 0's start (~35-40 us: W' plus chunk 0's z out of DRAM). Soak 300 calls, min PCC 0.99991.
- Tried, not kept: (i) the share multicasts flushed instead of acked - the matmul 8.4 -> 11.2 a chunk at the time
  (shares piling into L1 under the unpacker; later the matmul is 11.2 either way); (ii) one owner a row a chunk reading
  the whole row tile (224 KB) into its ring slot and one multicast: chunk 0's start 35-40 -> 60 us (ten owners' DRAM
  reads crowd W'), later chunks no wait. Its first version hung: the chunk-3 owner's BRISC waited for chunk 0's slot to
  be freed before loading its se tiles, which chunk 0's output needs (triage: that BRISC in the FREE wait, the others in
  CB_SE reserves) - fixed by loading se(c) before own(c + NSLOT); reverted for (ii)'s start.
- K split between the halves (the engine rewritten): core row r takes K tiles 0-55 of row tile r, row r + 5 tiles 56-111,
  both over all 56 output columns (column_split over 11: 5 or 6 a core); partners (same column, other half) swap the
  fp32 partials of each other's finished columns + their half's sum of squares (XSEND -> XRECV, a 16 KB column unicast
  a chunk), each finishing ~half its columns. Each z tile now has one destination core (no duplicate reads; step B's
  gather lands each K tile once), each A tile is reused over 5-6 W tiles: the matmul is math-bound - 11.2 -> 8.3 us a
  chunk. z's ring a slot a chunk (4: no FREE credits). Semaphores folded to 12 (the 16 limit): W' ranges a bit each in
  one word (1 << r added), the partner's exchanges one cumulative count.
- LoudBox (12 x 10 grid, 110 engine cores): device time 120-135 us (slowest chip; chip 0's kernels end ~104), chunk
  cadence ~15 us = matmul 8.3 + finish (s, partner add, x s, + se: ~20 tile ops) 3.6 + squares / s 2. PCC 0.99993+,
  300-call soak min 0.999910.
- Step B: the latent all-gather in the tail's program, landing straight in the engine's L1. Ports on the column right
  of the engine (fabric_all_gather's _plan with an explicit placement down that column - its placement probe needs a
  row of 4 free cores; balanced ring: the opposite block half each way): a port (direction d, link l) carries K half l
  of its blocks (own first, then the ones it relays) into the next chip's engine rows l RT .., share by share (row tile,
  column: its K tiles), packets of 3 tiles; a share's last packet a fused write + increment of the owner's landed word
  of the block's chunk position there. Each chip's chunk order: own, then by hop distance (ties: the previous chip).
  Relays read a share from this chip's owner once its landed word says so. The fence: a port's ready (a global
  semaphore, re-armed by its waiter) from the port that writes into it, sent at program start.
- Bring-up: (1) the relay's NoC-read copy of the landed word polled without invalidate_l1_cache: a stale cached zero.
  (2) the landed word had two consumers (the engine's share multicast, a relaying port) and the engine's re-arm wiped
  it before the relay saw it: hang (DPRINT on the ports: every sender's first share sent, every relay waiting on its
  first share). Now the landed words only count up (every landing of every call) and each consumer waits for >= its
  own call count (an L1 word it bumps at program start: the engine NCRISC's after the landed words, the port readers'
  in port_epoch) - no re-arm to race, trace-safe. (3) one packet a barrier / flush: ~9 GB/s a link (228 us); the
  ports now move groups of 8 packets (one read barrier, one flush; a relay waiting mid-group pushes the group early,
  padded, its packet count in a 16 B CB) - 148 us.
- LoudBox: the tail is one program, 148 us device time (slowest chip) vs all-gather 83 + norm / up 163 + add 17 = 263
  in 4 programs; gather-bound on the line (chip 3's last block at ~110 us; link floor for the line's busiest hop,
  3 half-blocks of 573 KB: 73 us). 300-call soak min PCC 0.999913.
- The exchange's tilize folded in: the tail takes the combine's row-major output (self.out, a 7 KB row a page) - each
  engine core reads its K share of its row tile's 32 own rows (32 segments of 320-384 B) and tilizes them (5-6 tiles,
  tilize_block) into its landing slot 0; the NCRISC marks slot 0 landed; the ports read their own block from the
  owners like a relayed one. Tiles stay on the wire (a remote row's K half is needed tiled by every core of its row:
  row packets would mean 320 B packets or a whole row tile tilized on one core per chunk, ~7 us on the path). 155 us
  (148 before; run-to-run +- 10), 200-call soak min PCC 0.999914.

### The fused K3 tail in MoEBlock.forward (MoEOptions.tail: auto | fused | ops)
- FusedTail (with the fused head, 640 rows a mesh row: K3 at chunk 1280 on the 2x4 = the Galaxy's per-chip load):
  the exchange hands the combine's row-major output (routed_prepped(tilize=False): self.out, persistent) and the tail
  engine runs the gather, the tilize of the own rows, norm, up, + the shared expert -> [S/SP, H/TP]. W_up's column
  slices with the norm weight folded uploaded at build; the engine built before the tail's warm-up (its persistent
  words with the tail's one-time L1).
- Per stage, LoudBox device time (slowest chip, median of 5), K3 chunk 1280, same session:
  ops tail:   head 538.9 | arena_zero 5.1 | dispatch 135.9 | experts_combine 2303.6 | tilize 7.8 | tail_ag 82.8 |
              latent_up 163.1 (2) | add 17.3 = 3255 us, 10 programs
  fused tail: head 532.5 | arena_zero 5.1 | dispatch 135.1 | experts_combine 2286.5 | tail 152.6 = 3112 us,
              6 programs (5 traced: no x copy)
  Traced replay 3171-3202 -> 3051-3073 us. Soaks: eager 300 (PCC 0.99624), traced 300 (min 0.99604, routing
  identical every replay); GLM-Flash unchanged (tail ops: no latent), soaked 100 + 100.
- CPU protocol model (tests/test_tail_protocol_model.py): ports with _schedule's line / balanced-ring schedules, the
  ready fence, shares as 1-2 packets (the last increments), monotonic landed words with per-consumer call counts, one
  in-order FIFO a directed link, 3 calls back to back per chip, random interleavings, G 2 / 3 / 4 / 8: no deadlock, no
  packet into a chip before it started that call, every landing at the chunk position expecting its block, every
  share landed exactly once a call; without the ready wait the fence check trips (43 pass). Mock Galaxy (8 x 4,
  FABRIC_2D_TORUS_X): the tail's ring plan (the opposite block half each way), its chunk order, the ports on column
  11, every chip's program built and enqueued (test_head_mock_galaxy: 3 pass with the head's two).
- Galaxy projection of the tail (TP ring of 4): a link direction carries 1.5 half-blocks (860 KB, ~36 us at 2 x 23.6
  GB/s for the two halves), so every remote block lands before the engine reaches it - compute-bound: chunk 0's start
  (~30 us, the slowest chip on the LoudBox: W' + its own rows tilized and row-multicast) + 4 x the compute-bound chunk
  cadence (~14.5 us: matmul 8.3 + finish 3.5 + squares / s / partner ~2.5) = ~88 us, vs the ops tail ~229 (all-gather
  on the ring ~41, norm / up 163, add 17, tilize 8). Block (Galaxy-projected): ~2043 -> ~1860 us.

### The arena's per-call zeroing folded into the dispatch (fused-head path)
- What l1_zero zeroed (sparse_combine_rf zero_runs + the writers' words): 51 cores a chip, 12-18 KB each - 4 of them the
  dispatch's port cores (their RISCs taken). Now kernels of the dispatch's program (l1_zero_kernels): each non-port
  core zeroes its runs (stores, then NoC copies of its first zeroed block), the port cores' runs by other cells (NoC
  writes from their zeroed block), then counts in (+1) on every dispatch port of its chip. A port sender sends no END
  marker before the count reached its call count x the kernels (a persistent counter that only grows, the call count
  in another word - no re-arm): a chip that has not zeroed holds every chip's dispatch from completing (the END(GMAX)
  wave), so no combine writes into its L1 before - the old program boundary's ordering, the rows' sends overlapping
  the zeroing. Only on routed_prepped's path (routed() with a shared expert dispatches before the arena is taken -
  there the zero program stays); K3_ZGATE=0 (probe) the old program.
- Device time noisy (K3 dispatch 140 / 148 with vs 140 / 135 + 5 without); traced, a win: K3 replay 3041-3045 vs
  3048-3055 us (back to back 2989-2996 vs 2995-2998). Programs a layer: K3 6 -> 5 (4 traced), GLM 8 -> 7 (6 traced).
  Soaks: K3 eager 300 + traced 300, GLM 200 + 200.

### Chunk scaling: chunk 2048 (1024 rows a mesh row = a Galaxy chip at chunk 8192) vs 1280 (640 = Galaxy 5120)
- What does not fit at 1024 rows: (1) the fused head - its engine needs one chunk of GY x MT = 32 row tiles for the
  fused reduce-scatter; GY 10 doesn't divide 32 and at GY 8 x MT 4 a pass holds 2 fp32 tiles, the gate / up pass split
  fails (K3: 8 tiles into 5 passes; GLM: 2 pairs into 2) - a re-plan, not an L1 limit; (2) the fused tail - 8 row tiles
  a chunk, two K halves: 16 core rows (10): it needs 2 row tiles a core (not built); its arena then ~1.55 MB at a ring
  slot a chunk (> 1.4), ~1.15-1.37 MB at 2-3 slots with FREE credits. The combine (S 256) fits. So 2048 runs the ops
  head and tail; 1280 below both ways (ops: the like-for-like ratio). Uniform routing (random router weights). The
  ops path's taps now keep a device copy (keep_taps "device"): it traces too.
- LoudBox device time (slowest chip, median of 5), ops head + ops tail, and the ratio 2048 / 1280:
  K3 (448 experts):  head 1121 -> 1580 (1.41) | dispatch 138 -> 197 (1.43) | experts_combine 2280 -> 2374 (1.04) |
                     tail 271 -> 373 (1.38) | block 3814 -> 4531 (1.19); traced replay (back to back) 3678 -> 4375
                     (1.19); fused 1280: 3137 device, 2991 traced
  GLM (288):         head 494 -> 676 (1.37) | dispatch 144 -> 206 (1.43) | experts_combine 1339 -> 1305 (0.97) |
                     tail 71 -> 95 (1.34) | block 2052 -> 2286 (1.11); traced 1868 -> 2154 (1.15); fused 1280 1693
- At the Galaxy's experts per chip (K3 224 = 28 a chip, GLM 64 = 8):
  K3:   head 1082 -> 1541 (1.42) | dispatch 133 -> 192 (1.44) | experts_combine 1276 -> 1414 (1.11) | tail 270 -> 373
        (1.38) | block 2767 -> 3526 (1.27); fused 1280: 2109
  GLM:  head 454 -> 629 (1.39) | dispatch 146 -> 211 (1.44) | experts_combine 466 -> 576 (1.24) | tail 70 -> 96 (1.38) |
        block 1141 -> 1518 (1.33); fused 1280: 934
- Galaxy-projected (the Galaxy-load runs; modelled: the TP collectives on the ring of 4 at half the line's time - x_ag,
  logits_rs, shared_rs, latent_a2a, tail_ag / tail_a2a; the dispatch x 1.12 for the 8 x 4 mesh's hops, the earlier
  model's 155 / 139; the fused head from galaxy_projection, the fused tail ~88):
  K3 ops:  head 859 -> 1205 (1.40) | dispatch 149 -> 215 (1.44) | experts 1281 -> 1419 (1.11) | tail 229 -> 311 (1.36) |
           block 2519 -> 3150 (1.25); fused 1280: head ~388, dispatch 162, experts 1288, tail ~88 = ~1925
  GLM ops: head 334 -> 462 (1.38) | dispatch 164 -> 236 (1.44) | experts 471 -> 581 (1.23) | tail 45 -> 64 (1.44) |
           block 1014 -> 1344 (1.33); fused 1280: head ~165, dispatch 167, experts 435, tail 43 = ~810

### Generalizing the fused head / tail - step 1: ring / torus eligibility
- FusedHead.fits refused any wrapped axis ("the LoudBox's lines, 2 links") though HeadSE3 / TailEngine carry the ring
  schedules (protocol models, mock builds); now it requires only 2 links (the exchange's ports and the tail's K halves
  are a link each) - lines or rings on either axis. FusedTail follows the head.
- The planner's / combine's accumulator count was a LoudBox constant (FusedHead.A 12 / 16): on the torus every chip
  has 8 dispatch ports (4 on the line's ends... ), the combine planned one fewer (11 / 15) and the head's planner
  disagreed. Now planned once in MoEBlock from the cells the experts' flat plan leaves free less the combine's ports
  (tt_k3_token_split_moe.planned_accumulators, the combine_cells / combine_port_cells helpers its own build uses), the
  head built with it and the exchange's combine pinned to it (ports-apart mode; asserted) - the LoudBox's 12 / 16 again.
- Host-only robustness for the mock: the dispatch's port placement falls back to the first free cell when the
  Ethernet probe reports nothing (no execution); flat_drisc_ensure skips its senders' ready wait on a mock cluster.
- tests/test_block_mock_galaxy.py (8 x 4, FABRIC_2D_TORUS_XY, TT_METAL_ENABLE_BLACKHOLE_DRAM_PROGRAMMABLE_CORES=1 so
  the experts' DRISC layout holds as on the Galaxy): MoEBlock with head / tail "auto" at chunk 5120 (640 rows a mesh
  row) - GLM-Flash (288 experts, 9 a chip): fused head; K3 (896, 28 a chip): fused head + fused tail; a forward's
  programs built and enqueued on all 32 chips. 2 pass; the head / tail mock builds 3 pass.
- LoudBox 2x4 line unchanged: accumulators 16 / 12, PCC K3 0.99610 / GLM 0.98668, traced back to back 2997 / 1695 us.

### Step 2 (tail): rows as a planned parameter
- The tail engine maps the call's row tiles (t = c RT + R in chunk-position order) round-robin over its PAIRS = GY / 2
  row pairs: pair t % PAIRS, step t / PAIRS. RT = PAIRS (640 rows a mesh row, TP 4): a step a chunk - today's mapping;
  1024 rows (RT 8): 7 / 7 / 6 / 6 / 6 steps. W' K ranges by pair. Per step a landing slot, a ring slot and an E slot
  (NSLOT: a slot a step where L1 has room, else fewer with the row's FREE credits - the steps done by every core of
  the row - before a ring slot's share multicast and before an E slot's unicasts). The per-step counters (landed,
  shares in, E tiles in) moved from program semaphores (16 max) to persistent count-up words compared with the call
  count (as the landed words were): no re-arm. Own rows: one tilize per own step. Ports walk a block's row tiles to
  (pair, step) on the receiving chip.
- 640: 4 steps a pair, 4 slots, arena 1199 KB; 159.5 us (148-158 before: neutral within run-to-run). 1024: 4 slots
  (credited), arena 1295 KB; 212 us standalone. 200-call soaks: min PCC 0.999914 / 0.999883.
- The fused tail no longer needs the fused head (both heads land the shared expert's output width split): MoEOptions
  .tail auto picks it for any latent model with 2 links; the ops path's exchange returns the combine's row-major rows
  (routed(tilize=False)). K3 block at chunk 2048 (ops head, fused tail): tail 212.8 us (ops 373), block 4531 -> 4412
  device, traced 4375 -> 4219 us; chunk 1280 unchanged (tail 156.6, traced 2993). Soaks 150 + 150 traced at both.
- Protocol models: the gather model at RT 2 / 5 / 8 (PAIRS 2 / 5 / 5), line + ring, G 2-8; a new engine-row model
  (GX cores a half, partners, NSLOT ring / E slots with FREE credits, the share token, the exchange, the compute's
  CB room): steps 4-7 at 2-4 slots, no deadlock, no slot overwritten in use; without the credits it trips (111 pass).
  Mock Galaxy: the tail engine built + enqueued at 640 and 1024 rows.

### The device profiler under trace (test_moe_block_traced[timing=device])
- The profiler tags a replay's programs (ProgramExecutionUID.trace_id / trace_id_counter): per replay and chip every
  program shows with its device duration and start / end. timing "device" (a test parameter; needs the profiler at
  process start) replays N times, reads once, groups by trace_id_counter.
- K3 chunk 1280 (fused head + tail): 4 programs a replay a chip, every replay seen at 10 / 40 / 100 replays a read
  (no overflow at 100). Per program (slowest chip, median): head 526 | dispatch 142.7 | experts_combine 2217 | tail
  157.7 us; sum 2972-2976, span (first start to last end) 2974-2978: ~2 us of gaps inside a replay; replay wall
  2991-2998 (back to back): ~20 us of dispatch around a replay. GLM: head 282 | dispatch 135.5 | experts_combine 1229 |
  tilize 8.2 | add 10.7 | tail_a2a 38.6; sum 1669-1670, span 1671-1672, wall 1693-1698.
- vs the eager per-stage numbers: head / dispatch / tail within ~1-10 us (the eager head has the x copy); the experts
  program 80-97 us shorter traced (K3 2217 vs 2299, GLM 1229 vs 1326) - the eager figure includes its launch skew
  across 120 cores (kernel duration = first core's start to last core's end). Traced device times are the better per-
  program numbers from here on. Profiler overhead on the fused kernels: none measurable (replay wall with it 2991-2998
  vs 2991-2997 without). The device CSV carries the trace id / counter columns; tt-perf-report not tried.

### Fused head at 1024 rows: what one chunk can't hold (step 2, head half)
- One chunk of 32 row tiles = GY 8 x MT 4. Two engine fixes (no change at 640: same code path, no define):
  dummy gate / up passes (MT 4 = one pair a pass; a column with fewer pairs repeats its first pair, the h past its
  pairs unread) and W-own as a ring of 5 block slots when the arena would not fit otherwise (one chunk: every block is
  used once a call; the DRAM layout unchanged) - K3 W-own 442 -> 85 KB.
- x chain alone at 1024 (parts test): GLM 267.9 us PCC 0.99978, K3 553.1 us PCC 0.99973.
- The full head (RS + router + latent + routing + prep) does not fit: K3 2016 KB at KB 7 (~1576 with every overlay),
  GLM 1452 at KB 1 (576 of it arrival zones sized for every slot). Per-core state scales with MT -> the chunked head
  (NCH 2, chunk m = block m of every owner) is the plan.
- 640 check after the two fixes (traced, b2b): GLM 1693 us PCC 0.98654, K3 2998 PCC 0.99604 (before 1695 / 2993-2997).

### Chunked fused head (NCH > 1), stage 1: x chain + down + SE reduce-scatter + latent (no router yet)
- Chunk m = block m of every owner (GROW(m, lr): an owner's rows stay contiguous - dispatch / combine / tail
  unchanged). The one-chunk body in a loop; every per-call counter counts up over the chunks (thresholds m x a
  chunk's count + the chunk's): RS arrival counters, port mailbox targets, latent words / counter, FINC pops.
- Reuse behind credits, not doubled: RS arrival zones - a core's NCRISC, before chunk m + 1's first slot, waits the
  compute's pop of chunk m's zone and +1s the zone-free word of the port facing that upstream; the port forwards a
  fabric atomic inc to the upstream's port (its zone-credit word), which waits >= m before chunk m's units. FINB -
  the ports' readers +1 a FINB-free word on each core they read after a chunk; the NCRISC hands the compute a page.
  Arrival-zone CB pointers kept at the zone's start: the NCRISC pads a chunk to the whole zone, the compute pops it.
- The latent's LAT + LATZ a chunk each (36 + 12 KB a chunk on K3: no zone reused in a call, no credits); dispatch
  prep's units chunk-major, the chunk's prep op runs its share.
- Two bugs found on the way: (1) the compute's chunk m + 1 gate / up ran with the RS fins' unpack formats (garbage,
  std 43x) - reconfig to (W, x) at each chunk's start; (2) OUTF straddle: a fin of 22 tiles left OUTF's pointer at
  6, the next chunk's 8-tile block straddled the end (pack_tile does not wrap) - OUTF blocks now end at multiples of 8.
- L1 at 1024 rows (GY 8 x MT 2 x 2 chunks): W-own ring (5 slots, re-read a chunk) when resident W does not fit,
  h-all 2 slots else 1, HOLD over ACC (compute-only both; chunk m's RS runs between the two gate / ups), no
  HOLD / FINB over XIN (the next chunk's x rounds land in XIN while the ports read FINB). K3 1274 KB (ring, h-all 1),
  GLM 1248 (resident, h-all 2).
- Parts test (LoudBox 2x4, eager us a call): 1024 rows GLM x chain 244.8, + RS 364.7 (PCC 0.99986); K3 + RS 781.4,
  + latent 843.2 (PCC 0.99984, latent 0.99993). Soaks 1000 calls each, fresh inputs, no hang. 640 unchanged: parts
  GLM 287.1 / K3 504.3, block traced b2b GLM 1695 / K3 2997 (PCC 0.98654 / 0.99604).
- Router chain: still one chunk (asserted). Its zones (K3 3 x 28 fp32 tiles a router core) cannot double in L1 and
  its sends are multi-hop - next: a credit from the owner's router core back to each source's (through the ports'
  logits stream), which also frees the source port's sub-block buffer.

### Chunked fused head, stage 2: router, routing exchange + planner, dispatch prep - the full head at NCH 2
- Router: not chunked. Its cores own a row tile each of all the call's owner rows (RB = rows / 32 / TP, 8 at 1024),
  as at one chunk - zones and port sub-blocks are a call's, nothing reused, no credits, and the chain is as short as
  at 640 (each core one row tile). More than the mover column's free cells: the row under the engine (GY 8 < 10
  rows) - W_r by a row multicast there (Mcast1D PerRow). On the engine (GLM) the logits phase covers every chunk's
  rows at the start (SE_MTR = MT x NCH rows: ETC x MTR <= 8 DEST tiles). Costs cells: the planner only picks a
  layout whose router cores fit (the column, or a free row). Instead of the per-chunk router with credits: same
  result, fewer protocol pieces. K3 at 896 experts (Galaxy) needs the ports' logits sub-blocks for 4 router cores a
  link: 3 x 4 x 28 fp32 tiles = 1.34 MB - check on the mock Galaxy (next).
- Hang 1 (K3 1024 full head, first call): the routing planner's instances - 2 x 8 router cores = 16 (A 16, EPC 56)
  - looped on garbage counts. Cause: RASend writes the call's parity / the source into a packet header's words 14 /
  15; words 0 .. M - 1 are the slices' counts - M 15 / 16 overwrote them (A/B: M 12 passes, 15 hangs). M <= 14 now
  (asserted). Pre-existing: at 640 M was <= 10.
- Hang 2 (K3 1024 full head, soak, after 4-100 back-to-back calls; triage: one chip's NOC1 stuck in write barriers,
  the rest waiting on its routing packets; once triage itself died - NOC1 wedged). A/B, one variable each: no prep
  400 pass; prep with 16-B W reads (NOWL) 400 pass; GLM with the W ring forced 600 pass; watcher row 4 vs 9, router
  row 8 vs 9, planner M 12: all hang; W reads gated behind the core's own prep writes: hangs. Every chunk's prep units
  in the last chunk's prep op (as at one chunk: after the last pass, all W read): 1000 + 1000 calls pass. Root cause
  not pinned (the chunk-0 prep op waiting on latent arrivals with chunk 1's W re-reads under it); kept as the design.
- Parts test 1024 rows, full head (RS, router, latent, routing exchange + planner, prep), eager a call: GLM 431.4 us,
  K3 886.7 us; plan / route words / dispatch rows bit-exact vs the host model; routing picks GLM 99.66%, K3 99.76%.
  Soaks 1000 calls each, fresh inputs.

### Chunked head, stage 3: block integration, router placement, protocol model
- MoEBlock: FusedHead at planned rows - head_layouts(rows, TP) lists (GY, MT, NCH) (whole owner blocks a chunk, MT <=
  2, fewest chunks, most core rows, the router cores' cells), HeadSE3 takes the first whose arena fits (an allocator
  OOM counts as a non-fit). chunk 2048 picks GY 8 x MT 2 x 2 chunks, head "auto" = fused (and K3's tail). The block
  logs the resolved head (engine cores, MT, chunks, router placement).
- Typed MoEOptions.router: auto | engine | router_cores -> RouterSpec.on_engine. auto = router_placement: the head
  ends at max(SE path, chain); router cores: 2 x 32 G tokens x E x H / G FLOP a core at 1.3 TFLOP/s (the Galaxy's K3
  E 896: 316 us) + 150 us of chain; engine: ~3 x its FLOPs on the SE path, the chain from there. K3 E 448 at 640 ->
  router cores (389 vs 423 us est.), K3 E 896 at 640 -> engine (440 vs 466), GLM -> engine, K3 at 1024 -> router
  cores (the SE path binds). auto takes the engine only at the preferred layout, else the router cores.
- K3 router on the engine now builds: the logits rounds past two reuse their two buffers behind receiver acks (+1 on
  the row's x lead / the column's W lead once the compute popped the round), K chunk the largest whose region fits
  past the RS zones (K3 E 896: 7 tiles, 8 rounds); DEST row groups (ETC x a group's rows <= 8). Not at 1024 rows
  on the engine: the ports' logits sub-blocks for 4 router cores a link (1.34 MB) - the engine writes them itself.
- Router cores at E 896 and 1024 rows: the ports' logits sub-blocks a window (NS slots a unit; core i's slot i % NS,
  the port +1s core i + NS's word once i's left) - the words per (direction, unit): one per unit only (the first try)
  let a bwd unit's credit satisfy a fwd one - a hang, fixed.
- LoudBox parts (no routing exchange, eager): K3 E 896 head 552.9 us at 640 (router cores) / 570.3 (engine); 1024:
  958 (router cores); E 448: 494.6 / 860.4. Soak 300 at E 896 1024. Mock Galaxy (8 x 4 torus): GLM 5120 / 8192
  auto (engine), K3 (896) 5120 auto (engine) / engine / router cores, 8192 auto (8 router cores) - 6 pass.
- Traced device time (timing="device", LoudBox 2x4, programs of a replay, slowest chip, median):
  K3 2048 fully fused: head 915 | dispatch 204 | experts 2298 | tail 218 = 3546-3550 us device, 3563 b2b (ops head +
  fused tail was 4412 / 4219). GLM 2048: head 408 | dispatch 200 | experts 1244 | 11.5 | 20.5 | tail 55.6 = 1904,
  1928 b2b (ops 2286). At 1280: K3 2974 / 2996, GLM 1672 / 1695 (unchanged).
- Protocol model (test_rs_protocol_model._run_chunks): NCH 2 / 3, line + ring, G 2 / 4 / 8, the zone and FINB
  credits, values tagged with their chunk. It found a real race: every core with an arrivals list signalled zone-free,
  but a core with only "nothing comes" slots runs ahead (its compute does not wait) - two of its signals counted for
  a slower core's one, a credit early, a zone slot overwritten before read (ring of 4, rows scattered, NCH 3). Now only
  a core something lands on signals (kernel: nfmax / nbmax; ports: zneed / zexp count those) - with that, a credit m
  needs every receiver's m signals (each is bounded by the credits so far). Negative tests: no zone credit, no FINB
  credit - caught. 1000 + 1000 soak calls after the fix.
- Open: the K3 NOC1 wedge with the prep op in every chunk (worklog above) - root cause not pinned; a traffic-only
  probe later.

### Ring of 4 on the LoudBox (one 2x2 square as a 1x4 mesh) - first runs
- p150_x4_ring_1x4 (LINE, RING; channels 2) + FABRIC_2D_TORUS_X, TT_VISIBLE_DEVICES of a square (2,3,6,7 / 0,1,4,5);
  p150_x4_line_1x4 the A/B. Proxy models: k3_ring4 (K3, 112 experts, router at E 896 folded 8:1 - RouterConfig.fold,
  a structured bias: one eligible candidate a group, distinct experts, uniform), k3_g224 (the same per-chip load on
  the 2x4: 224 experts, fold 4).
- Hang: the latent all-to-all on the ring - far-side ties went by p's parity; half the chips injected toward the
  neighbour the fabric does not route through. Now XGather3.ring_dir: shortest, ties as get_eth_forwarding_direction
  (latent and router logits). ARENA_MAX 1400 -> 1344 KB (K3 on the ring took KB 14 + the W ring, 1388 KB: below the
  port cores' static CBs).
- Ring numbers: tail engine K3 640 136.8 us (ring schedule; 159.5 on the line); head parts K3 E 448 x chain 364.0,
  + RS 413.0, + latent 447.5 (probe off) / latent on after the tie fix.
- k3_g224 on the 2x4 (traced, device): head 636-640 | dispatch 147-152 | experts 1214-1216 | tail 155-158 = 2083 us,
  b2b 2105 (Galaxy K3: 650 | 188 | 1421 | 151).
- Open: the experts + combine program at SP 1 hangs (ring: first call; line: in back-to-back replays; both squares;
  Tensix readers too - MoEOptions.experts_drisc off); the same per-chip load on the 2x4 passes.
- MoEOptions.experts_drisc (typed A/B); fewer planner instances than router cores handled (no instance there).

### Ring tail (ring of 4, K3 640 rows): what it waits on, first fixes
- Zones (TP-RUNIT / TP-SUNIT / TP-RWAIT on the ports, TL-RMREAD / TL-WGATE added): MM 8.3 us a step (~80% HiFi2
  peak, 4 steps a core); the gather's own-block sends start only after the own rows' tilize (their DRAM reads: 32
  640-B strided segments a core, 13-26 us behind the W' bursts), then run ~17-19 GB/s a link (80% of 23.6); the far
  block's relayed half lands ~70-85 us; the W' load (6.8 MB a chip) 30-80 us on NOC0 with the gather's reads and the
  half-column multicasts.
- Fixes: (1) every core's own-row reads before any W' load (a lead count + a multicast go): reads 4.8 us median,
  tilized <= 9 us; A/B 146.9 / 143.3 -> 129.1 / 134.9 us. (2) W' loads in K order a wave at a time (core p after
  its half column's range p - 1): 140.3 / 133.8 -> 124.1 / 121.1. (OUT behind the next step and batched landed-word
  polls: neutral, kept.) Ring tail 640: ~121-125 us (median of 5, slowest chip), 1024: 180 us; soak 400 calls each.
- Next: W' bandwidth (DRISC job / bank readers ~500 GB/s vs ~150 now), then per-step overhead (S + OUT + SQ ~4-5 us).

### Ring tail: the gather's send rate, fused increments, traced timing
- Fabric bench (zz_scratch_ring_wire.py, 1x4 TORUS_X, edge e0, 2 links, scatter over 55 receiving cores): every edge
  ~23.7 GB/s at 6 KB packets. A packet costs at least ~218 ns: 2 KB packets 9.4 GB/s, 4 KB 18.7, 6 KB 23.7. A fused
  write+inc with flush is ~2 packets (the receiver flushes its writes before the inc): every packet fused 11.8, every
  2nd 18.1, every 8th 23.7. A separate header-only atomic costs the same as a fused one (12.1 every packet).
  Bench gotcha: more than 12 packet headers on a RISC hangs by design (PacketHeaderPool exhaustion loops forever).
- Tail ports: the sender spent ~0.7 us of every ~2.3 us group waiting for the reader. With two staging groups the
  reader issues group X only once X - 2 was sent, so the next count waits on that issue. Three groups (PSTAGE 3 PG
  RP): the own block goes 3.3-42 us instead of 3.3-49.
- Landed increments in groups (TailEngine land_group, default 3). The fused inc goes on the last packet of every WG
  columns' shares, into that group's last core (the watcher). Its NCRISC pump counts the group's other cores landed;
  the flushed inc vouches for the earlier packets on the link. Ports end ~4 us sooner at 11 (whole row), but the
  engine sees a row later. Traced, ring 640, two batches of 20: lg1 127/129, lg2 121/126, lg3 127/118, lg11 130/124
  us - even within noise. Kept 3: fused ~1 packet in 6, in the free regime of the bench.
- Traced tail timing: test_tail_engine captures one call and replays it (replay_device_times). Eager medians
  swung 114-165 us with launch skew.
- What binds now (zones, 640): own rows tilized <= 9 us; first own packets ~12-15 us; own block ~26 us at ~22 GB/s;
  the far half lands 51-61 us. Both neighbours' row r land on pair r together, ~5.5 us a row: pair 4's steps 1-3
  all start after ~50 us. Each step's MM is ~10 us and OUT ~3.5 us, serial: (1, 6) ends 112 us.
- Next: spread the pairs' arrivals (bwd ports walk the rows in reverse), then step length.
- Tail fidelity (MoEDtypes.tail_fidelity, typed; default HiFi2). LoFi tail-alone PCC 0.99981-0.99990 vs HiFi2
  0.99990-0.99998. The K3 block (k3_g224, 1280) PCC 0.99611-0.99626 either way, routing identical. Traced time
  within noise, so LoFi buys nothing while the steps wait on data.
- Share chain: each column's share sets its own word (sa[s][c] = call count), via a set-multicast right behind the
  data multicast. Same NoC and VC, so they arrive in order: no write barrier per share. The chain's next column
  goes on that word. The NCRISC's write barriers pump while waiting, so a hop no longer waits behind an E-tile or OUT
  barrier. Hop ~0.8-1.1 us (was 1-2). Traced 640 117-120, 1024 172-179: neutral.
- Probes: no token at all (concurrent row multicasts) 164 / 234 us - the token stays. The matmul pushed share by
  share, with the squares after it, 130 / 190: S then waits for the whole row's E tiles on the critical path
  (patch kept in moe_block_runs/progressive_mm.patch).
- Landed-vs-share marks: the own step's chain runs 0.8 us a hop with everything landed. Neighbour steps are bound
  by the chain on one side (landing 0.4 us a column) and by landing on the other (1.2 us a column). Pair 4 (each
  chunk's last row) gets its steps 1 and 2 at ~62 us, then 3 serial steps of MM 10 + S 1 + OUT 3.5-4 (OUT waits on
  the partner's partials) end at ~113.
- Profiler gotcha: zones hash name + file + line into 16 bits, and TL-S at chlkc_pack 166 collided with
  PROFILER-DRAM-PUSH (TT_THROW at the read). moe_block_runs/zonehash.py finds such collisions.

### Head on the ring of 4 (K3 E 896 fold 8)
- test_head_se3 on the ring failed its planner check on the wrap pair only (chips 0 and 3: plan words 255 / 263, a
  0xFF port id, then token_index / table). The test's RoutingSpec had no torus, so port_luts gave no wrap ports, and
  plan_reference got no wrap. Both fixed in the test; the block already passes torus. Bit-exact on all 4 chips.
- Traced head mode in test_head_se3 (replay_device_times; batches of 4 replays fit the profiler's buffers).
- Ring head: 480 us traced (479.8 / 480.8; eager 505-510); the router on its own cores 570.
- One replay, slow chip: router MM on the engine 0-48, gate / up 48-324 (~275 us), chain (RPLAN) ends ~276-294,
  down + RS 320-476 (127-156 us). Down's compute ~74 us. The RS is 1.72 MB a link: ~73 us at 23.6 GB/s, ~90 at the
  audit's ~19. The latent all-to-all is ~215 KB a link (<= 5 us to gain).
- RS on the ring is not link-bound. Sender probe (SE_PROBE=RSSTAT, extended locally): ~249 packets a port in 90-118
  us (~13 GB/s a link). Waits: reader 4-23 us, fabric slot 5-14, batch flushes 7-20, stream polls 5-34; the rest
  ~230 ns a packet of NoC write into the EDM. A/Bs: poll every 4th packet 476 / 481 us, three staging batches
  474 / 487, vs 480 - the waits move, the end doesn't (paced by the engine's down + reduce chain). Not kept. The
  la_send lever is <= 5 us.
- Gate / up (K3, ring, traced): 36.4 GF a chip (28.2 gate / up + 8.2 latent down passes). The heaviest core has 12
  tile columns (124 / 11), 5376 tile multiplies, 127 us at HiFi2 peak; it measures ~276 us (46%). Probes (gate / up,
  head): none 275 / 480, NOMATH 255 / 450-454, NOX,NOW 235 / 456, NOXWAIT 254 / 447-454 - math and the data / sync
  rounds are each nearly the whole time. Math ~50 cycles a tile multiply (2 x 4, matmul_block a K tile a call); the
  shard-major ACC round trips ~18 us; a round (x 56 KB along the row, W ~60 KB down the column) ~4.8 us.
  Options: pass-major after the first pass (-10-12 us), KB 28 / lighter W handshake (-30-50 on the data side), math
  toward ~36 cycles (-50-70); Galaxy's 12 columns ~-8% for free.
- Correction: the earlier "~50 cycles a tile multiply (2 x 4 sub-blocks, HiFi2), MT 2 / DEST bound" reading of
  gate / up was wrong. zz_scratch_gu_mm.py runs the engine's exact call pattern on one core, x and W resident
  (matmul_block a K tile a call, KB 14, K 224 in fp32 DEST, full sync): 34.3 cycles a tile at HiFi2 (93% of
  peak) for 2x4, 4x2, 1x8, 8x1, 2x2 and 1x4 alike; LoFi 17.3, HiFi4 66.3; bf8 = bf16 W. The LLK is math-bound and
  fine. Gate / up is bound by waits: SE_PROBE=GUWAIT (unpacker cycles in its x / W cb waits, a core): x 46-171 us,
  W 7-75; the heavy columns (13 tiles: 5 pairs + 3 latent; the rest 10-11) wait least.
- K3 on the ring runs KB 7 (KB 14's arena is 1388 KB > 1344, even with the W ring at 5 slots): 4 passes x 32 steps =
  128 x rounds a call. KB 14 with a 4-slot W ring fits (1361 KB) but is slower: 494-500 us.
- Not kept: pass-major after the own shard (480.6 / 483.7, PCC same); x receivers acking ahead (kernel_lib
  SenderPipe send<_, ACK_AHEAD> with an atomic decrement of the acks, ReceiverPipe ack / wait_data, Counter
  data-ready, 4 XIN slots, arena 1369 KB): 479.5-490 us, x waits unchanged (patch in moe_block_runs/x_ack_ahead.patch).
  The ack -> send -> land chain is not what holds x back.
- x rounds by hand, owners sending ahead (persistent count-up words: per-slot landed stamp set behind the data, per
  (owner, slot) acks, 4 XIN slots; receivers ack as slots free, owners send their rounds in order once acked): 490-494
  us traced, x waits down a little, W waits up - not kept (moe_block_runs/x_stream_send_ahead.patch).
- The rounds' fixed cost: SE_PROBE=NOMATH,NOX,NOW,NOXWAIT (no matmul, no x / W data, no shard waits) still takes gate /
  up 165 us (128 rounds, ~1.3 us a round of protocol and loop overhead); NOMATH alone 255, NOMATH,NOXWAIT 254-263,
  NOMATH,NOX 237-255, NOMATH,NOW 230-266. The per-round machinery is the floor, not the data or the multiplies.
- Per-step waits (SE_PROBE=GUWAIT,GUSTEP, the heavy core (1, 2), dev 3): steady steps 1.3-1.9 us, about the math's
  1.42 (56 tile multiplies at 34.3 cycles) - the heavy core is math-bound between stalls. The stalls are x waits
  at the shard boundaries (13 us waiting for shard 1 at step 32) and ~18 us around steps 2-5, plus W's first round
  (13 us). So the gather's arrival is the lever, not the round protocol.
- Logits-vs-x priority on the ring (HeadSE3.logits_priority): x first 453-455 us traced vs logits first 479-483.
  The chain (RPLAN) still ends ~90 us before the SE path (~+20 for the Galaxy's N 32 routing). Now x first for a latent
  model on a ring; lines unchanged. Default ring head 447-454 us, soak 300 calls, mock Galaxy K3 builds.
- Tail lever 1 (each pair's steps ordered by when its rows land - per-chip step tables in the engine and ports,
  own rows first; bwd rows walked last first): traced 640 114-118 / 1024 194-198 us; without the reversal 123-125 /
  175. The base is 117-120 / 172-179: no consistent win. Not kept (moe_block_runs/tail_arrival_order.patch).

### K3 block on the ring of 4 (k3_ring4, chunk 640: the Galaxy's per-chip load), traced
- test_moe_block_traced[k3_ring4-640-device]: 48 replays checked (PCC min 0.99612). Per program, slowest chip,
  median: head 455-457 | dispatch 57-59 | experts + combine 1189-1191 | tail 113-114 us; replay span 1808-1813 us,
  back to back 1832 us wall. Galaxy budget: head 650-655 | dispatch 188 | experts 1421 | tail 151-154.
- The SP 1 hang is intermittent, not structural. Line (p150_x4_line_1x4): HANG in the eager wall loop twice (DRISC
  readers and Tensix readers, tail fused or ops) and once in the b2b replays after 48 checked replays; then the same
  config passed end to end. The wall loop's first call reuses eager call 0's x - same routing as a passing call.
  Triage at each hang: one chip still in the experts program with its flat kernels running (TRISC1 in matmuls,
  packers waiting for free tiles, y writers issuing), the others' flat kernels done, their combine kernels waiting.
  Standalone flat + combine (scratch_flat_combine_rf closeout, K3 Galaxy load) passes on both the line and the ring,
  traced 100 replays (fused 1182-1184 us).

### The SP 1 experts + combine hang: lost ring layouts (fixed)
- Hang words (sparse_combine_rf CombineProbes.hw / SPC_HW=1, read after a hang by tools/triage/dump_combine_hw.py):
  in every repro the line-end chip 0's two combine ports were blocked from the call's start on a row for home 3,
  whose ring layout word read 0. Chip 0's LAYR held homes 1 / 2's layouts; home 3's block was zero on both ports.
  Home 3's ports held them in LAYO with nothing pending - forwarded. Credits were not the issue (all returned).
- One variable at a time: zero gate off (K3_ZGATE=0) still hung. Re-forwarding every layout each 64K loops: 3 x 300
  checked replays clean, and one back-to-back phase took +5.7 ms (a rescued layout). A shadow copy of each forwarded
  block far from LAYR (MISC 384+) went missing with it, and the layouts present at the port kernel's start show they
  can land before the receiving chip's combine starts. With LAYR left out of the per-call zeroing (the port clearing
  it at its end): 3 x 300 clean. So the per-call zeroing on chip 0 ran after home 3's layouts landed: the dispatch's
  END markers do not order every chip's zeroing before another chip's combine forwards.
- Fix: the combine's start barrier (BAR: each port core counts in on every other chip's port cores, then waits for
  N - 1) now always on (TtK3TokenSplitMoE; it was only on with the shared expert in the combine). Nothing crosses
  into another chip's arena before every chip's combine started, i.e. after its zeroing.
- Validation: line 3 x 300 checked replays + eager + b2b clean; ring 1000 checked replays clean, per program head
  455-458 | dispatch 57-58 | experts 1189 | tail 112-116 us (the barrier hides under the experts); 2x4 k3_g224
  traced clean; mock Galaxy 6 / 6 build.
- Open (new): one 2x4 traced run hung in the fused tail on the first call (all chips: the tail engine's compute waiting
  z, its ports waiting a fabric slot / their reader); the next run with the same code passed. Triage in
  moe_block_runs/hang_blk_bar_2x4.log. To narrow in the stress phase.

### Generality: the model x rows x mesh map, fixes (Phase 1) and the device matrix (Phase 2)
- Tests: test_generality_map.py (host-only, mock clusters: 2x4 line, 1x4 ring, Galaxy 4x4 / 8x4; one JSON line a cell in
  generated/moe_block/gmap.jsonl - plan, arenas, tail slots, the error and where) and test_moe_block_matrix.py (device,
  tiers quick / nightly / soak; models x rows x routing scenarios, eager + traced replays checked vs the host model on
  the device's routing). Both fold a model past 64 experts a chip (the flat op's limit) as the device does.
- GLM-5.3 1024 rows (fused head NCH 4, router cores) hung in the warm-up: the FFREE CB (the ports' FINB-free credits)
  had 2 pages, and a core no port reads (fnr 0) credits every chunk up front - NCH - 1 = 3 - so its NCRISC blocked on
  the reserve inside its poll loop and stopped serving the W stream (trisc0 waiting WIN on every core). Sized
  max(2, NCH - 1). 2x4 traced: PCC 0.9867, span 2610 us (head 640 | dispatch 234 | experts 1642 | tail ops).
- GLM-5.3 512 / 640: the head's arena clashed with the movers' static CB (x gather port CB 2 x 4 x rp pages: rp 6 with
  GLM-5.3's fabric payload, end 210 KB vs ARENA_MAX's 177 KB allowance). The arena cap is now the free L1 at plan time
  less that CB (se_engine3.arena_cap). 2x4: PCC 0.9866; span 2120 / 2232 us (head 331 / 393).
- The ops fallback ran into the failed fused head's L1 (arena still allocated: its programs held it). The fallback now
  force-frees every L1 tensor of the partly built moe_block.tt objects (MoEBlock._free_partial_l1).
- K3 1280 rows on a 4x4: two planner regions (295 KB each) missed the router's overlay (xr + wr + hold, 576 KB) by 384 B.
  sc / idx / best (dead once the rows are made, like the bias) moved under the overlay; regions past the router's
  buffers when they still don't fit. Device K3 640 / 1024 (router on engine / cores): unchanged, PCC 0.996.
- K3 768 rows on Galaxy: the tail arena (1395 KB, under the 1400 KB cap) plus 36 KB the block keeps later exceeded the
  bank. Cap = free L1 at plan time less 44 KB (KEPT_LATER). Tail slots at 640 / 1024 / 1280 unchanged (4/4, 4/7, 4/8);
  768 5 -> 4.
- Ops router: the fold (top-K id / fold) - K3 at 2048 rows (no fused-head layout) now has a head.
- Top-K 4 / 6 (M3, DeepSeek V4.1): the planner unpacked joint rows by 8 / 12-byte NoC copies - a NoC copy needs source
  and destination equally 16 B aligned, so every other row was garbage; the home loop then indexed want[] with junk
  source chips and looped (M3's first call hung; triage: every planner instance at ++wl[src]). Rows under 16 B now
  unpack by words. K 6 also broke the staging alignment (24 B rows: back on 64 B every lcm(64, 24) / 24 = 8 rows),
  router_topk's row writes (r x 24 B destinations from 64 B slots) and dispatch_prep's landing slots (a row could run
  into the next one). zz_scratch_plan_k.py: the planner at K 4 / 6 / 8, 1 / 32 rows a page, bit-exact on every chip.
- DeepSeek V4.1: the ops router takes sqrtsoftplus (softplus, then sqrt fused into the bias add). The fused head stays
  sigmoid-only (falls back to ops).
- MoEBlock.close() frees the block's device tensors. test_moe_block_switch: K3 at 640 / 1024 / 128 / 2048 / 640 / 256
  / 640 rows, eager and traced, in one process - all PCC 0.996. test_moe_block_first_call: 24 builds + first calls in
  one process, clean; 12 fresh processes (builds 1) clean too - the 2x4 fused-tail first-call hang
  (hang_blk_bar_2x4.log) not reproduced in 36 first calls.
- 11-column boards (the 4xP150 mock): dispatch_prep's copy cores clamped to the grid.

Device matrix, LoudBox 2x4, traced (uniform routing unless noted; PCC min over 34 checks; span = slowest chip's
replay, us; all routings on the device's own routing):

| model | rows a mesh row: PCC / span | head / tail |
|---|---|---|
| K3 (E 896 fold 4, 28 a chip) | 128: .9962 / 1505; 256: .9962 / 1767; 512: .9961 / 1951; 640: .9960 / 2083; 1024: .9961 / 2762; 2048: .9962 / 5554 | fused (2048: ops) / fused |
| K3 640, routings | zipf spread .9961 / 1885; one hot .9960 / 2150; empty experts .9961 / 1650; zero-row chip .9961 / 2092 | fused / fused |
| GLM-5.3-Flash | 128: .9865 / 1394; 256: .9865 / 1446; 512: .9865 / 1612; 640: .9865 / 1676; 1024: .9867 / 1900; 2048: .9869 / 2919 | fused (2048: ops) / ops |
| GLM-5.3-Flash 640, routings | zipf spread .9863 / 1118; one hot .9865 / 1724; empty experts .9847 / 1104; zero-row chip .9865 / 1682 | fused / ops |
| GLM-5.3 | 128: .9864 / 1797; 256: .9866 / 1917; 512: .9866 / 2120; 640: .9866 / 2232; 1024: .9867 / 2608; 2048: .9870 / 4220 | fused (2048: ops) / ops |
| MiniMax M3 | 128: .9885 / 1418; 256: .9887 / 1542; 512: .9889 / 1838; 640: .9889 / 1986; 1024: .9891 / 2395; 2048: does not build | ops / ops |
| DeepSeek V4.1-Flash | 128: .9907 / 2293; 256: .9908 / 2408; 512: .9908 / 2667; 640: .9909 / 2767; 1024: .9909 / 3156; 2048: .9910 / 4110 | ops / ops |
| soak (10k replays, every 25th checked) | K3 640 .9960; K3 1024 .9961; GLM-5.3-Flash 640 .9865 - clean | |

zipf_packed / ties are skipped on the fused head (the planner's rows_cap overflow: the token-dropping policy is
pending). Not built: M3 at 2048 rows (the flat op's arena 1288 KB > the 1256 KB left).

Ring of 4 (LoudBox 1x4, torus X), traced, uniform: every model at 128 - 2048 rows passes (PCC as on the 2x4). Spans,
us: K3 1460 / 1698 / 1724 / 1814 / 2384 / 4530 (128 / 256 / 512 / 640 / 1024 / 2048); GLM-5.3-Flash 1370 / 1393 /
1499 / 1532 / 1720 / 2446; GLM-5.3 (64 experts a chip) 3294 / 3450 / 3602 / 3676 / 3892 / 4798; DeepSeek V4.1-Flash
2227 / 2317 / 2510 / 2581 / 2882 / 3602; M3 1024: 3106, 2048: 3921 (M3 128 - 640 built only after the ops warm-up fix,
below: PCC .9885 - .9886).
- The ops warm-up stranded L1: the logits reduce-scatter's first call places 8 persistent 64 B words under its large
  L1 intermediate (650 KB down on Galaxy); the exchange's arena then only got the L1 under them. The ops warm-up now
  runs with a free 16 KB block at the top held apart by a 64 B spacer (the words land there, the intermediates below).

Generality map (host-only mock, test_generality_map.py), rows 128 - 2048 a mesh row (320 excluded, below):

| mesh | builds | does not build |
|---|---|---|
| 2x4 line (12 x 10) | 39 / 40 | M3 2048 (flat arena) |
| 1x4 ring (an 11-column board: the 4xP150 mock) | 36 / 40 | GLM-5.3 1024 / 1280 / 2048 and M3 2048: combine (9 free cells) |
| Galaxy 4x4 torus X | 36 / 40 | K3 2048 (ops head planner static CBs 1.9 MB); M3 1024 / 1280 / 2048 (flat arena) |
| Galaxy 8x4 torus XY | 31 / 41 | K3 2048, GLM-5.3-Flash 2048 (planner static CBs); GLM-5.3 1280 / 2048 (combine: 31 sources, 25 / 3 landing slots); M3 512+ (flat arena) |

Structural (not fixed here):
- Rows not a multiple of 32 x TP a chip (320 a mesh row: 80 tokens a chip) - needs padded chunks.
- 2048 rows: no fused-head engine layout (row chunks of whole owner blocks) - the ops head; on Galaxy its combine
  planner's static CBs grow past L1 at S 512 a chip (K3, GLM-5.3-Flash on 8x4).
- Combine landing pool: one slot per source chip at the shallowest ring - 31 on the 8x4 - does not fit GLM-5.3's
  large rows at 1280 / 2048; on an 11-column board GLM-5.3's flat layout leaves 9 cells.
- The flat op's arena takes no L1 budget: M3 (H 6144, I 3072) plans 1288 KB at any rows, over the L1 left on the 8x4
  from 512 rows (the exchange now says so at build instead of a CB clash at the first call).
- Fused head: sigmoid router and SiLU / SiTU shared experts only - M3 (swiglu_oai) and DeepSeek V4.1 (sqrtsoftplus,
  clamped SiLU) run the ops head (correct, slower).
- More than 64 experts a chip (the flat op): the router fold proxy (K3 on the LoudBox).
- Packed routings (zipf packed, ties): the planner's rows_cap overflow - the token-dropping policy is pending.
- Ring of 4, routings at 640 rows (packed skipped on the fused head): K3 zipf spread / one hot / empty experts /
  zero-row chip .9961 - .9962; GLM-5.3-Flash .9869 / .9865 / .9847 / .9866. Ring soak K3 640: 10k replays clean
  (PCC min .9962).
- K3 1280 rows (fused head NCH 5, MT 1, 10 router cores) hung in the warm-up: the latent-arrivals credit CB (LRDY) had
  4 pages; the prep (every chunk's units in the last chunk's op) waits for all NCH unpopped, so at NCH 5 the NCRISC
  blocked on its fifth reserve inside its poll loop and stopped the W stream (triage: trisc0 waiting WIN on every
  core, ncrisc at cb_reserve_back(SE_LRDY)) - the FFREE pattern again. Sized 16 x max(4, NCH). Soak 10k replays clean,
  PCC .9962; head 1893 | dispatch 244 | experts 1474 | tail 252 us (the NCH 5 head is slow - perf later). Soak
  GLM-5.3 1024 (NCH 4, the FFREE fix): 10k clean, PCC .9867.
- K3 768 rows (fused head GY 6 x MT 2, 2 chunks) gave PCC .89 / nan (new nightly rows 768 / 1280). zz_scratch_block_ab
  (one stage ops a time): ops head passes, fused head fails (either router placement). test_head_se3: SE alone fine,
  + the fused reduce-scatter wrong on the first row tile of every odd owner block, every chunk; GY 6 / 10 x MT 2 with
  chunks only (owner blocks of CR / G = 3 / 5 row tiles: not a multiple of MT, so a core's two rows straddle two
  owners). The RS fin wrote a core's rows at y0 + r x stride - contiguous - but with chunks (rows owner-major) the
  second row is the next owner's first, rows apart. The protocol model passes (an address map, not a protocol bug).
  Fix: where rows straddle (SE_FIN_ROWS), each (chunk, row)'s own y base in the args. Parts test 768 / 1280 rows
  full head (RS, latent, router, routing + planner, prep) bit-exact / PCC .9998; block K3 768 PCC .9961, span 2389 us;
  K3 640 / 1024 unchanged (2082 / 2765).
- Switching models in one process (test_moe_block_switch[switch-models], one fabric payload: the largest row's, 12352):
  K3 1024 / 640, GLM-5.3-Flash 640, M3 512, DeepSeek V4.1 512, GLM-5.3 1024, K3 1024, eager + traced - all PCC as
  alone. The last K3 1024 first fell back to the ops head: words the earlier blocks' ops left mid-L1 split the free L1,
  the arena cap's probe landed above them and the arena below (into the movers' CBs). The free-L1 estimate is now also
  capped by the largest free block - fused again. K3 switching chunks (128 - 2048): clean.

### Overflow policy (b): worst-case capacity in both planners (user decision)
- Plan tensors (token_index, table; DRAM) at every row a chip can get: worst_rows = N S min(K, EPC) + 32 EPC (the fused
  head's default was 2 S K + 32 EPC - twice the average: zipf packed overflowed it, the dropped rows left zero table
  entries the combine waited on forever).
- Fused head (routing_a2a's planner, run by RoutingStream): an instance now reads only its experts' runs of each
  source's packet (per slice: [base + pre[e0], base + pre[e1]) - a slice is sorted by expert; each read keeps its
  source's 64 B alignment, run pointers in L1), after the headers' totals set its expert range. Its L1 holds its
  entries / rows only: inst_cap = ceil(N S min(K, EPC) / M) + N S (an instance's share of T plus one expert's - the
  boundaries are the expert edges nearest j T / M; an expert takes a token once), rows + 32 EPC. plan_l1 from those;
  build-time asserts (rows_cap >= worst_rows). Guards stay for a broken bound (drop + P_OVER, never an L1 overrun).
- Ops head (combine_plan's multi-core planner): rows_cap was already the worst case; each core's token_index / table
  L1 is now its share (rows_cap / 32 / M tiles) instead of all of rows_cap (12 B a row x the worst case did not fit
  Galaxy's planner cores).
- test_planner_caps.py: host model of both planners' splits over uniform / packed / hot-first / hot-last / alternate
  routings at LoudBox and Galaxy shapes - no instance past its cap; the old cap shown short of packed (not vacuous).
  zz_scratch_plan_k: the ops planner bit-exact at packed routing (K 4 / 6 / 8).
- Device, 2x4, fused head, 640 rows: K3 zipf packed PCC .9961 (experts 3779 us - the hot chip), ties .9961; GLM-5.3-
  Flash .9871 / .9880; uniform unchanged (head 640.6 us = before: the planner's time did not move).
- Ring of 4: K3 zipf packed / ties .9961; GLM-5.3-Flash .9870 / .9879. 2x4 regression at K3 / GLM-5.3 128 - 2048 rows:
  unchanged PCC, heads unchanged (K3 768 / 1024 / 1280: 903 / 1080 / 1910 us). Soak K3 640 zipf packed: 10k replays
  clean, PCC .9961 (experts 3.8 ms: the hot chip).
- Galaxy mock: at worst-case capacity two planner regions a router core no longer fit at K3 1024 / 1280 and
  GLM-5.3-Flash 1024 / 1280 on the 8x4 (428 / 506 KB an instance) - the block fell back to the ops head. RoutingStream
  now takes the most instances (M) whose regions fit: one a core (BRISC only) where two don't - those four fused again.
  The planner is slower at the lower M (not measurable here: Galaxy).

### The K3 chunked-head NOC1 wedge (prep in every chunk): reproduced, narrowed, fixed
- Repro (user: triage on, host risk accepted): test_head_se3, K3 1024 rows full head, SE_PREP_LAST off locally, soak.
  Intermittent - seed 7 with a sync each call failed at call 53 and at call 9; seed 11 at 200, input order 3 at 87,
  async runs in 0-49 / 100-149; one chip each time (1, 5, 6, 3...), NOC1 on its prep rows + 2 port cores; triage empty
  (NOC1 wedged) 6 of 10. The host survived all.
- Narrowing (>= 1200 calls a clean verdict): no prep writes clean; the writes into own L1 clean; posted DRAM writes
  clean; barrier per unit wedges; W window 1 wedges. Codex: NoC1 back-pressure (DRAM write responses + fabric credits);
  its window test came back negative.
- Fix: the prep's writes posted + an end-of-kernel fence (a non-posted 4 B rewrite a DRAM bank, then the barrier).
  test_posted_fence.py proves the fence (0 stale in 1.92 M iterations; 90 % without). Prep every chunk with the fix:
  3 x 1200 calls clean, dispatch rows checked every 200. Default stays prep in the last chunk (heads K3 768 / 1024 /
  1280: last 898 / 1078 / 1909 us, every 893 / 1073 / 1929 - no gain); the fix is perf-neutral (before 903 / 1080 /
  1910). Codex review: approve (fence source valid - PO not rewritten after the last unit; guard on pu).
- The hazard class and the audit: moe_knowledge_dump.md section 17. test_noc1_traffic.py (plain NOC1 write/read mix,
  no fabric) did not reproduce in 400k units a writer - fabric is the likely missing ingredient.
- Traffic-only probe with fabric (test_noc1_traffic fab_l1 / fab_dram: port cores streaming 8 KB packets with fused
  atomics into an X neighbour's L1 / DRAM, on top of the write + read mix): no wedge in 200 programs. Adding NOC1 column
  multicasts gave a stall at program 150-200 on logical column 11's row 0 / 1 cores of every chip in every variant -
  with no DRAM writes, with no lead on column 11 - a probe artifact around the movers column (never a W multicast
  column in the engine); not chased. The 20 audit sites stay unconverted suspects (knowledge dump section 17).
- Soaks, ops-head models (10k replays, every 25th checked): M3 640 PCC .9889, DeepSeek V4.1-Flash 640 .9909; with the
  earlier ones (K3 640 / 1024 / 1280, K3 640 zipf packed, GLM-5.3-Flash 640, GLM-5.3 1024, ring K3 640) every soak
  tier case is clean.

## 2026-10-25: combine shared landing pool (details in FLAT_COMBINE_OVERLAP_WORKLOG.md)
- `landing='shared'` (slot grants from one free list per accumulator) vs the default rings, ports apart, bk 4: at the
  Galaxy's ~11 slots per source the combine is 3-8% faster (fused 1-6%); with the 2x4's deep pools 7-12% slower.
  Rings stay the default. Soaked 20000 calls x 3 cases + 1M calls (GLM-5.3 8K zipf) clean.

## 2026-10-25: the head planner (layout + cost model), compact RS zones, the fused non-latent tail
Trigger: Kimi K2.7 at a Galaxy chip's load on the 2x4 (96 experts = router 384 folded 4, 12 a chip, 640 rows a mesh row):
head 762 us on an 11x4 engine, MT 1, 5 chunks - ~3x its shape's cost (K3's head, 3x its SE + router, is ~640 on the
2x4); the tail as ops (tilize 13 + add 17 + all-to-all 64 = 94 us).

Why the cliff: not the ranking - L1. The rows reduce-scatter (GLM, K2.7: rs_dim -2) kept a P-tile zone slot per arrival
of a core's (half, direction): 3 slots of MT x ceil(NTH / GX) tiles each way on a line of 4 - 504 KB at H 7168, MT 2.
11x10 MT 2 in one chunk was 28 KB over the cap at the smallest K step; the fit-first rule then took 11x4 x 5 chunks,
where every chunk streams all of W down the columns again (the cost model's 449 us of W stream).
- Compact zones (se_rs.RsPlan.compact, the rows scatter's default at G <= 4): a core's MT rows fall in one or two
  blocks, so a direction's arrivals are at most P tiles - each slot at the sum of the core's earlier slots' tiles. The
  ports compute a core's offset from its counts word and a per-unit prefix mask of the downstream's earlier slots'
  blocks; the NCRISC credits a slot's own tiles (threshold | pages << 16); the compute reads at the offsets. 1/3 of the
  zones' L1 on a line, 1/2 on a ring; the width scatter (K3) unchanged byte for byte. Protocol model:
  test_rs_protocol_model (ports' offsets = the cores' slots, inside the zone, never overlapping; negative: a dropped
  prefix block) - 102 pass.
- The router's logits K chunk (on the engine) sat in the zones' region: compact zones shrank it (K2.7 128 rows: KC 28 ->
  14, +8%); it may now grow the arena back by what the zones gave up (KC as before for every measured layout).

The planner (tt/head_plan.py): HeadShape (rows, H, I/TP, TP, E, K, latent, links, ring, grid) -> every layout (engine
GX x GY, MT, NCH, K step, logits on the engine / router cores, zone kind) sized by the engine's own arena code on the
host (SeEngine3(None, ..., arena_max)) against the device's cap (se_engine3.arena_cap a K step), ranked by a cost model:
per chunk max(the busiest column's HiFi2 tiles + W rounds, its W stream down the column, x row multicast) pipelined
with the x gather / RS link time (line: G - 1 shards on the end links, ring (G - 1) / 2), DRAM W (re-read a chunk with a
W ring), a chunk's fixed cost, the routing chain (fixed + expert tiles x K) and the logits (engine phase / router-core
matmul). Fit (tests/test_head_plan.py fit_cost) on 62 traced heads - K3 / K2.7 / GLM-Flash / GLM-5.3 at 128 .. 1280
rows, 2x4 line and ring of 4, P-tile and compact zones, K steps 1 .. 16: median error 6%, worst 23%.
- Inside the model's error (3%) it does not separate layouts (GLM-5.3 512 rows: K step 6 331 us, 12 349, 8 357): there
  the fewest row chunks (every chunk streams all of W again: K2.7 1024 rows 2 chunks 680 vs 4 750; GLM-5.3 1024 607 vs
  638), then the fit-first rule's layout (every measured head's), then the prediction. K3 / GLM-Flash: every layout as
  before; GLM-5.3: 128 .. 640 and 1280 as before.
- Generality: the engine width follows I/TP (columns with no gate / up pair asserted: I_s 1024 on TP 4); with the logits
  on the engine the router cores may take several free rows (1536 / 2048 rows: 12 / 16 router cores). Host sweep
  (test_head_plan_sweep, H 2048 .. 8192 x I_s 1024 .. 8192 x E 256 / 384 / 896 x rows 128 .. 2048, TP 4, 2x4): shapes
  with no fused layout 426 -> 33 of 1260; 6 cliffs left (predicted > 1.6x a neighbour per row): H 6144 / 8192 with
  I_s 8192 and K3-size heads at 2048 rows - L1 (the call's whole x resident: xown 448 - 896 KB) and the W restream a
  chunk. Levers (protocol changes, not done): an x slot ring (two chunks' x resident, credits back to the gather), a
  W-major chunk loop.
- Typed: MoEOptions.head_layout (a HeadLayout) for A/B sweeps (test_head_layout_ab); every build logs the planned
  layout, K step, predicted us and its arena against the device's cap.

The fused non-latent tail (tt/tail_a2a.py, FusedTail without a latent): one program a chip - workers (column slices of
every owned row tile: row-major reads, tilize, + the shared expert's tiles, own block straight to the output, the
others to a DRAM staging buffer, then +1 on the ports' z-ready word of that row tile) and ports (a core per direction x
link, fabric_all_gather's plan: per row tile their runs of each destination's block into its output pages, multi-hop 2D
unicasts, fused arrival increments; a ready fence from every destination first; arrivals awaited at the end).
Protocol model test_tail_a2a_model (no deadlock, every run once a call, never before its destination started the call,
always before it ended, re-armed; negatives: no fence, an arrival short, one z-ready count over the call - the device's
first bug: fast workers' later row tiles stood in for a slow one's, a row tile of garbage). Device (test_tail_a2a,
2x4, traced; ops tail = tilize + add + all_to_all_async_generic):
| model | rows | ops tail us | fused us |
|---|---|---|---|
| K2.7 | 640 | 94 | 65 |
| K2.7 | 1024 | 139 | 97 |
| GLM-5.3 | 640 | 92 | 58 |
| GLM-5.3 | 1024 | 128 | 87 |
| GLM-Flash | 640 | 59 | 46 |
| GLM-Flash | 1024 | 88 | 66 |
(link floor, line of 4: the middle links carry 4 blocks - K2.7 640 rows ~56 us.)
Validation (2026-10-25, bh-lb-09):
- 2x4 matrix, nightly uniform, every model at 128 .. 1280 rows (head us before -> after; PCC unchanged everywhere):
  K3 316 / 521 / 567 / 639 / 901 / 1079 / 1905 -> within +-1% (every layout as before). GLM-Flash: within +-1% (128:
  -4%). GLM-5.3: 1024 638 -> 607, the rest +-1%. K2.7: 640 762 -> 444 (block 1860 -> 1541), 768 554 -> 532, 1024 750 ->
  679; 128 / 512 +3% (inside the model's error). Blocks of the non-latent models 1-2% lower besides (the fused tail).
- Ring of 4 (1x4, torus X), 640 / 1024 rows: K3 457 / 856 (was 458 / 853), GLM-Flash 213 / 335 (213 / 335), GLM-5.3
  312 / 478 (311 / 466), K2.7 333 / 548; tails fused (K2.7 53 us, GLM-5.3 44, GLM-Flash 35 at 640 rows). The first ring
  run hung in the fused tail: the far destination's block was split both ways, and a 2D unicast injected toward the
  neighbour the fabric does not route by is never delivered (XGather3.ring_dir's lesson again) - now every
  destination and every ready goes the fabric's way (protocol model: tie cases).
- After that hang's reset the 2x4 came up with one routing plane (sparse_all_gather_2d: 1 link): a full reset of the
  8 cards (the pool's dirty flag) brought both links back.
- Soak, 2x4, 10000 traced replays, every 25th checked (402 checks), uniform + zipf packed: K2.7 640 / 1024 rows, GLM-5.3
  640 rows - clean (PCC min 0.98577 .. 0.98702, routing agreement >= 0.99).

## 2026-10-25: logits priority measured; the router's fin + top-K split over two cores (router_stream split)
- Ports' priority A/B (MoEOptions.logits_first), head us logits first -> x first: 2x4 K3 640 rows 638 -> 558, 1024 1120
  -> 1083, GLM-5.3 392 -> 383, K2.7 443 -> 433; ring of 4 K3 536 -> 458, 877 -> 852, GLM-5.3 312 -> 296, K2.7 331 -> 326.
  x first is the default on meshes of up to 8 chips (the line's logits-first came from Galaxy projections).
- Chain zones (640 rows, us from the head's start, 2x4): partials landed K3 219-273 / GLM-5.3 74 / K2.7 97-211; fin +
  top-K K3 106-109, K2.7 47-54, GLM-5.3 38; planner done K3 390-464. The Galaxy (coordinator): fin + top-K 114, planner
  done 490 vs the SE path 457 - the chain binds there.
- Split (MoEOptions.router_split, or HeadLayout.split): core (h, j) takes expert half h of row tile j - the engine writes
  each logits tile to its half's core (a count each), the ports carry 2 RB sub-blocks of ET / 2 tiles, the fin and the
  top-K run over half the tiles (global ids in the keys, the whole bias), half 1's best-K page goes to half 0 (its PART +
  a word, zeroed at program start - stale, it made the merge take the last call's page), half 0 merges it as one more
  pair round and makes the rows. The second cores take the column left of the movers: the engine 10 wide.
- Picks bit-equal to one core (deterministic sums: 0 of 24 chips' routing rows differ, K3 and K2.7); agreement as before.
- 640 rows, us (fin + top-K / planner done / head): 2x4 K3 109 -> 65 / 464 -> 412 / 558 -> 542, K2.7 49 -> 32 /
  326 -> 301 / 434 -> 439; ring K3 106 -> 63 / 391 -> 338 / 457 -> 456, K2.7 47 -> 30 / 248 -> 217 / 330 -> 325. The
  narrower engine cost the SE path nothing measurable (K3 its end 540 -> 516 2x4, 455 -> 445 ring). The LoudBox
  heads are SE-path bound: off by default there; on the Galaxy (chain 490 vs SE 457) ~-33 us expected.
- Soak (planner path, router_split): K3 / K2.7 640 rows, 2000 + 500 traced replays each, every 25th checked - clean.

## Host side of an untraced forward: prepared dispatch (2026-10-26, bh-lb-19 2x4, chunk 1280)
Untraced, the K3 g224 block was host-bound: 4310 us wall, 3839 us of host enqueue against ~1958 us of device time
(traced replay). py-spy (native) + cProfile of a warm forward loop (models/demos/moe_block/tests/zz_scratch_host_prof.py,
local only): ~30% the fused head's / tail's _rebind copying every chip's whole ProgramDescriptor (KernelDescriptor copy
and destruction) to rebind CBs to the arena taken again each call; ~31% generic_op's cache-hit slow path re-applying
every runtime arg of every core of every chip (apply_descriptor_runtime_args: per-word lookups and bounds checks);
~15% per-call tensor allocations (four L1 arenas, outputs); ~11% the enqueue itself. On main the descriptor-era
cache-hit mechanism is the program factory's override_runtime_arguments (mesh_device_operation_adapter.hpp); generic_op
has none, so Python-built programs always take the slow path.
- New bringup op ttnn.bringup.prepare_program / run_prepared (ttnn/ttnn/bringup/prepared_program): the descriptor is
  kept; prepare records the runtime-arg slots holding an io tensor's exact address and the CBs an io tensor backs; the
  factory's override_runtime_arguments patches only those on a hit. Cache key = structural hash + every unpatched
  runtime-arg value (a rebuilt descriptor with other frozen values must not hit the old program - the first version
  without it gave NaNs in the tail). Python wrapper ttnn.bringup.prepared_program.generic_op for call sites that keep
  their descriptor object.
- Moved: the fused head (rebuilt only when the arena base or a non-x io address moves; x patched per call), the tail
  (rebuilt when any address moves: its args point inside y / se / out), the token-split planner, dispatch prep and
  zero gate, the sparse all-gather's cached path.
| us, untraced | K3 g224 before / after | GLM-Flash before / after |
|---|---|---|
| host enqueue | 3839 / 1494 | 2757 / 1281 |
| wall (synced each call) | 4310 / 2573 | 3847 / 2438 |
| back to back (no syncs) | 4181 / 1966 | 2839 / 1672 |
Bit-exact block outputs on four inputs (both models); test_moe_block eager PCC 0.9961-0.9962 (K3), traced replays
unchanged (K3 2006-2011, GLM-Flash 1690-1695 us). Left on the host (K3): ~400 us of per-call arena allocations (SE,
tail, combine + its fit check), ~110 us the fused flat + combine op, the rest the enqueues.
- Next round (same day), toward the traced time in a synced forward (what a layer sees with nothing before it):
  host stage marks show everything after the head's launch is enqueued ahead of the device; the loss is the host
  work before the head reaches the device. Done:
  - prepared_program: an io tensor's slots are rewritten only when its first slot does not already hold its address
    (state kept in the program itself - all its slots are written together): ~1350 patch writes a chip on the head
    (every SE core carries the same dozen addresses) become ~12 lookups. Head launch 608 -> 426 us into the forward.
  - SE compute kernels: the per-group block and the RS / prep tables' prefix every core of a group shares go as common
    runtime args (sent once a chip) - the head's runtime-arg payload 39152 -> 29692 words a chip; the kernel indexes
    them through se_arg (indices stay absolute).
  - The fused head's per-call outputs (router rows, dispatch buffer, route words, latent, SE output; DRAM) are made
    once and reused when the block has its fused tail (nothing outlives a forward).
  - mesh_plan / manifest_spec memoized per mesh (~180 us of host a call, behind the device by then).
| us | K3 g224 | GLM-Flash |
|---|---|---|
| synced untraced wall: start of day / prepared / now | 4310 / 2573 / 2284-2306 | 3847 / 2438 / 2007-2029 |
| traced replay | 2006-2012 | 1690-1694 |
| host enqueue now | 920-1021 | 814-834 |
| head reaches the device (host mark) | ~320 | ~315 |
  Left: ~270-330 us before the head starts - its enqueue (~165 us: 116 KB of runtime args a chip, written chip after
  chip by the mesh queue), the SE arena taken again (~95 us: it shares its L1 with the exchange's arena), small items.
  Outputs: bit-exact against the start-of-day path on GLM-Flash; on K3 one input alternates run to run between the
  reference and one row (365) off by 0.0156 in 6 values - two outcomes only (accumulation order of one token's
  contributions), PCC unchanged (0.99624 / 0.99616 / 0.99611).

## GLM-5.3-Flash on the block, whole model (2026-10-26, bh-lb-19 2x4, chunk 5120, real weights)

The model's MoE layers ran on the all-gather path (GLM_EXPERTS_MODE=ag): none of the block / prepared-dispatch work
reached it. Now GLM_EXPERTS_MODE=block (tt/experts_block.py): router, shared expert and routed experts in one block call.
- Residual: GLM's split layout is the block's owners' rows already ([S/8, H] a chip, row-major chip order), so the block
  takes it as is (MoEOptions.residual "owners") instead of the model moving to [S/SP, H/TP]: the ops head gathers rows
  over TP (not width); the fused head (reads the width split) takes x through a TP all-to-all; the tail is the shared
  expert's add (its RS output is the owners' rows) - no tail all-to-all.
- One block for the 42 MoE layers (42 blocks do not fit L1): MoELayer holds a layer's routed experts (FlatRoutedExpert
  from GLM's flat weight cache - the block's layout keeps its key) and a host source for the router / shared expert;
  the fused head lays out each layer's weights at its first call and keeps a prepared program per layer. Other chunks:
  FlatRoutedExpert.replanned (same weights, a plan per capacity; asserted same packing / coordinators).
- Fused head at 2560 rows a mesh row (4x a Galaxy chip): HeadSE3 asserted the legacy layout list (no "router cores on
  the free rows" layout) before using the planner's; the logits ports' words were fixed at 16 (an end chip of the
  TP line sends 3 owners x 10 router cores). Planned engine 11x8, MT 2, 5 chunks, predicted 899 us.
- Clamped SiLU (limit 10) in the SE engine (SE_GLU 3, the flat op's calculate_clamped_silu_glu); GLM_53_FLASH now
  clamped (the checkpoint's swiglu_limit) - it was plain SiLU.
- Hang on the first in-model run: the spec's fabric payload is 8192 B, the combine's packet needs 2 Hr + 32 = 8224.
  Now asserted at build; runs use BRINGUP_FABRIC_PAYLOAD=8256 (ag unchanged at 8256: 9986 tok/s).
- The bring-up profiler did not see ttnn.bringup.run_prepared launches (the exchange's zero / prep / sparse dispatch:
  ~0.5 ms a layer were missing from "device" times); now counted.

| GLM-Flash 2x4, 5120 | ag | block, ops head | block, fused head |
|---|---|---|---|
| MoE a kda_moe layer (router + experts + SE), device | 4.73 ms | ~4.04 ms | ~3.15 ms |
| block alone, traced replay (test_moe_block owners) | - | 3974 us | 3104 us |
| 45-layer chain, traced (last chunk, at 51200) | 507.5 ms | 488.7 ms | 449.7 ms |
| warm 56k prefill (test_perf) | 5.64 s, 9986 tok/s | 5.52 s, 10199 tok/s | 4.99 s, 11287 tok/s |
| last-chunk top1 / top5 | 0.7107 / 0.9051 | 0.6997 / 0.8978 | 0.7064 / 0.9013 |

Ops-head breakdown in the model (us, 2560 rows a mesh row): x row gather 345, router chain ~205 (partition 29, logits 61,
RS 69, top-k 38), SE 403 (gate / up 2 x 111: the clamped activation is not fused into one matmul), SE RS 348, routing
gather 48, planner 73, prep 32, sparse dispatch 459, flat + combine 2071, tilize + add 63. Per MoE layer ~78 MB a chip
crosses the 200G links (gather 15, RS 15, dispatch ~24, combine ~24): on the LoudBox the head's collectives are the
lever - the fused head overlaps them (user: keep the shared expert / router TP-split, no replication).
Open: experts_during_dispatch at 5120 / owners gives PCC 0.951 (wrong output); cold prefill 52 s (each layer builds
its fused-head programs at its first call); the shared expert is bf8 / HiFi2 vs GLM's own bf16 / HiFi4 (top1 -0.4%).
- Follow-ups (same day): the fused head keeps one prepared program for every layer - its kernels take each weight tensor
  (SE w, the router's W / bias / index tiles) at its base address, so a layer switch is the prepared dispatch patching
  those slots (weights out of the rebuild keys) instead of a descriptor build a layer at its first call.
  test_moe_block_layers: two layers of their own random weights alternating on one block (owners and sp_tp, chunk
  5120): every call PCC 0.9869 against its layer's reference, ~0.000 against the other's.
- experts_during_dispatch with x read once (the default since e2ea8cc627b) read x before the dispatch landed (PCC
  0.951): with xdual the helpers read x and their x-ready arg slot held an arena address; now the relay's ready word.
  PCC 0.98694. Traced block (owners, fused head) 3105 -> 3181 us with it: off.

### GLM-Flash on the block: stress, routing over prompts, accuracy (2026-10-26)

- Eager stress: 100 back-to-back 56k prefills (real weights, shared expert at HiFi2 then): no hang, 4.965 - 4.975 s
  each (median 11335 tok/s). Cold first prefill 40 s (was 52 before the shared head program).
- Run-to-run: the block is not bit-reproducible (the ag path is: top1 picks 100% equal over 11 runs): ~89% of the last
  chunk's top-1 picks equal run 1's, top1 vs the text 0.665 - 0.722 (1024 rows). MoEOptions.deterministic (the router
  fin's order) does not change it: the combine's fp32 accumulators sum a token's rows in landing order and 45 layers
  amplify the last bits.
- Accuracy (6-run means of top1, first 1024 rows of the last chunk): shared expert bf8 / HiFi2 0.687, bf16 / HiFi2
  0.683, bf8 / HiFi4 0.719 (ops head), bf16 / HiFi4 0.714 (ops head); fused head bf8 HiFi3 0.708, HiFi4 0.700; ag
  0.705 - the fidelity, not the weights. GLM now runs its shared expert at HiFi3 (11295 tok/s).
- Routing over prompts (zz_scratch_routing_prompts, 56k each, profiled - the real-time profiler's runtime ids are 16
  bits: matched as op id & 0xFFFF in one pass):

| prompt | unique tokens | prefill | tok/s | MoE a chunk (42 layers, busiest chip) | MoE a layer | flat imbalance max |
|---|---|---|---|---|---|---|
| book (canonical) | 7293 | 4.99 s | 11280 | 153.4 ms | 3.14 - 4.74 ms | 1.18 |
| babilong (another book) | 7206 | 4.97 s | 11335 | 151.4 ms | 3.15 - 4.71 ms | 1.08 |
| code (ttnn python) | 4174 | 4.92 s | 11455 | 146.7 ms | 3.09 - 4.39 ms | 1.10 |
| numbers | 288 | 4.98 s | 11307 | 152.3 ms | 3.24 - 4.53 ms | 1.08 |
| one sentence repeated | 11 | 5.09 s | 11066 | 162.4 ms | 3.16 - 5.59 ms | 1.16 |
| random tokens | 47127 | 5.03 s | 11187 | 157.7 ms | 3.25 - 5.14 ms | 1.14 |

  Real text: the MoE share of a chunk moves +-3% (146.7 - 155.5 ms); degenerate inputs +6%. Per layer it is larger:
  L39 3.63 (code) - 5.18 ms (repeat), L23 3.54 - 4.89 (random), L30 3.47 - 4.52; most layers within 0.3 ms. The flat op's
  busiest chip is at most 1.18x the mean chip: EP 8 over 288 experts spreads even skewed routing.
- Traced stress (spec defaults, shared expert HiFi3): 2000 replays of the 45-layer chain (real weights, chunk 5120 at
  51200), 904 s, no hang; 452.1 ms a replay (eager 452.7).

## llk_helper_library merge: KDA fork, KDA projections, and a new flat+combine hang (2026-10-26, bh-lb-19 2x4)

- KDA precise gate re-cut on main's prepare_chunk_recurrence (ttnn/ttnn/bringup/prepare_chunk_recurrence, CHANGELOG):
  worst-head state 0.0269 (main's op 0.0548 fails the 0.05 bound); no perf cost (11442 -> 11482 tok/s).
- KDA input projection: main's ttKDA runs ttnn.linear, so GLM's minimal_matmul schedule had gone unused; back on
  minimal_matmul M4 K8 N8 (1.405 -> 1.169 ms a call, 34 a chunk): 11691 tok/s. Projections at HiFi3 (recurrence HiFi4;
  accuracy equal to HiFi4 to 1e-5): 4.70 s, 11978 tok/s, top1 0.709.
- The 320 KiB L1 path for the chunk terms is out of reach at GLM's geometry: 16 heads x 2560 rows a chip need 682 KiB
  a core, and summarize_chunk_recurrence's static CBs at 16 chunks a group end at 1.17 MB (GLM_KDA_PREP_L1=1 clashes
  by 166 KiB already at the component test's 1024 rows). Gain bound ~0.5 ms a layer of DRAM traffic; would need the
  chunk split into halves or a streaming summarize.
- NEW HANG on the merged tree (not on combine-lean-sender f57131f7774: 40/40 prefills clean, same box): 2 hangs in ~13
  prefills (~1 in 4000 block calls), both chip 2, FlatWithProgram (flat expert + combine_rf). Chip 2's flat expert is
  back-pressured end to end (y writer se6_dw waits for y-ring slots the combine sender drains; h / gate-up / x / DRISC
  grants all behind it); the other chips' flat finished, combine loops spinning; fabric routers fully drained (every
  sender channel at capacity, nothing in flight), links clean. So a credit / layout word is never seen, not stuck in
  the fabric. The bring-up kernels are unchanged apart from include paths: suspect main's kernel-side dataflow / NoC /
  fabric-adapter changes underneath. Triage + logs: scratch triage_stress1.txt, stress_fork_hang1_saved.log.
  The perf numbers above come from runs that did not hang.

### Block re-profiled on the merged tree, with the se_yrm trailer fence (2026-10-27, bh-lb-19)
- Device profiler on main needs TT_METAL_PROFILER_MID_RUN_DUMP=1 and TT_METAL_PROFILER_CPP_POST_PROCESS=1 besides
  TT_METAL_DEVICE_PROFILER=1, else ttnn.get_latest_programs_perf_data() is empty (replay_device_times read 0 programs).
- K3 ring of 4 (k3_ring4, 640, traced, 100 replays, slowest chip): head 455-460 | dispatch 57-58 | experts + combine
  1151-1152 (was 1189-1191) | tail 115-116 us; span 1768-1774 (1808-1813). PCC min 0.99612.
- GLM-5.3-Flash 2x4, 5120, owners residual, traced: x all-to-all 121 | 27 (not identified) | fused head 1084-1088 |
  dispatch 461 | experts + combine 1373-1376 | tilize 26 | add 37 us; device sum 3103 us (3104 before). PCC min 0.98686.
- K3 on a TP line of 4 (k3_ring4 640, p150_x4_line_1x4, no torus; PR 61481 puts TP on the tray half's line), traced:
  head 550-552 | dispatch 83-88 | experts + combine 1219-1227 | tail 145-147 us; span 1901-1904 (ring 1768-1774).
  Head + tail 697 vs 573 on the ring (+22%), block +7%: far less than the 2x the TP collectives' link bytes would
  suggest - the head's engine compute hides part of it. PCC as on the ring.
- K3 at 1280 rows a mesh row (a 4x4 stage's per-chip load at chunk 5120; k3_ring4 1280, 1x4; per-program times from
  the real-time profiler - the device profiler throws "Source location hashes are colliding" at this plan's zone count):
  ring head 1596 | dispatch 99 | experts + combine 1364 | tail 204 us; line 1866 | 150 | 1480 | 250 us. The head is
  3.5x its 640-row time: HeadSE3 plans 11x8, MT 1, NCH 5 - the shared expert's weights no longer stay in L1 next to x
  and stream once per row chunk (planner: w 718, wnoc 1365 us). Speed of light 376 (ring) / 665 (line) us. The window
  column is not a 4x4's (this proxy has 28 experts a chip at 183 rows, a 4x4 56 at 91). Scratch patch for the RT read:
  session scratchpad zz_rt_times.patch (not committed).

## Order-independent combine accumulation: offline calibration (2026-10-27)

Why: the team's main worry is run-to-run non-determinism of experts + combine (fp32 packer L1 accumulation of bf16
rows x routing weight, arrival order varies). Every deterministic option tried so far needed more accumulators or
staging, which made the landing rings shallower and killed perf. Idea: keep add-on-arrival and the 4 B accumulators, but
snap each term (bf16 w x bf16 y, exact in DEST fp32) to a fixed absolute grid q per layer and hidden channel before the
packer adds it. All terms are multiples of q and every partial sum is < 2^24 q, so each fp32 add is exact and the sum is
bit-identical in any order. The routing rows sum to 2.5 (sigmoid, norm_topk_prob), so any partial sum is bounded by
B_c = max_t sum_k |term| with no log2 K factor. The norm in front makes the scale token-independent (a static q works).
Int32 (q 7 bits finer) only if the packer can L1-accumulate Int32 (not confirmed).

Offline on GLM-5.3-Flash (models/demos/glm53_flash_d_p_lb/tests/zz_combine_snap_offline.py, exact weights, golden
ffn_norm + routing): calibrate B_c on s4096 (tokens 0-4095), test on s56320 chunk 10 (tokens 51200-56319). Layers 3, 4,
12, 20, 28, 36, 40, 44:
- B_c is set by token 0/1 (the sink) in almost every layer; channels spread 5-9 bits within a layer (per-channel q
  matters: one q per layer flips 5-10x more outputs).
- Test vs calibration: sum |term| never above 2x B_c (max 2^1.09 at L4, else <= 2^0.71); above 1x on 0-1.8e-4 of
  elements. So h = 1 bit of headroom: no saturation on this test.
- The weight-only hard bound (2.5 x L silu(L) x max_e ||Wd_e[c, :]||_1, swiglu_limit 10) is 5.4-14 bits loose (p50
  8-14): calibration, not a bound.
- fp32 grid, per channel, h = 1: error before the bf16 cast 0.0-1.7% of bf16 rounding; bf16 outputs differ from
  bf16(exact sum) on 0.07-1.8% of elements (1-ulp flips), rel L2 unchanged (1.659e-3 = bf16 rounding). h = 2 doubles the
  flips. Int32 h = 1: 1e-7 to 3e-5 flips, the same order as today's fp32 order dependence (two orders differ on ~1e-6).
- Not checked: bfp4 experts' outputs (gain 1.01, so same magnitudes), other texts, K3 (no weights here). Device cost
  (two SFPU adds a tile on the accumulators) not measured.

### Deterministic block: combine snap + fixed-order router fin (2026-10-27, bh-lb-19)
- Combine: MoEOptions.combine_snap (q, a power of two; sparse_combine_rf_prepare(snap_q=)): after the weighted multiply
  each term goes (v + C) - C on the SFPU (C = 1.5 x 2^23 q, two add_unary / sub_unary a tile) - a multiple of q, so
  every packer fp32 L1 add is exact while partial sums stay < 2^24 q and the sum is the same in any landing order.
  ISA docs: packer FP32 L1 accumulate and SFPMAD are IEEE RNE with denormal flush - the exactness argument holds.
  Guarantee for an element: |term| < 2^22 q and sum |term| < 2^24 q (q 2^-14: 1024 / ~3x GLM's calibrated max 2^8.4).
- Router fin: was in landing order unless MoEOptions.deterministic (slot order). Now always a fixed order by default:
  RouterStream._arrival_order - the predicted landing order (a source's send position - farthest first - then hops),
  passed as reader args; MoEOptions.deterministic default True (False: landing order, A/B only). The SE gate / up's
  shard order was already fixed (x_gather3's watcher raises ranks in a host-fixed order). First run hung at the fused
  head's first call: the routing planner read its args at a hard-coded 4 + G + 2 + 4 - now ab0 + nab + 2 + 4.
- Traced, device profiler, slowest chip (programs in order, us):
| | head | dispatch | experts + combine | tail | bitwise-different replays (same x, same routing) |
| GLM 2x4 5120 before (landing / no snap) | 1084-1086 | 461-463 | 1375-1378 | 26 / 37 | 0 of 34 |
| GLM snap + fixed fin | 1084-1086 | 460-461 | 1376-1378 | 26 / 37 | 0 of 34; 0 of 186 (200 replays) |
| K3 ring4 640 before | 456-458 | 57-58 | 1151-1152 | 113-119 | 4 of 34 (114 el.); 16 of 186 (162 el.) at 200 |
| K3 snap only | 452-459 | 55-58 | 1150-1152 | 111-117 | 0 of 34; 0 of 186 at 200 |
| K3 snap + fixed fin | 455-457 | 56-57 | 1149-1151 | 113-120 | 0 of 34 |
  PCC min unchanged (GLM 0.98686, K3 0.99612). Random weights: q 2^-14 not calibrated for them (PCC unchanged).
- Offline (GLM real weights, zz_combine_snap_offline.py + scratch combine_global_q.py): one global q 2^-14 costs
  0-7% combine rel error over plain fp32 (quiet layers 4 / 12 / 20 most), 2^-13 ~ bf16-rounded terms, 2^-12 +60% in
  quiet layers. main's chain (bf8_b y, bf16 product, bf16 running sum in slot order): 3.4x our error.
- Not yet: combine_snap default (per-model q; K3 real-weight bound unmeasured), model-level run-to-run check (GLM 56k
  twice, every (chunk, layer) compared), an overflow counter in the accumulator.

### Shared expert through the flat routed expert (2026-10-27, one chip, bh-lb-19)
Why: the head's SE engine (se_engine3) runs ~50% of HiFi2 peak with a ~1.3 us a round handshake floor; the flat expert
is a 1D DRAM-streaming design without grid-wide per-step syncs - is it a quicker route for the shared expert?
test_flat_balanced_sweep (new GLM_SWEEP_WDT=bf8), K3's per-chip shared expert: E 1, H 7168, I 1536 (6144 / TP 4), SiTU,
bfp8 weights, MIMO_FL_FIDELITY=HiFi2, pure DRISC defaults:
- M 640: 322 us (48 gate / up + 32 down cores); M 1280: 554 us. HiFi2 math floor on 90 compute cores ~170 / ~340 us:
  53-61% of it. The engine alone at 640 rows (Oct 22-23 probes): 317 us. No win - flat at this M is not DRAM-bound
  (32% / 25% of 512 GB/s) and only 80 cores work on one expert.
- Split along I into "experts" over the same rows (E 4 x I 384, E 2 x I 768): the flat plan has no gate / up split
  for 12 / 24 tile columns (TT_FATAL, flat_routed_expert_plan.cpp:486) - not pursued.
Reading: at this shape both designs sit near half of HiFi2 peak; the shared expert is math-heavy (M 640), the flat
expert is built for small-M, DRAM-bound experts. Levers stay inside the engine (fewer / bigger rounds, x arrival).
- Engine alone re-run (same day, test_se_engine3 k3, 640 rows, MT 2, KB 16, x families 7 + 7, 3 gate / up + 6 down
  passes, HiFi2): 316.8 us (317.7 with GUWAIT), PCC 0.9997 - unchanged since Oct 22-23. Zones, slowest chip, median
  over the 110 cores (us from the first kernel start): SE-XLOAD 0-21 (max 34) | SE-XROUNDS 22-237 | SE-GU 0-238 |
  SE-HX 237-252 | SE-DOWN 238-311 | NCRISC ends 313-316. HiFi2 math on 110 cores: gate / up 93, down 46 -> gate / up at
  39%, down at 64%. GUWAIT (the unpacker's waits inside gate / up, a core): x 66 us median (26-112), W 62 (15-92) -
  ~130 of gate / up's 238 us is waiting for data, x and W about evenly; edge columns (logical 0, 7, 10) wait most on x
  (81-101), the others more on W (60-77). Flat expert at the same shape (above): 322 us.

### SE engine bottom-up (2026-10-27, engine alone, K3 640 rows, MT 2, KB 16, slowest chip)
Why: rebuild the engine from math only, one ingredient at a time, to see where utilization breaks. New timing probes
(se_engine3 _probes): NOXR / NOWR - no x / W rounds at all (the CB pushed as if the round landed); SE_WPF=n - the W
prefetch window.
| stage | probes | us | gate / up end (median / max core) |
| s0 math only | NOXR,NOWR,NOWL | 227 | 131 / 161 |
| s1 + W column multicast from L1 (credits) | NOXR,NOWL | 245 | 139 / 169 |
| s2 + W from DRAM | NOXR | 304 | 210 / 227 |
| s3 + x round handshake (16 B) | NOX | 310 | 232 / 234 |
| s4 + x payload = the engine | - | 312 (317 checked) | 234 / 236 |
| a1 x rounds only (no W rounds) | NOWR,NOWL | 232 | 159 / 166 |
- x is cheap here (+5-8 us); the full run's x waits (GUWAIT 66 us) are x owners blocked by receivers waiting on W.
- Math only is already 227 vs 139 ideal: gate / up waits ~19 us for the x DRAM load (XLOAD) before the first step, the
  compute runs ~49 cycles a tile multiply (microbench 34.3) - ~20% loop overhead - and 48 pairs over 11 columns put 5
  pairs on 4 columns: those finish 30 us later and the h gather (SE-HX 43 us in s0) waits for them.
- The W DRAM stream is the big step (+59). NoC trace (full, ~8 reads a core): 35.1 MB in 858 NCRISC NOC1 reads of
  35-70 KB, issued 6.7-249.5 us: ~144 GB/s. tt-npe on that trace: congestion impact 0% (links are not the problem).
  Reads with the multicast payload 16 B (NOXR,NOW): 296 (reads alone are slow); reads with no W rounds at all
  (NOXR,NOWR: each core's reads straight into WIN): 345 - the cores pull the 35 MB at ~115-145 GB/s.
- Deeper window is worse: WPF 6 / 12: s2 331 / 349, engine 355 / 373 (vs 304 / 312 at 3).
- W reads on NOC0 (SE_WNOC=0): s2 291 (vs 304); the full engine HUNG (NCRISC on NOC0 beside BRISC's NOC0 traffic -
  shared command buffers / counters; opt-in only, default untouched; default re-run 316.8 us, PCC 0.99972).
Reading: ~14 cores read each DRAM bank (bank (c GY + cy) % 8, every core its own region) with 3 blocks in flight -
the DRAM delivers ~145 GB/s of 512, NoC0 or NoC1 alike, and more in flight makes it worse (bank conflicts / row
thrash is the likely cause; the flat expert's one-reader-a-bank pattern gets 98%). Next: W from a reader per bank
(the DRISC / optimal-bank cores) forwarded to the column holders, or W prefetched under the previous op; then the
compute loop's 49-vs-34 cycles and the 5-vs-4 pair imbalance.
- The compute loop: matmul_block (MT 2 x pw 4 = the full-sync fp32 DEST, one K tile a call, 16 calls a K step, init a
  pass, no reinit in the steps). With full-sync DEST the math waits through every pass's GLU (SiTU on the SFPU, pack
  thread) + pack: nothing overlaps the activation. Probe NOGLU: math only 227 -> 202, engine 312 -> 289 (span); with
  the GLU out the math-only gate / up loop is ~7% over the microbench (it was ~20%).
- Half-sync DEST (SE_HALFSYNC=1, A/B, engine-alone path: 4 fp32 tiles a half, 2 x 2 sub-blocks, gate / up 5 passes,
  down 11): gate / up ends 234 -> 205 but down 78 -> 99 (twice the W rounds); span 304. MT 1 (2 row chunks): 422 full
  / 438 half - W multicast twice, out. Half-sync + down K step 16 (SE_KBD=16): 289.8 us measured / 281 span vs 316.8
  / 312 (-27 us, -8.5%), PCC 0.99972, arena 1067 KB (vs 1229); KBD 24: 296; full-sync + KBD 16: 311.
  Not in the head yet: its RS fin (add_tiles blocks <= 8), the router on the engine (ETC x RG <= 8) and the latent
  passes assume 8 DEST tiles.

### SE engine alone: 289.8 -> 249.0 us (2026-10-27, K3 640 rows, MT 2, KB 16, slowest chip, 7-run median)
Starting point: half-sync + KBD 16 289.8 (default 316.8 / 316.9). Every step below is an opt-in switch (defaults
unchanged); PCC 0.99972 in every run. Final set:
SE_BRW=1 SE_BRW_BAR=0 SE_WPD=2 SE_BRW_DN1=1 SE_DN1_AT=0 SE_XPIPE=1 SE_XL=2 SE_HXINC=1 SE_PASSREV=1 SE_KBD=16 (full sync)
-> 249.3 / 246.1 / 248.7 / 252.3 / 247.1 / 249.0 / 250.3 us (median 249.0, span ~242-244).
- Why the W stream was slow (pure-read bench, zz_scratch_dram_mix test_edge_readers, whole grid, 35 MB): NOC0 386-391
  GB/s, NOC1 235-249, independent of page (70 KB vs 4 KB), bank pinning and depth. The engine forced its NCRISC onto
  NOC1 (dm RISCV_1 NOC_1) for the W reads + column multicasts: ~145 GB/s in the engine. Single-packet reads
  (SE_WPKT 4/8/16 KB): no change. Hand trace analysis (scratchpad noc_trace_an.py): reads balanced over the 8 banks
  (~42 in flight each), no hot bank; tt-npe's congestion figure was not used.
- SE_BRW: gate / up W blocks read by the BRISC on its own NOC0 (the NCRISC writes a table of its blocks in stream
  order into a per-core L1 words tensor; the BRISC issues them under transaction ids 1-12, paced to its own x round
  position + SE_WPD - issue-only, so no circular wait; the NCRISC waits on the issued count and polls NOC0's id status
  read-only before multicasting on NOC1). Unpaced (all at once) was worse (368: the 35 MB burst starved the x load
  and then the x row multicasts on NOC0); paced D 4: 278.9.
- SE_HXINC: the non-RS down starts on h columns as they land (as the RS path does): 274.8 with BRW.
- SE_XPIPE (+ SE_XL): x owners load their steps in need order (ids 13-15 by slot), SE_XL positions ahead, and send a
  step once its own load landed (no ports only; with ports the own shard loads first as before): 264.1.
- SE_BRW_DN1: the down's W blocks read by the NCRISC on its own NOC1 (its old prefetch, window 3) while the BRISC
  reads gate / up on NOC0 - both NoCs' read bandwidth: 260.0.
- SE_PASSREV: the narrow gate / up pass first (under the x load, which binds pass 0) and the full-width passes after
  it with only the W stream: 258-260; GUSTEP: passes 1-2 almost wait-free.
- Lookahead sweep: XL 2 / WPD 2 best (XL 3 / WPD 4 258, XL 1 / WPD 2 253, XL 2 / WPD 3 253).
- Tried, worse or no gain: 4 W slots (SE_NWS=4: 273-277); KB 28 full sync (arena 1700 KB); half-sync + KB 28 (286);
  half-sync on any of these (287-300: 2-wide passes make x rounds the bound); down pacing over the gate / up
  (SE_WDD) only moved the NOC0 contention; issuing the lookahead after each round (269-277); delaying the NOC1 down
  reads (SE_DN1_AT 7/14/28: 258-264); family-1 x loads on NOC1 (SE_XN1: 282, an unpaced NOC1 burst); dynamic NoC mode
  (never - removed).
- The head (k3_ring4 640 traced) with SE_BRW: 458-460 vs 456-458 - its gate / up is paced by the x gather, not the W
  stream; the other switches are not tried in the head yet (PASSREV / DN1 apply there; XPIPE / HXINC are no-ops).
- Left in the engine alone (GUSTEP): pass 0 bound by the x load + its W (NOC0 reads), a ~20 us step-0 bubble, the
  non-overlapped GLU (~20 us), the h gather's 11 latency-bound rounds (~22 us), 5-vs-4 pair imbalance.

### K3 head on the ring of 4: where the time goes, ports vs engine (2026-10-27)
Traced head replay (test_head_se3 k3 + RS fused + router + latent + routing, E 896 fold 8; 450-468 us), zones, chip 3:
router logits matmul on the engine 0-52 (~10 us of HiFi3 math: data movement / rounds) | gate / up + latent 52-316
(264 us: KB 7 -> 4 passes x 32 steps = 128 x rounds + 176 W rounds, ~2.06 us a step vs 1.42 of math) | down + RS
315-421 (max core 455), hand-offs 326-391 | routing chain ends ~362 (not critical).
Ports (NoC trace, hand analysis): the 4 port BRISCs write 8 KB packets (+96 B header) into the EDMs on NOC1 - north one
hop, then west along the Ethernet row (y 1), not through the engine. Per port while active: 18-21 GB/s (ports at
(15,2) / (15,3) -> eth (11,1) / (10,1), 5 / 7 hops) and 13-16 GB/s ((15,4) -> (12,1) 6 hops, (15,5) -> (4,1) 15 hops):
hop count does not set the rate. Phases: x gather 40-200 us, RS 360-500 (eager). The gather ends long before gate /
up does; the RS tail is paced by the engine's down / reduce chain (earlier RSSTAT probe: ports waiting, not
sending). DRAM reads ~40 MB a call (~90 GB/s average). The mover column is not what binds the head here.
- Engine-alone switches in the traced block (k3_ring4 640): SE_PASSREV 457-460, SE_BRW + SE_BRW_DN1 463-464, all three
  464 vs 456-458 base - no gain: the head's gate / up is bound by its 128 + 176 rounds at KB 7, not the W stream.
- KB 14 in the head: arena 1388 KB > 1344 (xown 224, wown 297 (ring), win 178, hall 220, acc 164, xin 112, ...).
Next candidates: the router matmul (52 us for ~10 us of math), fewer / lighter rounds in the head's gate / up (L1 for
KB 14: ~44 KB to find), the down tail (106-140 us for ~49 of math).

### The ports' Ethernet-row funnel (2026-10-27): port senders on NOC0 by default
Why: the head's x gather took 50-190 us (140) vs a ~69-86 us floor. NoC trace (traced K3 head replay): the 4 port
BRISCs (mover column, x 15) write the EDMs on NOC1 - north one hop, then west along the Ethernet row (y 1) - so all
four share the row links (15,1)->(12,1): 70-76 GB/s sustained there in the gather and in the RS (~86 raw), the four
ports together never above ~75 GB/s (13-21 GB/s each). Distance did not matter, the shared links did.
Standalone (zz_scratch_port_place.py: fabric_all_gather_py, K3's 640 x 1792 bf16 shard, ring of 4, 2 links, floor 69):
| ports | sender NOC1 | sender NOC0 (roles swapped) |
| one column, right edge (the head) | 138.1 us | 82.0 |
| one column, left edge | 138.4 | 81.7 |
| each under its Ethernet core ("auto") | 83.3 | 91.2 |
Any single column funnels on NOC1; swapping the port cores' roles (sender NCRISC / NOC0: east, then south - each port
its own row toward its own Ethernet column; reader BRISC / NOC1) matches ports under the Ethernet cores, no move.
- Head: x_gather3 PORT_NOC0 (default on; SE_PORT_NOC0=0 the old roles). test_head_se3 K3 traced 449-454 -> 436-438,
  eager 503 -> 484. Block traced: K3 ring4 640 head 452-459 -> 439-443; GLM-5.3-Flash 2x4 5120 fused head 1084-1085 ->
  967-968 (-117 us, -11%; its TP is a line); other programs unchanged; PCC unchanged (0.99612 / 0.98686), replays
  still bit-identical. GLM 56k prefill: 4.88 s / 11544 tok/s (4.93 s / 11421 before), top1 0.8699 / top5 0.980.
- fabric_all_gather_py: the swap by default when a chip's ports share a column (FAG_PORT_NOC0=0/1 forces).
- Also tried: the row leads' own-x load before the router's x (SE_XFIRST): the gather starts at ~9 instead of ~50 us
  but the head does not move (452-455) - gate / up's rounds bind it. SE_PAD=n: blank lines in the engine's kernels
  (a profiler zone-hash collision workaround).

### K3 head engine: 440 -> 390 us traced (2026-10-27, test_head_se3 k3 + RS fused + router + latent + routing, ring of 4)
Why: "engine still seems really slow". Bottom-up in the head (probes, per-step waits, a NoC trace of one call):
- With no x / W data at all (SE_PROBE=NOX,NOW, or NOXWAIT for the gather) gate / up still took 253-258 of 266 us: not
  the payloads, not the gather. GUSTEP: every call stalled 40-60 us at step 2 waiting for x, then ran wait-free.
- NoC trace: the gate / up W prefetch (3 of my blocks in flight a core) issued ~18 MB at once at ~52 us (after the
  router); the block each column needed first landed 10-55 us later (col 7 / 12 at ~108 us) and x round 2 waits for
  every column's step 0. Fix SE_WLA (default 6): a block is issued only once its round is within 6 of the rounds
  landed on the core - the grid's in-flight W is ~6 rounds, not ~30. Head 440 -> 397 (flat over 5-8, 3: 412).
  Gate / up 266 -> 226; the step-2 stall ~15 us.
- KB 14 now fits with a W ring of 4 slots (sizing tries rings 4 and 3): with WLA 390 vs 413 at KB 7 (no gain without
  WLA: the startup stall bound both).
- Router logits on the engine (0-47 us, ~12 us of HiFi3 math): 8 rounds of ~5.8 us; the W_r^T leads (NCRISC, NOC1)
  took ~3 us for a round's 21 reads vs ~0.6 for the x leads - all 11 sat on row 0, and NOC1 brings the DRAM responses
  back along the lead's row. SE_RWL (default): the lead on row NoC x % GY: logits 46 -> 37, head 397 -> 389.
- Tried, not kept: acks of a router round as soon as the compute popped it (leads read round rd + 1 while rd is in
  flight; logits 37 -> 32, head 380) - GLM-5.3's traced replays then picked different experts (291-1216 tokens): a
  race not found by inspection, removed. Latent tiles to the columns with fewer gate / up pairs (busiest 13 -> 12 tiles
  a K tile): 384 vs 380 - each pass's x rounds stream a whole row, so the sum of the passes' widest column (4+4+2+3)
  binds, not a column's total. SE_XFIRST with WLA: 405 (worse).
- Left: gate / up ~220 us vs ~170 (148 math at 34 cycles a tile + ~20 GLU) on the 5-pair columns: ~27 us x waits,
  ~13 W; the down / RS tail ~90-120 us, its end steps ~4.5 us a row (the RS exchange - fabric, ~20 GB/s a port).
- Block (traced): K3 ring4 640 head 439-443 -> 388-395 us, PCC 0.99612; MOE_BITS with MOE_COMBINE_SNAP=-14: 0 of
  186 replays differ (without the snap 4 of 34 - the combine's order, as before). GLM-5.3-Flash 2x4 5120: head 966
  (968 before), PCC 0.98686, 0 of 34 differ.
- SE_LOG=1: the engine logs its resolved layout (KB, W ring, router K chunk, regions).
- Gate / up partials by the packer's L1 accumulation (2026-10-28): each (shard, pass) packs its fp32 partial onto a
  fixed per-pass ACC slot (shard 0 writes, 1 .. G - 2 add in L1); only the last shard copies the sum back into DEST
  (UnpackToDestFp32 - full fp32; a zero-tile add would go through SrcA / SrcB, ~19 bits) - 1 read-back a pass, not
  G - 1, and no per-shard copy_init / reconfig / matmul re-init. ACC shrinks to one slot a pass (no FIFO straddle
  room). K3 head traced 390.4 -> 384.0 us, PCC 0.99985; GLM-5.3-Flash 2x4 5120 head 966 -> 959.5, PCC 0.98686.
- Half-sync DEST in the head (SE_HALFSYNC=1, opt-in): gate / up passes keep their width, each step's MT x pw block
  goes through DEST as <= 4-tile row sub-blocks packed onto the ACC slot with L1 accumulation (the last step reloads
  the sum, then the GLU on that half); router row groups, latent / prep blocks, the RS fin's add blocks and the RS
  plan's pwmax follow dest_tiles(). K3 head traced 397.0 vs 384.0 full sync (PCC same): gate / up 225 vs 223, down /
  RS 93 vs 90 (2-wide down passes). Not the default.
- One-core microbench (zz_scratch_gu_mm.py, x / W resident): 34.2-34.8 cycles a tile at HiFi2 (93%) for 2x4 / 4x2 /
  1x8 / 8x1 / 2x2 / 1x4, and the same with half sync + an L1-accumulated pack every step (2x4 as 1x4 halves 34.5,
  2x2 34.8, 1x4 34.6). The LLK, the unpacker and the L1 accumulation are not the bound.
- Gate / up build-up in the head (traced K3, KB 14: 64 x + 64 W rounds; median core): handshakes only (NOMATH, NOX,
  NOW, NOWREAD, NOXWAIT) 66 us (49 without the GLU - ~0.77 us a step of protocol + CB / DEST / ACC bookkeeping); 16 B
  W reads (NOWL) 69; + the W DRAM reads and shard waits 145; + payloads 182; + math 223. The W stream on NOC1 (~30 MB,
  ~210 GB/s) is the largest piece; the x round is lock-step (a receiver signals ready for round j only on entering
  receive(j), after round j - 1 landed) and the owner's loopback fence waits for all 11 acks. New probe NOWREAD (no W
  DRAM reads at all).

### SE engine round protocol (2026-10-28, K3 head traced, test_head_se3 ring of 4; median core)
Why: gate / up's handshake-only build-up (NOMATH, NOX, NOW, NOWREAD, NOXWAIT, NOGLU) took 49 us for 64 steps (~0.77 us
a step); the x round looked lock-step (a receiver acks round j only on entering receive(j)) with the owner's loopback
fence waiting for all 11 acks. Baseline today: handshake-only gate / up 49.5 us; traced head 386.2 / 386.4 / 386.8 us,
gate / up 224.3, PCC 0.99985.
- x rounds by hand, credit-based (SE_XCR=1, A/B): receivers credit round n to its owner as soon as its XIN slot is
  free (not on reaching round n), the owner multicasts data linked to a count-up stamp (XD[slot], loopback - the owner
  waits on its own stamp, no write barrier), owners send out of order (as soon as their credits are in, before the
  previous round landed on them). Handshake-only: 49.8 (credits, in-order owner) / 49.5 (send-ahead) / 50.8 (the
  same protocol lock-step, SE_XLA=0) / 49.8 (+ the owner's write barrier back) vs 49.5; traced head 388.2 / 384.6
  vs 386.2, gate / up unchanged (223-224). Per-step waits (GUWAIT, GUSTEP) in the handshake-only run: x 0.3-0.45 us a
  step, W ~0 - with or without credits. Not the x protocol.
- Probe NOXR (no x rounds at all) in the handshake-only run: 44.1 us (vs 49.5) - the x rounds are ~5 us of it, the W
  stream paces the rest. (Probe NOWR in the fused head HANGS by itself: its "push as if landed" blocks the NCRISC in
  cb_reserve_back(WIN) while the compute waits for RS arrivals that loop must push - triage: NCRISCs in
  cb_reserve_back, BRISCs past the x rounds. Do not use NOWR in the head.)
- New probe WLOOP (the NCRISC loop's passes and cycles while the first chunk's gate / up W rounds land): 175 passes in
  59 K cycles - ~340 cycles (0.25 us) a pass, ~2.7 passes a W round. Every pass re-ran blk() (4 runtime divisions:
  sps is a runtime arg) in the prefetch scan, the send's skip loop and the credit, plus offset_of's walk at a send.
- SE_WTAB=1: round -> block and my blocks' W-own offsets tabled once (uint16 on the stack): handshake-only 48.8 ->
  41.1 us (285 cycles a pass); traced head 381.5 / 385.5 / 382.0 vs 386.2 / 386.4 / 386.8 (~-4 us), gate / up 224
  (unchanged - see below), PCC same.
- SE_WGATE=1 (skip the RS / latent / prep / y checks while the first chunk's gate / up W rounds land): 147 cycles a
  pass, gate / up 41.3 (no change on top of WTAB); head 380.9. SE_WDEF=1 (the W ready set issued once the block left
  L1 - non-blocking, instead of flush / set / flush): 43.2 handshake-only (no gain). SE_WNB=1 (a send whose DRAM read
  has not landed is skipped this pass instead of blocking the loop in the trid barrier): head 382.9 (no gain).
- Why the real gate / up does not move (GUWAIT / GUSTEP per column, WTAB): the 5-pair columns (logical 0-3) wait x 41-51
  us and W 11-18 us a call, nearly all of it at step 0 (x ~5, W 4-10 - the first W blocks after the router phase),
  step 2 (x 5-11: round 2 waits for every row core's pop of round 0, and the lighter columns wait on W round 0 up to
  16 us) and step 16 (x ~18.5: shard 1 from the gather); between those they are math-bound (x / W ~0). The lighter
  columns' extra x waits (~1.3-1.5 us at the narrow-pass steps) are the row waiting for the 5-pair columns - load
  balance, not protocol. So the head's gate / up is heavy-column math + startup + shard arrival; the round protocol
  is hidden under math there.
- SE_WEARLY=1 (my blocks of the stream's first window issued at kernel start, under the router phase): router logits
  37 -> 48.5 us (the reads crowd NOC1 / the leads' read barrier), gate / up 224 -> 216.5; head 390.0 - worse.
- GLM-5.3-Flash 2x4 5120 traced head: base 957.7, WTAB 952.1, WTAB + XCR 956.0 (PCC 0.98686 each). An earlier GLM base
  run (same default code) timed out in the fused head's build (the test caught it and fell back to the ops head; cores
  15-2 / 15-3 - the gather's port column - did not finish; no triage: the timeout was swallowed); the repeat passed.
  Intermittent - see llk-merge-branch-hang; not reproduced in the 6 later GLM runs.
- Kept (default, no switch): the per-round tables. Deleted: SE_XCR (credit x rounds), SE_WDEF, SE_WNB, SE_WGATE,
  SE_WEARLY (the exploration diff: moe_block_runs/se_round_protocol_explore.patch). Probe WLOOP stays.
  Validation: K3 head traced 382.0 us (base 386.2-386.8), PCC 0.99985. K3 block (k3_ring4 640, traced, same day):
  head 390.6 vs 394.0 at HEAD, dispatch / experts + combine / tail unchanged (55.0 / 1151.3 / 115.2), PCC 0.99612;
  MOE_COMBINE_SNAP=-14 MOE_TRACE_REPLAYS=200: 0 of 186 replays differ (PCC min 0.99611; eager calls bit-identical
  PCCs to the earlier snap run). GLM: head 954.7 (base 957.7), PCC 0.98686, 0 of 34 differ.
Reading: the round protocol is no longer what the K3 head's gate / up waits on - it is the 5-pair columns' math plus
the startup (~16 us: the first W blocks land after the router phase; round 2 waits on every row core's pop of round
0) and the gather's shard arrival (~18.5 us at shard 1). The handshake-only floor is now ~0.64 us a step (41 us).
x-round credits would only pay with more XIN slots (2 now; the arena has 16 KB free at KB 14, a slot is 56 KB).

## Random stress (2026-10-28, bh-lb-09, HEAD 23ea8fa2d1f)
Why: hunt hangs and wrong / non-deterministic results in the block and the fused head on the default build before
the model sees them. One run = one random draw (weights seed, x seed, x scale 0.001-16, x normal / per-row lognormal
scale spread / sink rows (x50, x20, zero rows) / 90% sparse; routing by router bias: uniform, skewed (bias sigma 0.05-3),
K hot experts on one chip, K hot spread, K/2 hot, 50-85% experts never picked). Untracked scratch tests
models/demos/moe_block/tests/zz_scratch_stress_{block,head}.py (block: traced + eager, combine snap 2^-14 on every
block draw, every same-x repeat compared bitwise, output hashes for cross-process checks; head: test_head_se3's
setup + traced replays + same-x bitwise check). Drivers, logs, triage: session scratchpad stress/ (draws.log, NOTES.md).
- 121 runs, ~127 min of device runs; 68 random / long draws: K3 block ring 18 (4 at 1280), K3 block line 8 (3 at 1280),
  GLM-5.3-Flash 2x4 23 (5120 / 2048 / 1280), K3 head ring 12 (3 at 1280 rows) + line 7. Passing runs: ~67K traced
  replays (up to 5000 a run, back to back with a check every 25-100) and ~5.5K eager calls (up to 500 a run, fresh x
  each call). Controls SE_WLA=0, SE_RWL=0, SE_PORT_NOC0=0: 8 runs, clean.
- Determinism (snap on): K3 ring 0 of 2668 same-x repeats differ, GLM 0 of 958, head ring / line 0 of 1117.
  Cross-process: the same draw in separate processes is bit-identical (K3 ring 16 / 16 x, GLM 16 / 16, 8 starts x 2 x,
  head 8 / 8).
- F1 (wrong results, K3 on the TP line): 680 of 844 same-x repeats differ bitwise, every line draw, snap on or off.
  Whole rows (every column on every chip) of tokens owned mostly by line chips 0 / 1; the combine output itself differs
  (tapped before the fused tail): diff rows 0.1-0.27 rel L2 = a whole expert term. Same x every call (NX 1): bit-
  identical. Several wrong rows match "one pick's term computed on the previous call's x row of the same token"
  (PCC 0.98, offline vs the ring's bit-exact output): the experts read dispatch rows that have not landed yet. A
  device sync between the dispatch program and experts + combine removes it (0 of 8, worst row 0.099 as the ring);
  MOE_TAIL=ops or MOE_HEAD=ops also 0 of 8 (timing or the exchange's config). Present at b8212572997 and in the pre-merge
  tt-metal-envclean f57131f7774 (no snap: same 0.1-0.2 whole-row diffs) - old, not the merge. The block test's PCC
  (0.9957 > 0.98) does not see it; per-row error on tiny rows reaches 1500x. MOE_EDD=1 on the line asserts ("dispatch
  phase: port (0, 1) X- 0 has no stream into (0, 0)"). Suspect: the sparse dispatch's END markers / per-link fused
  increments vouch for a row's send, not its landing in the destination's DRAM, on multi-hop line paths.
- F2 (hang, GLM 2x4): the fused head's first call (MoEBlock build) timed out once (d103, 1 of ~18 GLM process starts;
  the coordinator saw one earlier today on the same cores). The block caught it and fell back to the ops head on a
  wedged mesh (eager 0 PCC nan). Triage (scratch triage_d103.txt): head op running on all 8; port-column NCRISCs stuck
  in wait_for_empty_write_slot (EDM full) - x senders on TP positions 1 / 2, router logits senders (LStream) on the two
  line ends - while BRISCs wait in rs_gather / CB space; 15-11 waits x shard counts. Fabric sender channels at 0 free
  slots. No closed software cycle found (codex agrees): credits stop coming back. Not reproduced: same draw x 3, ring
  -> GLM alternation x 3, 8 bare GLM starts. Next: many fresh-process starts per arm with SE_PORT_NOC0=0 vs default.
- F3 (accuracy, not order): GLM PCC < 0.98 with large-magnitude rows - x16 0.97934, sink rows 0.97243, row spread
  0.975-0.979 - identical with snap off.
- F4 (snap range): q = 2^-14 is absolute: x 0.01 -> PCC 0.45 (ring and line, 0.98098 without snap), GLM x 0.01 0.838,
  x 0.5 0.981-0.985; rows with small norm lose their routed terms (worst row rel err 1.0). An under / overflow counter
  in the accumulator would flag it.
- Harness notes: MOE_TAIL=ops traced captures fail ("program not yet in program cache" - the ops tail builds a program
  in the capture), safe-pytest then reports a HANG. Head HXC_ROWS=1280 with the test defaults (MT 2, GY 10): L1 OOM
  (arena 1199 KB a bank, largest free block 1178 KB); at b8212572997 it fails earlier (router cores < row blocks);
  GY 8 / MT 1 (the block's plan) passes at both - not a regression.

### F1 fixed: every dispatch link ends in a ring of the chip it leads to (2026-10-28, bh-lb-09)
Why: the random stress's F1 (K3 on the TP line: same-x replays differ by whole expert terms; stale x rows).
- Mechanism (sparse_all_gather_2d): the dispatch is its own program; a chip's experts start when its port kernels are
  done, and a port is done when every input ring delivered END(GMAX) - END follows the link's rows on the same link,
  so a chip is done only after the rows of every link that leads into one of its rings landed. On a 1-D line the end
  chip's only port has no input (nothing behind it, no other axis): the link from its neighbour's port toward it
  (rows on their last hop) has no continue / turn stream there, so no END - the end chip's dispatch finished on its own
  rows and its experts read x rows still in flight (the previous call's rows at those slots). Mid chips and every 2-D
  mesh (2x4, Galaxy 4x4 / 8x4, rings or lines) always have a port to continue or turn into: covered already. The same
  hole made MOE_EDD=1 on the line assert ("port (0, 1) X- 0 has no stream into (0, 0)").
- Fix: such a link carries a terminal stream - only END(GMAX), staged after its last row, into ring 0 of the receiving
  port of the same link (free: no chip behind it). It gates that port's completion, not its END order (ring_mask bit
  3 + k; no new dependency chain). One wiring helper (_wiring) for the standalone op and dispatch_phase. Checked offline
  on 1x2 / 1x4 / 1x8 / 4x1 lines, 1x4 ring, 2x4, 4x4 (ring x line both ways), 8x4 torus: every link ends in a ring,
  every sender stream matches an input ring; terminal streams only on the 1-D lines (2 a line).
- Result (snap 2^-14, same-x replays bitwise): line 0 of 1199 (before: 222 of 299), line 1280 hot_mix 0 of 399, ring
  0 of 999, GLM-5.3-Flash 2x4 0 of 299; MOE_EDD=1 on the line now runs, 0 of 199. Line worst row rel err 0.129 -> 0.098
  (= ring); PCC min 0.99611 -> 0.99618; ring 0.99619-0.99628.
- Perf (test_moe_block_traced device timing, 100 replays, slowest chip): ring dispatch 55.4 -> 55.3 us, span 1702 ->
  1700; line dispatch 82.3 -> 87.4 (+5: the end chips now wait for their incoming rows), experts + combine 1215 -> 1163,
  span 1863 -> 1865 (+0.1%); GLM 2x4 dispatch 464.8 -> 462.9, span 2981 -> 2987 (experts + combine 1380 -> 1389: the
  combine accumulator's fp32 unpack change landed in the tree between the runs; the 2x4's dispatch wiring is unchanged).
- Also: the dispatch descriptor cache key (dkey) held buffer addresses and options only - an output of another width
  at a freed one's address reused the old program (row bytes, CBs, accessors). Scratch repro on the line: W 3584 then
  3072 at the same address -> every chip wrong; with shape / dtype / layout / page size / memory config in the key, 4 of
  4 calls right.

## Codex review fixes: combine unpack, router weights, prepared-program key, latent_norm, guards (2026-10-28, bh-lb-09)
Why: codex's read-only review of e49692459bd (block + head) found these; verified against the code, fixed one commit
each. Checked against the stress draws: K3 ring4 640 / GLM-5.3-Flash 2x4 5120 with snap -14, 300 replays + 3 eager
calls each, and the head (K3 ring, 300 replays). Before / after outputs were dumped and compared bitwise.
- Combine accumulator (714cbaa98d3): CB_ACC is now UnpackToDestFp32 (the out pass's copy_tile read the fp32 sums as
  TF32). The compiled unpack dst format went from Tf32 to Float32, but 0 bitwise changes on K3 / GLM, snap on and
  off: the TF32 step never changed a bf16 output here. GLM experts + combine 1379-1384 -> 1386-1388 us (+0.4%); K3
  unchanged.
- Router weights (d14d9d1d522): each weight is now the pick's exact fp32 score, which the fin packs back over HOLD. It
  used to be rebuilt as the key's truncated (score + bias) - bias. Picks are bit-identical in every run. Uniform
  routing: 0.6% of weights move by 1 ulp, and PCC is unchanged (head weights PCC 0.99911 is limited by logit
  precision). hot_spread (bias 50): 28-50% of weights change; max |w - host| drops 1.57e-3 -> 0.60e-3 (K3) and
  5.6e-3 -> 2.9e-3 (GLM). test_router_split_equal k3 / k27 is bit-exact. The fin now runs one score tile a pass;
  the traced head is unchanged.
- Prepared-program key (c7bc5d7ac9f): the key now holds each slot's io index (tagged), per-core place and arg count,
  and frozen CB buffers and offsets. Outputs are bit-identical. A scratch check keys apart codex's [a, a, b] /
  [a, b, b] case.
- latent_norm=False (2e1e4d68451): "auto" now takes the ops tail and the summary says why. K3 PCC 0.988 -> 0.994.
- Guards (1f591587d7a): fabric_all_gather_py replays its port protocol on the host and raises on the CB-group /
  inc_every deadlock. It also raises for a page larger than the payload. se_engine3 raises when ceil(NB / GY) > 64
  (trid_of / slot_of), or when the SE_BRW table overflows. Not fixed in the kernels: the reader can't push a part
  group without breaking the CB's group alignment. The fabric_all_gather_ttnn copy of the module is unchanged.
- Perf (K3 ring4 640 traced, device sum, slowest chip, 3 reads each): 1700-1704 us before, 1697-1706 us after. Head
  388-391, dispatch 55, experts + combine 1151, tail 113-116 us: all unchanged.

## GLM-5.3-Flash 2x4 chunk 1280: the fused head builds again; build failures raise (2026-10-28, bh-lb-09)
Why: the stress draws rd006_glm, d201 and d203 (GLM 2x4, chunk 1280) logged a host TT_FATAL in the fused head's build
(kernel_lib/mcast/host/mcast.cpp:66, "rotating sender grid needs multiple senders per group"). `MoEBlock` (head
"auto") then quietly ran the ops head. 2x4 at chunk 1280 is our Galaxy per-chip test shape.
- Cause: a regression from the llk merge (70ba8c8d940). At 640 rows the planner picks KB 16 for GLM: 8 K steps,
  2 a shard over TP 4, so `_subsets(1, 11)` = (1, 1). Each x family (x0 / x1) has one owner column. Mcast1D took a
  rotating sender line of one core and built it as a fixed sender (`rotating()` = more than one sender). The unified
  `ttnn.Mcast` refuses a rotating grid with one sender a group. K3 640 / 1280 and GLM 5120 have families of 2 to 4
  columns, so they never hit it.
- Fix (se_engine3.attach_mcast): a family of one column is `McastFixedSenderConfig(sender_index=column)`, a fixed
  sender a row. That is what Mcast1D built before the merge. The kernel calls are the same (that column always sends,
  so it never calls receive). Families of more than one column are unchanged, so the other shapes build the same
  programs.
- Loud fallback: `MoEOptions.head_fallback` (default False). A fused head that `FusedHead.fits` allows but that fails
  to build now logs an error with the reason and raises. True takes the ops head, as before, with a warning naming the
  option. A shape `FusedHead.fits` refuses still takes the ops head under "auto", and the summary says why. The fused
  tail had no build fallback, so a failure there already raised. Checked on device with the pre-fix x families
  patched in: default raises the TT_FATAL; head_fallback=True builds the ops head with the reason in the summary.
- GLM 2x4 1280 (test_moe_block_traced wall, owners residual, snap -14, 300 replays): fused head 11x10 MT 2 KB 16,
  PCC min 0.98654 (pre-merge 0.98654), 0 of 286 same-x replays differ bitwise. Replay 1666 us wall, 1644 b2b. The ops
  head (MOE_HEAD=ops, same run setup): 1825 / 1799 b2b, PCC 0.98655. So the fused head saves 155 us (-8.6%).
- Unchanged checks (same test, snap -14, 100 replays): K3 ring4 640 PCC 0.99612, 0 of 86 differ, 1717 us b2b. GLM 2x4
  5120 PCC 0.98686, 0 of 86 differ, 3030 us b2b. Both PCCs match the earlier runs. The planner's owners there are
  2 / 4 (the soak logs show no one-column family outside GLM 640), so their mcast configs and programs are unchanged.
- Still open: router_stream's W_r family is a rotating group over the router cores. One router core (RB = CR / TP =
  1) would hit the same check. No planned layout has one today.

## Soak (2026-10-28, bh-lb-09): replay soak, fresh-process starts (SE_PORT_NOC0 A/B), random stress on the fixed tree
Scratch tests zz_scratch_stress_{block,head}.py (now removed from the tree; copies, drivers, logs in the session
scratchpad soak/: results.log, NOTES.md, run.sh / cycle.sh / tally*.py). Profiler off, combine snap 2^-14, safe runner.
- Replay soak, e49692459bd: K3 ring 640 (3000 eager + 100K replays), GLM 2x4 5120 (2000 + 100K), K3 head ring
  (1000 + 100K): no hang, 0 bitwise-differing same-x compares (checked every 500 replays, every eager call); PCC
  0.99612-0.99625 / 0.98684-0.98700 / SE 0.99987-0.99989. K3 line: F1 shows, 193 of 1184 compares differ.
- Same soak on the fixed tree (ea0aaa2ddfa, F1 line-end termination af4a8a89bc8 + review fixes): K3 line 3000 + 100K
  0 of 3184 differ (worst row 0.098 = the ring); ring, GLM, head as above, 0 differ, same PCC ranges. F1 fixed.
- F2 (GLM fused-head first-call hang, ~1 in 18 starts in the morning sweep) did not reproduce: fresh-process starts,
  SE_PORT_NOC0=1 vs 0 interleaved: GLM 5120 back to back 0/100 vs 0/100; GLM right after a K3 ring run 0/52 vs 0/52;
  GLM with a random draw per pair 0/32 vs 0/31; GLM 1280 (fused head since c888fc73612) 0/15 vs 0/15; K3 ring
  0/52 vs 0/52. All GLM 5120 starts bit-identical across processes and arms. With p = 1/18, 0 of ~200 a arm has
  P < 1e-5: the rate today is far lower (95% upper bound ~1.5% a arm) or F2 needs a condition not recreated. The
  A/B cannot separate the arms. d103 hit with a warm kernel cache (no kernel compiled before the hang; its JIT
  misses were the ops-head fallback). The safe runner does not reset cards after a passing run, so "no reset
  between starts" was already the condition throughout.
- Random stress on the fixed tree (19:07-22:47, 22 iterations: a random draw of each of K3 ring / line / head ring /
  head line / GLM, plus the start pairs above): 116 draws, 0 hangs, timeouts or NaN; 0 of 13,298 same-x compares
  differ (line 0 of 4044). PCC failures only the known ones: GLM rowmix / sink x 0.969-0.980 (F3), K3 x 0.01
  0.4535 and x 0.1 sparse 0.975 (F4, snap range). GLM draws at 1280 / 2048 / 5120 all ran the fused head.
- Before c888fc73612 the fused head never built at GLM 2x4 chunk 1280 (host TT_FATAL mcast.cpp:66, rotating
  senders) and the block silently took the ops head (morning d201 / d203 too); now it builds, or raises.

## Fused head generality: a host plan that matches the build, refusals with numbers (2026-10-29, bh-lb-09)
Why: `FusedHead.fits` was a coarse filter (layouts exist, activation / router / links / bf8). Shapes it accepted failed
at build (GLM-Flash 2x4 1280 yesterday; the HEAD mock sweep below shows more), and a refusal said little. Marko: "fits"
must mean "builds", say why otherwise, keep the fast path where it fits.
- `head_plan.fit(shape, arena_max)` = every requirement the build checks, no device: the matmul cores' arena per (layout, K
  step, W ring, row chunks), the RS plan (counts word: TP <= 4, P < 256), the mover column (ports, the watcher, router
  cores - or free rows under the matmul cores), the router cores' arena / logits sub-blocks / runtime args, the routing
  exchange (RoutingStream's own paths on a host ring plan: X / Y senders, forwards, port and watcher buffers, the
  planner regions at the most instances that fit), packet headers. Ranked by the cost model as before. `FusedHead.plan`
  adds the model checks and runs it on the machine (grid, payload, free L1 - measured 1424 KB at the plan, 1383 KB at
  the build: the plan assumes the free L1 less 56 KB less an 8 KB margin) and the combine's real accumulator count
  (planned before the head now). HeadSE3 builds from fit() on the device's own cap (>= the plan's), so an accepted
  shape has a layout there; a build failure after an accepted plan raises HeadBuildError ("a planner gap").
- Refusals: e.g. "L1: the least arena any layout needs is 1431 KB (engine 11x8 MT 1, 10 row chunks, K step 7, router on
  the engine), a core has 1326 KB; tried K steps [1, 2, 4, 7, 14], row chunks [5, 10, 20], W rings off / 5 / 4 / 3;
  ... the largest rows a mesh row that fit: 1280 (chunk 1280)". What L1 forced in an accepted plan is logged against
  the cost model's unlimited-L1 pick ("L1-limited: K step 16 -> 8, row chunks 5 -> 10 ... +462 us").
- Generalized (commit a9d8ec234c7): SE activation constants are kernel defines from the model (SiTU betas, clamp limit)
  and SwiGLU-OAI (MiniMax M3, moe_gpt's swiglu_sfpu.h) is SE_GLU 4; shared weights bf4 / bf8 / bf16 plumbed from
  MoEDtypes; the NCRISC's W-block tables are sized NB / GY + 1 (the "> 64 blocks a core" refusal at small K steps is
  gone, NB <= 1024); one router core (logits on the router cores) reads W_r itself (RT_RW_SOLO - the unified Mcast
  refuses a rotating family of one). Every assert in SeEngine3's sizing names its values.
- Fixed at HEAD (a99fa23e9f8): the head's port cores hold a static packet CB from the L1 base (2 x 4 x rp pages); in the
  one-arena forward the exchange's arena could start below its end - MiniMax M3 (6-page packets, CB end 210304, arena
  at 192000) and K2.7 2x4 / 4x4 (HEAD mock sweep) TT_FATAL'd "static circular buffers clash". The head now runs in its
  own arena when the exchange's would start below `static_top`.
- Chunk padding (4f1b2d272f2): a chunk that is not whole tiles a chip runs at the next that is (zero rows at the end of
  each mesh row / chip, sliced off the output). One ttnn.pad where the zeros are whole tiles past the last; through
  row-major where they fall inside the last tile (ttnn.pad's tile path throws there, then segfaults). K3 ring 600,
  K3 2x4 2000, GLM-Flash 2x4 2000 (sp_tp) / 1000 (owners): PCC as unpadded, bitwise stable; pad + slice ~25-45 us.
- Checked (snap 2^-14, traced, every replay vs the host on the device's routing): K3 ring / line 640 / 1280, GLM-Flash
  2x4 5120 / 2560 / 1280 - layouts, PCC, 0 bitwise-differing same-x replays as before; heads within noise (K3 ring 640
  interleaved A/B: HEAD 392.1 / 389.2 / 391.1 us, new 390.4 / 391.7 / 390.2). MiniMax M3 ring 640 now fused: PCC
  0.98874 (its usual), head 264.5 us.
- Harness notes: the device profiler throws "Source location hashes are colliding" (16-bit FNV of zone name, file,
  line) as kernels change lines: zone_src_locations.log keeps stale entries (pruned of ../chlkc_ zones before each run)
  and two live zones can collide (SE-RPWAIT vs SE-GU at one config) - the scratch harness shifts the SE compute's lines
  (GEN_ZSHIFT) when it happens. Not a code issue.

## The fused head's DRAM fallback (4d93bf0ce6c, 2026-10-29, bh-lb-09)
Why: at large rows a mesh row no resident layout fits L1 - x own holds every chunk's x (K3 2560 rows: 560-1120 KB a
core). Marko: fall back to DRAM only where needed, keep the resident path where it fits.
- Only when no resident layout fits (MoEOptions.head_dram_fallback, default on): x all-gathered over TP into a [rows, H]
  DRAM buffer ahead of the program; x own a ring of 2 chunk slots reloaded from it (chunk m + 2 into chunk m's slot at
  the end of chunk m, behind its rounds and - no latent - its dispatch prep: XFREE credits); the ports carry no x but
  keep the call fence; router cores may take a rectangle of the free rows. The plan logs it with its cost ("x from DRAM
  (the fallback: L1 ...)").
- K3 2x4 chunk 5120 (2560 rows a mesh row - the quarter Galaxy; resident: 1455 KB > 1332): replay 6491 us vs the ops
  head 6734, PCC 0.99626; K3 ring 2560: 5063 vs 5392; DeepSeek-V4.1 2x4 5120: 3998 vs 4450. Soak: 300 eager + 20000
  replays each (K3 2x4 5120, ring / line 2560), PCC 0.9962-0.9963, 0 of 228 same-x replays differ, no hang.
- Refused, with the reason: K3 past 3200 rows a mesh row (LATZ and the router cores grow with the rows). GLM-5.3-Flash
  2x4 10240 / ring 4096 stop in the combine ("landing pool"), the ops head too - not the head's.

## Routing exchange: more than 16 forwards a port (2026-10-29)
- A forwarding port's slots were a fixed 16 (RAFwd sent[16], the flags + landed count at word 16 x 16 in the mailbox).
  The 8x4 torus on one link needs 22 (8x4 line with the SP ring: 30). RoutingStream(fslots) sizes them by the mesh's
  routing paths (head_plan.routing_paths, >= 16): the kernel's RA_NF_, the mailbox (lw_off), the args check; the plan
  refuses past 64.
- Checked: mock 8x4 torus links 1 (Galaxy cluster descriptor) build GLM-5.3-Flash 5120 / 10240 and K3 5120 and run a forward, fused;
  the LoudBox meshes (<= 12 forwards) unchanged (regression set below).

## TP 8: the reduce-scatter's counts word, the LoudBox 1x8 ring / line (2026-10-29, bh-lb-09)
- se_rs.counts_word packed a byte a block (TP <= 4). Past TP 4 a core's rows fall in at most two blocks (rows scatter,
  MT <= 2): two (block, count) byte pairs, x_gather3's cnt_of reads either form. The latent's all-to-all (K3) still
  needs TP <= 4 (3 destination headers a port) - refused with that reason; TP > 8 refused.
- MGDs p150_x8_ring_1x8 / p150_x8_line_1x8 (flat_combine_tools) for the 8 P150s as one TP axis.
- 1x8 chunk 1024 (2048 for GLM-Flash), fused vs ops head (traced replay us; PCC same as ops): GLM-5.3-Flash ring 1645 vs
  2038, line 1881 vs 2140, ring 2048 2288 vs 2365; DeepSeek-V4.1 ring 2616 vs 2785; GLM-5.3 ring 2294 vs 2390; K2.7
  ring 3605 vs 3710. MiniMax M3 1x8: the flat expert op's arena does not fit (1288 KB > 1284) - not the head's.
- Open: with the router's logits on the router cores at TP 8, the picks of one x differed call to call (200-360 of 1024
  tokens; same x, snap on); on the matmul cores bit-identical (0 of 20). Not narrowed; the plan takes the matmul cores
  at TP > 4 ("the router's logits on the router cores differ call to call") - GLM-5.3 ring 8 1024: 2289 vs 2294 us.
- Not lifted (#3b): more than 2048 tokens a chip - the routing planner's S bound, shared by the ops head and the
  exchange; the refusal names it ("chunk <= 2048 x N").

## The router's buffers over the reduce-scatter's zones: a cross-chip race, fixed with a "router done" credit (2026-10-29, bh-lb-09)
Why: the DRAM fallback (4d93bf0ce6c) ran in its own arena because inside the exchange's one GLM-5.3-Flash 2x4 call gave
wrong rows and a router-rectangle run hung. Two arenas hid both without a cause. Narrowed on the repro until the overlap
showed.
- Repro (scratch harness zz_scratch_head_gen.py, uncommitted; scratchpad gen/run.sh): `run.sh oa_samex 2x4
  GEN_MODEL=glm53_flash GEN_CHUNK=2560 GEN_XDRAM=1 GEN_XDRAM_ON=1 MOE_TRACE_REPLAYS=2 NOPROF=1 GEN_SAMEX=1` (GEN_XDRAM
  forces the fallback, GEN_XDRAM_ON=1 its layout with the router's logits on the matmul cores, GEN_SAMEX the same x
  every call; before the fix with the head in the exchange's arena). The first eager call was wrong: PCC 0.964-0.970,
  192-224 of 2560 rows (owner tiles 0 / 1 / 10 / 40 / 41 / 50 / 51); every later call 0.98685.
- One variable at a time: the arena zeroed before the call - same (not stale contents); a device sync after the x
  all-gather, or x gathered before the first call - same (not the gather); the ports' x streams kept (x still from
  DRAM) - fixed; two arenas - fixed. DPRINT on chip 0's matmul cores: the router's x rows and W_r^T rounds read the same
  in the bad and the good call, the partial logits (RO) of rows 0-3 differed - the inputs changed under the compute.
  The router's RX / RW / RO moved past the arena, one arena otherwise unchanged - fixed, PCC 0.98674 every call.
- Cause: SeEngine3 puts the router-on-the-matmul-cores buffers (RX / RW / RO) in the arena's tail over the
  reduce-scatter's arrival zones (ARRF / ARRB), the latent's LATZ and the prep's buffers: "written only after gate / up,
  the router's phase is over before it". True of this chip's own writes; ARRF / ARRB / LATZ are written by other chips'
  ports, which wait only for "the chip started this call". In the resident head a chip's reduce-scatter needs every
  chip's x, which a chip sends only behind its own router phase: forced skew (2 ms before TP position 0's router phase,
  GLM 2x4 5120, old protocol) - PCC right, every chip waited (replay 3044 -> 4953 us). In the fallback no chip waits for
  another's x: on the first call (program load skew) an upstream's chunk-0 units land in the zones while the router
  compute still reads RX / RW there. Forced skew in the fallback (GLM 2560, the same 2 ms, old protocol): every eager
  call wrong (PCC 0.965, the same owner tiles), no hang. Two arenas moved the router region off the zones; x streams
  kept re-coupled the chips.
- The router-rectangle hang (01:28 that night, router on 10 router cores, nothing over the zones): its triage had the
  BRISCs in cb_wait_front(SE_XFREE), the x loads behind them - the x ring's per-chunk credits left over from the previous
  call, fixed 8 minutes later (the XFREE drain at the end of a call, in 4d93bf0ce6c) and never re-run in one arena. Now:
  4 runs in one arena (8-16 traced replays + eager each), PCC 0.98674, bitwise stable. Not an arena bug.
- Fix, a protocol (a layout keeping RX / RW / RO off the zones would take K3 640's router ~600 KB it does not have):
  SeEngine3 marks a layout whose router buffers overlap ARRF / ARRB (rtr_zones) or LATZ (rtr_latz). Then every matmul
  core, its router phase over (RO written out, RX / RW consumed), adds 1 on every port core's "router done" word
  (SE_RDONE; the head's mailbox, zw_off + 8). A port, once all GX x GY are in, sends its upstream peer chunk 0's zone
  credit (the credits that gated chunks 1 .. NCH-1 now also gate chunk 0: rs_send waits m + 1, zexp + 1) and/or the
  latent's "started" messages (LATZ's fence moves from "started" to "router done"); every port re-arms the word at the
  end. With row chunks the port also polls it between its x gather's packets (the upstream's chunk 0 comes before the
  gather's last chunk ends - without it GLM 2x4 5120 / 2560 lost 1.3%); with one chunk it does not (a blocking header
  between x packets cost K3 640 ~3 us of gather). No overlap: nothing changes.
- The RS protocol model (test_rs_protocol_model) runs every chunked case with the router phase too (1-3 chunks) and
  catches a zone write during a router phase without the credit (the device's first-call rows).
- One arena again for the fallback (MoEBlock.forward takes the exchange's arena for every head).
- Checked (snap 2^-14): the repro in one arena PCC 0.98674 every call, 0 of 21 same-x replays differ; forced skew in the
  fallback with the fix: eager 0.98682 / 0.98685, 0 of 20 differ. Heads, A/B interleaved against the HEAD tree (device
  us, slowest chip, median of 40): GLM-Flash 2x4 5120 960.1 / 960.8 vs 960.6 / 960.6, 2560 526.6 / 528.4 vs 525.4 /
  526.8, K3 ring 640 393.1 / 393.5 vs 394.6 / 393.4. Soak (300 eager +
  20000 replays, 232 checks, 0 of 228 same-x replays differ, no hang): the fallback GLM-Flash 2x4 2560 (router on the
  matmul cores) PCC 0.98683, GLM-Flash 2x4 5120 0.98688, K3 ring 640 0.99616; 4 fresh-process starts of the repro clean.
- F2 (the GLM 2x4 first-call hang, d103): not this race. d103's layout overlays (RX over ARRF / ARRB), but what its
  triage waits on is outside the overlap - arrival counters (global semaphores), zone credits / mailboxes, CB counts;
  the overlap holds the zones' data tiles only, so the race gives wrong values, never a missing increment (the forced
  skew above: wrong rows, no hang). And d103 hung before any RS reached devs 0-5: their matmul cores in gate / up waiting
  for x shards (xready), router cores waiting for other chips' logit partials, port NCRISCs in
  wait_for_empty_write_slot (EDM full); only devs 6 / 7 (SP row 1, TP 2 / 3) were in the RS. F2 stays open: a fabric
  flow stall in the x gather + the logits stream.

## Fused head device matrix after the generalization (2026-10-29, bh-lb-09 - 12x10 grid, 200G, snap 2^-14)
Every model on the 1x4 ring / 1x4 line at chunk 640 / 1280 / 2048 and the 2x4 line at 1280 / 2560 / 5120 (5120: the
8x4 Galaxy's per-chip load, 640 rows a mesh row), traced, every replay checked against the host on the device's routing
(scratch harness zz_scratch_head_gen.py; scratchpad gen/matrix.sh, mx_table.py). Traced replay us of the whole block;
the ops head where it was run (MiniMax M3, K2.7 everywhere, every 2x4 5120). 0 of 20 same-x replays differed in every
run that built; no hang.
- K3: ring 1727 / 3216 / 4070 (2048: x from DRAM), line 1901 / 3550 / 4883 (x from DRAM), 2x4 1949 / 3649 / 6512 (x
  from DRAM) vs ops 6733; PCC 0.9961-0.9963.
- K2.7: ring 3168 / 3697 / 4812 vs ops 3414 / 3902 / 4570, line 3413 / 3990 / 5493 vs 3626 / 4306 / 5200, 2x4 3453 /
  4053 vs 3684 / 4390; PCC 0.9856-0.9860. At 2048 rows the fused head is 5-6% slower than the ops head: the plan is
  L1-limited (K step 14 -> 2, compact zones, W streamed); the cost model ranks it above the DRAM fallback (1021 vs 1098
  us predicted) and has no ops-head estimate to compare - open. 2x4 5120: the combine does not fit, either head.
- GLM-5.3-Flash: ring 1473 / 1829 / 2116, line 1585 / 2013 / 2390, 2x4 (owners) 1666 / 2126 / 3042 vs ops 3972.
- GLM-5.3: ring 3511 / 3941 / 4399, line 3674 / 4248 / 4802, 2x4 2127 / 2737; 2x4 5120: the exchange does not fit
  beside the fused head (the combine at A 12: landing pool -15 < 7) - the ops head runs it, 5873 us (a28131dc614 now
  raises HeadBuildError naming it).
- DeepSeek-V4.1: ring 2279 / 2667 / 3070, line 2400 / 2872 / 3418, 2x4 2436 / 2906 / 4000 (x from DRAM) vs ops 4449;
  PCC 0.9908-0.9911.
- MiniMax M3 (fused for the first time): ring 640 2467 vs ops 2719, 1280 2975 vs 3225; line 640 2603 vs 2860; 2x4 1280
  1630 vs 1901; PCC 0.9887-0.9890. Ring 2048 / line 1280: the flat op's arena (1288 KB) does not fit the L1 left
  beside the fused head (1276 / 1278 KB) - the ops head runs them (3744 / 3520 us); line 2048, 2x4 2560 / 5120: not
  with either head.
- Open from this: the head plan does not size the exchange next to it (its kept L1, the combine at its pinned
  accumulators) - three configurations where the ops head builds and the fused one does not; and a crossover check
  against the ops head for L1-limited plans (K2.7 2048 rows).
