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
