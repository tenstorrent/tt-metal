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
