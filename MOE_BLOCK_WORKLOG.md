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
