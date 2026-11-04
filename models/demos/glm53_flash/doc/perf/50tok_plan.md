# GLM-5.3-Flash on the Galaxy: the 50-100 tok/s per user program (branch `arg/glm53_perf`, 2026-11-03)

Direction (2026-11-03 21:00): reach 50-100 tok/s per user **without** speculative decoding, keeping coherence (the
decode-id chain gate, `tests/test_mesh_model.py`) and accuracy (GPQA Diamond / LiveCodeBench / MMMU through the TIS
release flow); user support may drop below 32. The release branch `arg/glm53_served_v2` (#63440, frozen at
30c0f414775) is untouched: everything here lands on this branch.

## Where the step goes today (frozen tip, TP4 x DP8, 4 users per row)

| layer kind (count) | measured | byte floor | overhead (launch + sync) | programs |
|---|---|---|---|---|
| KDA + MoE (34) | 958 us | 601 us | 357 us | 66 |
| DSA + MoE (11) | 1360 us | 762 us | 598 us | 152 |
| step (32 users) | 44.2 ms = 22.6 tok/s/u | ~29 ms | ~15 ms | ~2,900 |

Two consequences: fusion alone cannot reach 50 tok/s on TP4 with bfp8 experts (the floor is 29 ms), and the
overhead is the per-program launch + synchronisation cost of ~2,900 programs per step (~5 us each).

## The reference (aperezvicente-TT/glm-5.3-flash, 8 chips as a 2x4 mesh)

TP8 attention / dense + EP8 routed experts (36 of 288 per chip); one persistent program per run of same-kind layers
(the 45-layer loop inside the kernels; 64 owner cores + DRAM-bank reader cores streaming weights through a ring, the
next phase's / layer's weights prefetched during the tail); in-program fabric all-reduce on 2 lane cores per chip;
embed / head / sampling inside the same trace; routed experts bfloat4_b. Measured (its code comments): sparse KDA layer
283 us, dense KDA 237 us, MLA program 444-560 us at 1 user: ~18-19 ms per step at 1-4 users = the 50 tok/s/u claim
(32 users = "user groups", unmeasured there). Its tt-metal build (a6b89afae56) and the ttnn ops it depends on
(`experimental.deepseek.fused_hyperconnection`, `experimental.kda.*`, `experimental.deepseek.moe.fused_experts`) are
not in tenstorrent/tt-metal, so the design is taken, not the code.

## Levers, in order, each gated on the decode-id chain + evals

| # | lever | expected step | status |
|---|---|---|---|
| 1 | TP8 per instance (DP4 on the Galaxy: 4 columns of 8 chips as 1x8 rings), 4-8 users per instance | floor 29 -> ~17 ms | probe running: `tests/test_mesh_moe.py` on a (1, 8) column ring |
| 2 | routed W13 (and W2) bfloat4_b, `GLM53_EXPERT_W13_DTYPE=bfp4` (separate weight cache) | floor -20..30 % | queued after 1; accuracy via evals, not the id chain |
| 3 | program count per layer 66 / 152 -> ~10: the MoE glue (router gate + top-k + the gathers + RS/AG), the KDA attention chain and mHC pre/post as single generic_op programs with in-program collectives, modelled on the reference's phase table (P1 mhc_pre .. P10 mhc_post, weight ring, semaphore hand-offs) | overhead 15 -> ~3 ms | design next |
| 4 | DSA layers on the same scheme | | after 3 |
| 5 | MTP on top (`tt/spec_decode.py`) for the 80-100 band | | after 3 |

Target arithmetic: 1 + 2 + 3 = ~13 ms floor + ~3 ms overhead = ~16-20 ms per step = 50-60 tok/s/u at 4-8 users per
instance, 16-32 users on the Galaxy.

## Gates (refined 2026-11-03 21:50)

The frozen release used "decode ids identical" as the coherence proxy. For this program the gate is the quality
itself: a change that keeps the math (a layout, a TP width, a bit-exact fusion) must keep the DECODE-IDS line; a
change that moves rounding (bfp4, a fused op whose reduction order differs) is judged by (1) its per-op PCC against
the composite on real activations (>= 0.9999 for glue, the expert ops at their existing bars), (2) the chain's
per-user PCC within noise of the base, and (3) the served coherence smoke plus GPQA / LiveCodeBench / MMMU through
the release flow. Under that gate the fused DSA decode-state program (`GLM53_DSA_DECODE_STATE=1`, closed on
2026-11-03 for 6 of 16 ids at chain PCC 0.8469 vs 0.85; -1.6 ms per step at 4 users per row) is a candidate again.


* coherence: `DECODE-IDS` line of the chain test identical to the frozen tip for levers 1 and 3 (TP changes keep the
  math up to reduction order; a program fusion must be bit-exact or PCC-gated per op as today's custom ops are);
* accuracy: lever 2 is judged on GPQA Diamond + LiveCodeBench + MMMU through the local release flow (not the id chain);
* every served check at the extreme ISLs first ([[test-extreme-isl-first]]).

## Log

* 2026-11-03 21:36 lever 1 probe: `tp8_moe_1x8` (TT_VISIBLE_DEVICES=0,1,2,3,27,26,25,24, descriptor
  `_logs/mesh/bh_galaxy_1x8_ring.textproto`, GLM53_MESH_SHAPE=1,8): the 1x8 ring opened under FABRIC_2D; PASS in 2:52: the hybrid MoE layer 4 on 8 columns, worst PCC vs HF 0.999957 at 2 rows and 0.999930 at 32 rows (the same bars as the (2, 4) block). Our mesh code takes cols = 8 for the MoE as is.

## Where the per-layer overhead sits (roofline rows, 4 users per row, frozen tip)

The launch gaps themselves are small (`layer.gaps`: 24 us over 33 gaps per KDA layer, 43 us over 76 per DSA layer).
The 357 / 598 us of "overhead" per layer is inside under-utilised or sync-bound ops and in the 8-row MoE collectives:

| cluster | KDA layer measured - floor | DSA layer measured - floor | what it is |
|---|---|---|---|
| `mhc.pre_block_fused` x2 | 141 us | 141 us | 82 us each for 11 us of bytes: the fused pre-block (coefficients + norms over 4 streams) runs latency-bound |
| row collectives (`gather_x`, `gather_routing`, `rs_rows`) | 111 us over 77 us floor (188 us total) | 102 us over 77 (174 total) | the 8-row MoE exchange: vanish entirely on a 1-row instance |
| `all_reduce` (attention + MoE) | 36 us | 33 us | 2 CCL ops each |
| `kda.decode` | 49 us | - | DRAM under-utilised |
| `dsa.state_composite` (29 ops) | - | 74 us | the paged-state glue |
| `dsa.pool_gather`, `dsa.sdpa` | - | 91 + 49 us | L1/NoC bound |
| `dsa.*_view` (7 ops) | - | 44 us | layout glue |

Projection for one (1, 8) instance at 4 users, from the rows: KDA layer 958 - 188 (row CCL) - ~135 (half the
weight bytes) = ~635 us; DSA layer 1360 - 174 - ~150 = ~1035 us; step ~33 ms = 30 tok/s/u. Then, in order:

| step ((8, 4) hybrid, 32 users at 4 per row, bfp8) | saving per step | step | tok/s/u | status |
|---|---|---|---|---|
| frozen tip 4d98abdb1d3 | | 44.2 ms | 22.6 | released |
| dest-resident Sinkhorn in the pre-block (default) | -2.6 | 41.7 | 24.0 | measured, chain pcc 0.955 (base 0.947) |
| fabric row gather (`GLM53_HYBRID_GATHER=fabric`) | -2.2 | 39.5 | 25.3 | measured, bit-identical ids |
| fabric combine + TP all-reduces (`GLM53_HYBRID_COMBINE=fabric`, `GLM53_TP_ALLREDUCE=fabric`) | -1.8 | 37.7 | 26.5 | measured; decode pcc gate rerunning on the zero-fill fix; served smoke PASS at 19.5 tok/s/u (tip 16-17) |
| mHC: post-mix merged into the next pre-block, fan-in, round-2 scalars | ~-2.2 | ~35.5 | ~28 | next |
| KDA: the two 288-wide projections into the qkv matmul, decode op input prefetch | ~-1.5 | ~34 | ~29.5 | |
| DSA glue (fused decode state, views, pool gather) | ~-1.6 | ~32.4 | ~31 | `GLM53_DSA_DECODE_STATE=1` exists (-0.7 ms, decode-gate pending) |
| router + shared expert as one program with the gather | ~-1 | ~31.4 | ~32 | |
| 8 users (1 per row): the expert op at ~150 us | ~-2 | ~29.4 | ~34 | the user's compromise option |
| the persistent per-layer program (the reference's design) or bfp4 under an accuracy gate | the remaining ~10 ms | ~20 | 50 | the only route to 50 at bfp8 x 32 users |
| MTP (K = 2-3) on top | | | 80-100 | after the above |

Every row is a per-op change behind the existing PCC / decode-id gates; no megakernel is required for the 50 band.
The reference's persistent-stack design stays the lever for the 100 band without MTP (it removes the remaining
~100 us of collective + gap cost per layer).

## Serving layout for lever 1 (DP4 x TP8)

The model's row axis is not data parallel for the MoE: the hybrid expert layout splits each expert's intermediate
dimension over the mesh rows and exchanges tokens between rows (`gather_x` / `rs_rows`), so a (4, 8) mesh would be
one 32-chip instance with 4-row collectives, not four instances. Four independent TP8 instances are therefore four
engines: the vLLM TT plugin's standard multi-process DP (`--data-parallel-size 4`, per-rank mesh and
`TT_VISIBLE_DEVICES` discovery), each rank on one Galaxy column opened as the (1, 8) ring (the topology-only
`bh_galaxy_1x8_ring.textproto` applies to every column; the ring closes through the torus wrap). Users per rank 4-8;
prefix cache, slot placement and the page table stay per rank. To validate after the (1, 8) step numbers: the plugin's
DP discovery on the Galaxy's columns, one rank served, then four.

## Log (continued)

* 21:40 `tp8_e2e_1x8_B4`: the 45-layer chain + perf on column 0 as a (1, 8) ring, 4 users, bfp8 (the (1,8) weight
  cache converts first, ~50-80 s per layer).
* 21:47 `tp8_e2e_1x8_B4_bfp4`: the same on column 1 with `GLM53_MODEL_EXPERT_DTYPE=bfp4` (routed W13 bfp4, W2 bfp8)
  into `_cache/mesh_bfp4`, chain PCC bar lowered to 0.9 so the run reports its step and ids; accuracy comes from the
  evals, not this chain.
* 21:46 pre-block profile on chip 8 (tracy over `test_mhc_pre_block.py::test_pre_block_real`, B = 1/4/32, FPU and
  exact variants): the op's device time per case.
* 21:47 pre-block profile (chip 8, 1x1 mesh, no fabric): the generic op's device kernel time is **101.8 us for B = 1, 4
  and 32 alike** (FPU projection; 183.8 us with the exact SFPU projection); the standalone `mhc_split_sinkhorn` op
  51.9 us. The cost is serial latency, not work: projection passes, the 10 + 12 tile fan-in of two tree rounds on
  single cores, 43 Sinkhorn dest passes with a CB round trip each, two multicasts. Plan for a v3 (per wrap, from 82-102
  us): (C) fold the h-norm into round 1 through the 4x4 Gram matrix G_ij = x_i . x_j of each user (sum_c h_c^2 =
  sum_ij pre_i pre_j G_ij, exact in real arithmetic; the coordinator finishes it and ships rsqrt in the pre multicast:
  no second tree round, no second multicast, -15..20 us); (F) the projection without the 3-way bf16 split (-7 us, the
  FPU rounds the operands anyway per the op's README); (A) a shallower fan-in (-5 us); (D) the attention post-mix and
  the FFN pre-block as one program (-8 us per layer); (B) a DST-resident SFPU Sinkhorn (-15 us) last. C + F + A + D
  = ~-35 us per wrap = -3.2 ms per step; with B -4.5 ms.
* 21:58 pre-block attribution (chip 8, served variants, x_parts 1): Sinkhorn iterations 1 / 5 / 20 = 46.4 / 52.4 /
  74.3 us, so the 20 iterations cost 28 us (1.4 us each: two dest passes with a CB round trip); the remaining 46 us
  = projection (4 FPU passes), the two tree rounds (10 + 12 serial tile adds on the leaders / coordinator, twice),
  two multicasts, the coefficient heads, mix and the write. **Design, dest-resident Sinkhorn**: the comb tile's 16
  columns of a user row are the left face's even slot (lanes c <-> columns 2c) and odd slot (columns 2c + 1) of its
  4-row band (`mhc_sfpu_x8.h` geometry); the 4x4 entry (i, j) sits at column 4i + j, so a row sum over j is the
  lane pair {2i, 2i+1} of both slots and a column sum over i is the stride-2 lanes j/2 + 2i of one slot: both are
  subvector rotate-adds (ROR1 / ROR2 / ROR4), no matmul, no pack; reciprocal by Newton on the SFPU. The kernel
  takes the live row count and processes ceil(rows / 4) bands only (pad rows are ignored downstream already), so
  at 4-8 users the 40 normalisations touch one or two bands: 28 -> ~3-6 us. Then round 2 (sum of h^2) as 32
  scalars per core summed on the RISCs and the rsqrt shipped in the pre multicast (-7 us), and the FFN pre-block
  merged with the attention post-mix (-8 us per layer). Target per wrap: 74 -> ~36 us.
* 22:13 **pre-block v3, step 1 landed**: `MHC_X_PARTS=1` (81.4 -> 74.3 us) and the dest-resident Sinkhorn
  `MHC_SINKHORN_DST=1` on the live rows' bands (`mhc_sfpu_sinkhorn.h`): 74.3 -> **46.4 us per wrap at 1-4 users**,
  59.9 at 32 (chip 8, unit numerics within the bars). Per step at 4 users: 90 x 35 us = **-3.2 ms**. Knobs
  `GLM53_MHC_PRE_XPARTS=1 GLM53_MHC_PRE_SINKHORN=dst`; defaults flip after the chain's decode-id gate on a (1, 8)
  column with the converted cache. Next in the wrap: round 2 as RISC scalars (-7), the fan-in (-5), the merge with
  the attention post-mix (-8 per layer).
* 22:15 **KDA decode attribution** (chip 8, device profiler zones, the TP4 shard H = 16): kernel span 69.6 us at
  B = 4 (256 at B = 32). Phase spans at B = 4: `kda_conv` 43 us (per-core median 30, max 43: the conv + SiLU phase
  also absorbs the wait for the part-0 reader's multicast of the inputs; the NCRISC reader runs 33 us median / 65 max),
  then `kda_gates` 5, `kda_derived` 2, `kda_user` 8 per user (21 span), `kda_out_norm` 9-13 on the 32 part-0 cores.
  At B = 32 the recurrent-state traffic (2 x 2.1 MB per user) sets the slope (`kda_user` span 148 us); the fixed
  ~45 us is the serialized input movement + the phase chain. Lever: start the state and history reads before /
  during the conv phase on every part (today they wait on part 0's multicast), and run the out-norm on all parts:
  69 -> ~40 us at B = 4 = -1 ms per step over 34 layers. Scheduled after the pre-block items.
* 22:20 KDA reader zones (chip 8, B = 4, H = 16, async prologue): item reads 1.5 -> 30-35 us on the 16 part-0 cores
  (DRAM-rate bound per core, ~230 KB each), multicast 3, conv 8, gates 5, derived 2, user 8 (compute), gather +
  out-norm 10 on part 0: span 69 us. Design for the next step: every part reads a fifth of the item inputs and
  multicasts its share (reads ~6 us), the out-norm per part on its own rows: ~40 us per op, -1 ms per step.
* 22:25 KDA item reads split over all 5 parts (loopback multicast of each part's share, the history shift split
  too): all 12 unit tests pass but the span stays 69.9 us: the reads still end ~30 us into the kernel (per-core
  median 19 us, max 30) with 80 issuing cores as with 16, so the ~3.7 MB of 2 KB tile reads per step are a
  DRAM / NoC latency floor for this access pattern, not an issue-rate limit. Variant not kept. The lever that
  remains is prefetch: the 112 KB per slot of step-invariant KDA inputs (history taps, conv taps, channel
  constants) read into a persistent L1 buffer on the slot's part-0 core by the preceding mHC pre-block program
  (its readers idle for ~40 us), the KDA reader copying L1 -> CB in ~2 us: item phase ~32 -> ~10 us, -0.7 ms per
  step. Needs host-side persistent L1 tensors allocated before the trace capture.
* 22:24 **lever 1 constraint**: the (1, 8) bfp8 chain died at layer 33 with DRAM OOM (4.18 of 4.24 GB per bank):
  a single-row TP8 instance holds 36 FULL experts per chip. Per chip over the 42 MoE layers (H 4096, I 2048, 288
  experts): (8, 4) hybrid 7 GiB (72 experts x I/8), (1, 8) bfp8 28 GiB, (1, 8) bfp4 14 GiB, (2, 8) bfp8 14 GiB,
  (4, 8) bfp8 7 GiB. So TP8 on one column needs bfp4 routed experts (lever 2 becomes the enabler; the column-1 run
  decides), or a (2, 8) instance of 16 chips at bfp8 (DP2 on the Galaxy, descriptor `bh_galaxy_2x8_torus.textproto`
  prepared). Running meanwhile on column 0: the first 32 layers on (1, 8) bfp8 (the converted prefix fits), base vs
  the pre-block v3 knobs, for the per-layer TP8 step cost and a 32-layer decode-id gate of the knobs.
* 22:40 **lever 1 measured** (`tp8_l32_1x8_B4_base`): the first 32 layers (24 KDA + 8 DSA) on a (1, 8) column ring
  at 4 users, bfp8: traced step **34.47 ms = 1.08 ms per layer**, the same per-layer cost as the (8, 4) hybrid at
  4 users per row (the 958 / 1360 us rows give 33.9 ms for the same prefix). The single-row TP8 instance saves
  nothing by itself: the 8-row collectives (188 us per layer) vanish, but the two 8-ring all-reduces and the
  full-expert streaming op (36 full experts per chip instead of 72 eighth-slices) cost as much; the per-op profile
  on column 0 (`tracy_l4_b4_1x8`, `tracy_l3_b4_1x8`) attributes it. The ladder row "(1, 8) instance -11 ms" is
  withdrawn: at 4 users the step is per-op fixed latency on either layout, so the program-count levers (3, 4) carry
  the target, and the layout choice follows the fused design (one collective domain of 8 chips as the reference, or
  the hybrid with in-program row exchange). Warm TTFT 493 ms for 128 tokens on the prefix.
* 22:40 **pre-block v3 on the TP8 prefix** (`tp8_l32_1x8_B4_mhcv3`, `GLM53_MHC_PRE_XPARTS=1 GLM53_MHC_PRE_SINKHORN=dst`):
  34.47 -> **31.92 ms** (-2.55 ms over 64 wraps = -40 us per wrap; the unit measured -35), chain pcc 0.807 vs 0.782
  (both a 32-layer prefix against HF's rows: not the gate), host-argmax agreement unchanged (19 % of 32 positions).
  The 45-layer gate runs on (8, 4) at 4 users per row once the bfp4 run frees column 1 (`gate84_B4_base` /
  `gate84_B4_mhcv3`): the base ids must equal the frozen tip's line (10:58 `e2e_8x4_B4_postmcast`, S = 16: the KDA
  reader prologue change moves no rounding), the knobs are judged by the chain pcc against 0.86 at S = 16, the
  first-token match and the step (44.23 ms - ~3 ms expected).
* 22:45 **(1, 8) per-op profiles** (`tracy_l4_b4_1x8`, `tracy_l3_b4_1x8`, column 0, 4 users, bfp8): KDA+MoE layer
  1.228 ms (0.934 on the 8x4): the expert stream op takes **626 us** for ~4 full experts per chip (2.2x its byte
  floor; 36 full experts per chip instead of 72 eighth-slices) and the per-chip expert count is unbalanced (chips 5-7
  idle 0.58 ms in the MoE reduce-scatter); the row collectives are gone (-187 us) and the attention shards halve
  (126 vs 188 us) but the two 8-ring all-reduces cost 67 + 143 us. DSA+MoE layer 1.57 ms (1.316): experts 362 +
  the reduce-scatter wait 319. The (1, 8) bfp4 45-layer chain (`tp8_e2e_1x8_B4_bfp4`): fits, **45.05 ms** (22.2
  tok/s/u), 128-token chain pcc **0.78** (0.947 at bfp8 on the 8x4): no step gain and an accuracy loss. Levers 1
  and 2 are closed: the (8, 4) hybrid at 32 users stays the serving layout and the program-count levers carry the
  target there.
* 23:06 **pre-block v3 on the full mesh** (`gate84_B4_base` / `gate84_B4_mhcv3`, (8, 4) 45 layers, 4 users per
  row): base (the perf branch's KDA reader prologue) is id-identical to the frozen tip (16 / 16 ids, chain pcc
  0.946814, first token match, 44.36 ms); the v3 knobs give **41.18 ms (24.3 tok/s/u, -3.2 ms as predicted)** but
  the 128-token chain pcc 0.9185 and a different first token: one knob moves the rounding too far (suspected
  `x_parts=1`, which drops the 3-way bf16 split of the projection operands). A/B running: `gate84_B4_dst`
  (x_parts 3 + dest Sinkhorn) and `gate84_B4_xp1` (x_parts 1 + fused Sinkhorn); the accurate knob flips its
  default, the other stays off.
* 23:15 **lever 3, first program: the row exchange** (`custom_ops/row_exchange`, draft, device test pending the
  A/B): the hybrid MoE's gather (untilize + all_broadcast + concat + tilize for x and the routing: 8 ops, 113 us) and
  combine (reduce-scatter + all-gather + selector matmul: 74 us) each as ONE generic op with in-program fabric
  unicasts between the rows of a column (after the DeepSeek-V3 b1 fused ops' protocol in this tree:
  `ttnn.setup_fabric_connection` per routing direction, `WorkerToFabricEdmSender`, fused write + atomic increment
  into a persistent per-instance receive staging in DRAM and a global arrival semaphore). One exchange core per chip
  sends 7 peers x 3 packets of 12 KB; 9 worker cores untilize / cut / assemble / tilize their 512-column chunk (the
  reduce sums the 8 rows' pieces by one exact selector matmul in fp32 DST). Knobs `GLM53_HYBRID_GATHER=fabric`,
  `GLM53_HYBRID_COMBINE=fabric` (`tt/moe.py`, defaults unchanged). Expected: 187 -> ~30 us per layer, about -7 ms
  per step.
* 23:25 **pre-block v3 A/B verdict** ((8, 4) 45 layers, 4 users per row, 128-token chain pcc / step): dest Sinkhorn
  alone **41.72 ms, pcc 0.9547** (base 44.36 / 0.9468) -> default (`GLM53_MHC_PRE_SINKHORN=dst`); `x_parts=1` alone
  43.56 ms, pcc 0.9147, first token and decode ids changed from position 1 -> rejected (the projection keeps its
  3-way bf16 split); both 41.18 / 0.9185 (the savings do not add). Step 1 of lever 3 lands **-2.6 ms**: 24.0 tok/s/u.
* 23:26 **row exchange measured** (`rowx_8x4_r2`, the (8, 4) torus, T = 4): gather (x + routing, 9 chunks, 7 peers x
  3 packets of 12 KB per chip) bit-exact on all 32 chips and after 20 trace replays, **40 us per traced call**
  (the composite: 113 us); reduce exact to one bf16 ulp, **42 us** (74 us). About -105 us per MoE layer, -4.4 ms
  per step expected; chain gate running (`gate84_B4_fabgather`: the gather alone must reproduce the dst run's ids;
  `gate84_B4_fabric`: both knobs, judged by the chain pcc against 0.9547).
* 23:40 **lever 3, second program: the TP all-reduce** (`custom_ops/row_exchange` mode `allreduce`, commit
  86d57a1dd6d, device test queued): the attention and MoE all-reduces over a row's 4 columns (RS + AG composites, 50
  + 39 us per layer) as one two-phase fabric program each: every column owns a quarter of the hidden columns (2 of the
  8 chunks), receives the 3 peers' T-row pieces of its chunks (phase 1, one 8 KB packet per peer), sums them with
  the selector matmul in fp32 DST, writes its output chunks and fans the sums out (phase 2, one packet per peer).
  Plugged into `GLMMesh.all_reduce_cols` behind `GLM53_TP_ALLREDUCE=fabric` for decode shapes (`[1, 1, T, W]` bf16,
  `cols * T <= 32`), one op instance per call site (`tag` = attn / moe / mlp: a peer's next call of one site lands
  only after this chip's intervening call of the other site, so the persistent staging is reused safely). Expected
  ~30 us per call: about -2.2 ms per step. Lesson booked: a chain run JIT-compiles kernels minutes after it imported
  op.py, so kernel edits wait for the running job (`gate84_B4_fabgather` died on that; rerun queued).
* 23:44 **TP all-reduce measured** (`rowx_8x4_r3`, the (8, 4) torus, T = 4, 8 chunks / 2 per owner / 1 packet per peer
  per phase): exact to one bf16 ulp on all 32 chips and after 20 replays, **34 us per traced call** (the composites:
  51 us attention, 30-39 us MoE); gather 40 / reduce 41 us again. Per layer: 2 x ~10 us = about -0.9 ms per step on
  top of the row exchange's -4.4. The first full-fabric chain gate died on a shape mismatch (the combine returned 32
  logical rows where the add with the shared expert wants T): outputs now carry their logical rows; the gate reruns
  (`gate84_B4_fabar`) after the gather-only one.
* 00:00 **full-fabric chain gate** (`gate84_B4_fabar`: gather + combine + TP all-reduce): **37.69 ms = 26.5 tok/s/u**,
  849 tok/s aggregate (the gather alone 39.51, the dst base 41.72), prefill untouched (chain pcc 0.954698, first
  token 44210) -- but the S = 32 decode chain pcc fell to 0.794 (0.878 with the gather alone) and the decode ids
  diverge from position 8. Found by desk-check: the all-reduce owner assembles `cols * T = 16` rows and left rows
  16..31 of its block unwritten; in the unit test that L1 held finite leftovers, in the model NaN / Inf garbage, which
  the selector's zero weights turn into NaN sums (0 x NaN). Fix 9727301310f: every mode zero-fills the rows past the
  assembled users (the unit test now also replays with fresh inputs per replay to expose stale-staging races). The
  served smoke (`served_fabric_1m`, all three knobs, 1 M profile: first token vs HF, 4 concurrent greedy x 2, bench
  128/128 at 32 users) and the isolating gates (combine only / all-reduce only) run on the fixed kernels, then the
  full gate again (`gate84_B4_fabar_r2`).
* 00:51 **per-layer profile with the fabric knobs** (`tracy_l4_b4_fab`, chip (0,0), KDA+MoE layer 4, 4 users per row,
  dst Sinkhorn default): **22 programs, 776 us** (the tip: 33, 934). mHC 2 x 52.6 + 2 x 18.7 = 143; attention qkv 66.8,
  small projections 13.8 + 9.5, KDA decode 72.9, out 27.2, fabric all-reduce 42.2 (composite 50); router 30; fabric
  gather 56 (40 in isolation: the collective absorbs the chips' skew), experts 215 (202 at the tip), fabric combine 38,
  shared 33, add 3, fabric all-reduce 24 (composite 39). The 37.7 ms step follows (34 x 0.776 + 11 DSA layers). What
  is left per KDA layer: experts ~50 us over the bfp8 floor, KDA decode ~28 over, mHC ~100 over a merged target,
  the four exchanges ~160 (skew-bound), router + shared 66 in 8 programs.
* 00:54 **DSA+MoE layer with the fabric knobs** (`tracy_l3_b4_fab`): 65 programs, 1167 us (tip 1316): the attention
  block is still 50 programs (~550 us: the decode-state generic op 133, flash-MLA 73, o-proj 44, the q / absorb
  matmuls 49 + 21 + 32 + 20 + 19, a layer norm 19, a view 20, ~38 small ops), fabric all-reduces 48 + 21, mHC 143, router
  30, fabric gather 58, experts 198, fabric combine 64, shared 33. The fused decode-state block that exists behind
  `GLM53_DSA_DECODE_STATE=1` (29 + 8 ops -> 1 + 2) measured -0.7 ms per step on 2026-11-03 but moved the decode chain
  (pcc 0.847 vs 0.860 at S = 16); its decode gate runs after the combine diagnostics.
* 01:50 **the combine's chain drift, resolved to the metric**: with the fabric gather + combine, the S = 32 decode
  chain pcc is 0.826 (0.878 with the composite combine) and the ids diverge from position 6 -- yet (a) the op is exact
  in isolation, with fresh data per replay and in the gather -> scale -> combine -> add -> all-reduce sequence under
  one trace, (b) the same chain run EAGERLY reproduces 0.826341 and the same ids (not trace-specific), and (c) across
  all 1386 calls of that eager chain the fabric combine is within 1 bf16 ulp of the composite computed on the same
  input (max 1.4 % of |ref|, none over 0.1). The S = 32 chain through the KDA recurrence amplifies ulp-level
  perturbations: the fabric all-reduce alone moved the same metric to 0.913 (+0.035), the combine to 0.826 (-0.05);
  the prefill chain (0.9547) does not move. Rounding-moving changes are therefore judged by the per-call agreement
  (this check), the prefill chain and the served evals; the S = 32 metric's noise band is measured next with the
  composite's own alternative summation (`gate84_B4_ccl`). The device rounds the sums to nearest (unit diagnostic:
  95 % of elements equal the round-to-nearest fp32 sum, the rest differ by summation order).
* 01:59 **fused DSA decode state on top** (`gate84_B4_dsastate`: fabric gather + all-reduce + `GLM53_DSA_DECODE_STATE=1`):
  38.27 ms (-0.84 vs the same knobs with the composite state), S = 32 chain pcc 0.891 (0.913 composite: inside the
  band), prefill chain unchanged. A -0.8 ms rounding-moving candidate for the same eval gate as the combine.
* 02:08 **the S = 32 chain's noise band** (`gate84_B4_ccl`: the fabric gather with the composite's generic
  `ttnn.all_reduce` combine instead of the reduce-scatter + all-gather form): pcc min 0.809, median 0.862, max 0.863
  (per-user spread, the generic composite is not bit-identical across rows), and the same decode ids as the fabric
  combine. Summation-order variants of the same math span 0.79-0.91 on this metric (rs + ag 0.878, generic 0.81-0.86,
  fabric combine 0.826, fabric all-reduce 0.913, all three fabric 0.794), so it does not grade rounding-moving
  programs. Verdict for the fabric exchange: correct to one bf16 ulp per call (1386 calls), prefill chain unchanged,
  served smoke coherent; the served GPQA Diamond run with the three knobs (`served_fabric_gpqa`, reference 90 on the
  tip) decides the defaults.
* 02:39 **GPQA Diamond, served with the three fabric knobs** (`served_fabric_gpqa`, 1 M profile, the release flow's
  `--ci-mode` subset = 10 of 198 questions): exact match 0.90, the same 9 / 10 the frozen tip's release run scored on
  that subset. No regression at the CI sample count; the full 198-question run follows as the arbiter before the
  defaults flip.
* 04:30 **persistent program, step 1 built** (branch `arg/glm53_persistent`, worktree `wt_persist`, commit
  e6e52769e4c): `custom_ops/moe_persistent` = the MoE block's three programs as ONE program per chip: the row gather
  on the 10 cores the expert placement leaves idle, a multicast release into the 112 expert cores' semaphores once
  the gathered tiles are written, the fused gate|up + down kernels unchanged (wait hooks at their start), the down
  cores counting their written output tiles on the combine workers, and the row combine on the same 10 cores. The
  data still crosses the gathered / expert-output tensors; the gain is two launches and their fills (~25-35 us per
  layer) and, more importantly, the phase-chaining pattern (program semaphores + a multicast release + completion
  counters) every further phase of the persistent layer reuses. Model knob `GLM53_MOE_PERSIST=1`; its chain gate
  (`gate84_B4_persist`) must reproduce the all-fabric run's ids bit for bit (the same math, one program less two).
  Next phases: the gathered rows multicast straight into the expert cores' L1 (no x_g tensor), then the attention
  half (pre-block -> qkv -> KDA recurrence -> o-proj -> all-reduce -> post-mix) on the same pattern.
* 04:43 **GPQA Diamond, all 198 questions, served with the three fabric programs** (`served_fabric_gpqa_full2`, 1 M
  profile, 32 concurrent): exact match **0.869 +- 0.024**. The release flow's CI subset (10 questions) scored 9 / 10
  on both the tip and this branch; the full set has no tip reference run (the tt-shield releases run the subset). With
  the per-call agreement, the unchanged prefill chain and the served smoke, the fabric gather, combine and TP
  all-reduce become the defaults (`GLM53_HYBRID_GATHER=fabric`, `GLM53_HYBRID_COMBINE=fabric`,
  `GLM53_TP_ALLREDUCE=fabric`; the composites stay behind the knobs). Step on the (8, 4) at 32 users: **37.7 ms,
  26.5 tok/s per user** (tip 44.2 / 22.6); served decode at 32 users 19.5 tok/s per user (tip 16-17).

- 2026-11-04 04:52 `gate84_B4_persist` (step 1) failed at construction: on the Galaxy chips' 12x10 compute grid the
  fused expert op (12 gate|up groups x 8 banks + 16 down cores) leaves 8 free cores; the exchange needs 10 (1 + 8 x
  chunks + 1 routing chunk at H = 4096, E_loc = 72). Fix 4bb7e41c2c7: `persistent_fused_groups` picks the largest
  group count (cap 12) whose placement leaves the cores -> 11 groups (88 + 16, 16 free) when `GLM53_MOE_PERSIST=1`.
  Re-gated as `gate84_B4_persist_g11`, with `gate84_B4_fabar_g11` (same tree, no persistent program, 11 groups) as the
  ids reference in case the group count moves the expert rounding, then `tracy_l4_b4_persist_g11`. The chained
  `tracy_l4_b4_persist` (rc=0) profiled the non-persistent fallback: void.
- 2026-11-04 05:08 `gate84_B4_persist_g11` failed at construction too: `'ExpertStream' object has no attribute
  'down_xy'` (the fused op's attributes live on the wrapper's `fused_op`). Fix 9b4aeaf46b5. Re-queued as
  `gate84_B4_persist_g11b` strictly behind `gate84_B4_fabar_g11` and `gate84_B1_fabar` (one chip waiter at a time);
  on a pass the same chain runs step 2 (`gate84_B4_persist2`), the 8-user persistent point (`gate84_B1_persist_g11b`)
  and the two layer profiles.
- 2026-11-04 05:17 `gate84_B4_fabar_g11` (11 gate|up groups, no persistent program): ids bit-identical to
  `gate84_B4_fabar_r2` (12 groups), per-user pcc min 0.794 / chain 0.9547 both, step 37.34 vs 37.69 ms. The group
  count moves neither the rounding nor the step at 32 users; the persistent gates keep fabar_r2's ids as the target.
- 2026-11-04 05:18-05:35 the first device launch of the persistent program (`tracy_l4_b4_persist_g11`, the layer
  profile picked the fixed tree up) HUNG. tt-triage on the live hang: the gather phase completed (workers in phase B,
  x_g written), the exchange core stuck in `noc_async_write_barrier` after the four release multicasts, the release
  semaphores (12-15) set on four cores only, every gate|up core still waiting. The chips' worker grid is NOC x 2-7 and
  10-15 (8-9 are not Tensix) and the rect spanned it; a NOC multicast may target Tensix cores only. Fix 9f921e1de41:
  `grid_mcast_rects` splits every grid multicast into Tensix-only contiguous rects (the sender's own rect uses the
  loopback variant); the step-2 landing multicast takes the same rects (71cf14a9ee0). SIGTERM freed the chips;
  `gate84_B1_fabar` opened the mesh fine afterwards.
- 2026-11-04 05:57-06:08 `gate84_B4_persist_g11b` (rect split) HUNG with the same signature: the release landed on
  nine edge cores (top and bottom rows), the exchange core in its write barrier. The Blackhole kernel API does not
  mirror coordinates per NOC (`NOC_0_X(noc, size, x) = x`), but a multicast rect is walked from its start corner along
  the NOC's direction and tt-metal's factories swap the corners for NOC_1 (matmul 2D: `std::swap(in0_mcast_start,
  in0_mcast_end)` when the kernel's NOC is NOC_1); a mis-oriented rect is a torus wrap-around multicast that reaches
  the grid's edges only. Fix 62da7b05527 / f31e1f1fec1: the kernels swap the corners when `noc_index == 1`. Reset #2
  (same procedure), the unit test `unit_moe_persistent` now runs first; `chain_persist_g11c` re-gates on a pass.
- 2026-11-04 06:22 `gate84_B1_fabar_r2` (8 users = 1 per row, today's fabric path, 11 groups): traced decode step
  32.95 ms = 30.4 tok/s per user (32 users: 37.3 ms = 26.8). Four times fewer users save only 4.4 ms: the step is
  launch / latency bound, not bandwidth bound. The 50 tok/s/u floor (20 ms) needs 13 ms off the 8-user step: the
  persistent layer's whole budget (launches, the exchanges' round trips, the attention half's ~20 small ops per layer).
- 2026-11-04 06:24 `unit_moe_persistent_r2` PASSED (wt_persist @8afb4a2b941, NOC-1 corner swap): the persistent program
  is bit-exact against gather + fused experts + reduce on every chip, traced replays with fresh inputs match the
  golden. Traced wall per replay: persistent 0.646 ms vs three ops 0.605 ms (T = 4, random routing with 8 picks per
  user, i.e. nearly every expert active per chip). No launch gain at this shape: the program adds ~70 us somewhere
  (the x_g write + "written" count + release multicast + the down -> reduce signal, or the 120-core launch itself);
  the layer profile (`tracy_l4_b4_persist_g11d`) and step 2 (resident x: no x_g write / read) decide.
- 2026-11-04 06:28 step 2's unit test (`unit_moe_persistent2_r2`): the eager persistent program matched the three-op
  sequence bit-exactly, the first traced replay (fresh inputs) HUNG. tt-triage: on most chips 3-7 gate|up cores had
  their compute waiting for weight tiles with the reader finished (reader vs writer expert-count mismatch), the down
  cores waiting on those experts, the exchange cores in phase B. Cause: `writer_gate_up.cpp`'s resident-x branch
  skipped the `noc_async_read_barrier` that the x-read branch placed after the routing reads, so the writer scanned
  stale routing; eager matched only because the previous program had left the same routing in that L1 region. Fix
  675eefb3f4c (wt_persist2). Reset #3; `chain_persist2_r3` re-tests step 2 after step 1's gate and profile.
- 2026-11-04 06:43 `gate84_B4_persist_g11d` PASSED: step 1 (gather + fused experts + combine in one program per chip)
  reproduces `gate84_B4_fabar_r2` bit-exactly over the 20-step chain (ids, per-user pcc 0.794, chain 0.9547). Step
  38.37 ms vs 37.69 / 37.34 ms: the block is ~20 us per layer SLOWER than the three programs it replaces, i.e. the
  in-program handshakes (x_g write -> "written" count -> release multicast; down -> reduce signal) cost more than the
  three dispatch boundaries they remove. Next: the per-core timeline of the profiled unit test (`tracy_unit_persist`)
  to see the gaps, then step 2 (resident x removes the x_g write and the gate|up x reads).
- 2026-11-04 06:52 `gate84_B1_persist_g11d` (8 users, persistent block) PASSED bit-exact vs `gate84_B1_fabar_r2`;
  step 32.59 vs 32.95 ms (-0.36 ms, ~8 us per layer) while at 32 users it is +20 us per layer: the block's fixed
  handshake cost is roughly constant and the expert phase inside it scales with the active experts, so the gain
  appears only where the three programs' launch cost dominated. Still far from the 13 ms the floor needs.
- 2026-11-04 07:00 CORRECTION from the profiled unit test (`tracy_unit_persist`, device 0, traced): consecutive
  programs show ~0 us op-to-op gaps (the dispatcher writes a program's config while the previous one runs), so in a
  traced decode the launch count is NOT the cost; the persistent block's period (~560 us) is the sum of its three
  parts (~488 us) plus ~30 us of in-program handshakes. The step is bound by the ops' device time at tiny batch:
  layer 4 at 32 users = MHC 52 + 19 + 52 + 19, all-reduces 42 + 24, qkv 69, KDA 72, experts 215, shared 36,
  router 30, o-proj 27, small matmuls 23 (us). At 8 users the experts shrink, the rest stays: ~730 us/layer.
  The 50 tok/s/u floor (440 us/layer) therefore needs the small-op chain cut: the four MHC phases (142 us), the two
  all-reduces (66-80 us), the shared expert overlapped with the routed experts (36 us), the router folded (30 us).
  The persistent program remains the vehicle (phases on disjoint cores overlap, exchanges fused), but the order of
  work changes: MHC + all-reduce latency first. `chain_tracy_b1` profiles layer 4 at 8 users (today's path and the
  block) for the budget in the floor's regime.
- 2026-11-04 07:07 `gate84_B4_persist2` (step 2, resident x by multicast) PASSED bit-exact but SLOWER: step 40.42 ms
  vs 38.37 (step 1) / 37.69 (today); unit 0.691 vs 0.646 (step 1) / 0.603 ms. Eight 16-tile chunk multicasts into
  two 60-core rects (path-reserved, serialized on the workers' NOC) cost ~45 us per layer more than the x_g write
  plus the gate|up cores' own 256 KB reads they replace. Unless the layer profile (`tracy_l4_b4_persist2`) shows the
  loss elsewhere, step 2 is shelved: step 1's x path stays.
