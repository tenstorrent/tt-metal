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

| step | saving per step | step | tok/s/u |
|---|---|---|---|
| (1, 8) instance, DP4 on the Galaxy | -11 ms | 33 ms | 30 |
| `mhc.pre_block_fused` 82 -> ~20 us (90 calls) | -5.4 ms | 27.6 | 36 |
| bfp4 routed experts (accuracy-gated) | -4 ms | 23.6 | 42 |
| DSA glue: `state_composite` + views + `pool_gather` as 1-2 programs | -2.8 ms | 20.8 | 48 |
| `kda.decode` to its DRAM floor | -1.7 ms | 19 | 52 |
| MTP (K = 2-3) on top | | | 80-100 |

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
