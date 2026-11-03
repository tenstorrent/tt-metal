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
