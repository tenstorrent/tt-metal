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
