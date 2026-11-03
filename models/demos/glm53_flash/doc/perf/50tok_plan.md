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

## Gates

* coherence: `DECODE-IDS` line of the chain test identical to the frozen tip for levers 1 and 3 (TP changes keep the
  math up to reduction order; a program fusion must be bit-exact or PCC-gated per op as today's custom ops are);
* accuracy: lever 2 is judged on GPQA Diamond + LiveCodeBench + MMMU through the local release flow (not the id chain);
* every served check at the extreme ISLs first ([[test-extreme-isl-first]]).

## Log

* 2026-11-03 21:36 lever 1 probe: `tp8_moe_1x8` (TT_VISIBLE_DEVICES=0,1,2,3,27,26,25,24, descriptor
  `_logs/mesh/bh_galaxy_1x8_ring.textproto`, GLM53_MESH_SHAPE=1,8): the 1x8 ring opened under FABRIC_2D; result below.
