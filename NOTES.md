# t249: DiffVAE next lever toward 1 s (after NA approx exp)

Base: t48 @946a37952bd, DiffVAE 1080p 145f 4x8 decode 3.378 s (blx01 job 889).

## Lever picked: NA query chunk (2,1,1) -> (1,1,1)  (DIFFVAE_NA_CHUNK_BRICKS=1,1,1, env already exists)
The bricked NA op computes every query brick of a chunk against the chunk's WHOLE gather, so the score
tile pairs per shard are chunks x bricks/chunk x gather. Measured with the real host planner
(ttnn.transformer.neighborhood_plan on blx01 t238/b, t249drv/plan249*.py):

| stage (volume, brick)            | chunk (2,1,1) pairs | chunk (1,1,1) pairs | change |
|---|---|---|---|
| s5 (145,272,480) b(2,4,4) H4xW8 key-phase | 2,113,440 (gather 112) | 1,787,040 (gather 96) | -15.4% |
| det (21,68,120) b(8,4,1)  | 64,260 | 48,195 | -25% |
| det (41,68,120) b(16,2,1) | 91,800 | 68,850 | -25% |
| det (81,136,240) b(8,2,2) | 440,640 (gather 36) | 302,940 (gather 27) | -31% |

The (2,1,1) default dates from t12 (it halved K/V reads and work items when the op was read-bound).
NA is now softmax/SFPU bound (~12% matmul util), so fewer score tiles should win; K/V reads double.
Bigger chunks (1,2,1), (1,1,2), (4,1,1), (1,2,2) all add pairs. Risk: gather 27 -> tiles_per_kv_chunk 3
(largest divisor <= 8), short kv chunks; DIFFVAE_NA_KV_CHUNK_TILES is the follow-up knob.

## A/B job (blx01, detached driver)
Driver /var/tmp/fasth3/t249/drv/driver249.sh (copy in t249drv/), started 2026-10-08 02:29 UTC; it waits for
t247's driver (job 898) to end and the broker health check, then submits ONE broker job, -t 360:
run249.sh "base c1" on t238/b @34a571c5f47 (both arms DIFFVAE_NA_APPROX_EXP=1 = t48 defaults), one process per arm,
warm-up + seeds 0,1 timed + host-noise seeds 0,1 scored vs #214 refs (diffvae/ref) and c1 vs base.
Marker: /var/tmp/fasth3/t249/drv/driver.marker; log driver.log; outputs /var/tmp/fasth3/t249/outAB.

## Next step on wake
Read driver.log (DECODE_MEAN per arm, cmp_*.json). Pass = faster, PCC >= 0.9999, PSNR within ~0.5 dB of base
(base ~55.0/54.6 dB vs refs). If pass: change `_query_chunk_bricks` stride-1 default to (1,1,1)
(DIFFVAE_NA_CHUNK_BRICKS=2,1,1 restores old), add a default test, land on t48 via ttp/t249-land + ttp push --detach.
If mixed (s5 vs det): add a per-volume keyed DIFFVAE_NA_CHUNK_BRICKS like DIFFVAE_NA_BRICK and A/B per stage.

## Result (blx01 job 900, 2026-10-08 02:36 UTC, no drops)
- base (2,1,1): DECODE_MEAN_S 3.378 s; c1 (1,1,1): 5.762 s (+71%). Quality equal (PCC 0.99995, PSNR 54.6-55.0 dB both).
- REJECTED: doubled K/V gather reads and per-chunk overhead outweigh the 15-31% fewer score tiles. Do not redo. No code change.
- Next: pick another lever (fused stage-5 blocks / stage-5 conv+norm path; profile first). Bigger chunks (e.g. (4,1,1) or (2,2,1)) may be worth one A/B given this direction.

## Run 2: chunk (4,1,1) A/B (c4), started 2026-10-08 02:44 UTC
Why: c1 cost far more (+298 ms per stage-5 block) than a pairs/reads model allows, which points at
per-work-item (chunk) overhead or K/V reads. (4,1,1) halves the work items on every volume
(host planner, plan249*.py on blx01):

| volume, brick | (2,1,1) pairs / chunks / gather | (4,1,1) pairs / chunks / gather |
|---|---|---|
| s5 (145,272,480) b(2,4,4) | 2,113,440 / 9435 / 112 | 2,790,720 (+32%) / 4845 / 144 (K/V reads -34%) |
| det (21,68,120) b(8,4,1) | 64,260 / 510 / 63 | 64,260 / 255 / 63 |
| det (41,68,120) b(16,2,1) | 91,800 / 1020 / 45 | 91,800 / 510 / 45 |
| det (81,136,240) b(8,2,2) | 440,640 / 6120 / 36 | 660,960 (+50%) / 3060 / 54 |
(2,2,1) and (2,1,2) give about the same s5 pairs as (4,1,1); (8,1,1) +100%: not tried.

Driver: same driver249.sh (now job AB4 "base c4", -t 330 = job 900's 213 s +50%), outputs
/var/tmp/fasth3/t249/outAB4, marker /var/tmp/fasth3/t249/drv/driver.marker (run-1 marker kept as
driver.marker.ab1). It waits for the broker to be free of smarton jobs (t252 was running at start).
On wake: read driver.log (DECODE_MEAN per arm, cmp_*.json). If c4 is faster and passes, change the
stride-1 default in `_query_chunk_bricks` to (4,1,1) (DIFFVAE_NA_CHUNK_BRICKS=2,1,1 restores the old one) and land.
If mixed, a per-volume chunk knob (det stages 2-3 lose nothing at (4,1,1)) is the follow-up.

## Profile facts for the lever after this (t240 deep profile, before packed lanes and approx exp)
Decode 4212 ms: det stages 1425 ms, stage 5 2786 ms. Per stage-5 block 325 ms: neighborhood-sdpa 168,
qkv-lanes slice+norm+rope 48, halo+brick-permute (k,v) 46. Det stage 1 (21,34,60) dim 2048: 621 ms,
of which linear-order attention 299 (gather + dense masked SDPA), qkv-to-volume 93, MLP 102 (replicated).
Stage-1 bricked replicated NA would be slower (full volume per chip). The real fix is P6(b): split stage 1
over the mesh (e.g. sp on mesh axis 0, W 60/4 = 15, plus heads TP on axis 1) with the W-sharded bricked kernel.

## Result chunk (4,1,1) (blx01 job 905, 2026-10-08 02:56 UTC, no drops)
base 3.374 s, c4 6.142 s (+82%). Quality same (PCC 0.99995, PSNR 55.04/54.59 vs refs; c4 vs base 58 dB).
Rejected. Both (1,1,1) and (4,1,1) are much slower: the (2,1,1) default is the optimum; chunk size is not a lever.
Next: stage-1 de-replication / P6(b) (see above).
