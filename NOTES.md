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
