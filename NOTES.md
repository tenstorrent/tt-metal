# t244 notes (notes branch only)

A/B of the NA compute-config knobs DIFFVAE_NA_APPROX_EXP=1 and DIFFVAE_NA_FIDELITY=lofi on t48 @34a571c5f47 (blx01).

- Scaffolding: tt-project/t244/ (from t243). decode244.py runs the three arms in ONE process (the NA op reads both
  env vars on every call), so only one decoder load and one cold warm-up; ends with a def re-time for drift.
  Copied to blx01 /var/tmp/fasth3/t244/drv. Driver checks out 34a571c5f47 (bundle) in /var/tmp/fasth3/t238/b.
- Pass: PCC >= 0.9999 vs #214 host-noise refs and PSNR within 0.5 dB of def (55.07-55.51 dB).
- 2026-10-08 01:47 UTC: driver started on blx01 (setsid), broker job 887 (-t 330), b at 34a571c5f47.
- Next: `ssh g15blx01 cat /var/tmp/fasth3/t244/drv/driver.marker`; read drv/driver.log (summary, cmp lines),
  out/cmp_{def,approx,lofi}.json. If an arm passes and is faster: flip its default on t48 (=0 off switch, unit test
  in models/tt_dit/tests/unit/test_diffvae_ops.py like test_packed_lanes_default_on), land via a -land branch + ttp push.
  If 887 dropped, the driver reruns it itself (second drop -> skipped).

## Result (job 887, completed, no drops)
- Warm 1080p 145f decode, seeds 0,1: def 3.511 s, approx 3.512 s, lofi 3.510 s, def recheck 3.513 s. No gain.
- PCC/PSNR identical in all arms (0.999956/55.51 dB s0, 0.999957/55.07 dB s1), md5-identical to def.
- INVALID (review #245): NeighborhoodSDPAOperation::compute_program_hash does not hash compute_kernel_config,
  so with all arms in one process (def first) approx and lofi reused the cached HiFi2/exact-exp program.
  The knobs were never applied. Rerun with one process per arm: t246 below.

## t246: rerun, one python process per arm (blx01)
- tt-project/t246/: decode246.py (asserts one arm per process, prints the NA config it built), run246.sh
  (ARMLIST -> one process each), driver246.sh (job A: def approx lofi, -t 480; job B: both def, -t 330, only
  if approx and lofi both pass). Copied to blx01 /var/tmp/fasth3/t246/drv; b = t238/b at 34a571c5f47 (= t48 tip).
- 2026-10-08 01:58 UTC: driver started (setsid), job A = broker job 889.
- Next: `ssh g15blx01 cat /var/tmp/fasth3/t246/drv/driver.marker`; read drv/driver.log (MD5 lines, cmp lines),
  outA/run.log (DECODE seed lines per arm), outA/cmp_*.json, outB/ if run.

## t246 result (blx01 broker job 889, t48 @34a571c5f47, one process per arm, seeds 0,1)
| arm | warm s (s0, s1) | mean | vs def | PCC vs ref | PSNR vs ref (s0/s1) | md5 s0 |
|---|---|---|---|---|---|---|
| def | 3.510, 3.515 | 3.513 | - | 0.999956/0.999957 | 55.51/55.07 | ef3940530a17 |
| approx (DIFFVAE_NA_APPROX_EXP=1) | 3.381, 3.376 | 3.378 | -0.135 s | 0.999951/0.999952 | 55.04/54.59 (-0.47/-0.48 dB) | 13802b012e19 |
| lofi (DIFFVAE_NA_FIDELITY=lofi) | 3.568, 3.628 | 3.598 | +0.085 s | 0.999874/0.999871 | 49.21/48.65 | 64e9dd4de0a6 |
All arms md5-distinct (knobs apply in fresh processes). Lofi fails quality and is slower: rejected.
Approx passes the rule (PCC>=0.9999, within 0.5 dB, borderline) and is 135 ms faster: to land default-on with =0 off switch.
Outputs: g15blx01:/var/tmp/fasth3/t246/outA. Job B (combo) not run since lofi failed.
The earlier #244 'both knobs inert' result was wrong: one process reused the cached program (compute_program_hash omits compute_kernel_config).
