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
