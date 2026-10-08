# t244 notes (notes branch only)

A/B of the NA compute-config knobs DIFFVAE_NA_APPROX_EXP=1 and DIFFVAE_NA_FIDELITY=lofi on t48 @34a571c5f47 (blx01).

- Scaffolding: tt-project/t244/ (from t243). decode244.py runs the three arms in ONE process (the NA op reads both
  env vars on every call), so only one decoder load and one cold warm-up; ends with a def re-time for drift.
  Copied to blx01 /var/tmp/fasth3/t244/drv. Driver checks out 34a571c5f47 (bundle) in /var/tmp/fasth3/t238/b.
- Pass: PCC >= 0.9999 vs #214 host-noise refs and PSNR within 0.5 dB of def (55.07-55.51 dB).
