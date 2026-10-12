# t423 notes (trace the FastH3 video VAE decode waves)

Code commit: 5a8c59583a9 (traced_decode flag, per-wave dispatch/wait profile, test_decode_trace_ab).
blx01 tree: /var/tmp/fasth3/t423/b (setup423-blx01.sh, python-only on the t284 build, marker rc=0).
Job script: /var/tmp/fasth3/t423/run423.sh <stage|ab> (copy in tt-project/t423/run423.sh); env423.yaml.
Logs: /var/tmp/fasth3/t423/out_<mode>/run.log on blx01.

## Step 1
blx01 broker job 852 (submitted 2026-10-11 23:02 PDT, -t 600, unmeasured): `run423.sh stage`
= test_decode_stage yuv420 5s with MINIMAX_H3_VAE_PROFILE=1. Read the "dispatch per wave" and
"wait per wave" lines (and DECODE_STAGE total).

## Next step
If dispatch per wave is well above the device wait: device-lint, then submit
`bash /var/tmp/fasth3/t423/run423.sh ab` (-t = step-1 time +50%, <= 600) on blx01; the A/B gate
is bit-identical or PSNR >= 40 / PCC >= 0.999 on seeds 0-4. If it passes, cherry-pick 5a8c59583a9
onto a -land branch off origin/ttp/t48-ltx25-integrated and `ttp push --detach`; `ttp push --own --detach`.

## Drops
2026-10-12 05:43-05:55 UTC, blx01, broker jobs 842-848: chip 11 bridge-reset/health-gate, recovered
by the broker (hold 848 ended 05:55 UTC). Not our job (no t423 job was running).
