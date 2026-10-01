# t87 notes
- Branch ttp/t87-vae-trace-2x4-check @7d877a1e498 (pushed), from t86 8cfb050c34a. Adds
  models/tt_dit/tests/models/ltx/test_vae_ltx_trace_ab.py + tmp/blx03/{run87,stage87}.sh.
- No new build: t86 differs from the blx03 t48 build (a613d669eef) only in Python. run87.sh uses
  TT_METAL_HOME=~/fasth3/t48 and a models/ overlay at /var/tmp/fasth3/t87/src (staged by stage87.sh, 131 MB).
- blx03 job 063 submitted 17:57 UTC (full 4x8 open + create_submesh(2,4); eager warmup + 3 eager, capture, 3 replays;
  trace_region_size 500 MB). Broker log /var/log/tt-device-broker/2026-10-01_175754_063.log, our log /var/tmp/fasth3/t87/run87.log.
- Pre-submit: blx03 power-cycled 10:26 PDT after ltx-host job 051 dropped tray 1 (chips 8-15); not ours. Our t83 job 062 ran fine after.
- Next: ssh g14blx03 tt-device-mcp status -j 063; grep -E "AB |AB_CMP|VAE_DECODE_SPLIT|T87_EXIT|passed|failed|Error" /var/tmp/fasth3/t87/run87.log.
  Check the server.log for DEAD CHIP / power-cycle between 17:57 and job end (if during our job: stop all device work).
- Cleanup after: ssh g14blx03 rm -rf /var/tmp/fasth3/t87 (keep trimmed log copy in tmp/blx03/t87res/).
