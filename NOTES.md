# #81 LTX_SDPA_EXP_APPROX A/B — status

Code: commit 8db2db59115 on ttp/t81-ltx-ring-sdpa-exp-approx-mode-a-b-ltx-sd (pushed). Env LTX_SDPA_EXP_APPROX
(default 0) sets exp_approx_mode on all 5 SDPA configs in attention_ltx.py. CPU test
test_sdpa_exp_approx_env_reaches_every_sdpa_config passes.

Device job: blx03 broker job **040** (queued 2026-10-01 14:45 UTC).
- Script: g14blx03:~/fasth3/t81/run81.sh (copy in tmp/t81/). Build/kernels = ~/fasth3/t48 @a613d669ee
  (contains #57180, so flag 0 really is accurate exp); models/tt_dit = tmp/t81/src (rev 8db2db59115).
- Why blx03, not g15blx02: g15blx02's built tree (fasth3-opt, Sep 30) lacks #57180, so it would
  run approx exp in both arms.
- Logs: g14blx03:~/fasth3/t81/run81.log and /var/log/tt-device-broker/2026-10-01_144544_040.log

Log lines (grep T81_):
- T81_BLOCK flag=F F,H,W=.. ms_per_block=..  traced AV block 0, real 22B weights, Linear 2x4 sp1/tp0
- T81_AB S1|S2 off_ms on_ms delta pcc_video pcc_audio  (flag 1 vs flag 0)
- T81_TORCH flag=F pcc_vs_torch rel_rmse  (video-only diffusers block, S1 grid, scaled random weights)
- T81_TORCH_AB pcc_flag1_vs_flag0;  T81_FAIL ..;  T81_EXIT=rc

Projection to 4x8 (sp=8/tp=4 does ~1/4 the per-chip SDPA work of 2x4 sp=4/tp=2):
  e2e saving ≈ (8·Δ_S1 + 3·Δ_S2)/4 × 48 blocks. Ring SDPA overlaps the CCL, so 2x4 may hide some gain.
Accept for eval pack if block time drops ≥5% and PCC ≥0.999. Default stays 0 either way.

Next step on resume: check job 040 status; check the broker log for chip drops during our job
(if any: stop-all procedure); grep T81_; compute numbers; gzip the log back to tmp/t81/;
rm -rf g14blx03:~/fasth3/t81; write result.json done.
