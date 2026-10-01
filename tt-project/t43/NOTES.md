# t43 notes: conv VAE decode op breakdown (2x4, blx03)
Files: test_prof_conv_vae_2x4.py, run43.sh, analyze43.py (copies in blx03:~/fasth3/t43).
Setup: 544x960/145f on 2x4 = same per-chip shard and pad masks as 1080p/145f on 4x8. Random weights, real
2.5 latent (t37 job 931 gen0, center crop). Tracy device profile, one forward, AICLK cap 1150.
Job 988: setup error (root conftest not loaded from outside tree) -> fixed with -p conftest. Job 989 = real run.
Output: blx03:/var/tmp/fasth3/t43/prof/reports/<ts>/ops_perf_results_*.csv, log ~/fasth3/t43/run43.log.
Next: ssh g14blx03 'cd ~/fasth3/tt-metal && python_env/bin/python ~/fasth3/t43/analyze43.py <csv>'.
Cleanup after: blx03 /var/tmp/fasth3/t43/jit (keep the csv locally), ~/fasth3/t43.

## 2026-10-01 02:11-02:14 UTC: job 989 failed, blx03 went down (see INCIDENT.md)
Bare (2,4) mesh open on the galaxy failed fabric router sync on device 1 (chans 4,5 never handshook). The broker then
ran its all-link fabric check; blx03 stopped answering ssh/ping by 02:14. Device work stopped per the 22:10 rule.
Fixed locally (NOT yet copied to blx03): test opens (4,8) and runs on create_submesh(2,4), the pattern of
test_vae_ltx.py::test_prof_vae_ltx_devicetime; run43.sh sets LTX_FUSE_YUV_OUTPUT=1.
Rerun only after the user OKs device work on blx03 again:
  scp tt-project/t43/{test_prof_conv_vae_2x4.py,run43.sh,analyze43.py} g14blx03:fasth3/t43/
  ttp lock g14blx03-device -- ssh g14blx03 '~/fasth3/tt-metal/tmp/blx03/submit.sh 1200 bash /home/smarton/fasth3/t43/run43.sh'
  then: ssh g14blx03 'cd ~/fasth3/tt-metal && python_env/bin/python ~/fasth3/t43/analyze43.py $(ls -t /var/tmp/fasth3/t43/prof/reports/*/ops_perf_results_*.csv | head -1)'
Off-device estimate: conv3d = 295 ms/decode from the 4x8 _BLOCKINGS per-call times x call counts (42% of the
~0.70 s decode window in job 879, which also holds YUV + 454 MB readback).

## #57 rerun, 2026-10-01 07:14 UTC
Health before submit: blx03 up since 02:14, broker /health status=ok fsm=healthy, not held, no jobs; post-reboot
startup (992, 32 chips ARC ok) and fabric check (993) passed. (pre-step is root-only, so /health was used.)
Copied the fixed test/run43.sh/analyze43.py to blx03:~/fasth3/t43 (full 4x8 open + create_submesh(2,4); submit.sh
dry-run guard passed). Submitted as broker job 994 (cap 1200s). ONE job only; no retry on failure.
Check: ssh g14blx03 'tt-device-mcp status -j 994; tail -5 ~/fasth3/t43/run43.log'
Then analyze (command above), copy csv + analysis here, clean blx03 /var/tmp/fasth3/t43/jit.
Result (07:14-07:16 UTC): job 994 PASSED, exit 0, 116 s. Full 4x8 open + create_submesh(2,4) worked: no fabric errors,
no chip drop. After the job blx03 was the same boot (up since 02:14), had 33 /dev/tenstorrent entries, and the broker
/health showed ok/healthy. Real 2.5 latent (1,128,19,17,30), AICLK 1150. Analysis: analysis_994_fwd.txt
(analyze43.py now keeps only the ops between the start/stop signposts; the first pass, analysis_994.txt, also counted
weight-prep permutes from load time).
Per chip, forward only: 2.30 s device kernel time. conv3d 1785 ms (77%), everything else 520 ms.
CAVEAT: the conv3d number is NOT production. The _BLOCKINGS key includes (h_factor, w_factor), so 544x960 on 2x4 looks up
(2,4,...,147,34,30). That key has no entry, and every conv3d ran on fallback blockings (e.g. s3_res 46.8 ms/call vs
6.1 ms tuned on 4x8, a 7.6x gap). The non-conv ops do not depend on blocking and are valid per-chip production numbers
(at 1150 MHz).
Production estimate: conv3d ~295 ms (4x8 tuned table) + non-conv ~450-520 ms => conv3d is ~36-40% of decode device time.
Largest non-conv items:
  output path after conv_out: 208 ms total. ReshapeView [..,32071680,1]->[..,221184,145] 139 ms (row-major, rows 1
    element wide), Permute 3x1x4x4->64x4x145x1 55 ms, ReshapeView 668160x64->8017920x4 15 ms.
  NeighborPadAsync halo 118 ms (42 calls); MUL by a [1,1,8,1] mask before each conv 67 ms (42 calls);
  temporal-pad Concat 38 ms (42); LayerNorm 47 ms; ADD (resnet residual, 1024 ch) 21 ms; RgbToYuv 3 ms.
Cleaned on blx03: /var/tmp/fasth3/t43 (3.9 GB), ~/fasth3/t43. Local: ops_perf_994.csv.gz, run43_994.log.gz.
