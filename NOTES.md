# t17 notes

Branch reset onto the t36 line (16ba9a383dc, what blx03 runs). Commit 9a02249fdfb = the A/B change.

## Finding (CPU, tmp/t17/check_keys.py on blx03)
- The LTX-2.5 conv decode path uses the 2.3 monolith's conv decoder (LTX25_VIDEO_VAE = ltx-2.3-22b-distilled-1.1;
  the real 2.5 conv VAE file is not downloaded). Its 11 conv3d sites at 1080p/145f (and 153f) ALL hit exact
  swept _BLOCKINGS entries (misses=0). Nothing untuned to sweep.
- DiffVAE (NA decoder) uses no conv3d at all.
- The fallback/T-relaxed warnings in conv145 logs are the I2V image encoder (built at num_frames=1, T=3) and the
  audio Conv2d/1dViaConv3d (keyed without geometry, h=w=1). Not on the T2V video decode path.
- Lead: the newer 153f sweep (upstream #57265) picked H_out=16,W_out=2 at the SAME per-device H/W as 5 of the
  145f sites (s1_res, s1_up, s2_res(+compress_time), s3_res, s3_chg), with 30-40% lower per-frame time than the
  older 145f entries (which were 4x8/8x4). C_in/C_out/T blocks identical -> expect bit-identical output.

## Device A/B (single config)
Job 884 FAILED (pytest timeout 480s): run25.sh exports TT_METAL_HOME=$W, overriding env.yaml -> t17 kernel
paths -> cold JIT. Log kept as conv145_t17/run_884_cold.log. Rerun via detached blx03:~/fasth3/drive17.sh
(tmp/t17/drive17.sh): waits for the one project slot, then submits with trailing TT_METAL_HOME=main tree,
PYTHONPATH=t17:main/ttnn:main/tools (verified: ttnn from main, conv3d.py from t17). Job id in blx03:~/fasth3/drive17.log.
Done check: tmp/t17/done17.sh. Then run compare.sh as below.

blx03 job 884 (882 was killed: new TT_METAL_HOME made the JIT cache cold; 884 runs t17 python with TT_METAL_HOME=main tree):
baseline job 879 (conv145_t20, 16ba9a383dc): gen#1 VAE decode (conv) 0.72s, E2E 8.761s.
Output: blx03:~/fasth3/out/ltx25_1080p_6s/conv145_t17/{run.log,ltx_av_fast_*.mp4}
Check: ssh g14blx03 tt-device-mcp status 1 | grep -w 884
Next: grep -E "LTX_TIME|decode|E2E" run.log for gen#1; compare mp4 vs conv145_t20 (ffmpeg psnr / md5 of decoded
frames). Faster + identical -> keep, fix comments with measured numbers. Slower -> revert 9a02249fdfb.
Cleanup after: ssh g14blx03 'cd ~/fasth3/tt-metal && git worktree remove --force ~/fasth3/t17'.
Compare: ssh g14blx03 bash -s < tmp/t17/compare.sh

## 2026-10-01 rescope (no full-mesh runs on blx03)
drive17.sh died with the ~23:33 blx03 reboot before submitting (no drive17.log, queue empty). The conv145 4x8 A/B is
barred now. Replaced by a single-chip (1x1) microbench: tmp/t17/test_blk_ab.py times the 5 changed sites with old vs
new blocking (trace-based, HiFi2, per-device 4x8 input shapes) and checks old/new outputs are identical.
Job: blx03 broker 987, wrapper tmp/t17/run_ab.sh, log blx03:~/fasth3/out/t17_ab.log (AB_ROW / AB_TOTAL / T17AB_EXIT).
Check: ssh g14blx03 tt-device-mcp status 1 | grep -w 987
Next: new faster at every site + identical -> keep 9a02249fdfb, put measured us in the table comments.
Any site slower -> revert that row. Chip drop -> stop, report job/chips/time.

## Result, job 987 (2026-10-01 02:03, 1x1 on blx03, AICLK capped 1150 MHz, no chip drop)
| site | old (4x8/8x4) us | new (16x2) us | identical |
|---|---|---|---|
| s1_res | 2090 | 2974 | yes |
| s1_up | 19035 | 25498 | yes |
| s2_res | 14875 | 17611 | yes |
| s3_res | 6640 | 7481 | yes |
| s3_chg | 13146 | 14706 | yes |
| total | 55786 | 68271 (+22%) | PCC 1.0, maxabs 0 |
New blocking is slower at every site -> reverted to the 145f table (no net change to conv3d.py vs 16ba9a383dc).
The 153f-sweep H16xW2 win does not carry over to 145f: T and the T block differ, so the per-frame numbers aren't comparable.
Old per-call times run ~9% above the table comments, consistent with the 1150 MHz clock cap.
Raw rows: tmp/t17/job987_result.txt. blx03 worktree ~/fasth3/t17 to be removed (done below).
