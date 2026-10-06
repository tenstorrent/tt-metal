# t170 — t95/t164 eval pack on blx01 (follow-up of #164)

Branch ttp/t170-... = ttp/t164-eval-pack-g15 @ a4b6a835d1a (+ t170 scripts in tmp/t170).
Box: blx01 (g15blx01), everything under /var/tmp/fasth3/t170 (nothing under blx01 /home).
- tree/: hardlinked /var/tmp/fasth3/t48 (bf7db12a149) models/ + the 8 python files that differ in a4b6a835d1a
  (OVERLAY_COMMIT). Build, kernels, JIT cache: t48 tree + /var/tmp/fasth3/cache (job 621's, warm).
- Default warmup (no LTX_WARMUP_T2V_ONLY/ENCODERS cuts: those dropped tray 3 on blx01 in jobs 625/640).
- driver.sh: pack (configs.txt) -> post.py PCC/PSNR vs own baseline -> pick 1-2 -> baseline5 + <cfg>5
  (seeds 0-4, default prompt) -> post.py. Marker /var/tmp/fasth3/t170/driver_pack/DRIVER.done; pid in driver.pid.
  Results /var/tmp/fasth3/t170/res/<label>/, summaries res/summary_configs{,5}.md.
- Timeouts: baseline 300 s; knobs = 1.5 x measured baseline wall + 120 (<= 600); one retry at 600 on timeout.

## Run log
- 2026-10-06 22:35 UTC: driver started on blx01 (pid 203047). Baseline = broker job 665.

## Next (after DRIVER.done)
- scp the mp4s of baseline5/<cfg>5 (and baseline + winner) to g15 tt-project/data/g15/t170/ (mp4s only),
  run VBench + per-seed PCC/PSNR vs baselines/ltx25_1080p_6s/ref_dv145 (ltx_eval batch --vbench-ref) on g15,
  look at stills, write the table + recommendation, commit, land on ttp/t48-ltx25-integrated.
