# t221: LTX-2.3 bf16 e2e, BH 4x8 (blx01), 1088x1920 145f, seeds 0-4 (user #156)

Counterpart of #220/#251 (8-bit, broker job 909, mean 5.756 s, steady 5.37 s). Same tree (blx01
/var/tmp/fasth3/t220/src, commit d791f4e9497), same caches/JIT dir, same env (LTX_QUALITY=medium schedule:
6 S1 + 1 S2 steps, LTX_TRACED=1, LTX_ATTN_FABRIC_AGMM=0, BWE/VOC trace off), same prompt and test. Only change:
`LTX_QUANT=` (empty), which apply_quality_env's setdefault keeps, so the DiT runs bf16/HiFi2 baseline and its cache
goes to cache/dit-ltx23/ltx-2.3-22b-distilled-1.1/transformer (~45 GB new, on blx01 root fs, 181 GB free at start).

Files on blx01 /var/tmp/fasth3/t221: run221.sh, env.yaml, drv221.sh, cmp221.py (copies here).
Driver (started 2026-10-08 ~03:07 UTC, setsid): fill1..3 (seed 0, until one completes) -> time (seeds 0-4,
out_time/) -> cmp221.py bf8 vs bf16 seed 0 (out_time/cmp.txt, still_bf16_seed0_f72.png, still_bf8_vs_bf16_seed0_f72.png).
Marker: /var/tmp/fasth3/t221/drv221.done; log drv221.done.log (job ids, statuses).

Next: when the marker exists, read it; if TIME_DONE completed, parse out_time/run.log (E2E_WALL_S lines + stage
tables) and cmp.txt and report the table vs #251. If FILL_NOT_OK, read out_fill*/run.log (OOM? timeout? drop?).

## Attempt 1 result (2026-10-08 03:15 UTC): DROP, job 914 killed
- Fill job 913 completed (03:07-03:12 UTC, 286 s): bf16 DiT cache + JIT built, so bf16 fits on 4x8.
- Timed job 914 (03:12:39 UTC) killed by broker device recovery at 03:15:13 UTC: tray 4 (chips 16-23,
  0000:c1..c8) left the PCIe bus during seed 4 (gen 5, stage 1 step 4). Our job; no crash before it.
  Broker escalated to its own full galaxy reset, then blx01 dropped off the network (~03:20 UTC, likely power cycle).
- Partial warm walls before the drop (out_time/run.log): gen0 seed0 31.915 (cold), gen1 seed0 7.225,
  seed1 5.719, seed2 5.655, seed3 5.700 s. Stage 1 step ~275 ms (bf16). Videos out_time/ltx_av_fast_1920x1088_{0..4}.mp4
  (file index = gen index, not seed).
- Rerun: drv221b.sh (time2 tag -> out_time2/, marker drv221b.done) started 03:19 UTC; it waits for the broker to be
  free of hold/health/reset jobs. If blx01 rebooted, the driver is dead: restart it with
  `ssh g15blx01 'cd /var/tmp/fasth3/t221; setsid nohup bash drv221b.sh > drv221b.out 2>&1 < /dev/null &' < /dev/null`
  (only after checking `pgrep -u smarton -f "bash drv221b.sh"` is empty and no drv221b.done exists).
- Same config dropping twice in a row on blx01 -> skip there and report the partial numbers above.

## Attempt 2 (2026-10-08 05:18 UTC)
- blx01 back (rebooted ~04:53 UTC, broker recovered, DiffVAE jobs 946/947 ran clean after). blx03 HELD (tray 2,
  chips 8-15 off the bus since 05:01 UTC), no t221 job there.
- drv221b.sh copied to blx01 and started (pgid 85242). Timed job 948 (seeds 0-4, out_time2/) running from 05:18:17 UTC.
- Marker /var/tmp/fasth3/t221/drv221b.done. Next: parse out_time2/run.log + cmp.txt, report vs #251.

## Result (2026-10-08, standard wake): DONE
Job 948 (blx01, 05:18:17-05:21:47 UTC, commit d791f4e9497) vs #251 job 909 (bf8, same tree). Stage times from
log timestamps (parse221.py; s1 includes ~0.1 s denoise init, audio = mel-VAE + vocoder, export = mp4 mux).

| run | gen/seed | wall | enc | S1 (6 st) | upsample | S2 (1 st) | VAE | audio | export |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1/0 (first warm) | 7.348 | 1.731 | 1.786 | 0.144 | 0.965 | 0.698 | 1.217 | 0.798 |
| bf16 | 2/1 | 5.624 | 0.189 | 1.766 | 0.148 | 0.970 | 0.699 | 1.194 | 0.649 |
| bf16 | 3/2 | 5.721 | 0.181 | 1.780 | 0.148 | 0.972 | 0.699 | 1.189 | 0.744 |
| bf16 | 4/3 | 5.744 | 0.180 | 1.764 | 0.145 | 0.973 | 0.694 | 1.240 | 0.737 |
| bf16 | 5/4 | 5.600 | 0.184 | 1.775 | 0.139 | 0.965 | 0.696 | 1.182 | 0.647 |
| bf16 mean 5 seeds | | 6.007 | 0.493 | 1.774 | 0.145 | 0.969 | 0.697 | 1.204 | 0.715 |
| bf16 mean seeds 1-4 | | 5.672 | 0.184 | 1.771 | 0.145 | 0.970 | 0.697 | 1.201 | 0.694 |
| bf8 mean 5 seeds (909) | | 5.756 | 0.554 | 1.599 | 0.152 | 0.877 | 0.693 | 1.211 | 0.661 |
| bf8 mean seeds 1-4 (909) | | 5.451 | 0.257 | 1.597 | 0.151 | 0.877 | 0.691 | 1.202 | 0.665 |

Step: bf16 S1 275 ms, S2 814 ms; bf8 S1 247 ms, S2 722 ms (+11-13%). Cold gen0: bf16 31.9 s, bf8 32.4 s.
Quality: bf16 vs bf8 seed 0 PCC 0.425, PSNR 13.6 dB. Visual check of frame 72: both clean, same prompt content
(singer, guitar, stool), different pose/framing. Divergent sampling from the weight precision change, not a
broken bf16 output. Artifacts on g15blx02 tt-project/t221/: bf16_seed0.mp4, still_bf16_seed0_f72.png,
still_bf8_vs_bf16_seed0_f72.png (untracked). blx01 originals: /var/tmp/fasth3/t221/out_time2/.
Cleanup: removed the 37 GB bf16 DiT cache (t220/cache/.../ltx-2.3-22b-distilled-1.1/transformer, built by fill 913).
Drops: 1 (2026-10-08 03:15:13 UTC, blx01, job 914, tray 4 chips 16-23, ours). Rerun 948 clean.
