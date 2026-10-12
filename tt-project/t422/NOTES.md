# t422: upstream #58676 (tile-row AdaLN gather) on the Turbo branch

Code: branch ttp/t422-port-upstream-58676-tile-row-adaln-gathe, commit 12aff07c462 (on bff7d82040c).
- Cherry-pick of fbd0af7e10f without the schedule cache (trace-replay hazard #59645).
- Conflicts resolved: the op keeps our preserve_rope_rounding (CBs c_22-c_24), and the tile-row map moved to
  CB c_25. With MINIMAX_H3_ADALN_PRECOMPUTE=1, the precomputed tables are used and the norm-weight fold and the
  tile-row map are off (the tables do not include the norm weight).
- Host test test_adaln_tilerow_minimax_h3.py passes (7 tests).

A/B on blx01 (fl2va, 768p, 5 s, seed 0, kf_t2va_f0.png, CALIBRATED_FOX_PROMPT, v1.0 4-step adapter, 2 replays, warm
deadline 400 s):
- off: /var/tmp/fasth3/t408/b (bff7d82040c, t284 build).
- on: /var/tmp/fasth3/t422/b (12aff07c462, its own Release build: the port changes C++, so the t284 build
  cannot be hardlinked).
- Judgment call: the two arms run as adjacent jobs, not one job. One 5 s arm takes about 250-330 s of process
  wall (t400/t408), so a pair at +50% needs about 750-1000 s, over the 600 s cap. Same call as t400/t421.
- Steps: setup422-blx01.sh, detached on blx01 (log /var/tmp/fasth3/t422/detach/setup422.log, marker
  /var/tmp/fasth3/t422/setup.marker). Then chain422.sh, detached on g15blx02, waits for the marker and for no
  other smarton job on blx01, and submits specs/off.txt then specs/on.txt. Its log is
  tt-project/t422/state/chain.log and ends with "T422_DONE <rc>".
- Results land in /var/tmp/fasth3/t422/out_{off,on}/run.log, with md5sum lines for the mp4s.
- Next: compare per-step denoise times (relative, at the clamp) and the mp4 md5s.

## 2026-10-12 run 1598
- off arm: blx01 job 860, completed (264 s). Per step 1.4935 s / 1.5264 s (2 replays), Total Pipeline 8.0932 s /
  8.8756 s, process wall 255 s, mp4 md5 c79b7d37ba9c0f5034b3392af1f1a841 (out_off).
- on arm: refused twice by the job lint (setsid in run422.sh; the lint got stricter after the off arm ran).
  run422.sh now runs pytest in the broker's group (no setsid; TERM to timeout, then KILL). Deployed to blx01.
- submit_on.sh (detached on g15blx02, run dir 1598, t422on.rc) queues the on arm behind job 864 (another
  smarton job) and logs to state/on.log, ending with "T422ON_DONE <rc>".
- Next: read state/on.log; compare per-step times (relative, clamp) and md5 against the off arm.

## 2026-10-12 run 1604
- on arm job 865 failed (440 s, exit 1): "[t286] warm deadline 400 s passed in Warming audio decode lengths"
  at 6/12, 58 s/it (off arm: 12/12 at 4 s/it). Cause: cold JIT for the new build (1889 kernel compiles vs 199
  on the off arm), not a code error. No drop (fabric-check 866 passed). Cold log kept as out_on/run.cold.log.
- Rerun: blx01 job 867 (submit_on.sh, detached as run 1604 t422on2), log state/on.log.
- Next: read state/on.log; compare per-step times and md5 with the off arm (job 860).

## 2026-10-12 run 1605
- on arm job 867 failed again at the 400 s warm deadline, now in "Warming prompt encoder keyframe layouts". The JIT
  cache grew: 3446 kernel lookups (1889 hits) vs 6329 for a full warm run on the off arm, so about 2700 kernels
  are still cold. Audio decode is now cached (+2180 programs). No drop; not a code error.
- Judgment call: warm deadline 500 s, late limit 430 s (pytest timeout 560 s, job -t 600 s) for the on arm, to fill
  more cache per job. The replays are timed after warmup, so the deadline does not change the measurement.
- submit_on.sh now tries up to 2 jobs; the second only if the first stopped at the warm deadline (log kept as
  out_on/run.fill<try>.log). Detached as run 1605 t422on3; log state/on.log ends with "T422ON_DONE <rc>".

## 2026-10-12 run 1606
- Drop: 2026-10-12 07:30 UTC, blx01, after job 868 (smarton, task t421, not ours). Fabric-check 870 failed (-9),
  hold 871 "chip(s) left PCIe bus", bridge-reset 872 of chip 24 failed; broker holds the device (degraded, 1/32 off).
  Our on-arm try (run 1605) refused at submit, no job of ours ran.
- submit_on.sh now also waits (2 min steps, up to 5 h) while the broker holds the device or the box is unreachable.
  Relaunched detached as run 1606 t422on4; log state/on.log ends with "T422ON_DONE <rc>".
- The submit then passed health but the lint refused: /var/tmp/fasth3 on blx01 was 453 GB (cap 450). Deleted
  /var/tmp/fasth3/t400/b (3.0 GB; #400 done, referenced only by t400's own scripts) -> 450 GB. Kept t284/b (base build
  for t286/t408/t409/t420/t421 setups) and t408/b (our off arm). cache/dit-h3hf (152 GB) and
  cache/tt-metal-cache-h3hf (45 GB) are most of the footprint.
- Relaunched as run 1606 t422on5: on arm is blx01 job 875 (queued 07:34 UTC). Log state/on.log, "T422ON_DONE <rc>".
- Next: read state/on.log; compare per-step times (relative, clamp) and md5 with the off arm (job 860).

## 2026-10-12 run 1610
- on arm job 875 failed at the 500 s warm deadline (530.7 s, exit 1), now in "Initializing bucket buffers: 120320,41984",
  further than job 867. Still cold JIT for the new build, not a code error. No drop: post-job health passed (fabric OK).
- The fill retry was refused by the lint: /var/tmp/fasth3 on blx01 at 455 GB (cap 450). The shared JIT cache
  cache/tt-metal-cache-h3hf grew 45 -> 50 GB with our fills.
- Nothing safe to free: t409/b (#416 queued), t420/b (#426 queued), t428 main+t48 (12.6 GB, #428 queued), t48 (used by
  t420/t421/t428 envs), t421/t423/t427 (queued). home/.cache/tt-metal-cache (5 GB) is live; its entries untouched since
  10-08 total only 0.2 GB. Judgment: no deletion of other tasks' data; handed off waiting on disk with a cleanup followup.
- Next (once du /var/tmp/fasth3 < 448 GB on blx01): relaunch tt-project/t422/submit_on.sh detached (same as run 1606),
  then compare per-step times and md5 with the off arm (job 860: 1.4935 / 1.5264 s/step, md5 c79b7d37ba9c0f5034b3392af1f1a841).

## 2026-10-12 run 1616
- /var/tmp/fasth3 on blx01 back to 439 GB (others cleaned up). Relaunched submit_on.sh detached as run 1616 t422on6
  (rc file state/runs/1616/t422on6.rc; log tt-project/t422/state/on.log). At launch another smarton job held blx01's
  queue, so the submitter waits in 2 min steps.
- Next: read state/on.log; compare per-step times and md5 with the off arm (job 860).

## 2026-10-12 run 1622 (result)
- On arm blx01 job 886 passed (08:29 UTC, exit 0, no drop). Per-step 1.4042 / 1.3866 s vs off (job 860) 1.4935 / 1.5264 s:
  -6.0% / -9.2%, mean -7.6% (900 MHz clamp, relative only). mp4 md5 on = off = c79b7d37ba9c0f5034b3392af1f1a841: bit-identical.
- Code 12aff07c462 on ttp/t422-port-upstream-58676-tile-row-adaln-gathe. Done.
