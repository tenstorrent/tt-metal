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
