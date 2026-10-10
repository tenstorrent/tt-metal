# t288: AdaLN precomputed vs on device, FastH3 HyperFlow 8-step, 4x8 blx03

## 2026-10-10 ~06:10 UTC (run 1292)
- Code: ttp/t288-adaln-ab = a99af1c7dea + cherry-pick acebc7d39e2 (test import fix, cc3b6de2dc7)
  + f7266236d90 (MINIMAX_H3_ADALN_PRECOMPUTE=0 keeps AdaLN on device under HyperFlow; the turbo
  test skips the two-time embedder check under the table). Python only.
- `ttp push --own` refused: its push check runs LTX tests (test_vae_ltx_*_ref.py) that do not
  exist on this H3 branch. Branch not on origin yet. Applied on blx03 as patches instead:
  ~/fasth3/t286 detached at 03848fa4df (same tree as f7266236d90; build from #286 reused).
- HyperFlow adapter: HF videorebirth/hyperflow -> blx03 /var/tmp/fasth3/models/hyperflow
  (2.8 GB, sha256 matches hyperflow.json, READ_OK). Header has hyperflow sigmas (9 pts),
  gate 0.25, shifts 12/3.
- Driver /var/tmp/fasth3/t288/run288.sh <tag> <on|off> (copy here). fl2va, 5 s, 1344x768,
  NFE 8, seed 0, kf /var/tmp/fasth3/t209/kf_first.png, caches dit-h3hf + tt-metal-cache-h3hf
  (shared with #286). Off arm reuses transformer_resident_adaln cache; on arm writes a new
  transformer_precomputed_adaln cache (cold: expect on-a to stop at the warm deadline, on-b warm).
- Queued (FIFO): t288-off-a, t288-on-a, t288-on-b, t288-on-c, t288-off-b. ONCE guard per arm
  (out_on/PASS, out_off/PASS), so extra jobs exit at once after a pass. off-b re-runs only if
  off-a failed.
- AICLK on blx03 is clamped at 900 MHz: numbers are relative only.

## Next step
Wait on probe.sh t288-off-b. Then: done markers of all five; out_{on,off}/PASS; grep
"MINIMAX-H3 PERFORMANCE RESULTS" table from /var/tmp/fasth3/t288/out_{on,off}/run.log;
PSNR/PCC of out_on vs out_off mp4 (ffmpeg psnr + numpy on decoded frames); still frames;
copy small artifacts to tt-project/t288/. Cleanup: dit-h3hf/minimax-h3/transformer_precomputed_adaln
if not kept, hyperflow model if no follow-up needs it.

## 2026-10-10 05:10 PDT (run 1359): blx03 runs all failed, blx03 down, move refused
- blx03 serial runner (900 MHz clamp), every arm failed on the #286 warm deadline (400 s), none
  reached denoise, so there are no timings or videos:
  - t288-off-a broker 853: 395.9 s call, "warm deadline 400 s passed in Initializing bucket buffers".
  - t288-on-a broker 855: 399.1 s call, "warm deadline 400 s passed in Warming prompt encoder keyframe layouts".
  - t288-on-b broker 857: 411.9 s call, same bucket-buffer deadline.
  - t288-on-c broker 860: DROP 06:34:37 UTC (status=killed, chips 12 and 8, trays 1-4), then
    broker-kill 06:47:43 UTC (chips unknown); T288_EXIT=137. Our job; blx03 later went out of service.
- Removed our leftovers from the blx03 runner: queue/...t288-off-b.job and running/t288-on-c.{job,state}
  moved to /var/tmp/fasth3/runner/parked/t288-moved-blx01/ (nothing else touched).
- blx01 refused (coordinator 05:02 PDT, #330): MiniMax-H3 is 135 GB (blx03), dit-h3hf cache 152 GB;
  blx01 footprint already ~140 of 150 GB. No weights copied.
- Branch ttp/t288-adaln-ab pushed to origin at f7266236d90 (ttp push --own passed this time).

## Next step (when blx03 is back: tt-project/state/ready/g14blx03.READY)
The cold test does not fit 600 s at the 900 MHz clamp: warmup alone passes 400 s. Before requeueing,
cut warm work so one arm finishes in ~400 s: e.g. init only the 1344x768/5 s bucket buffer instead of
all 26 buckets, and skip keyframe-layout warmups the fl2va 5 s run does not use (or run at full AICLK).
Then requeue off then on (on needs its precomputed-AdaLN cache: a cache-only job first, then the timed run).

## 2026-10-10 05:10 PDT (run 1370): still blocked on blx03
- g14blx03.READY is stale (04:36 UTC, before the outage). blx03 broker at 12:05 UTC: "device HELD
  (degraded): device is dirty and unverified: gate/post-job: 8/32 chip", health gates 884/886/888/892
  failed, bridge-reset chips 8-15 failed. Wait probe: tt-project/t288/probe_blx03.sh (exits 0 once the
  broker is active and not holding the device).
- No device work, no copies to blx01 (coordinator 05:02 PDT). Branch unchanged at f7266236d90 on origin.
- Before requeueing, the warm-only-the-used-bucket flags (coordinator decision 05:04 PDT) must land on
  ttp/t288-adaln-ab; the cold test does not fit 600 s at the 900 MHz clamp otherwise.
