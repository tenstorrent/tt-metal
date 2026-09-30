# t18 notes (resume here)

Finding (attempt 1): ltx-rt already has zero-copy readback + native planar stitch
(models/tt_dit/utils/yuv_d2h.py, planar_concat_cpp). The remaining serial host cost was the
libx264 encode (~0.7 s at 1080p/145f) after the device audio decode. Commit 63902277007 moves
the encode to a worker thread under the audio decode (LTX_ASYNC_EXPORT=1 default, =0 restores serial).
Host unit test models/tt_dit/tests/unit/test_yuv_video_export.py: 3 passed (mp4 byte-identical).

Device A/B (1080p/145f, BH 4x8 ring, seed 0, traced), queued behind live service 2026-09-30 13:22:
- job 597: serial  (LTX_ASYNC_EXPORT=0) -> tmp/out/serial, log /var/log/tt-device-broker/2026-09-30_132228_597.log
- job 598: async   (LTX_ASYNC_EXPORT=1) -> tmp/out/async,  log /var/log/tt-device-broker/2026-09-30_132231_598.log

Next step once both are done:
1. grep both logs for "generate() wall", "Audio decode", "Export"/timings table; before = serial, after = async.
2. cmp tmp/out/serial/*.mp4 tmp/out/async/*.mp4 (expect byte-identical; else PCC on decoded frames).
3. Write result.json, delete tmp/out mp4s after recording.

## Attempt 2 (2026-09-30 13:44)
- job 597 serial DONE: steady-state replays Total 6.41 s / 6.27 s (first warm 7.92 s); Video export 0.9 s,
  Audio decode 0.4 s, VAE decode 0.68 s. 4 mp4s in tmp/out/serial.
- job 598 async was KILLED by broker device recovery (8 chips dropped off PCIe during weight load; broker
  did glx_reset). Not our code. Broker caps timeouts at 600 s, so A/B cannot share one job.
- Resubmitted async alone: job 610 -> tmp/out/async, log /var/log/tt-device-broker/2026-09-30_134408_610.log
Next: grep 610 log for "Total (compute)" / "Video export" / "Audio decode"; compare to 597;
cmp tmp/out/serial/ltx_av_fast_1920x1088_0.mp4 tmp/out/async/ltx_av_fast_1920x1088_0.mp4; write result.json; rm tmp/out.

## Attempt 2 result (2026-09-30 14:42) — DONE
job 610 async done. E2E_WALL_S steady state: serial 7.33/7.21 s -> async 6.67/6.66 s (-0.6 s).
Video export 0.9 -> 0.3 s; other stages unchanged. Video streams bit-identical across all 8 mp4s
(serial gen#2 audio differed from the other 7 = run-to-run audio nondeterminism in the serial job, untouched path).
Kept tmp/keep/t18_async_seed0.mp4 + still; tmp/out deleted.
