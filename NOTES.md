# t183 NOTES: LTX_AUDIO_OVERLAP prototype

## State (2026-10-07 04:50 UTC, run 739)
- Branch ttp/t183-prototype-audio-decode-overlapped-with-v. Code commits on top of t48 87e4b0732df:
  - 0fb6ddfa38f ltx: opt-in LTX_AUDIO_OVERLAP (deferred YUV assembly on the export worker)
  - 4bf7c675899 ltx test: LTX_E2E_SEEDS tokens "N+" run that seed with LTX_E2E_AB_ENV set
- Mechanism: device concurrency between the VAE and the audio decode is impossible (both use all 32 chips
  on CQ0, so the device FIFO serializes them). With LTX_AUDIO_OVERLAP=1, `_yuv_planar_d2h(defer=True)`
  returns a DeferredYuvPlanar right after the readback sync. The YuvVideoExport worker runs the C++ planar
  concat (GIL released) before x264, so the main thread queues decode_audio earlier. Same bytes.
  The expected gain is only the host assembly time (tens of ms), likely <100 ms, so the default probably stays off.
- CPU checks: unit tests pass (deferred export gives the same mp4 bytes and resolves on "yuv-video-export";
  DeferredYuvPlanar slice/reshape). Eager vs deferred assembly is byte-equal on the python and C++ paths (fake 2x4).
- blx01 setup is done: /var/tmp/fasth3/t183/{files,tree,mkoverlay.sh,run_cfg.sh}. The tree is t48 bf7db12a149 plus
  python at 4bf7c675899 (OVERLAY_COMMIT).
- No device job submitted. At 04:46 UTC blx01 was HELD (degraded): chips 16-23 (tray 3) fell off PCIe
  during broker job 715 (smarton, task t185 s2x2), not ours.

## Next step
1. Health: `ssh g15blx01 tt-device-mcp status`: nothing HELD, no 🔧 or smarton job running.
2. Submit (one job, one A/B pair):
   ssh g15blx01 "tt-device-mcp run-bg 'bash /var/tmp/fasth3/t183/run_cfg.sh ab5 PYTEST_S=330 LTX_E2E_SEEDS=0,0+,1,1+,2,2+,3,3+,4,4+ LTX_E2E_AB_ENV=LTX_AUDIO_OVERLAP=1 LTX_E2E_EXTRA_REPLAYS=0 LTX_FRESH_PROMPTS=0 LTX_WARMUP_T2V_ONLY=1 LTX_WARMUP_ENCODERS=0' -w /var/tmp/fasth3/t48 -e /var/tmp/fasth3/t159/env.yaml -t 360"
   (Pass the args to run_cfg.sh, not as an env prefix: run_cfg exports EXTRA_REPLAYS=1 and FRESH_PROMPTS=1 first.
   PYTEST_S is read before the arg loop, so put it in an env prefix too: `env PYTEST_S=330 bash ...`.)
   Odd gens (#1,#3,...) are off, even gens are on. Outputs: /var/tmp/fasth3/t183/res/ab5/ltx_av_fast_1920x1088_<gen>.mp4.
3. Compare: video frames off vs on and against tt-project/data/g15/ref_t48_f6b8 (same seed, DEFAULT prompt).
   Audio: decode the PCM with ffmpeg (-f s16le) and compare between arms. Timings: E2E_WALL_S gen#N, STAGE_SPLIT, "VAE decode", "Audio decode".
4. Land via a -land branch from origin/ttp/t48-ltx25-integrated (cherry-pick both code commits), `ttp push --detach`.
   Default off unless bit-identical AND mean gain >=100 ms.

## Drop log
- 2026-10-07 05:02 UTC, blx01, broker job 732 (smarton, task t188), chips 16-23 (tray 3) left PCIe; broker holding/recovering (bridge resets 734/735 failed). t183 not submitted.
- 2026-10-07 05:30 UTC: light wake, blx01 unreachable over ssh (No route to host; likely rebooting after tray 3 drop). A/B not submitted.
- 2026-10-07 08:42 UTC: blx01 healthy; A/B submitted as broker job 769 (queued behind ltx-host 768). Next: when done, do steps 3-4.

## Result (2026-10-07, job 769, blx01 4x8, warm, DEFAULT prompt, build bf7db12a149 + py 4bf7c675899)
Odd gens off, even gens on (LTX_AUDIO_OVERLAP=1). Runtime 174.7 s, exit 0, no drops.
| seed | off E2E | on E2E | off VAE-end->audio-end | on |
| 0 | 5.707 | 5.651 | 398 ms | 407 ms |
| 1 | 5.645 | 5.676 | 395 | 407 |
| 2 | 5.728 | 5.708 | 386 | 408 |
| 3 | 5.703 | 5.688 | 399 | 405 |
| 4 | 5.681 | 5.704 | 383 | 401 |
Mean off 5.693 s, on 5.685 s: -7 ms, within noise. VAE forward 0.5 s both arms. STAGE_SPLIT mel_vae 36-42 ms, vocoder+bwe 336-358 ms both arms.
Bit identity (decoded rawvideo + s16le PCM md5): off == on == ref_t48_f6b8 for all 5 seeds, video and audio.
Verdict: lands default off (gain < 100 ms). Host YUV assembly is not on the critical path in any meaningful way; real overlap needs device concurrency (audio on a disjoint submesh), which costs VAE chips.
