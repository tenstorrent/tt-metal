# t27 notes (resume here)

Branch fast-forwarded onto #18 (63902277007). Commit 0495a8e666c: mp4 encode moved to a persistent child
process (models/tt_dit/utils/video.py: YuvVideoExportProcess/_EncoderProcess), frames via shared memory
(yuv_d2h._get_planar_out_buf -> shared_frame_buffer). Default LTX_EXPORT_PROCESS=1; =0 restores #18 thread
export; LTX_ASYNC_EXPORT=0 still restores serial. LTX_EXPORT_CPUS=32-63 optionally moves the encoder to SMT
siblings (count must equal inherited mask or libx264 thread count/bitstream would change; refused otherwise).
Build reused via symlinks to t10 (no C++ diff; repointed from t7 by #35 disk cleanup): build, build_Release, runtime, ttnn/ttnn/_ttnn.so (untracked).

Why: after the VAE, the libx264 encode is the critical path (audio decode 0.41 s is already hidden). Streaming
frames per VAE chunk is not possible on the conv decoder: one whole-clip traced decode, and the YUV planes on
device are CHWT (T innermost), so frames cannot be read back in T-chunks without a device layout change.
Host evidence (tmp/bench_gil.py): ttnn.synchronize_device holds the GIL (device.cpp binding has no
gil_scoped_release); a 400 ms GIL hold on the main thread stretches the thread export tail 0.29 -> 0.78 s,
process export 0.32-0.37 s. Video md5 identical. Unit tests: 9/9 pass (test_yuv_video_export.py).

Device A/B queued 2026-09-30 14:52 (600 s broker cap -> one mode per job), outputs in tmp/out/<mode>:
- 641 thread   (LTX_EXPORT_PROCESS=0, = #18 baseline path)
- 642 process
- 643 siblings (process + LTX_EXPORT_CPUS=32-63)
Next: grep each log (/var/log/tt-device-broker/*_64{1,2,3}.log) for "E2E_WALL_S gen#2/#3", "Video export",
"Audio decode", "VAE decode (forward)". Baseline job 610: E2E 6.665/6.660, export tail 0.3 s.
Bit-identity: md5 of demuxed video packets across tmp/out/*/ *.mp4 (see tmp/bench_pin.py for the snippet).
If process/siblings is not faster than thread, flip the default back to thread (LTX_EXPORT_PROCESS default 0).
Keep one mp4 + still in tmp/keep, delete tmp/out.

## 2026-09-30 16:10 (attempt 2, device pause)
Jobs: 641 timeout (410 s, no E2E lines; broker health gate flagged it dirty), 642 killed by chip drop,
643 started after the 16:03 pause, killed by us. No device numbers exist for the process export yet.
Off-device analysis: warm C++ planar reassembly = 30 ms, libx264 encode = 0.77-0.87 s (tmp/bench_host.py).
Critical path after the VAE is VAE end -> encode; audio (0.41 s) is already fully hidden. So
(a) starting audio before the readback finishes gains 0, and pipelining the scatter into the encoder
gains <= 30 ms. (b) streaming per VAE chunk needs a T-chunked device decode + T-outer yuv layout: device work.
The only remaining bit-identical host lever is the GIL fix (process export) in 0495a8e666c.
Next: after the pause lifts, run tmp/READY_27.md (process, then thread), compare, pick the default.

## 2026-09-30 (attempt 3, pause still on)
Branch pushed to origin (ttp/t27-overlap-vae-output-readback-encode-with-, 0495a8e666c). No code change.
Still waiting for the device pause to lift; then run tmp/READY_27.md.

## 2026-09-30 18:45 (attempt 4, blx03 open)
g15blx02 still paused; device pause lifted for blx03 only. A/B moved there.
blx03 setup: worktree ~/fasth3/t27 (detached 0495a8e666) of ~/fasth3/tt-metal, C++ (ttnn/cpp, tt_metal,
ttnn/ttnn/__init__.py) checked out from that checkout's HEAD so kernels match its build; build/runtime/_ttnn.so
symlinked to ~/fasth3/tt-metal (same as t32). Unit tests 9/9 there. JIT cache ~/fasth3/cache/t27-tt-metal-cache.
600 s broker cap -> driver ~/fasth3/t27/tmp/drive27.sh (detached): prewarm wrapper (capture job 869, offline
compile, then process run), wait, then thread run. Log ~/fasth3/t27/tmp/drive27.log, ends DRIVE27_DONE;
job logs tmp/job_process.log / tmp/job_thread.log; job ids in drive27.log (PROCESS_JOB=, THREAD_JOB=).
Jobs 868/870 were cancelled duplicates from a botched first launch.
Next: grep -E "E2E_WALL_S gen#[23]|Video export|Audio decode:|VAE decode \(forward" in both job logs; md5 of
demuxed video packets tmp/out/process/*_2.mp4 vs tmp/out/thread/*_2.mp4 (tmp/bench_pin.py snippet, on blx03).
Baseline 6.66 s is g15blx02; compare process vs thread on blx03 (same box). If process not faster, default
LTX_EXPORT_PROCESS to 0. Cleanup on blx03 after: keep one mp4 + still (copy to g15 tmp/keep), then
git -C ~/fasth3/tt-metal worktree remove --force ~/fasth3/t27; rm -rf ~/fasth3/cache/t27-tt-metal-cache.

## 2026-09-30 19:00 (attempt 5, blx03 run failed at load)
Jobs 872 (process) and 873 (thread) on blx03 both exit in ~30 s: AutoTokenizer fails, Gemma-3-12B snapshot on
blx03 (~/.cache/huggingface/hub/models--google--gemma-3-12b-it-qat-q4_0-unquantized/.../68f7ee4f...) has only
README.md; /home/losullivan on blx03 is not readable. g15 runs used /home/losullivan/... (23 GB, readable on g15).
LTX distilled ckpt is present on blx03 (~/.cache/ltx-checkpoints). blx03 /home free 140 GB (< 150 GB rule), so
copying Gemma there needs the user's OK. Blocked on that. Kept on blx03: ~/fasth3/t27 (453 MB) and
~/fasth3/cache/t27-tt-metal-cache (491 MB, prewarmed). Resume: once Gemma exists on blx03, fix GEMMA_PATH in
~/fasth3/t27/tmp/blx03_ab.sh and resubmit via drive27.sh stage 3 only (process, then thread).

## 2026-09-30 19:59 (attempt 6, Gemma now on blx03)
Gemma is at /var/tmp/fasth3/models (from #36). Fixed ~/fasth3/t27/tmp/blx03_ab.sh on blx03: TT_METAL_CACHE used
$FASTH3_DATA before it was set. LTX-2.3 DiT cache is warm in ~/.cache/tt-dit on blx03 (pre-existing).
Driver ~/fasth3/t27/tmp/drive27b.sh (detached): process job 887, then thread job, one at a time.
Log ~/fasth3/t27/tmp/drive27b.log ends DRIVE27_DONE; job logs tmp/job_process.log, tmp/job_thread.log.
Next: same greps and md5 check as attempt 4. Then the #20 follow-up (t24 export_async on t20 conv path,
faster x264 settings) is still open; it was not started.

## 2026-09-30 20:32 (attempt 7)
Job 887 (process) hit the 600 s broker cap: the JIT cache moved to /var/tmp/fasth3/cache/t27-tt-metal-cache (cold),
so it spent the run compiling. That cache is now 5.1 GB and warm. Job 889 (thread) started 20:28 on the warm cache.
The blx03 chip drop at 20:16 (holds 891-903) was under job 888 (task t37), not ours.
Detached driver ~/fasth3/t27/tmp/drive27c.sh waits for drive27b (job 889), then reruns process mode.
Log ~/fasth3/t27/tmp/drive27c.log ends DRIVE27C_DONE; job logs tmp/job_thread.log, tmp/job_process2.log.
Next: same greps and md5 check as attempt 4, comparing job_thread.log with job_process2.log.
Cleanup after: rm -rf /var/tmp/fasth3/cache/t27-tt-metal-cache (5.1 GB) and the ~/fasth3/t27 worktree on blx03.

## 2026-09-30 20:45 (attempt 8, after host reboots)
Both hosts rebooted. Job 889 (thread) never started (log ends at "Waiting to start"); drive27c died with the reboot,
so job_process2.log does not exist. blx03 broker has the device HELD (degraded): 8/32 chips (8-15) off the bus,
bridge-reset / glx_reset jobs 906-910 failed. No device timings yet. JIT cache /var/tmp/fasth3/cache/t27-tt-metal-cache
(5.2 GB) is still warm. Next, once the broker is healthy: on blx03 run
`cd ~/fasth3/t27 && setsid nohup bash -c 'tt-device-mcp run-bg "bash tmp/blx03_ab.sh thread" -w $PWD -t 590 -e tmp/blx03_env.yaml' ...`
i.e. edit drive27c.sh to submit thread first (drop the drive27b wait), then process; same greps and md5 check as attempt 4.

## 2026-09-30 21:13 (attempt 9, blx03 healthy again)
Broker power-cycled blx03 (job 918), all 32 chips back, fabric check 921 passed, queue empty.
Detached driver ~/fasth3/t27/tmp/drive27d.sh on blx03: thread job 922, then process job, one at a time.
Log ~/fasth3/t27/tmp/drive27d.log ends DRIVE27D_DONE; job logs tmp/job_thread_d.log, tmp/job_process_d.log.
Next: same greps and md5 check as attempt 4 on those two logs/outputs; then cleanup (see attempt 7) and the #20 follow-up.
