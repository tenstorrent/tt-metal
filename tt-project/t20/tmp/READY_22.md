# t22: device runs ready to launch (DO NOT submit until the user lifts the g15blx02 device pause)

One project device job at a time. Run each only when `tt-device-mcp status` shows no smarton job queued/running.
Working dir for all: /home/smarton/fasth3/tt-metal/tt-project/worktrees/t22 (branch ttp/t22-..., head >= 5d993cd2f7c).
Kernel cache for the t22 tree is warm (job 686 compiled it; the changes below are Python-only), so no separate warm job.

## 1. S2 prompt reuse + noise prefetch (single config, ~5 min)
    tt-device-mcp run-bg "bash tmp/e2e.sh s2reuse" -w $PWD -t 450 -e tmp/e2e_env.yaml
Check: `bash tmp/cmp.sh s2reuse <job>`.
Expect: all latents and mp4s byte-identical to ../t14/tmp/e2e/safe (the same as job 686),
S2 "denoise init" prompt ~0 ms (job 686: 40-54 ms), S2 denoise ~2.38-2.40 s (686: 2.42-2.43 s).
Byte check: `for f in tmp/e2e/s2reuse/*; do cmp $f ../t14/tmp/e2e/safe/$(basename $f); done`

## 2. (optional, after 1) device prompt handoff, plan item #2 alternative
    tt-device-mcp run-bg "bash tmp/e2e.sh noisepf_ph LTX_DEVICE_PROMPT_HANDOFF=1" -w $PWD -t 450 -e tmp/e2e_env.yaml
Check: `bash tmp/cmp.sh noisepf_ph <job>`. Skips the S1 upload as well; needs a byte-identical result to be kept.

## 3. Ring SDPA chunk re-sweep under the 8 KB payload (plan #6), one config per job (~2-4 min each)
Current table (attention_ltx.py ring_sdpa_chunk_by_n): S1 N=9728 -> (96,256), S2 N=38912 -> (192,512).
Exactness: a Q-chunk change keeps each query row's K-chunk order, so it should be bit-exact (check with cmp_blk).
A K-chunk change reorders the online-softmax accumulation: NOT bit-exact, so it needs the 5-seed eval
before it can land (the sampler amplifies any change, see denoise_plan.md).
First the baselines, then one candidate per job; log line `BLOCK_TRACE_MS`:
    tt-device-mcp run-bg "bash tmp/chunk_ab.sh stage_1 base" -w $PWD -t 600 -e tmp/block_env.yaml
    tt-device-mcp run-bg "bash tmp/chunk_ab.sh stage_2 base" -w $PWD -t 600 -e tmp/block_env.yaml
    # exact (Q only): S1 128,256 and 64,256; S2 256,512 and 128,512
    tt-device-mcp run-bg "bash tmp/chunk_ab.sh stage_1 128,256" -w $PWD -t 600 -e tmp/block_env.yaml
    tt-device-mcp run-bg "bash tmp/chunk_ab.sh stage_2 256,512" -w $PWD -t 600 -e tmp/block_env.yaml
    # non-exact (K): S1 96,512; S2 192,1024 may overflow L1 (2x4 did at K=1024)
Compare: `python tmp/cmp_blk.py tmp/blk/chunk_stage_1_base.pt tmp/blk/chunk_stage_1_128_256.pt`
Keep a candidate only if BLOCK_TRACE_MS drops >= 1% (x48 blocks x steps) and, for Q-only, exact=True.

## Device stop procedure and drop guards (added 2026-10-01, #64)

- **At every device stop** (pause, chip drop, reboot, fabric failure): on each box, list our own queued
  broker jobs (`tt_device_queue_status` / `tt_device_recent_jobs`) and kill each with `tt_device_job_kill`.
  The broker re-queues jobs after a reboot (QUEUE-RESTORE): blx03 job 000 ran at 07:29:16 that way.
  Never kill other tenants' jobs. Skip any box whose broker is being upgraded.
- **`hostfmax.py 1150` is not a guard against drops.** ltx-host job 995 ran at aiclk 1150 and still dropped
  tray 1 (see tt-project/research/blx03_drop_0722.md). Rely on the rules instead: open the full mesh, then
  `create_submesh(2,4)`; one project job at a time per box; no 4x8 runs.
