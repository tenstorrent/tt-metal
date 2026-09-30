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
