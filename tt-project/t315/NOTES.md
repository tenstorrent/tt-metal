# t315: headline LTX-2.3 8+3 warm timing of ltx-rt f6547442b30 at full clock (follow-up of #313 job 363, 7.32 s clamped)

## Run 1 (2026-10-10 05:15-05:35 UTC)
- Box check: NEITHER box at full AICLK.
  - blx01: job 370 (t293 main, 05:00 UTC) shows "AICLK failed to settle ... Expected 1350, observed 900 ... clamped by
    max-arbiter index 10 at 900 MHz" on all 32 chips (same since ~03:59 UTC 10-09, see t305).
  - blx03: READY, broker active, our runner busy with t286 jobs (845 running, t286-t10-dlb queued). Every device-opening
    job since its reboot (up since 00:05 UTC) shows the same 32-chip 900 MHz clamp, incl. live ltx-host job 837.
    The clamp has come and gone on blx03 since 10-06 (runs of clamped and clean jobs in /var/log/tt-device-broker).
- Not run clamped (spec). Setup done meanwhile (non-device):
  - blx03 worktree ~/fasth3/t315 @ f6547442b30 (detached, from origin/ltx-rt), release build via
    ~/fasth3/t315drv/setup315.sh (blx03-launch.sh); log /var/tmp/fasth3/t315/build.log, marker
    "T315_DRIVER_DONE setup <rc>" in /var/tmp/fasth3/t315/driver.log.
  - Job script /var/tmp/fasth3/t315/run315.sh (copy here), env /var/tmp/fasth3/t315/env315.yaml. Uses the local
    checkpoint ~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors (btrfs /home, local), upsampler from
    HF_HOME=/home/sulphur/hf (local), gemma /var/tmp/fasth3/models. Own cold JIT cache /var/tmp/fasth3/t315/jit
    (uncached 8+3 job ~379 s on blx01 per #303; -t 570).
- Probe: tt-project/t315/clkprobe.sh (exit 0 = blx03 build ok and newest blx03 device job unclamped, or newest blx01 job unclamped).

## Next (on probe pass)
- blx03 clean: write spec (ID=t315-e2e-r1, TASK=t315, CONFIG=t315-ltxrt-f654-e2e, TIMEOUT=570,
  WORKDIR=/home/smarton/fasth3/t315, ENV=/var/tmp/fasth3/t315/env315.yaml, CMD=bash /var/tmp/fasth3/t315/run315.sh,
  NEEDS=/home/smarton/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors,/var/tmp/fasth3/models/gemma-3-12b-it-qat-q4_0-unquantized,
  COLD_S=380) and enqueue with tt-project/harness/templates/blx03-runner/blx03-enqueue.sh.
- blx01 clean first: blx01 has no f6547442b30 build; t313's overlay (/var/tmp/fasth3/t313/ov_next on the t48 bf7db12a149
  build, job 356 config) is the fallback: copy run313d.sh to t315 without the PROMPT override, -t 570, device-lint first.
- After: quote the last PERFORMANCE table of out/run.log verbatim, check its clamp count is 0, still from the mp4, compare
  with job 363 (7.32 s clamped, ltx-rt prompt) and job 356 (7.33 s clamped, test prompt, same config as this one).
- Cleanup: blx03 ~/fasth3/t315 worktree (git -C ~/fasth3/tt-metal worktree remove --force), /var/tmp/fasth3/t315/jit, tmp.
- 05:24:52 UTC: blx03 build done (setup rc 0), ~/fasth3/t315 @ f6547442b30. Device lint of run315.sh passes
  (--device --timeout 570 --cold 380, needs checkpoint/gemma). run315.sh logs "[t315] AICLK clamp warnings: N"; N must be 0.
- Probe at 05:26 UTC: still clamped on both boxes. Handed off waiting on clkprobe.sh.
