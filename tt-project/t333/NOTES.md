# t333: LTX-2.5 standard e2e with conv VAE (user #228), blx01

State at 2026-10-10 06:28 UTC (run 1305, split at the context line):
- Box: blx01, chosen over blx03 because of disk headroom (blx03 /var/tmp had ~209 GB free, too tight for 72 GB weights
  plus caches) and an idle broker. All files under /var/tmp/fasth3 on blx01 (/ was 43% used before the copy).
- setup333.sh (pid group 2836247), started 06:24 UTC: (1) copies the 5 LTX-2.5 split files (72 GB) from the MLPerf
  snapshot 28dac7acdc to /var/tmp/fasth3/models/ltx-2.5 with sha256 checks (SHA256SUMS); (2) worktree
  /var/tmp/fasth3/t333/b at ttp/t48-ltx25-integrated f6547442b30 and a Release build. Marker: t333/driver.log line
  `T333_DRIVER_DONE setup <rc> ...`. Logs copy.log, build.log. (The first launch at 06:22 got an empty script and did nothing.)
- drv333.sh (pgid 2849082), started 06:27 UTC: waits for setup, then broker jobs one at a time, -t 600 each, on the
  blx01 broker: c6a (145 f, cold JIT), c6 (145 f warm: the headline), c10 (241 f = 10 s). A drop is rerun once.
  Marker /var/tmp/fasth3/t333/drv333.done; log drv333.log; per-job logs run_<tag>_job<id>.log; videos and a t=3 s
  still in out_<tag>/.
- run333.sh is linted (device-lint.sh g15blx01 ... --timeout 600 --cold 600: rc 0).
- Conv VAE: the HF 2.5 conv file is gated (401, no HF token on any box), so the run uses the LTX-2.3 conv VAE swap
  (LTX25_VIDEO_VAE = ltx-2.3-22b-distilled-1.1.safetensors, same arch, #207/#332). Label it that way.
- Env: LTX_VERSION=2.5 LTX25_DIFFVAE=0 LTX25_ROOT=local copy, defaults 1088x1920 24 fps seed 10, 8+3,
  RUN_VBENCH=0 RUN_CLIP=0 (to fit the 600 s cap; #332 suggested defaults 1), TT_DIT_CACHE_DIR unset.
- blx01 clock: no AICLK-clamp lines in the newest broker log at 06:26 (user #230: run anyway; label clamped runs relative only).

Next step on wake: read drv333.done and drv333.log; for each job quote the last 20 lines and first error line if it failed;
quote the test's timing table from run_c6_job*.log verbatim (box blx01, job id, commit f6547442b30, command), take the
VAE decode stage vs DiffVAE 2.313 s (t48 @a5a774ea17f); copy the mp4 and still to tt-project/t333/; then clean up blx01
(t333/jit, t333/b worktree via `git -C /var/tmp/fasth3/t48 worktree remove`, out_* videos after copying; decide on the
72 GB models/ltx-2.5 copy: keep only if the conv VAE optimization follow-up runs on blx01 soon, and say so).
