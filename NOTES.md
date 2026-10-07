# t195 notes

Code: b21f12b93a2 (S2 default 0.909375,0.421875,0.0; LTX_S2_SIGMAS still overrides), pushed to
origin/ttp/t48-ltx25-integrated (a1da2e19509..b21f12b93a2). Host tests pass (test_ltx_quality, test_ltx_euler_tail).

Device: blx01 broker job 764, queued 06:18:41 UTC (behind nothing project; ltx-host 763 running), -t 240
(job 758, same shape: 156.5 s). Cmd: env PYTEST_S=220 bash /var/tmp/fasth3/t195/run_cfg.sh ref5
LTX_FRESH_PROMPTS=0 LTX_E2E_SEEDS=0,1,2,3,4 LTX_E2E_EXTRA_REPLAYS=0; -w /var/tmp/fasth3/t48 -e /var/tmp/fasth3/t159/env.yaml.
Overlay /var/tmp/fasth3/t195/tree = t171 tree (f6b806516cc) + 7 files changed f6b8..b21f12b93a2 (mkoverlay.sh).
Output: /var/tmp/fasth3/t195/res/ref5 (gen#N = seed N-1, gen#0 = cold seed 0).

Post: detached waiter (run 789, t195post.{log,rc}) runs tmp/t195/post.sh 764: fetch to
tt-project/data/g15/ref_t48_s2x2/seed<N>.{mp4,json} + run.log, identity.txt (cmp vs t185_s2x2), seed0_t3s.png.
Next: read POST.done + identity.txt, E2E_WALL_S gen#1..5 from run.log, write meta.json, symlink
baselines/ltx25_1080p_6s/ref_t48_s2x2 -> ../../data/g15/ref_t48_s2x2, hand off done.
If job 764 dropped: log it, rerun once on blx01 when the broker is healthy (2nd drop -> blx03 runner/exabox).

## Result (2026-10-07)
Job 764 exit 0, 156.7 s, 0 JIT compiles, no drops. Warm seeds 0-4: 4.779 4.806 4.822 4.774 4.788 s
(mean 4.794, worst 4.822; was 5.520 at 4194cd98852 with 3-step S2). Cold gen#0 32.5 s.
All 5 mp4s byte-identical to t185_s2x2 (job 758). Registered baselines/ltx25_1080p_6s/ref_t48_s2x2 ->
data/g15/ref_t48_s2x2 (meta.json). blx01 overlay tree removed; res/ref5 + scripts kept (/var/tmp/fasth3/t195).
