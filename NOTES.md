# t133: cherry-pick #56023 (BH NoC MID-register skip) onto t48

Branch ttp/t133-cherry-pick-56023-bh-noc-mid-register-sk = t48 c4409b1fa24 + 27a9c2f95c9 (cherry-pick of upstream
27cecf3f7c1) + A/B scripts in tmp/t133/.

Conflict: blackhole/noc_nonblocking_api.h. t48 predates upstream's noc_init_one() refactor; the two MID clears the
commit adds there went into t48's noc_init() loop instead. The added lines match upstream's exactly.
Checked: every PCIe-addressed kernel on t48 either was updated by the commit or is unchanged from the commit's
parent upstream (so upstream already covered it). drisc_relay.cpp was renamed on t48
(tools/profiler/kernels/streaming_profiler_relay.cpp); git applied the hunk there.

Off-device: tmp/t133/build_and_test.sh (Release build + LTX CPU tests, /dev/tenstorrent hidden), logs in tmp/t133/.

## Device A/B (#135: full 4x8, per the user's 2026-10-05 update)
Traced conv VAE decode on the full 4x8 mesh at 1088x1920/145f (tmp/t133/test_t133_4x8.py), real 2.5 latent,
LTX_CONV3D_BLOCKING_MESH=4,8; 3 eager + capture + 8 traced decodes per process. A = c4409b1fa24, B = branch tip.
Two short broker jobs, one A/B pair each: j1 = A B, j2 = B A. The first A run records the 1088x1920 halo-off
reference (/var/tmp/fasth3/t133/vae_ref). Drops: wait for the health check and rerun; skip after 2 in a row.
Launch, from g15blx02 in this worktree:
  ssh g14blx03 'bash -s' < tmp/t133/blx03_setup133.sh        # two worktrees + lean Release builds (~5 GB)
  ssh g14blx03 mkdir -p fasth3/t133drv && scp tmp/t133/driver.sh g14blx03:fasth3/t133drv/driver.sh
  tt-project/harness/templates/blx03-launch.sh t133 /home/smarton/fasth3/t133drv/driver.sh
Result: /var/tmp/fasth3/t133/driver.log on blx03 (T133_RUN/T133_CMP lines, DROP lines). Relaunching the driver
skips jobs whose run133_<j>.log ends with T133_EXIT=0.
Cleanup after: worktree remove ~/fasth3/t133a, ~/fasth3/t133b; rm ~/fasth3/t133drv, ~/fasth3/t133-setup.*,
/var/tmp/fasth3/t133/{jit_A,jit_B,out_*}.

## Off-device result (2026-10-05)
Release build: BUILD_EXIT=0. LTX CPU tests (models/tt_dit/tests/models/ltx/): 207 passed, 393 skipped, 0 failed
(t48 c4409b1fa24 added tests since the 189-pass baseline). build_Release and .cpmcache removed.
Remaining: the device A/B above, once device work is allowed.

## Status (#135 run 1, 2026-10-06 02:50 UTC)
Setup (t133a/t133b builds) and driver started on blx03 (driver pid 39509, boot 02:26:22). Broker was HELD/degraded
(8/32 chips off-bus, tray 2, recovery failing) and t141/t140/t119 drivers also wait for it; ours queues via
submit.sh (one project job at a time). Next run: if driver.log has T133_DRIVER_DONE, read T133_CMP lines and act
(identical + faster: cherry-pick 27a9c2f95c9 onto t48 and push; then cleanup). If the driver died (reboot), relaunch
it with the same command; finished jobs are skipped.

## Status (#135 run 2, 2026-10-06 03:12 UTC)
blx03 rebooted ~03:02 UTC (no t133 job running); the run-1 driver died before submitting. Relaunched: job 244 (j1)
failed at once. Arm A refused to run because `git rev-parse --short` gives 10 chars on blx03 vs the 11-char pin;
arm B then had no reference to compare against. B alone ran fine (traced median 0.5558 s, min 0.5438; eager med
0.5351). Fixed in 8e45440b891 (full-hash compare); t133b on blx03 moved to it (only tmp/ + NOTES differ, no rebuild).
Old logs: /var/tmp/fasth3/t133/{driver.run1.log,run133_j1.job244.log}. Driver relaunched 03:09:10 (pid 24636).
Drop (not ours): 03:10:17 UTC, job 246 (smarton, another task's t48 driver), chips 8-15 / tray 2 left PCIe;
broker HELD and recovering. Our driver waits for the health check, then submits j1.
