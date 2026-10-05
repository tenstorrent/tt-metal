# t133: cherry-pick #56023 (BH NoC MID-register skip) onto t48

Branch ttp/t133-cherry-pick-56023-bh-noc-mid-register-sk = t48 c4409b1fa24 + 27a9c2f95c9 (cherry-pick of upstream
27cecf3f7c1) + A/B scripts in tmp/t133/.

Conflict: blackhole/noc_nonblocking_api.h. t48 predates upstream's noc_init_one() refactor; the two MID clears the
commit adds there went into t48's noc_init() loop instead. The added lines match upstream's exactly.
Checked: every PCIe-addressed kernel on t48 either was updated by the commit or is unchanged from the commit's
parent upstream (so upstream already covered it). drisc_relay.cpp was renamed on t48
(tools/profiler/kernels/streaming_profiler_relay.cpp); git applied the hunk there.

Off-device: tmp/t133/build_and_test.sh (Release build + LTX CPU tests, /dev/tenstorrent hidden), logs in tmp/t133/.

## Device A/B (NOT run; device work is stopped)
Traced conv VAE decode (test_vae_ltx_trace_ab.py: full 4x8 mesh, then create_submesh(2,4)), 544x960/145f,
LTX_CONV3D_BLOCKING_MESH=4,8. A = c4409b1fa24, B = branch tip, run A B A B in one broker job.
Launch, from g15blx02 in this worktree, once device work is allowed:
  ssh g14blx03 'bash -s' < tmp/t133/blx03_setup133.sh        # two worktrees + lean Release builds (~5 GB)
  ssh g14blx03 mkdir -p fasth3/t133drv && scp tmp/t133/driver.sh g14blx03:fasth3/t133drv/driver.sh
  tt-project/harness/templates/blx03-launch.sh t133 /home/smarton/fasth3/t133drv/driver.sh
Result: /var/tmp/fasth3/t133/driver.log on blx03 (T133_CMP lines: traced/eager delta, A vs B identical=).
Expect identical=True. Cleanup after: worktree remove ~/fasth3/t133a, ~/fasth3/t133b, ~/fasth3/t133drv, t133-setup.*

## Off-device result (2026-10-05)
Release build: BUILD_EXIT=0. LTX CPU tests (models/tt_dit/tests/models/ltx/): 207 passed, 393 skipped, 0 failed
(t48 c4409b1fa24 added tests since the 189-pass baseline). build_Release and .cpmcache removed.
Remaining: the device A/B above, once device work is allowed.
