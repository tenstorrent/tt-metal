# t133 (#135) state, 2026-10-07 10:01 UTC — moved to blx01 (g15blx01)
- blx03: the two queued t133 runner jobs moved to g14blx03:/var/tmp/fasth3/runner/parked/ (never started). Not rerun there.
  Stale blx03 data in /var/tmp/fasth3/t133 on blx03 (jit_A, jit_B, vae_ref, stale_pre_runner): delete after this A/B.
- blx01: everything in /var/tmp/fasth3/t133 (scripts = tmp/t133/blx01/). A = /var/tmp/fasth3/t48 @ bf7db12a149
  (c4409b1fa24 + host-only conv3d guard). B = worktree /var/tmp/fasth3/t133/b = bf7db12a149 + #56023 (18a7875813e),
  sharing A's host build by symlink (#56023 changes only device headers/kernels). Both arms
  TT_METAL_DISABLE_PRECOMPILED_FW=1 and own JIT caches, so firmware + dispatch kernels build from each tree.
- driver.sh (pid 1404567 on blx01) runs j1 (A B, broker job 773, -t 600) then j2 (B A, -t = j1 time +50%), then cmp133.py
  -> /var/tmp/fasth3/t133/cmp.txt; marker /var/tmp/fasth3/t133/driver.marker (driver.marker.killed2nd = stale, from a
  duplicate driver I killed; ignore).
- Next: read driver.marker, driver.log, cmp.txt. If B identical and faster beyond noise: cherry-pick 27a9c2f95c9 onto t48, push.
  Cleanup on blx01: git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t133/b; rm -rf jit_A jit_B out_* in
  /var/tmp/fasth3/t133; rm -rf /var/tmp/fasth3/t48/tmp/t133.
