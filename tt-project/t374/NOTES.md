# t374 notes (resume here)

Branch ttp/t374-conv-vae-conv3d-vol2col-input-reuse-in-t, rebased on origin/ttp/t48-ltx25-integrated 90ed8257bac.
- 0426c25ad80 notes: DESIGN.md (byte counts, row ring design, rolling T deferred).
- 8a365da8129 CODE: opt-in TT_CONV3D_ROW_RING=1 (factory: W_shard_max = full padded W, L1 check, reader CT arg 47,
  TensorAccessorArgs now at 48; reader: ring gather + vol2col slot map) + test
  models/tt_dit/tests/models/ltx/test_conv3d_row_ring.py (both arms in one process, program cache cleared between;
  asserts md5-identical outputs on all 8 shards of a 2x4 submesh of the 4x8 mesh; prints off/on trace µs).
  Not yet compiled or run anywhere.
- later commits: drivers (tt-project/t374/drv374.sh, run374.sh, env374.yaml). Keep off the push branch.

## Running (started 2026-10-10 14:31 PDT)
blx01 driver: ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t374/detach drv374 -- bash /var/tmp/fasth3/t374/drv374.sh
- builds /var/tmp/fasth3/t374/b (worktree of /var/tmp/fasth3/t48 repo at 8a365da8129, Release), log t374/build.log
- then one broker job (-t 600, lint passed): run374.sh s3_res. Logs: t374/drv374.log, t374/run_s3_res_job<id>.log,
  marker t374/drv374.done ("DONE s3_res:job=<id>:<status>" or BUILD_FAILED rc=N).
- check: ttp detach --check --host g15blx01 /var/tmp/fasth3/t374/detach/drv374

## Next
1. Read drv374.done + run_s3_res_job*.log: line "[row_ring] s3_res ... speedup=X identical=True".
   Build failure: build.log (first error). Kernel compile error: the job log.
2. identical=True and speedup >= 1.15: run LAYERS="s2_res" (same code path) then decide s4_res (its ring, 761 KB,
   does not fit L1: needs a W-tiled ring; the factory falls back, so s4_res arm "on" = off). Then a module A/B.
   Then land 8a365da8129 on t48 via a -land branch + ttp push (code commit only).
   speedup < 1.15: record and stop (no landing of a non-default flag unless useful).
3. Clean up blx01: git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t374/b; rm -rf /var/tmp/fasth3/t374.

## Result 1 (job 537, blx01, 900 MHz clamp, relative)
s3_res: off 6379.5 us, on 7000.5 us -> 0.911x (slower), identical=True (md5, max_abs_diff 0).
Hypothesis: full-width row gather (32 cols) at the first w_block of each h_block is not overlapped with compute,
while the default path spreads the gather over 8 w_blocks (144 then 96 sticks each).

## Iteration 2 (started 2026-10-10 15:22 PDT)
1ac9a44b0fa CODE: rows new to an h_block fill their columns one w_block at a time (cols [c_prev, c_cur)),
in-bounds gathers skip the per-stick padding check. Kernel-only change: no rebuild; t374/b checked out at
1ac9a44b0fa, JIT dir wiped. Driver: env SKIP_BUILD=1 JOB_T=120 drv374.sh (job -t 120 = 70 s measured +50%+).
Old outputs: t374/drv374.{done,log}.job537, detach/old537/.
check: ttp detach --check --host g15blx01 /var/tmp/fasth3/t374/detach/drv374
Next: same as above. speedup >= 1.15 -> s2_res, then module A/B, land. Else record, stop, clean up blx01.

## Result 2 (job 545, blx01, 900 MHz clamp, relative) — REJECTED
s3_res: off 6378.0 us, on 6120.1 us -> 1.042x, identical=False (max_abs_diff 498.75, PCC 0.380).
Iteration 2 has an indexing bug (per-w_block column fill), and even its buggy (possibly work-skipping) time is
only 1.04x, far below the 15% bar for step 3. The correct version (iteration 1) is 0.911x.
Decision (15:40 PDT): reject the row ring prototype; no landing, no step 3. The flag stays opt-in on this branch only.
Why it does not pay: conv3d's reader already overlaps the full-volume gather with compute across w_blocks; the
L1 row ring saves DRAM/NoC reads but moves the gather into a serial stall at each h_block start, and the
remaining gain (<5%) cannot reach 15% without restructuring compute (e.g. folding norm+SiLU, PLAN item 5 note
from #375), which is a larger change than this task.
blx01 cleaned: t374/b worktree removed from /var/tmp/fasth3/t48, /var/tmp/fasth3/t374 deleted. No drops.
