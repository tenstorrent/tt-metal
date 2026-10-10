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
