# t380 notes

Code: 3718fada887 (on t48 aa2846bd191): TT_FATAL in Conv3dDeviceOperation::validate_on_program_cache_miss
rejects T*H*W > 64 and not a multiple of 32; device test models/tt_dit/tests/unit/test_conv3d_vol2col_ring.py.
Evidence the rule is exact: blx03 bisect (#114, run 593): hangs (64,128,5,4,4) 273, (5,8,2) 276, (7,4,4) 280,
(7,8,2) 285 (n=80/112); passes n=192 aligned and n=48. Reader always inits ChunkWriter with the full num_patches.

blx01 driver (started 2026-10-10 17:46 PDT): /var/tmp/fasth3/t380/drv380.sh 3718fada887 via
`ttp detach --remote g15blx01 drv380` (run 1462). Build log t380/build.log, driver log t380/drv380.log,
marker t380/drv380.done ("DONE job=<id>:<status>:T380_EXIT=<rc>"), job log t380/run_job<id>.log.
Probe: ttp detach --check --host g15blx01 /home/smarton/.ttp-detach/1462/drv380

Next: read marker + job log. On pass: `ttp checks`, then git switch -c ttp/t380-...-land origin/ttp/t48-ltx25-integrated,
cherry-pick 3718fada887, ttp push --detach. Then remove /var/tmp/fasth3/t380/b worktree + jit on blx01
(git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t380/b) and ~/.ttp-detach/1462 on blx01.
