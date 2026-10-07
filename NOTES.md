# t232 notes: DiffVAE key-phase A/B (t228 code b95861393fe)

Box: blx01 (g15blx01). Everything under /var/tmp/fasth3/t232 (tree b = worktree of t48 repo at b95861393fe, own build).
Driver: /var/tmp/fasth3/t232/drv/driver.sh (copy in tmp/t232/drv), started 2026-10-07 22:40 UTC (pid 275245, sid 275245).
Flow: setup232.sh (cold Release build, log /var/tmp/fasth3/t232/build.log) -> one broker job run.sh (-t 600):
host-only planner tests (hosttests.py runs test_choose_sharded_brick_regression with mesh_device=None, since its (1,1)
fixture would open a bare mesh; pytest -k refuses_stride_with_h_split / key_phase_pins / divides_h_shard), then arms
kp0 (DIFFVAE_NA_KEY_PHASE=0) and kp1 (=1), DIFFVAE_S5_2D=1, 2 timed seeds + deep profile + host-noise seeds 0-4.
Score: cmp_kp{0,1}_vs_ref.json (vs /var/tmp/fasth3/diffvae/ref, floor ~43.7 dB) and cmp_kp1_vs_kp0.json in drv/.
Marker: /var/tmp/fasth3/t232/drv/driver.marker ("T232_DRIVER_DONE stage=.. rc=.. jobs=..").
gtest NeighborhoodPlanBuild.* not built (build_metal.sh without --build-tests).

Next: on marker, read drv/driver.log, out/run.log, cmp_*.json. If kp1 faster and within floor (vs ref) and close to kp0:
git switch -c ttp/t232-land origin/ttp/t48-ltx25-integrated; cherry-pick b95861393fe; ttp push --detach.
