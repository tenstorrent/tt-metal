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

## Result (2026-10-07/08, blx01 broker job 867; job 865 timed out at 432 s, exit 124, rerun)
A/B DIFFVAE_S5_2D=1, 2 timed seeds: kp1 4.494 s vs kp0 4.931 s (-0.44 s). PSNR vs host-noise ref: kp1 55.1-55.8 dB,
PCC 0.99995; kp0 about the same; kp1 vs kp0 55.3-55.9 dB (floor 43.7).
test_choose_sharded_brick_regression 5/7: 1080p_decode (168 vs pinned 147) and det_stage4 (36 vs 27) fail identically
on t212/b (planner .cpp blob a9a4ba4 = parent 5c1635d733d), so the pins are stale before t228; not caused by key phase.
Cherry-pick onto origin/ttp/t48-ltx25-integrated (8b7e9a9a5b2): test-file conflict (both sides add tests), kept both
-> ttp/t232-land 1a4d14830c9. Host-only guards on the merged tree (blx01 t232/b, no C++ change vs b95861393fe): 11 passed.
Landing via ttp push --detach from ttp/t232-land.
