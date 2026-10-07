# t219 notes (C4: DiffVAE stage-5 2-D split)

Head (code): c12f7f34eff on ttp/t219-c4-diffvae-stage-5-2-d-split-na-tile-nar (base a40d78b8bae).
- feb44de2529 stale DIFFVAE_NA_UNSAFE_CHUNK references removed (docs + one test setenv).
- 3d0729e48e1 NA planner/executor take h_axis: H x W shards, all heads per chip, H halo then W halo
  (corners ride the W exchange). Unit test: test_neighborhood_bricked_w_sharded.py::test_bricked_2d_sharded_matches_host (4x8).
- c12f7f34eff DIFFVAE_S5_2D=1 (opt-in): stage 5 puts H on the TP axis, tp off, one band. RoPE tables,
  device/host x_t, context (_h_partition), pixel pull handle the tile.
NOT YET RUN ON DEVICE. Python-only: overlay on blx01 /var/tmp/fasth3/t212/b (8b1167ef43 = a40d78b8bae code).
P4 (NA tile narrowing): not implemented. #213 shows <=4% at S5 with one-brick chunks; multi-brick
gains are per-brick slots the M=query-rows matmul cannot skip; kv_chunk_count is compile-time.
Next: overlay tree (cp -al t212/b/models, replace changed files by write+mv, never in place), job 1 unit
test, job 2 decode_ref.py with DIFFVAE_S5_2D=1 on #214 latents + cmpS.py (t212 branch tt-project/t212/score).

# t222 (device test + land of t219's DIFFVAE_S5_2D=1)
Branch ttp/t222-... reset onto t219 notes head 4f3bcfa30fd. Code commits to land: feb44de2529 3d0729e48e1 c12f7f34eff
(origin t48 head is a40d78b8bae = their base, so the cherry-pick is clean).
blx01: overlay /var/tmp/fasth3/t219/ov (cp -al of t212/b/models + conftest.py + pytest.ini; the 6 changed
models/ files replaced by write .new + mv; b's git status still clean). Scripts: tt-project/t222 -> blx01 /var/tmp/fasth3/t219/drv.
Driver: blx01 /var/tmp/fasth3/t219/drv/driver.sh (pid 3677629, started 20:01 UTC), log drv/driver.log, marker drv/driver.marker.
- Job U = broker 832: test_bricked_2d_sharded_matches_host win5 + stage5_window on 4x8 PASSED (22 s runtime).
- Job D: runD.sh = decode222.py (decode_ref.py + one profiled decode -> out/stage_tree.txt) with DIFFVAE_S5_2D=1,
  outputs /var/tmp/fasth3/t219/out; then cmpS.py vs diffvae/ref -> drv/cmp.json, still drv/still_seed0_f72.jpg (top: 2-D, bottom: ref).
Baseline C1 (job 824, same script + tree, 1-D): mean 10.313 s, PSNR 55.1-55.8 dB.
Next: when the marker exists, read cmp.json + stage_tree.txt; if correct and faster, land the 3 code commits on t48.
2026-10-07 light wake: job D dropped twice on blx01 (833: chip 14 dead at 20:05:50 UTC; 843: chip 11 dead at 20:20:58 UTC; both ~80 s in, whose=smarton t222 D). Driver skipped D on blx01 per the two-drop rule. No cmp.json/stage_tree. Next (standard): judge whether the 2-D path itself causes the drops (two different chips, same point in the run), then move D to blx03 runner or exabox.
2026-10-07 20:30 standard wake: D's real failure was a code bug, not the drops. Both 833 and 843 died at stage 5 with
`AssertionError: H shard 68 is not whole 8-site bricks` (neighborhood_attention.py): _choose_sharded_brick ignored the
H split and picked (2,8,2). The chip drops (14, 11) came AFTER, in the broker's post-job fabric check. blx01 tray 2
(chips 11-14) also dropped at 18:35, 18:46 (chips 13, 12) before our jobs, and 19:19 chip 25: box flakiness.
Fix 5772489e0a0 (code): chooser takes height_local/h_shard_count, requires brick_h | H shard, plans all H x W shards.
2-D now picks (2,4,4): 200 gathered bricks over 18615 query bricks per chip (1-D: (2,8,2), 168 over 74460, but 1-D also
splits heads 4 ways, so NA work per chip is ~3.72M x 4 heads-equivalent vs 12.5M: 2-D NA may be ~20% slower; gains must
come from dropping TP collectives). Host-only test test_choose_sharded_brick_divides_h_shard passed on blx01.
Note: test_choose_sharded_brick_regression's pinned 1080p_decode (84,272,480)->(8,2,2) returns (2,8,2) on this base (stale pin).
Driver patched: a traceback in run.log is our failure, not a drop. Run 2 started 20:28 UTC (pid 3768509), new config
(brick fix), D only (U already passed); waits for blx01 health (fsm was down). Marker drv/driver.marker (run 1's kept as .run1).
Next: when marker exists, read drv/cmp.json, out/stage_tree.txt, still; if correct and faster than C1 10.313 s, land
feb44de2529 3d0729e48e1 c12f7f34eff 5772489e0a0 on t48 (-land branch, cherry-pick, ttp push --detach).
2026-10-07 light wake: job 853 (blx01, brick fix 5772489e0a0, DIFFVAE_S5_2D=1) completed, no drops.
Correct: PCC 0.99995-0.99996, PSNR 55.1-55.8 dB on all 5 seeds (floor 43.7). But SLOWER: mean 14.032 s
(14.08/14.10/14.01/13.97/14.00) vs C1 10.313 s. Not landed. Stage tree: tt-project/t222/stage_tree_job853.txt.
Next (standard): find where stage-5 2-D loses ~3.7 s (H neighbor_pad exchange? brick (2,4,4) vs (2,8,2)?), fix or drop.
2026-10-07 standard wake (attempt after spend-limit stop): job E = blx01 broker 855 (driverE.sh/runE.sh/decodeE.py),
production path (device stage-5 noise), 3 seeds per arm, no drops, rc 0:
  1-D (C1 default): 5.295/5.278/5.279 s, mean 5.284 s; stage 5 4208 ms (blocks ~490 ms each).
  2-D (DIFFVAE_S5_2D=1): 4.927/4.927/4.928 s, mean 4.927 s (-0.357 s, -6.8%); stage 5 3840 ms (blocks ~457 ms).
So job 853's 14.0 s (vs 10.3) is the host-noise test path only (2-axis sharded_from_torch upload of host x_t), not production.
Correctness: 853 host-noise 2-D vs unoptimized ref PCC 0.99995+, PSNR 55.1-55.8 dB, all 5 seeds (floor 43.7).
Device noise in 2-D draws one full tensor per band (same seed on every chip) and mesh_partitions it, like 1-D: no repeated
noise across chips. Decision: land feb44de2529 3d0729e48e1 c12f7f34eff 5772489e0a0 on t48 (opt-in, default off).
Stage trees: tt-project/t222/stage_tree_jobE855_{1d,2d}.txt.
