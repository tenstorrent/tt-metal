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
