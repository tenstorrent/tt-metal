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
