# t228 notes: DiffVAE key-phase K/V grid (DIFFVAE_NA_KEY_PHASE=1, opt-in)

Code: b95861393fe on ttp/t228-diffvae-key-phase-k-v-brick-grid-opt-in- (base t48 @5c1635d733d). Pushed (branch only).

Done (host only):
- Planner: key phase = shard_origin not brick-aligned. Gather measured from raw window start, floor-div bricks.
  Standalone host check (g++ on neighborhood_plan.cpp): brick (2,4,4) gathers 112 (7,4,4) on all 32 shards at
  1080p stage 5 under the 2-D split, vs 200 at phase 0; every query window is inside its chunk's gather.
- Reader: query_phase compile args added to query-site math. Python: key_phase_geometry -> resident (146,80,72),
  low (1,5,5); rephase K/V after the halo exchange (untilize, natural, slice, zero T frames, rebrick, tilize).
- Tests added: gtest NeighborhoodPlanBuild.KeyPhaseGathersFewerBricksAndCoversEveryWindow; pytest
  test_key_phase_pins_brick_and_gather (host-only).

Not done: nothing compiled or run on a device yet. No #226 follow-up test results yet.

Next steps:
1. blx01 (g15blx01): new worktree /var/tmp/fasth3/t228/b from /var/tmp/fasth3/t48 repo, checkout b95861393fe,
   cold build detached (template: /var/tmp/fasth3/t212/setup212.sh). /var/tmp has 280 GB free. Do not touch t48/t212 trees.
2. Broker job 1 (<=600 s): pytest test_neighborhood_sdpa.py::test_choose_sharded_brick_regression,
   ::test_choose_sharded_brick_refuses_stride_with_h_split, ::test_key_phase_pins_brick_and_gather; gtest if built.
3. Broker job 2 (one 4x8 A/B): DIFFVAE_S5_2D=1, DIFFVAE_NA_KEY_PHASE=0 vs 1, seeds 0-4 host noise; template
   /var/tmp/fasth3/t227/drv/run.sh + decode227.py + common.sh (swap B to the t228 tree). Score vs
   /var/tmp/fasth3/diffvae/ref (floor ~43.7 dB).
4. If exact and faster: land on ttp/t48-ltx25-integrated (cherry-pick b95861393fe on a -land branch, ttp push --detach).
