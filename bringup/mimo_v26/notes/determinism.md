# Decode MoE race: hand-over from the moe agent (E261007-0818-033 is the decode agent's localisation)

## What I reproduced (MoE only, no attention, no model)
Test: `models/demos/blackhole/mimo_v26/tests/test_moe_race_stress.py` (commit 5c4b1ee7013+; 4x8, real layers 1-6 chained in ONE trace of `MiMoMoEDecode`, real dump hidden states distinct per mesh row,
8 live slots per row, 3000 replays on identical inputs, every layer output compared bitwise with the first replay, per mesh row). Launch (private lib needed, see below):
`launch.py --worktree <wt> --commit <c> --host <free> --log <log> --env LD_LIBRARY_PATH=/mnt/tt-data/ssinghal/mimo/artifacts/moe_B_9a8fd8582e5/lib --env MIMO_REPS=3000 --env MIMO_ROUTER=exact -- python_env/bin/python -m pytest -x -s -q models/demos/blackhole/mimo_v26/tests/test_moe_race_stress.py`
(runs 90 s after the ~6 min weight build; env knobs MIMO_NL, MIMO_LIVE, MIMO_ROUTER=exact|fast, MIMO_STAGE=99|5|4..., MOE_DISPATCH_SPARSE_MCAST=0, MIMO_DISP_LINKS/WORKERS).

Results (18000 layer-calls per run, logs /mnt/tt-data/ssinghal/mimo-logs/moe_I_*_5c4b1ee7013.log):
| variant | glitching layer-calls | rows | max abs diff |
|---|---|---|---|
| router exact, default (dispatch sparse_mcast ON) | 5 | 1, 2 | 0.0023 .. 0.0242 |
| router exact, MOE_DISPATCH_SPARSE_MCAST=0 (per-expert dispatch path) | 4 | 1, 2, 3 | 0.0023 .. 0.0242 |
| router fast (router_select kernel), default | 2 | 3 and ROW 0 | 0.0020, 0.0242 |
So it reproduces with the MoE alone (~2-5e-4 per layer-call here; the decode agent saw ~1e-3), it is NOT the sparse-multicast dispatch feature, NOT the exact router (fast router also glitches), and row 0 is not immune (one row-0 glitch with the fast router).
The max-diff VALUES repeat exactly across runs/layers/variants (0.0022955, 0.0219879, 0.0241923, 0.0242269, 0.0156631): looks like one deterministic quantity per event, e.g. one slot's expert contribution
missing/duplicated or one expert's rows read stale, not random bit noise. Diffs are small here (hidden of the dump, 8 live slots; the model saw up to 14.8 on massive-activation tokens).

## Stage timings (traced, per layer, host-timed) to know what is in the window
router+pad+spread 0.29 (fast router 0.03-0.1), counts (masked_bincount+offset_cumsum, CCL on cluster axis 0) 0.09, dispatch 0.03, experts (unified op) 0.78 (HiFi3: ~1.0), combine 0.10, post_combine_reduce 0.19, all_reduce 0.05 ms.

## Not yet done / ideas
* Per-stage bitwise stress (MiMoMoEDecode.forward(upto=0..5), outputs kept alive): mask the garbage rows (dispatch/combine buffers are not initialised; compare only valid rows from counts/region offsets) to find the first stage that differs between replays.
* Experts-stage race candidates: unified op reads counts/region offsets on device while offset_cumsum (CCL) may not be complete for rows 1-3? (all consumers are ordered by the program queue, but fabric writes into other chips' buffers are asynchronous: a chip can start dispatch/combine reads before
  a neighbour's remote writes land; look at the semaphores in combine and dispatch end-of-op barriers).
* post_combine_reduce accumulates with packer L1 accumulate in bf16 and `read_tile_value` mailbox reads of the indices: a stale indices page would drop one expert (matches the repeating diff magnitudes).
* The decode agent's finding that attention outputs are identical until the MoE layer and the all_reduce stress is clean stand.
* Libraries: private relink builds with tools/priv_relink.py; moe_B_9a8fd8582e5 has the precise numerics; the dispatch op now has MOE_DISPATCH_SPARSE_MCAST (this branch).
