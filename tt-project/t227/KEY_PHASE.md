# t227: offset the K/V brick grid by one site ("key phase")

## Why

Job F (blx01 863): stage-5 neighborhood-sdpa is 294 ms of each 471 ms block, 2.35 s of the 4.93 s decode.
LoFi barely moves it, so it is not math-fidelity bound. #213 showed tile narrowing saves only ~3% at
stage 5. The real waste is that the gather counts whole K bricks, and the 11-wide window starts 5 sites
before the query brick. That puts the window's start mid-brick on every axis, so each axis pays one extra brick.

Fix: brick K/V on a grid shifted against the query grid, so that interior windows start on a K brick
boundary. Q, the output, the window rule and the math do not change, and the result is exact: the same
keys are attended, with fewer wasted ones.

## Numbers (CPU, `key_phase_calc.py`, global volume (145,272,480), window 11^3, stride 1)

Slots = K bricks gathered per query chunk (worst case over chunk positions, like `gather_bricks`).

| brick | chunk | slots today | slots shifted | K phase | per Q brick today | shifted | gain |
|---|---|---|---|---|---|---|---|
| (2, 4, 4) | (1, 1, 1) | 175 | 96 | (1, 1, 1) | 175.0 | 96.0 | 1.82x |
| (2, 4, 4) | (2, 1, 1) | 200 | 112 | (1, 1, 1) | 100.0 | 56.0 | 1.79x |
| (8, 2, 2) | (2, 1, 1) | 196 | 144 | (0, 1, 1) | 98.0 | 72.0 | 1.36x |
| (4, 4, 2) | (2, 1, 1) | 210 | 120 | (1, 1, 1) | 105.0 | 60.0 | 1.75x |
| (1, 4, 8) | (2, 1, 1) | 180 | 144 | (0, 1, 0) | 90.0 | 72.0 | 1.25x |

The production 2-D split uses chunk (2,1,1) and needs a brick that divides H 68 and W 60. Today's best is
(8,2,2) at 98 slots per Q brick. With the shift, (2,4,4) at phase (1,1,1) needs 56: 1.75x fewer score
tiles, PV tiles, K/V reads and masks. If kernel time scales with slots, 294 -> ~168 ms per block, about
-1.0 s on the decode (4.93 -> ~3.9 s). The shift should also help det stages 1-4 (T brick 8 against
T window 3); not measured.

Phase (1,1,1) for brick (2,4,4) means K brick boundaries sit at global site = 1 (mod brick). Per axis:
- H (shard 68, brick 4): resident starts 7 below the owned rows and ends 5 above them (80 rows; today
  8 + 68 + 8 = 84). `_halo_exchange` already takes separate pad_left and pad_right.
- W (shard 60, brick 4): same, 7 + 60 + 5 = 72 (today 76).
- T (not sharded, brick 2): one zero frame below frame 0, so resident = 146. The mask hides it, since a
  clamped window never reaches below 0.

## What has to change (every phase-0 assumption found)

1. Planner `neighborhood_plan.cpp`
   - `validate_config`: drop "shard_origin brick-aligned" and "query_origin brick-aligned". Require
     instead that (shard_origin + query_origin) be brick-aligned, so the query grid stays global.
     Make the fit check work in sites.
   - `build_plan`: compute misalignment and gather-origin rounding in the RESIDENT-local frame
     (`floor_mod(union_global - shard_origin, b)`), not the global one. With an aligned shard this is
     identical to today.
   - Add `query_phase = query_origin - query_origin_bricks * brick` to the plan.
   - gtest: brute-force coverage at phase != 0, plus the 112-slot count for (2,4,4) chunk (2,1,1).
2. Reader `neighborhood_reader.cpp`: new compile args `query_phase_{t,h,w}`. Add the phase wherever a
   query site comes from bricks: line ~230 (word-0 encode), ~584 (`chunk_origin_site`), ~743
   (`query_origin_site`). `classify_brick` and `fill_mask_tile` are site-based already, so they are
   phase-safe.
3. Shortcuts that assume phase 0. Turn them off while phase != 0 for the first version, then make them
   phase-aware:
   - `relative_span_low/high` + `relative_table_index` (reader) and `_build_relative_masks` (Python): the
     relative tables.
   - `_build_regime_masks` (27 uploaded regime masks).
   - `gather_is_canonical` / `mask_writes_skippable`. This is mask reuse across chunks. With a constant
     phase the masks still depend only on (key brick - query brick, clamp), so reuse should stay valid
     once the expected span low accounts for the phase.
   Risk: with the tables off, every Mixed tile is filled on the reader (~1024 window tests). The reader
   may then become the limit. Check that with the device profiler before calling the gain.
4. Python `neighborhood_attention.py` / `_plan.py`
   - Opt-in env `DIFFVAE_NA_KEY_PHASE=1`: halo low = 7 / high = 5 (from the phase), a 1-frame T pad on
     K/V, `shard_origin` and `query_origin` from the phase, resident shape.
   - `_choose_sharded_brick` scores with the phase (it asks the real planner, so it follows).
   - Update the `test_choose_sharded_brick_regression` pins only under the env.
5. Quality: exact math, so expect PCC ~1 against the 2-D hifi2 output and the same 48+ dB against the
   unoptimized reference (host noise, seeds 0-4).

## Plan for the device side

One build on blx01 /var/tmp/fasth3, at/after t48 5c1635d733d plus this change. Then:
- job 1: gtest plus `test_neighborhood_sdpa.py` (incl. `test_choose_sharded_brick_regression`, the GNA+2-D
  guard, a new phase case vs the torch reference).
- job 2: decode A/B with DIFFVAE_S5_2D=1, phase off/on, 2 timed seeds + host-noise seeds 0-4.
