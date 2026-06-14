# conv_bench tools — UNIFIED (non-pin) branch port

3-mode conv trace profiler ported from `origin/wransom/conv_bench_v2`, re-wired for the unified
non-pin conv2d helper path (`wransom/conv_unify_trm`). Pin is fully removed on this branch; the TRM
lever is the helper's TileRowMajor auto-select (`CONV_TILE_PACK_ROW_MAJOR` define), no pin.

## Modes (TT_CONV_BENCH_MODE)
- **main**       — main's verbatim no-helper kernel `conv_bmm_tilize_main.cpp` (SubblockMajor,
  hand-written matmul). Factory inserts a vestigial `TEMP_SUM` CB index at compute-arg slot 25
  (after OUT@24) so the verbatim kernel's arg layout lines up; never read for non-depthwise.
- **helper_sbm** — unified kernel `conv_bmm_tilize.cpp`, TileRowMajor auto-select forced OFF →
  pure SubblockMajor non-pin.
- **helper_trm** — unified kernel with TileRowMajor auto-select ON. `TT_CONV_BENCH_FORCE_TRM=1`
  (set per-mode by cb_bench.sh — NEVER globally, it FATALs main/sbm) skips the ROI gate so every
  hard-eligible conv engages TRM. Hard-ineligible → graceful fallback to SBM (logged
  `trm_fallback_sbm=true`); a helper_trm row equal to helper_sbm = fallback, not a null result.

## Run
- Smoke probe one mode:  `MODE=helper_trm bash conv_bench_tools/cb_probe.sh`
- One label, all 3 modes: edit env, then `bash conv_bench_tools/cb_bench.sh <reps> main helper_sbm helper_trm`
- Full BH sweep:         `bash conv_bench_tools/cb_bh_all.sh`  → `conv_bench_data_unify.csv`

cb_bench.sh exports `TT_METAL_HOME=$WT` so Tracy device-kernel durations land in THIS worktree's
`generated/profiler/reports`, and `PYTHONPATH=$WT/ttnn:$WT` so the worktree ttnn is imported.
Build with profiler: `./build_metal.sh` (Tracy ON by default on this tree). Gate device runs behind
`scripts/run_safe_pytest.sh`.
