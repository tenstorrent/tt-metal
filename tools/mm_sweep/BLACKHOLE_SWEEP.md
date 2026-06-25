# minimal_matmul — Blackhole benchmarking handoff

You're picking up `minimal_matmul` perf work on a **Blackhole p150b** (single chip, compute grid **11×10**
= 110 cores, 1.35 GHz). The op kernels/factory are considered **done** — your job is **measurement**:
find the best **slicing `(S,Pk)` + blocking** for the **FLUX and LTX** shapes, then (later) extend to a
larger sweep and fit BH-specific heuristics. If even the *best* config is sub-par on a shape, flag it —
that's the signal to reopen kernel optimization; otherwise the current code is expected to be enough.

## Setup
- Branch: `cglagovich/minimal-matmul-wh-sweep` (pull). Build: `git submodule update --init --recursive`
  then `bash build_metal.sh`. Use the **Blackhole** env (`ARCH_NAME=blackhole`, e.g. `source bh_env.sh`).
- Tool: **`tools/mm_sweep/joint_sweep.py`** (single-device, resumable).
- Background: BH grid.y=10 is **not** a power of 2. The op was made grid-generic for exactly this
  (divisor-based `(S,Pk)` via `largest_divisor_leq`; the old pow2 gate that crashed BH skinny shapes is
  gone). The WH-tuned NoC/DRAM levers do **not** carry to BH, so the sweep runs with them **off**.

## Run it
```
MM_CLOCK_HZ=1.35e9 TT_METAL_DEVICE_PROFILER=1 \
  python tools/mm_sweep/joint_sweep.py "" bh_joint.json flux     # FLUX only; use ltx / all likewise
```
- Sweeps, **per shape**, the cross-product of **all feasible `(S,Pk)`** (divisors of grid.y, K-feasible)
  × the **pruned block set**, plus an **AUTO baseline** (current heuristic + auto-block). Slicing and
  blocking interact, so this joint sweep is the ground truth — don't decompose.
- **Queries the device grid** (auto-adapts to 11×10 / harvesting) and computes `PEAK = GX·GY·2048·clock`.
- `MM_NO_LARGE_LEVERS=1` (default) isolates partition+blocking from the WH-fit levers.
- **Resumable**: writes one JSON entry per shape and skips shapes already present — safe to kill/restart.
  Per-shape it opens/closes the device (profiler CSV flushes at close), so partial results survive.
- Output per shape: `grid`, `peak_tflops`, `heuristic_SPk` (what the heuristic *would* pick),
  `auto` (baseline util), `best` (oracle `S,Pk,mb,kb,nb,subblock,us,util,pcc`), `best_vs_auto`, and `all`.

## Block-candidate prune rules (already in the script — don't re-add a divisibility constraint)
Blocks need **not** divide the per-core tile counts (the kernel does `div_up` + a clamped partial last
block — this was the main win on WH: e.g. `mb=6` on per-core-M=17 beat auto by 1.19×). Rules: max-DST
subblock (`sbh·sbw==4`, fp32 DST) except blocks too small to reach it; L1 footprint ≤ `1310720` B;
candidates capped at the smallest value covering per-core (no wastefully-large blocks); K-blocks are
divisors of per-band-K (no K-padding). No block-count lower bound — **L1 is the only upper bound**, which
keeps large shapes OOM-safe.

## VERIFY on BH (the script can't be tested on a single WH chip where it was written)
1. **L1 budget** — `L1_CB_BUDGET = 1310720` (1.25 MiB) is the WH-safe value. BH has more L1/core, so this
   is conservative (you may be leaving big blocks unswept). Bump it once you confirm BH's per-core CB
   limit; configs that OOM are caught (PCC fail / exception) so it's safe to push.
2. **Divisor-(S,Pk) actually runs on grid.y=10** — before the big sweep, sanity-check that explicit
   `TT_MM_NUM_SLICES=5` and `TT_MM_NUM_SLICES=5 TT_MM_K_SLICES=2 TT_MM_K_FUSED=1` PCC-pass on a couple
   shapes (this is the path the BH study had crashing pre-fix).
3. **`d.compute_with_storage_grid_size()`** returns the real BH grid (expect 11×10) so PEAK/util are right.
4. **clock** — 1.35 GHz is the BH peak; if you measure a different sustained clock, set `MM_CLOCK_HZ`.

## Method
- **Smoke first**: run 3–5 shapes (or `flux`/`ltx` subset), confirm PCC-pass across regimes, that `S>1`
  and `Pk>1` configs actually run, and that timing/util look sane. Then let the full FLUX/LTX run
  (~60–400 configs/shape, ~hours single-chip; resumable).
- **Read the results**: `best_vs_auto` tells you how much the oracle beats the current (WH-fit) heuristic
  on BH. `heuristic_SPk` vs the oracle's `best.S/Pk` divergence is the input for **re-fitting the BH
  (S,Pk) heuristic** (the thresholds in `sp_heur_backtest.py` were fit for WH 8×8 — they're a starting
  guess on BH). Low `best.util` even at the oracle ⇒ a structural shape worth flagging.
- **Then**: extend to a larger shape suite and parallelize across chips (see `block_sweep_mesh.py` for the
  submesh+threading pattern), and fit BH heuristics (S/Pk thresholds + the stage-2 block heuristic)
  against this oracle.

## Operational rules
- **Tracy-free** (don't enable the tracy profiler — it wedges large captures). Use
  `TT_METAL_OPERATION_TIMEOUT_SECONDS=60` to catch a hung dispatch instead of a wall-hang.
- **Always verify PCC** — an empty error grep is NOT a pass. A non-subblock-multiple block can silently
  corrupt to PCC~0.1; the sweep already gates on `pcc>0.99`, but double-check before trusting a number.
- Profiler CSV: `$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv`, flushes at device close.
- Reset a hung BH p150b with `tt-smi -r`.
- **Only commit when the user explicitly asks.**
