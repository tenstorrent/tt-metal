---
name: gptoss-blaze-profiling
description: How to profile GPT-OSS-120B (and other fused decoder layers) in tt-blaze on a Blackhole loudbox — device-profiler setup, per-op timing method, run recipe, known blockers/fixes, report scripts, and established findings. Use when profiling blaze fused layers, interpreting per-op device times, or wiring profiling into CI.
---

# GPT-OSS-120B Blaze Profiling on Blackhole Loudbox

End-to-end playbook for per-op profiling of tt-blaze fused decoder layers on a single Blackhole loudbox. Covers the environment, the device-profiler method, the exact run recipe, every blocker hit and its fix, the report-generation scripts, and the findings established so far.

## 1. Environment & access

- **Host:** `ssh bh-lb-02` (Blackhole loudbox, 8 chips = 4×2 submesh for TP=8).
- **Container:** `docker exec -it blaze-djordje bash` — the dev container. Repo mounted at `/localdev/divanovic/tt-blaze` (same path on host and in container; `/localdev` is bind-mounted).
- **tt-metal:** `TT_METAL_HOME=/localdev/divanovic/tt-blaze/tt-metal`; build via `cmake --build tt-metal/build && cmake --install tt-metal/build` (needed after any tt-metal C++ patch).
- **Reset device:** `tt-smi -r` **inside the container** (not on host PATH). Reset between runs / after crashes.
- **Chip clock:** 1350 MHz (device-profiler timestamps are in cycles → µs = cycles/1350).
- **Dispatch regime:** blaze runs **slow dispatch** (`TT_METAL_SLOW_DISPATCH_MODE=1`). NEVER profile in fast dispatch — it does not reflect the real setup. This is a hard rule.
- **SSH gotcha:** SSH commands from this tooling frequently auto-background and return empty output initially. Prefer `run_in_background: true` and read the task output file after the completion notification; use `scp` to pull files/logs (more reliable than capturing ssh stdout). Large CSVs: run `gen_csv.py` *on the box*, then scp the small `ops.csv`.

## 2. The per-op timing method (how numbers are produced)

blaze runs the whole fused layer as one `program.run()`, so tracy's `ops_perf_results.csv` is empty. Timing comes from the **tt-metal device profiler**:

- Kernel codegen emits `DeviceZoneScopedN("<OP_NAME>")` around each op fragment → raw per-core/RISC cycle stamps land in `<outdir>/.logs/profile_log_device.csv`.
- Rows: `PCIe_slot, core_x, core_y, RISC, timer_id, time[cycles], ..., zone_name, type(ZONE_START|ZONE_END)`.
- **Pairing:** per `(chip,core,RISC,zone)`, sort events, pair each START with next END → duration = (end−start)/1350 µs. Drop durations above a clamp (mis-pairs).
- **Pool RISCs:** per `(chip,core,zone)`, pool all RISC instances, take **median** (robust to DM/other RISCs that sit waiting inside the zone). Taking max-over-RISC instead catches a waiting RISC and inflates the number — don't.
- **Busy core:** median > 0.3 µs.
- **Reduce-root chip:** the chip with the largest `MOE__REDUCE_TO_ONE` core (this is the MoE reduce root — the compute reference; on the loudbox it's PCIe slot 3). It is NOT "the slowest chip" — all 8 chips are lock-stepped at the same wall.
- **Per-op time = max over the reduce-root chip's busy cores** of the median duration (the op's critical core, where real compute lands; peer chips wait).
- **% wall = op_time / wall.** These do NOT sum to 100 — ops overlap across cores. The Σ of per-op times (~90 µs) exceeds the wall (~67.5 µs); the Σ is for *attribution* (which op is biggest), not an additive latency budget.

### Wall = the layer's per-iteration latency
- A `DeviceZoneScopedN("STAGE_CHECKPOINT")` wraps the whole while-loop body → its span = one full iteration on each core.
- **Wall = the reduce-root chip's STAGE_CHECKPOINT.** For steady-state layers (128/1024 ctx) all ~129 root cores agree tightly (e.g. 67.5 / 74.3 / 70.3 µs) → median works.
- **At very long context (64K) the layer is NOT lock-stepped** — the few SDPA cores run ~155 µs while most cores finish in µs. There the *median* STAGE_CHECKPOINT is wrong (≈5 µs); use the **max over root cores** (the critical SDPA core). Sanity check: wall(64K) ≈ wall(128) − SDPA(128) + SDPA(64K).

## 3. The run recipe

Driver script `/localdev/divanovic/run_perop.sh <nodeid> <outdir>` (run inside the container):

```bash
cd /localdev/divanovic/tt-blaze && source env.sh
export GPTOSS_WEIGHTS=synthetic                    # HF-free deterministic seed weights; timing is value-independent
export TT_METAL_SLOW_DISPATCH_MODE=1               # blaze's real regime (130-core grid)
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0  # copy_host_to_device workaround (see blockers)
export BLAZE_SKIP_LAYER_ASSERTS=1                  # skip PCC checks — TIMING ONLY (see caveat)
export BLAZE_WORKER_L1_SIZE=1421824                # shrink allocatable L1 → grow kernel-config ring for full per-op zones
unset BLAZE_PROFILE_MINIMAL_ZONES BLAZE_PERF_DUMP  # full per-op zones (STAGE_CHECKPOINT is always emitted now)
python -m tracy -r -p -v -o "$OUT" -m pytest "$NODEID"
# CSV -> $OUT/.logs/profile_log_device.csv  (device data auto-dumped at close_mesh_device)
```

- **tracy flags:** `-r` ops report, `-p` only-profile-enabled-zones, `-o <outdir>` artifacts folder, `-m pytest <nodeid>`. Output: `<outdir>/.logs/profile_log_device.csv` + `reports/<ts>/`.
- **Select the test by exact node-id** — `-k "a and b"` gets space-split by `python -m tracy` and selects 0 tests. Example node-id: `tests/blaze/fused_ops/gptoss_layer/test_gptoss_global_layer.py::test_gpt_oss_120b_global_layer[blackhole-True-pos127-num_iterations5-fabric_2d]`.
- **num_iterations:** use `num_iterations5` for robust steady-state medians (drops the cold iter-0 via median). Use `num_iterations1` when the iteration is long (≥~150 µs, e.g. global @ 64K) because multi-iteration overflows the per-core profiler buffer and corrupts ~50% of timestamps. num_it=1 is the *cold* iteration (one-time fills) but fine for compute-op timing.
- **The 28 µs input broadcast** shows every iteration in the standalone test (no upstream layer), so num_it=5 median still ≈ 28 µs — it's real per-layer cost, not just iter-0.

### kernel-config ring (why worker_l1_size)
- Full per-op zones make the kernel program larger than the default kernel-config buffer → "Program size (N) too large for kernel config buffer (70656)".
- ring = `max_worker_l1_size − worker_l1_size`. `max_worker_l1_size=1,532,416`; default `worker_l1_size=1,461,760` → ring 70,656. Lowering worker_l1_size grows the ring (costs allocatable L1).
- `worker_l1_size=1,421,824` → ~108 KB ring, enough for full per-op zones + STAGE_CHECKPOINT. Lower further only if a bigger program needs it.

## 4. Known blockers & fixes

**tt-metal C++ patches (`tt_metal/impl/profiler/profiler.cpp`) — CI BLOCKER:**
1. `useFastDispatch()` must return `MetalContext::instance(ctx).rtoptions().get_fast_dispatch() && !isGalaxyMMIODevice(...)` (was `is_dispatch_firmware_active()`). Without it → `std::bad_cast` at `close_mesh_device` / `ReadDeviceProfiler`, because blaze weight-upload calls `setup_fast_dispatch`.
2. Zone-source 16-bit hash-collision `TT_THROW` (~line 237): replace with `zone_src_locations.erase(...); continue;` (make non-fatal) — else num_it=1 full-zone runs throw.
- Requires rebuild+install: `cmake --build tt-metal/build && cmake --install tt-metal/build`. **The loaded lib is stale until `cmake --install`.**
- **CI implication:** CI downloads a PREBUILT tt-metal from the pinned SHA (`tenstorrent/tt-metal`), so these local patches are NOT present in CI. They must be upstreamed to tt-metal (and pinned) OR CI must fetch a tracy/profiler-enabled build. This is the main thing to resolve before CI profiling works.

**blaze changes:**
- `blaze/kernel_codegen.py`: `STAGE_CHECKPOINT` `DeviceZoneScopedN` emitted unconditionally at top of the while loop (was gated behind `BLAZE_PROFILE_MINIMAL_ZONES`). No-op without the profiler; gives per-iteration wall + per-op in a single run.
- `_PROFILE_MINIMAL_ZONES = os.environ.get("BLAZE_PROFILE_MINIMAL_ZONES")=="1"` gates the per-op zones (minimal mode = only STAGE_CHECKPOINT).

**Test edits** (`tests/blaze/fused_ops/gptoss_layer/test_gptoss_{global,windowed}_layer.py`):
- `variant = os.environ.get("GPTOSS_WEIGHTS","real")` (was hardcoded `"real"`); `provider=(gpt_oss_weight_provider if variant=="real" else None)`.
- Module-level `_DEV_PARAMS`/`_WDEV` dict; env-gated `["worker_l1_size"] = int(os.environ["BLAZE_WORKER_L1_SIZE"])`; parametrize uses `[_DEV_PARAMS]`.
- Added `position_id` params: global has `pos127/pos1024/pos2048/pos64k(65535)`; windowed has `pos127/pos1024/pos2048/pos64k(65535)`.
- `conftest.py`: `gpt_oss_weight_provider` returns None when `GPTOSS_WEIGHTS != "real"`.
- `harness.py`: `BLAZE_SKIP_LAYER_ASSERTS` early-return in `assert_layer_correctness`.

**Runtime crashes/fixes:**
- `copy_host_to_device` "Command sequence size mismatch 128 vs 96" → `TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`.
- Bus error / sysmem-held (zombie from a crash) → `pkill -9 -f "pytest|tracy"` then `tt-smi -r`.
- Timestamp corruption / buffer wrap: long iterations (≥~150 µs) or num_it=5 corrupt absolute timestamps and mis-pair long zones. gen_csv clamps mis-pairs; use higher `--clamp/--cpcap` for long-context runs, and prefer num_it=1 there.

## 5. Report scripts

Working copies at `/localdev/divanovic/report/{gen_csv.py, gen_html.py}`; committed copies under `scripts/perf_report/` (branch `divanovic/gpt-perf-report`) — the `/localdev/divanovic/report/` versions are NEWER (auto-PFX detection, STAGE_CHECKPOINT-derived wall, `--clamp/--cpcap`). Reconcile before relying on committed ones.

- **`gen_csv.py <profile_log_device.csv> --out <dir> [--wall 0] [--clamp 40] [--cpcap 500]`** → `ops.csv` (op,phase,time_us,pct_wall,busy_cores,n_chips,prev,next) + `coregrid.csv`. Auto-detects layer prefix (`GPTOSS_GLOBAL_LAYER__` / `GPTOSS_WINDOWED_LAYER__`) from a GPTOSS zone containing `MOE__REDUCE_TO_ONE`. `--wall 0` derives wall from STAGE_CHECKPOINT on the reduce-root chip. Raise `--clamp`/`--cpcap` for long-context (e.g. `--clamp 500 --cpcap 1000` at 64K); wall may need max-over-cores handling when not lock-stepped.
- **`gen_html.py ops.csv --out x.html --wall <us> --title/--h1/--subtitle`** → the 2-section report (relative-cost bar chart + per-op table with bold Σ and Layer-wall total rows). Theme-aware.
- **`coregrid.csv`** lets you prove serial-vs-parallel: ops sharing the same physical cores CANNOT overlap → they're serial. (gate/up/down_proj + SDPA all share the same 8 cores → serial.)

## 6. Established findings (GPT-OSS-120B, TP=8, decode)

**Per-layer wall:** global@128 ≈ 67.5 µs, global@1024 ≈ 74.3 µs, global@64K ≈ 152 µs; windowed(SWA) ≈ 70.3 µs at ANY context (128 and 64K both ~70 µs).

**Per-op (reduce-root, steady-state) — the layer is mostly SERIAL:**
- Input broadcast+RMSNorm+mcast-sync: **~28 µs** — but this is ~all barrier WAIT (RMSNorm compute ~1 µs). It's the `BroadcastRMSNorm` op = `CclBroadcast` (single-core, single-link torus forwarding tree, per-token fabric open/close + full barrier) → metadata mcast → RiscSync → RMSNorm. Cost is fabric latency/handshake, not bandwidth (6 KB payload). ~10 µs/hop; root 3.3 → 1-hop 13.4 → 2-hop leaf ~28 µs.
- MoE expert matmuls dominate: gate_proj ~12, down_proj ~12, up_proj ~7.3 µs — **context-independent**, all on the same 8 cores → serial (~31 µs). MXFP4 weights, d_model=2880, expert d_ff=2880 (back-derived: 36×128×3×2880² ≈ 115 B params). Memory-bound (weight streaming from DRAM) at batch=1.
- Reduces: MOE reduce ~6.9, ATTN_O reduce ~4.7 µs (fabric-latency bound, context-independent).
- **SDPA is the only context-scaling op:** global 2.6 (128) → 6.5 (1024) → 87.5 µs (64K, ~33×). Sliding-window (SWA) SDPA is **flat** (2.66 µs at 128 and 64K) — it reads a fixed 128-token circular window (`sliding_window_size=128`, `circular_size_tokens=128`), so cost is context-independent by design.
- Overlapped infra (optimizing yields ~0): CB_RECONFIG ~4, prologue ~1.6, mcast-syncs ~1.6 µs.

**Architecture (deployment):** blaze is **pipeline-parallel** — each decoder layer is its own stage on an 8-chip TP group, wired to the next by device-to-device fabric sockets; a single galaxy (8×4=32 chips) = 4 stages × 8 chips, 36 layers ring-mapped. Per-layer handoff = `MOE__REDUCE_TO_ONE` (8→1 core) → socket (1→1, one 6 KB page, overlaps via `n_slots=32` pipelining) → next layer's `broadcast_rmsnorm` (1→8). **The 1→8 broadcast recurs every layer and is serial inside each stage's wall (~38% of it)** — it is NOT hidden by pipelining (pipelining only hides the cross-stage socket hop). Global layers scale with context; SWA layers are flat.

**Roofline (assumption-dependent — VERIFY DRAM BW ≈512 GB/s and the expert weight-sharding factor):** global@128 layer ≈ 67.5 µs measured → ~22 µs theoretical floor. MoE matmuls memory-bound: `weight_bytes_per_chip / DRAM_BW` ≈ 4 µs each (~2 MB/chip/proj). Broadcast theoretical ~2 µs but practical floor ~6–8 µs (fabric protocol). SDPA memory-bound on KV read.

**Optimization priority:** broadcast (persistent fabric connections instead of per-token open/close; flatter tree) and the MoE expert matmuls are the real targets (on the critical path in every layer). Don't optimize overlapped infra.

## 7. Caveats

- **Timing is value-independent**, so synthetic weights + `BLAZE_SKIP_LAYER_ASSERTS=1` give valid op-times but do NOT verify numerical correctness (PCC). To verify correctness at a given position, run WITHOUT skip-asserts and with `GPTOSS_WEIGHTS=real`.
- Standalone single-layer profiling misses cross-stage fabric contention present in a live pipeline ring; absolute times could shift. Structural conclusions (which op is biggest, serial vs parallel, context scaling) hold.
- At num_it=1 the two context-independent comm ops (broadcast, ATTN_O reduce) can be undersampled on the reduce-root single instance — substitute their context-independent value from a num_it=5 run.
