# Dispatch + Combine ethernet measurement: LoudBox 8x1 vs Galaxy 8x4

End-to-end workflow for measuring how ethernet/fabric bandwidth affects MoE prefill **dispatch** and **combine** ops in DeepSeek-V3 prefill on Tenstorrent hardware.

## TL;DR

1. **On Galaxy 8x4**: capture each MoE layer's gate `indices` tensor during a real prefill (KB per layer).
2. **On LoudBox 8x1**: replay each of Galaxy's 4 dispatch groups (cols) separately on the 8 LB chips → "no contention" per-col times.
3. **On Galaxy 8x4**: replay all layers in-situ → "with contention" per-col times.
4. **Compare**: LB per-col vs Galaxy per-col matches within ~1-3% (the residual is the actual 4-col fabric contention overhead). Hot col (heaviest routing share) matches on 57/58 dispatch and 58/58 combine layers.

## Approach

### Why captures are small

Dispatch's only load-driving input is the gate `indices` tensor (`(dispatch_group_size, seq_len_per_chip, num_experts_per_tok)` int32). Everything else either:
- Falls out of `indices` via `get_gate_outputs()` (expert_offsets, counts, region_offsets), or
- Is static config (`expert_dispatch_table` via `ExpertMapping.create_dispatch_table()`), or
- Is filler whose content doesn't affect kernel cycle count (`x`, `weights`)

So one `(8, sl, 8)` int32 tensor (~32 KB at 1K ISL, ~800 KB at 25K ISL) per layer is enough. Total across all 58 MoE layers: ~2 MB at 1K, ~46 MB at 25K.

### LB 8x1 replay design (per-col, isolated)

For Galaxy col k:
- **`num_routed_experts = 256`** on LB (= Galaxy global; the table indexing space, NOT physical experts)
- **`experts_per_chip = 8`** explicit override (LB hosts 8 experts/chip × 8 chips = 64 experts, same per-chip layout as Galaxy col k)
- **`dispatch_table = col 0's row` of `ExpertMapping.create_dispatch_table(256, 8, 4)`** — all cols on LB use col 0's table; per-col differentiation comes from the indices remap (next).
- **Indices remap (LB only)**: `indices_lb = (indices_galaxy - k * 64) % 64`, with out-of-col routings replaced by sentinel `255` whose table entry is `-1` → kernel skips. This is essential because LB's combine kernel uses `first_expert_id = 0` for its single-col mesh, so expert IDs must land in `[0, 64)`.

End-to-end: each col-k LB replay does **the same routings Galaxy col k did** (true 1:1), just relabeled into the col-0 expert range so the kernel's combine logic works correctly.

### Galaxy 8x4 replay design (full mesh, with contention)

- Mesh shape `(8, 4)`, `FABRIC_1D`, `Topology.Linear`, `num_links=2` — production config
- Full `(4, 256)` dispatch table; all 4 cols dispatch simultaneously
- Indices used as-is (no remap)
- One run per layer captures all 4 cols' per-chip times via the profiler

### Per-col attribution on Galaxy (critical)

Galaxy's physical Dev IDs are **permuted, not arithmetic** — `DEVICE_ID % 4` is wrong. The actual mapping is discovered at runtime via `mesh_device.get_device_id(MeshCoordinate(r, c))` and saved to a JSON sidecar (`meshmap_8x4.json` next to the capture dir). The aggregator reads the sidecar to group chips into the correct dispatch groups. Example mapping (varies per machine):

```
col 0: [0, 1, 2, 3, 27, 26, 25, 24]
col 1: [4, 5, 6, 7, 31, 30, 29, 28]
col 2: [12, 13, 14, 15, 23, 22, 21, 20]
col 3: [8, 9, 10, 11, 19, 18, 17, 16]
```

### Aggregation semantics

For each (layer, op, col), the wall-clock metric is:

```
per-iter:  max DEVICE KERNEL DURATION across the 8 chips in this col (this iter's wall-clock)
across iters: median over timed iters (warmup dropped)
```

This matches `models.tt_transformers.tests.test_utils.merge_device_rows` for non-collective ops. Then per-layer wall-clock = `max` across the 4 cols.

## Files (current state)

| File | Purpose |
|---|---|
| `tt/moe/tt_dispatch.py` | `TtDispatchModule` kernel wrapper. `_capture_indices()` hook saves the gate indices when `TT_DS_CAPTURE_DISPATCH_LAYERS` env var matches the layer. |
| `tt/moe/tt_combine.py` | `TtCombineModule` kernel wrapper. Legacy `_capture_inputs()` for the deprecated combine-only flow lives here too. |
| `tests/perf/test_dispatch_combine_replay.py` | **Replay test.** Loads `indices.pt`, runs dispatch+to_layout+combine for one combo. Handles both `(8,1)` (LB) and `(8,4)` (Galaxy) mesh variants. On the 8x4 variant it writes the `meshmap_8x4.json` sidecar. |
| `tests/perf/run_dispatch_combine_replay_sweep.py` | **Sweep runner.** Iterates over (layer × col) on 8x1 or (layer) on 8x4. Wraps pytest under tracy with `-r` (+ `shlex.quote(kfilter)` to dodge the shell-quoting bug), grabs the timestamped `ops_perf_results_*.csv` tracy auto-produces, copies just the ops CSV to `--out-dir`, deletes the big tracy artifacts. `--summary-only` re-aggregates without re-running. |
| `tests/perf/plot_lb_vs_glx.py` | Bar-plot generator. Reads both summaries; emits 8 standalone PNGs (2 ops × 4 cols) plus one combined PNG. |
| `tests/perf/analyze_galaxy_per_col.py` | Standalone Galaxy per-col aggregator from a single `ops_perf_results_*.csv`. Pre-mesh-map era — uses `DEVICE_ID % 4` so it's only correct if Galaxy's mesh happens to be row-major; otherwise use the sweep runner's `--summary-only` path. |
| `tests/perf/test_combine_replay.py` + `run_combine_replay_sweep.py` | Legacy combine-only flow. Functional but superseded. |
| `tests/perf/analyze_combine_perf.py` + `fix_captured_metadata.py` | Legacy combine-only tools. |
| `CLAUDE_HANDOFF_LB.md` | This document. |

## Commands

All commands are single-line for copy-paste. Set `TT_METAL_HOME` to the local `tt-metal/` root (`/data/nmilicevic/tt-metal` on Galaxy, `/localdev/nmilicevic/tt-metal` on LB).

### Step 1 — Galaxy capture (one-time)

On Galaxy 8x4:

```bash
TT_DS_CAPTURE_DISPATCH_LAYERS=all TT_DS_DISPATCH_CAPTURE_DIR=/data/nmilicevic/dispatch_captures_25k TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528 TT_METAL_HOME=/data/nmilicevic/tt-metal python -m pytest -xvs models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -k "pretrained and smoke and e256_device_fp32 and mesh-8x4 and 61_layers and longbook_qa_eng and 25600 and iter1 and balanced and right_pad"
```

Writes one `L<NN>/indices.pt` per MoE layer (~58 files) plus `meshmap_8x4.json` (after the first replay test runs on the same machine — see Step 3).

Rsync to LB:

```bash
rsync -av /data/nmilicevic/dispatch_captures_25k/ bh-51:/localdev/nmilicevic/dispatch_captures_25k/
```

### Step 2 — LB 8x1 sweep (no contention, per-col)

On LB:

```bash
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TT_METAL_HOME=/localdev/nmilicevic/tt-metal python models/demos/deepseek_v3_d_p/tests/perf/run_dispatch_combine_replay_sweep.py --mesh 8x1 --layers $(seq -s, 3 60) --cols 0,1,2,3 --capture-dir /localdev/nmilicevic/dispatch_captures_25k --out-dir /localdev/nmilicevic/lb_8x1_dispatch_combine_results_full_correct --warmup 1 --timed 3 --skip-existing |& tee lb_sweep.log
```

Produces 232 CSVs (58 layers × 4 cols) + `summary_linear-8-2link.csv`. ~4-6 hours wall-clock at 25K ISL. Use tmux to survive ssh disconnects.

### Step 3 — Galaxy 8x4 sweep (with contention, one CSV per layer)

On Galaxy:

```bash
TT_METAL_HOME=/data/nmilicevic/tt-metal python models/demos/deepseek_v3_d_p/tests/perf/run_dispatch_combine_replay_sweep.py --mesh 8x4 --layers $(seq -s, 3 60) --capture-dir /data/nmilicevic/dispatch_captures_25k --out-dir /data/nmilicevic/glx_8x4_dispatch_combine_results_full --warmup 1 --timed 3 --skip-existing |& tee glx_8x4_sweep.log
```

Single run per layer. The first run also writes `meshmap_8x4.json` into `--capture-dir` — needed for correct per-col aggregation. ~1-2 hours wall-clock at 25K ISL.

### Step 4 — Re-aggregate (no re-profile needed)

If you ever want to redo aggregation (changed `--warmup`, fixed a bug, re-rendered):

```bash
TT_METAL_HOME=/localdev/nmilicevic/tt-metal python models/demos/deepseek_v3_d_p/tests/perf/run_dispatch_combine_replay_sweep.py --mesh 8x1 --layers $(seq -s, 3 60) --cols 0,1,2,3 --capture-dir /localdev/nmilicevic/dispatch_captures_25k --out-dir /localdev/nmilicevic/lb_8x1_dispatch_combine_results_full_correct --warmup 1 --summary-only

TT_METAL_HOME=/data/nmilicevic/tt-metal python models/demos/deepseek_v3_d_p/tests/perf/run_dispatch_combine_replay_sweep.py --mesh 8x4 --layers $(seq -s, 3 60) --capture-dir /data/nmilicevic/dispatch_captures_25k --out-dir /data/nmilicevic/glx_8x4_dispatch_combine_results_full --warmup 1 --summary-only
```

### Step 5 — Comparison plots

After both summaries exist, copy the LB summary somewhere both Galaxy & LB can see (e.g., into the tt-metal root on Galaxy) and run:

```bash
python3 /data/nmilicevic/tt-metal/models/demos/deepseek_v3_d_p/tests/perf/plot_lb_vs_glx.py --lb-summary /data/nmilicevic/tt-metal/lb_summary_linear-8-2link.csv --glx-summary /data/nmilicevic/glx_8x4_dispatch_combine_results_full/summary_mesh-8x4-2link.csv --out-dir /data/nmilicevic/tt-metal/lb_vs_glx_plots
```

Produces 8 standalone PNGs (`dispatch_col0.png` … `combine_col3.png`) plus one combined PNG (`all_lb_vs_glx.png`). Each plot has blue bars (LB 8x1) and orange bars (Galaxy 8x4) side-by-side per MoE layer.

## Diagnostic: print the Galaxy mesh layout

On Galaxy, quick standalone (~5 s, no captures needed):

```bash
TT_METAL_HOME=/data/nmilicevic/tt-metal python -c "import ttnn; mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4)); ttnn.visualize_mesh_device(mesh); print('=== per-col device IDs ==='); [print(f'  col {c}: {[mesh.get_device_id(ttnn.MeshCoordinate(r, c)) for r in range(8)]}') for c in range(4)]; ttnn.close_mesh_device(mesh)"
```

Prints ttnn's canonical mesh visualization plus the 8 device IDs per dispatch group. The `[MESH-MAP]` block in the test's pytest output (when the env var is set) shows the same info.

## Final results (58-layer comparison, 25K ISL)

From `lb_vs_glx_comparison.csv` (produced by ad-hoc cross-comparison; both `summary_*.csv` files together):

| Metric | Value |
|---|---|
| Hot-col match LB↔GLX (which col is slowest per layer) | Dispatch 57/58, Combine 58/58 |
| Galaxy per-col overhead vs LB — Dispatch | mean +2.45%, median +2.41%, p90 +4.20%, max +6.44% |
| Galaxy per-col overhead vs LB — Combine  | mean +0.95%, median +0.89%, p90 +1.79%, max +3.04% |
| Galaxy layer_max overhead vs LB layer_max | Dispatch +1.67% mean, Combine +0.55% mean |

Galaxy's 4-col fabric contention overhead is real but small (~1-3%); per-col routing-share signature is preserved on both machines.

## Gotchas worth remembering

1. **Tracy `-r` shell-quoting bug.** `python -m tracy -r` reassembles its child command via `" ".join(args)` and runs it with `shell=True`, which strips quotes from `-k "..."` filters. **Fix:** the runner uses `shlex.quote(kfilter)` so the filter survives the round-trip. Don't drop `-r` — it's what triggers tracy's `generate_report()` and gives you `OP CODE` + `DEVICE ID` columns in the CSV.

2. **Without `TT_METAL_DEVICE_PROFILER=1`, no per-op CSV.** The runner sets it; if running pytest by hand, set it.

3. **LB combine kernel needs expert IDs in `[0, 64)`.** The kernel uses `first_expert_id = 0` on a single-col mesh, so for col k > 0 we **must** remap indices on the host before pushing (`indices - k*64`, with sentinel `255` → table `-1` for out-of-col). The test does this automatically when `mesh_cols != ndg_galaxy`. Skipping this remap leaves col 1/2/3 combine processing nothing → bogus ~1ms times.

4. **Galaxy mesh layout is permuted.** Per-col attribution **must** use `meshmap_8x4.json`, not `DEVICE_ID % 4` or `DEVICE_ID // 8`. The sidecar is written automatically by the test on the first 8x4 run. If you ever re-image the machine or change the mesh shape, regenerate the sidecar by running one test combo.

5. **Don't use `ttnn.empty` / `ReplicateTensorToMesh` for `dispatched_buffer`** in the legacy combine-only flow. Both produce tensors without the per-device shard spec the combine kernel needs → kernel queues then `synchronize_device` hangs. Use `ttnn.from_torch(host_tensor, mesh_mapper=get_ep_mesh_mapper(...))`. Not an issue in the unified flow.

6. **`num_links` is always 2 on Blackhole.** The 1-link parametrize entry was removed; only `linear-8-2link` (LB) and `mesh-8x4-2link` (Galaxy) exist.

7. **`num_routed_experts=256` + `experts_per_chip=8 explicit` + `num_devices=8`** is an untested kernel config (doesn't appear in any tt-metal CI parametrize). It works for our use case but isn't on the validated config matrix. If you switch ISL or layer config and hit a hang at `synchronize_device`, this is the likely cause.

## Environment

- `TT_METAL_HOME = /data/nmilicevic/tt-metal` (Galaxy) or `/localdev/nmilicevic/tt-metal` (LB)
- Branch: `nmilicevic/ds-glx-lb-measure`
- Python: `source python_env/bin/activate` from the repo root
- Galaxy hostnames: e.g. `bh-glx-110-c05u02`, `bh-glx-d03u02` (varies per reservation; you'll generally be on slurm-login first, then `srun --pty bash` onto a Galaxy node)
- LB hostname: `bh-51-special-nmilicevic-for-reservation-<N>`
- 8x4 Galaxy: 32× p150b BH chips. LoudBox: 8× p150b BH chips, both at 200G ethernet per fabric link

# Legacy combine-only flow (deprecated)

The previous session built a combine-only capture-and-replay flow before the unified one. Still works, kept for backward compat:

- Captures: `combine_captures/` / `combine_captures_1k/` with `L<NN>/col<K>.pt` files (4 per layer, MB-sized at 25K).
- `dispatched_metadata[..., 0]` (src_chip) had a global-coord bug — patched via `fix_captured_metadata.py`. The flag `src_chip_remapped=True` in each `.pt` confirms the patch.
- Test: `tests/perf/test_combine_replay.py`. Sweep: `run_combine_replay_sweep.py`. Analyzer: `analyze_combine_perf.py`.

Use the unified flow for any new work.
