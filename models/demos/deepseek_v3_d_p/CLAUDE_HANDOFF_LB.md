# Claude handoff: Dispatch + Combine ethernet measurement on LoudBox

You are picking up an in-progress experiment to measure how ethernet/fabric bandwidth affects the MoE prefill **dispatch and combine** ops for DeepSeek-V3 prefill on Tenstorrent hardware. The capture step is done on a Galaxy 8×4 machine; the **replay step on LoudBox 8×1 is what's running on this machine**. This document is the handoff from a previous Claude session.

## RECOMMENDED PATH: unified dispatch+combine replay (new)

The flow described in **§ Unified flow** below is the **current recommended approach** — it captures only the gate's `indices` tensor (KB-sized) and runs both dispatch and combine on LB in sequence. This replaces the older combine-only flow (which captured combine's inputs directly and needed a `src_chip` global-coord remap).

The older combine-only flow (§ Legacy combine-only) is still available and described later, but you should default to the unified flow for new work.

# Unified flow (dispatch + combine)

## Goal in one paragraph

Capture the gate's `indices` tensor from each MoE layer during a real DeepSeek-V3 prefill on Galaxy 8×4, ship the small `.pt` files (KB each) to an 8-chip LoudBox, and replay both **dispatch** and **combine** there in sequence on each of the 4 conceptual Galaxy columns. Compare LB per-column kernel times against Galaxy actual per-layer times to quantify ethernet bandwidth sensitivity for both ops. The user already finished the capture step; only the replay sweep + aggregation remains on this LB.

## Why this design

Dispatch's only load-driving input is the gate `indices` tensor (`(dispatch_group_size, seq_len_per_chip, num_experts_per_tok)` int32). Everything else either falls out of indices via `get_gate_outputs()` (the expert_offsets and counts), is static config (the expert_dispatch_table), or is filler whose content doesn't affect kernel cycle count (`x` and `weights`). So a single `.pt` per layer captures the routing pattern; everything else is reconstructed on LB.

This is much cleaner than the older combine-only flow because:
- Smaller captures (KB vs MB-GB)
- No `src_chip` global-coord remap needed — dispatch on LB writes its own LB-local src_chip values into metadata, and combine consumes them naturally
- Measures BOTH ops with one run; profiler CSV separates `DispatchDeviceOperation` from `CombineDeviceOperation`

## Current state for unified flow

- 🟡 Galaxy capture script ready (`tt_dispatch.py:_capture_indices`, env `TT_DS_CAPTURE_DISPATCH_LAYERS`). User needs to **run the dispatch capture on Galaxy** to produce `L<NN>/indices.pt` files. The unified flow does NOT use the old `combine_captures*/` data.
- ✅ Replay test ready: `tests/perf/test_dispatch_combine_replay.py`.
- ✅ Sweep runner ready: `tests/perf/run_dispatch_combine_replay_sweep.py`.
- 🟡 Captures location once user runs them on Galaxy: typically `/localdev/nmilicevic/dispatch_captures/` (settable via `TT_DS_DISPATCH_CAPTURE_DIR`).

## Galaxy capture (one-time, run by user before any replay)

```bash
TT_DS_CAPTURE_DISPATCH_LAYERS=all TT_DS_DISPATCH_CAPTURE_DIR=/data/nmilicevic/dispatch_captures TT_DS_PREFILL_TTNN_CACHE=/mnt/models/.../prefill_secure DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528 python -m pytest -xvs models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py -k "pretrained and smoke and e256_device_fp32 and mesh-8x4 and 61_layers and longbook_qa_eng and 1024 and iter1 and balanced and right_pad"
```

`TT_DS_CAPTURE_DISPATCH_LAYERS` accepts `"all"`, a comma list `"5,20,40,55"`, or empty/unset (no capture). One `indices.pt` per MoE layer is written; total ~5 MB at 1K isl, ~45 MB at 25K isl (across 58 MoE layers). Rsync to LB:

```bash
rsync -av /data/nmilicevic/dispatch_captures/ bh-51:/localdev/nmilicevic/dispatch_captures/
```

## What's NEXT (the immediate task)

Run the sweep at the layers of interest, examine the per-layer LB dispatch+combine times. Optionally compare against Galaxy actual perf in a future iteration.

```bash
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python models/demos/deepseek_v3_d_p/tests/perf/run_dispatch_combine_replay_sweep.py --layers 5,20,40,55 --cols 0,1,2,3 --num-links-id linear-8-2link --capture-dir /localdev/nmilicevic/dispatch_captures --out-dir /localdev/nmilicevic/lb_dispatch_combine_results --warmup 2 --timed 5 |& tee sweep.log
```

Note: `num_links` is hard-coded to 2 (Blackhole always uses 2 links). The `--num-links-id linear-8-2link` arg only controls the output filename suffix; it's not a configurable link count.

The runner script does, per (layer × col):
1. `python -m tracy -p -v --disable-device-data-push-to-tracy -m pytest -v test_dispatch_combine_replay.py -k "L<L> and col<C> and linear-8-2link"`
2. `python tools/tracy/process_ops_logs.py -n L<NN>_col<K>_linear-8-2link` to produce `ops_perf_results_*_<name>.csv` (has rows for both `DispatchDeviceOperation` and `CombineDeviceOperation`)
3. Copies that CSV to `out_dir/L<NN>_col<K>_linear-8-2link.csv`
4. Deletes the source `generated/profiler/reports/<ts>/` dir and `.logs/tracy_profile_log_host.tracy`, `.logs/tracy_ops_times.csv`, `.logs/tracy_ops_data.csv`

After all combos, prints + saves `out_dir/summary_linear-8-2link.csv`:

```
Layer        col0           col1           col2           col3      LAYER max
L05      <ns>           <ns>           <ns>           <ns>          <ns>
L20      ...
L40      ...
L55      ...
```

## Files to know (unified flow)

| File | Purpose |
|---|---|
| `models/demos/deepseek_v3_d_p/COMBINE_GLX_LB_MEASUREMENT.md` | Older full workflow doc (combine-only) — has background context |
| `models/demos/deepseek_v3_d_p/tt/moe/tt_dispatch.py` | `TtDispatchModule` — kernel wrapper. Has `_capture_indices` for Galaxy capture, controlled by `TT_DS_CAPTURE_DISPATCH_LAYERS` env var. |
| `models/demos/deepseek_v3_d_p/tt/moe/tt_combine.py` | `TtCombineModule` — kernel wrapper. Old `_capture_inputs` is still there for the legacy flow but unused by unified flow. |
| `models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_replay.py` | **Unified replay test** — loads `indices.pt`, runs dispatch + combine on 8x1 LB. |
| `models/demos/deepseek_v3_d_p/tests/perf/run_dispatch_combine_replay_sweep.py` | **Unified sweep runner** — main tool for the LB measurement. |
| `models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py` | Legacy combine-only replay (still works, but newer flow is preferred). |
| `models/demos/deepseek_v3_d_p/tests/perf/run_combine_replay_sweep.py` | Legacy combine-only sweep runner. |
| `models/demos/deepseek_v3_d_p/tests/perf/analyze_combine_perf.py` | Aggregation CLI — currently combine-focused but easily extended to dispatch. |
| `models/demos/deepseek_v3_d_p/tests/perf/fix_captured_metadata.py` | Legacy one-shot patch script for OLD combine captures (unused by unified flow). |

## Methodology (how the aggregation works — same for dispatch and combine)

For one Galaxy layer L and one op (dispatch OR combine), the LB no-contention estimate is computed in 3 levels:

```
layer_L_op_LB_time =
  max over 4 captured columns of (
    max over 8 chips of (
      median over timed iters of DEVICE_KERNEL_DURATION_ns
    )
  )
```

- **median per chip across iters**: smooths run-to-run noise; max would overestimate by latching on the worst iter
- **max across 8 chips per column**: dispatch and combine both have implicit synchronization barriers — kernel completes when slowest chip finishes
- **max across 4 columns**: on Galaxy all 4 columns run dispatch/combine concurrently; the layer's wall-clock for each op = slowest column

The unified sweep runner produces TWO numbers per layer (dispatch_max and combine_max). Default `warmup=2 timed=5` is fine; lower if you want faster runs (3 timed minimum for a stable median).

## Gotchas the previous session hit (so you don't repeat them)

1. **`python -m tracy -r` has a shell-quoting bug.** It reassembles its child command via `" ".join(args)` and runs with `shell=True`, which strips quotes from `-k "..."` filters. Pytest then sees `and` as a positional filename and collects 0 items. The runner script avoids `-r` entirely; it does manual `process_ops_logs.py` post-processing instead. If you need `-r` for some reason, use nested quotes: `-k "'L05 and col0 and linear-8-2link'"`.

2. **Without `TT_METAL_DEVICE_PROFILER=1` set, no per-op CSV.** It's just an env var, set it in your environment or rely on the runner script which sets it for you.

3. **`OP CODE` and `DEVICE ID` columns only exist in `ops_perf_results_*.csv`** produced via the full `process_ops_logs.py` path (which needs `tracy_ops_times.csv` + `tracy_ops_data.csv` from a `python -m tracy` run). The smaller `cpp_device_perf_report.csv` produced by `--device-only` doesn't have them.

4. **LoudBox physical fabric is not a true linear-8 chain** (degree histogram `{3:8}` in fabric init logs — it's a 3-regular graph, probably a hypercube). But `FabricConfig.FABRIC_1D` + `Topology.Linear` works on it for combine at topk=8, as verified by `test_prefill_combine.py -k "linear-8-2link and perf_no_pcc"` running successfully.

5. **Old combine-only flow had a `src_chip` global-coord issue** — captured metadata had `src_chip ∈ {0, 4, 8, 12, 16, 20, 24, 28}` (Galaxy global LinMeshCoords) instead of `[0, 8)` (within-group). The combine kernel would try to fabric-write to non-existent chips and hang. The fix divided by `num_dispatch_groups` (4). **The unified flow doesn't have this problem** because dispatch runs on LB and writes LB-local src_chip values into the metadata that combine consumes.

6. **Don't use `ttnn.empty` or `ReplicateTensorToMesh` for `dispatched_buffer` in the old combine-only flow.** Both produce tensors without the per-device shard spec the combine kernel needs. Symptom: kernel queues fine, `synchronize_device` hangs forever. Use `ttnn.from_torch(host_tensor, mesh_mapper=get_ep_mesh_mapper(mesh_device), ...)`. The unified flow sidesteps this entirely because dispatch produces the buffer with the right shard spec already.

7. **`expert_id` field in metadata is global Galaxy expert ID** (0-63 for col0, 64-127 for col1, …) in the old flow. The combine kernel doesn't use it for routing — only `src_chip`, `token_idx`, `topk_idx`, `weight` matter.

8. **LB replay is TRUE 1:1 with Galaxy col k via skip-by-table.** `num_routed_experts=256` (Galaxy global) is the table indexing space, NOT the number of physical experts hosted on LB. `experts_per_chip=8` is passed explicitly to `TtDispatchModule.__init__` (overrides `compute_constants` which would otherwise derive `256/8=32`). LB physically hosts `experts_per_chip × dispatch_group_size = 64` experts (8 per chip × 8 chips), same per-chip layout as Galaxy col k. The dispatch_table on LB is **col k's row of `ExpertMapping.create_dispatch_table(256, 8, 4)`** — `(1, 256)` shape with 64 valid chip IDs (in `[0, 8)`) for experts at global IDs `[k·64, (k+1)·64)`, and `-1` for the other 192 experts. Captured indices are pushed to LB **as-is** (no remap, values in `[0, 256)`). Every routing that Galaxy col k did is replayed; every routing that Galaxy col k skipped via -1 is also skipped on LB via -1.

9. **Untested kernel config risk**: the combination `num_routed_experts=256, experts_per_chip=8 explicit, dispatch_group_size=8, num_devices_physical=8` doesn't appear in any tt-metal CI parametrize that I've found. `TtDispatchModule.__init__` plumbs all four through, but if internal kernel code asserts something like `num_routed_experts == experts_per_chip * num_devices_physical` (which would be `64 == 64` in this config and fail), the kernel could reject the combo. First-run failure mode: hang at `synchronize_device`. Mitigation if it hangs: switch to a smaller-scale approximation — see git history for the previous "Path B" with `num_routed_experts=64` + remap + 4× over-routing.

## How to extend / things you might be asked

- **Vary num_links?** Not on Blackhole — always 2 links. The 1-link parametrize entry was removed; only `linear-8-2link` exists.
- **Compare against Galaxy actual per column**: use `analyze_galaxy_per_col.py` (next to `analyze_combine_perf.py`) to get Galaxy per-(layer, op, col) times. LB sweep produces per-(layer, col, links) CSVs. Both are 1:1 ("max over 8 chips for one column") so direct comparison is apples-to-apples — no over-routing fudge needed in Path A.
- **More layers**: `--layers 3,4,5,...,60` for all MoE layers. 58 × 4 = 232 runs at 1K isl ≈ 5-8 hours wall-clock.
- **Larger captures (25K isl)**: same workflow with `--capture-dir` pointing at the 25K dispatch_captures dir. Replay run host RAM is bounded (~MB for indices + per-iter buffer allocation on device only).
- **Legacy combine-only flow**: still present in case you want to compare results, but the unified flow is preferred for all new work.

## Environment

- `TT_METAL_HOME = /localdev/nmilicevic/tt-metal`
- Branch: `nmilicevic/ds-glx-lb-measure` (or whatever's checked out — look at `git branch --show-current`)
- Python: use `python_env/bin/activate` from the repo root
- LB hostname: `bh-51-special-nmilicevic-for-reservation-74201` (reservation-specific). User: `nikolamilicevic`.
- LoudBox: 8× p150b BH chips. Plenty of RAM (~500 GB).

## How to invoke me effectively here

When asking for help: paste the exact command you ran, the last ~50 lines of output, and what you expected to see. I'll diagnose. Things I'd want to know if a run fails:

```bash
# inspect one indices capture
python -c "
import torch
b = torch.load('/localdev/nmilicevic/dispatch_captures/L05/indices.pt', weights_only=False)
print('keys:', list(b.keys()))
print('config:', b['config'])
print('indices shape/dtype:', b['indices'].shape, b['indices'].dtype)
print('indices value range:', b['indices'].min().item(), b['indices'].max().item())
"
# expect: keys = ['indices', 'config']
# expect: indices shape = (8, seq_len_per_chip, num_experts_per_tok), dtype=int32
# expect: indices value range in [0, num_routed_experts)

# free RAM
free -h

# git branch + latest commit
git -C /localdev/nmilicevic/tt-metal branch --show-current
git -C /localdev/nmilicevic/tt-metal log -1 --oneline

# what's in the output dir
ls -la /localdev/nmilicevic/lb_dispatch_combine_results/
```

If anything in `test_dispatch_combine_replay.py` or `run_dispatch_combine_replay_sweep.py` code looks off, those are the main files on this machine. The Galaxy capture hook lives in `tt_dispatch.py:_capture_indices` (only needed when re-running capture).

# Legacy combine-only flow (still works, but deprecated)

The previous session built a combine-only capture-and-replay flow before the unified one. It's still present and functional:

- Captures: `combine_captures/` and `combine_captures_1k/` with `L<NN>/col<K>.pt` files (4 per layer, ~MB each at 25K).
- Captures **have been patched** (`fix_captured_metadata.py`) so `dispatched_metadata[..., 0]` is within-group chip ID `[0, 8)`. The config flag `src_chip_remapped=True` in each `.pt` confirms the patch.
- Test: `test_combine_replay.py` loads `.pt`, pushes to device, runs combine N times. Uses `ttnn.from_torch(..., dtype=ttnn.bfloat16, mesh_mapper=get_ep_mesh_mapper(...))` for the buffer — do NOT use `ttnn.empty` or `ReplicateTensorToMesh` here (combine hangs without proper shard spec).
- Sweep: `run_combine_replay_sweep.py` mirrors the new unified runner but for combine only.

Use the unified flow for new measurements. The legacy flow is kept for backward compat and result comparison.
