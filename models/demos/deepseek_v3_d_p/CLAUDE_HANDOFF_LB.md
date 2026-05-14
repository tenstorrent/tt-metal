# Claude handoff: Combine ethernet measurement on LoudBox

You are picking up an in-progress experiment to measure how ethernet/fabric bandwidth affects the MoE prefill **combine** op for DeepSeek-V3 prefill on Tenstorrent hardware. The capture step is done on a Galaxy 8×4 machine; the **replay step on LoudBox 8×1 is what's running on this machine**. This document is the handoff from a previous Claude session.

## Goal in one paragraph

Capture combine inputs (metadata, counts, region offsets) from each of Galaxy 8×4's 4 dispatch groups during a real DeepSeek-V3 prefill at the layers of interest, ship the per-column `.pt` files to an 8-chip LoudBox, and replay the combine kernel there in isolation (no concurrent 4-way fabric contention). Compare LB per-column kernel times against Galaxy actual per-layer times to quantify ethernet bandwidth sensitivity. The user already finished the capture step; only the replay sweep + aggregation remains on this LB.

## Current state (assume this is already done)

- ✅ Captures generated on Galaxy at two ISL sizes — sitting on this LB at:
  - `/localdev/nmilicevic/combine_captures/` (25K isl, ~24 MB per `.pt`)
  - `/localdev/nmilicevic/combine_captures_1k/` (1K isl, ~1.3 MB per `.pt`)
- ✅ Layout per capture dir: `L<NN>/col<K>.pt` for 58 MoE layers × 4 columns = 232 files.
- ✅ Captures have been **patched** so `dispatched_metadata[..., 0]` is within-group chip ID `[0, 8)` instead of Galaxy global LinMeshCoord. Check via `config["src_chip_remapped"] == True` inside any `.pt`. (Without this patch the combine kernel hangs forever — it tries to fabric-write to non-existent chip IDs.)
- ✅ Test code at `models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py` works end-to-end with one combo (verified). Uses `ttnn.from_torch(..., dtype=ttnn.bfloat16, mesh_mapper=get_ep_mesh_mapper(...))` for the buffer — **do not use `ttnn.empty` or `ReplicateTensorToMesh` here**; both produce tensors without the shard spec combine needs and hang.
- ✅ Sweep runner at `models/demos/deepseek_v3_d_p/tests/perf/run_combine_replay_sweep.py` — loops over layer×col, saves per-combo CSVs, deletes tracy artifacts, prints per-layer max-of-4-cols summary.

## What's NEXT (the immediate task)

Run the sweep at the layers of interest, examine the per-layer LB combine times. Optionally compare against Galaxy actual perf in a future iteration.

```bash
# from /localdev/nmilicevic/tt-metal, with python_env activated
TT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python models/demos/deepseek_v3_d_p/tests/perf/run_combine_replay_sweep.py \
  --layers 5,20,40,55 \
  --cols 0,1,2,3 \
  --num-links-id linear-8-2link \
  --capture-dir /localdev/nmilicevic/combine_captures_1k \
  --out-dir /localdev/nmilicevic/lb_replay_results \
  --warmup 2 --timed 5 \
  |& tee sweep.log
```

The runner script does, per (layer × col):
1. `python -m tracy -p -v --disable-device-data-push-to-tracy -m pytest -v test_combine_replay.py -k "L<L>_col<C> and linear-8-2link"`
2. `python tools/tracy/process_ops_logs.py -n L<NN>_col<K>_linear-8-2link` to produce `ops_perf_results_*_<name>.csv`
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

## Files to know

| File | Purpose |
|---|---|
| `models/demos/deepseek_v3_d_p/COMBINE_GLX_LB_MEASUREMENT.md` | Full workflow doc (read for context if needed) |
| `models/demos/deepseek_v3_d_p/tt/moe/tt_combine.py` | `TtCombineModule` — the kernel wrapper. Has `_capture_inputs` for Galaxy capture, controlled by `TT_DS_CAPTURE_COMBINE_LAYERS` env var. Includes the `src_chip //= num_dispatch_groups` remap at capture time. |
| `models/demos/deepseek_v3_d_p/tests/perf/test_combine_replay.py` | Pytest entry point that loads `.pt`, pushes to device, runs combine N times. |
| `models/demos/deepseek_v3_d_p/tests/perf/fix_captured_metadata.py` | One-shot script that patches old captures (already run; idempotent via `config["src_chip_remapped"]` flag). |
| `models/demos/deepseek_v3_d_p/tests/perf/analyze_combine_perf.py` | Aggregation CLI with `galaxy <csv>`, `lb <csv>`, `compare` subcommands. |
| `models/demos/deepseek_v3_d_p/tests/perf/run_combine_replay_sweep.py` | The sweep runner — this is the main tool you'll use here. |

## Methodology (this is how the aggregation works)

For one Galaxy layer L, the LB no-contention estimate is computed in 3 levels:

```
layer_L_LB_combine_time =
  max over 4 captured columns of (
    max over 8 chips of (
      median over timed iters of DEVICE_KERNEL_DURATION_ns
    )
  )
```

- **median per chip across iters**: smooths run-to-run noise; max would overestimate by latching on the worst iter
- **max across 8 chips per column**: combine has an implicit synchronization barrier, kernel completes when slowest chip finishes
- **max across 4 columns**: on Galaxy all 4 columns run combine concurrently; layer combine time = slowest column

The sweep runner implements this in its `summarize()` function. Default `warmup=2 timed=5` is fine; lower if you want faster runs (3 timed minimum for a stable median).

## Gotchas the previous session hit (so you don't repeat them)

1. **Don't use `ttnn.empty` or `ReplicateTensorToMesh` for `dispatched_buffer`.** Both produce tensors without the per-device shard spec the combine kernel needs. Symptom: kernel queues fine, `synchronize_device` hangs forever. Use `ttnn.from_torch(host_tensor, mesh_mapper=get_ep_mesh_mapper(mesh_device), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)`. The host tensor is ~24 GB transient at 25K isl, ~1 GB at 1K isl — LoudBox has plenty of RAM (check `free -h`).

2. **The metadata fix has already been applied to existing captures** via `fix_captured_metadata.py`. If captures get regenerated on Galaxy with the latest `tt_combine.py`, the remap happens automatically and no further patching is needed. If somehow captures get regenerated WITHOUT the patched `tt_combine.py`, the kernel will hang again on replay — run `fix_captured_metadata.py /path/to/captures` to patch them in place.

3. **`python -m tracy -r` has a shell-quoting bug.** It reassembles its child command via `" ".join(args)` and runs with `shell=True`, which strips quotes from `-k "..."` filters. Pytest then sees `and` as a positional filename and collects 0 items. The runner script avoids `-r` entirely; it does manual `process_ops_logs.py` post-processing instead. If you need `-r` for some reason, use nested quotes: `-k "'L05_col0 and linear-8-2link'"`.

4. **Without `TT_METAL_DEVICE_PROFILER=1` set, no per-op CSV.** It's just an env var, set it in your environment or rely on the runner script which sets it for you.

5. **`OP CODE` and `DEVICE ID` columns only exist in `ops_perf_results_*.csv`** produced via the full `process_ops_logs.py` path (which needs `tracy_ops_times.csv` + `tracy_ops_data.csv` from a `python -m tracy` run). The smaller `cpp_device_perf_report.csv` produced by `--device-only` doesn't have them.

6. **LoudBox physical fabric is not a true linear-8 chain** (degree histogram `{3:8}` in fabric init logs — it's a 3-regular graph, probably a hypercube). But `FabricConfig.FABRIC_1D` + `Topology.Linear` works on it for combine at topk=8, as verified by `test_prefill_combine.py -k "linear-8-2link and perf_no_pcc"` running successfully.

7. **`expert_id` field in metadata is global Galaxy expert ID** (0-63 for col0, 64-127 for col1, …). The combine kernel doesn't use it for routing — only `src_chip`, `token_idx`, `topk_idx`, `weight` matter. Leave it as-is.

## How to extend / things you might be asked

- **Add `linear-8-1link` to the sweep**: re-enable the param in `test_combine_replay.py` (currently commented out — the user disabled it because they only want 2link). Then add `--num-links-id linear-8-1link` to a second sweep run.
- **Larger captures (25K isl)**: change `--capture-dir` to `/localdev/nmilicevic/combine_captures/`. Each replay run uses ~22 GB host RAM transient (fine on this box; verify with `free -h` before scaling).
- **Compare against Galaxy actual**: the user would need to provide an `ops_perf_results_*.csv` from a Galaxy 8×4 run with `TT_METAL_DEVICE_PROFILER=1`. Then `analyze_combine_perf.py compare --galaxy <csv> --lb-glob "/localdev/nmilicevic/lb_replay_results/*.csv"` builds the comparison table.
- **More layers**: `--layers 3,4,5,...,60` for all MoE layers. 58 × 4 = 232 runs at 1K isl ≈ 5-8 hours wall-clock.

## Environment

- `TT_METAL_HOME = /localdev/nmilicevic/tt-metal`
- Branch: `nmilicevic/ds-glx-lb-measure` (or whatever's checked out — look at `git branch --show-current`)
- Python: use `python_env/bin/activate` from the repo root
- LB hostname: `bh-51-special-nmilicevic-for-reservation-74201` (reservation-specific). User: `nikolamilicevic`.
- LoudBox: 8× p150b BH chips. Plenty of RAM (~500 GB).

## How to invoke me effectively here

When asking for help: paste the exact command you ran, the last ~50 lines of output, and what you expected to see. I'll diagnose. Things I'd want to know if a run fails:

```bash
# verify captures are patched
python -c "import torch; b=torch.load('/localdev/nmilicevic/combine_captures_1k/L05/col0.pt', weights_only=False); print(b['config'].get('src_chip_remapped'))"
# expect: True

# free RAM
free -h

# git branch
git -C /localdev/nmilicevic/tt-metal branch --show-current

# what's in the output dir
ls -la /localdev/nmilicevic/lb_replay_results/
```

If anything in the test_combine_replay.py or run_combine_replay_sweep.py code looks off, those are the only two files I expect to touch on this machine. The Galaxy capture side (`tt_combine.py._capture_inputs`, `fix_captured_metadata.py`) is done.
