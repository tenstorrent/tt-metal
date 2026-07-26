# HANDOFF — verify the LTX-2.3 TP×SP per-layer sweep on BH Galaxy 4×8

You are picking up a perf characterization. One measured point exists (the shipped
`TP=4/SP=8` anchor); every other split is currently a **model calibrated through that
anchor**. Your job: run the real per-layer sweep and replace the modeled points with
measured ones, then confirm (or correct) the conclusion that **TP=4/SP=8** is the floor.

## What's here
- `../../../models/tt_dit/tests/models/ltx/test_perf_ltx_layer_tpsp.py` — the harness
  (one video-only `LTXTransformerBlock`, random weights, signposted forward). Params:
  6 splits `tp{1,2,4,8,16,32}_sp{32,16,8,4,2,1}` + 2 axis-placement variants
  `tp4_sp8_altaxis`, `tp8_sp4_altaxis`, × stages `stage_1` (N=9728), `stage_2` (N=38912).
- `model_v2.py` — the calibrated roofline + `ltx_tpsp_results.json` writer.
- `bucket.py` — stacked-CSV → 3-bucket record. `plot.py` — records → PNGs.
- `ltx_tpsp_results.json` — current (modeled + anchor) records; **overwrite entries with measured**.
- `../ltx_4x8_ring_profiles/` — the measured (4,8) anchor summaries this was calibrated on.

## Environment (matches the branch this was built on)
- BH Galaxy 4×8, profiler build (`ENABLE_TRACY=ON` — already on in this tree).
- Denoiser code = this branch's working tree, **including the uncommitted-then-snapshotted
  WIP** (split-forward ring-SDPA in `attention_ltx.py`, ccl nanobind/CMake, normalization).
  If verifying against a clean tree instead, expect ring-SDPA numbers to differ.
- venv: `./python_env/bin/python`. Run from `$TT_METAL_HOME`.

## CRITICAL first step
**Reset the Galaxy before anything** — the mesh won't open otherwise (this is why the
sweep is unfinished: `open_mesh_device` hung on a wedged eth core):
```
tt-smi -glx_reset
```
Serialize device jobs; run in background; never block-wait. Do NOT pipe the command
through `tail` — it buffers and the broker reaps the job as hung after 300 s of silence.

## Run one config (per (id, stage))
```
cd $TT_METAL_HOME && ./python_env/bin/python -m tracy -p -r \
  -o generated/profiler/ltx_tpsp/<id>_<stage> -a device_kernel_duration -t 5000 \
  -m "pytest models/tt_dit/tests/models/ltx/test_perf_ltx_layer_tpsp.py \
      -k '<id> and <stage>' -x -q -s"
```
Use precise `-k`, e.g. `-k 'tp4_sp8 and stage_2 and not altaxis'`. IDs:
`tp1_sp32 tp2_sp16 tp4_sp8 tp8_sp4 tp16_sp2 tp4_sp8_altaxis tp8_sp4_altaxis`.
**Skip `tp32_sp1`** — SP=1 is architecturally invalid for LTX video self-attn (can't mask
padded keys without ring `logical_n`); it will error. That's expected, record it as unsupported.

## Post-process one config
```
CSV=$(ls -t generated/profiler/ltx_tpsp/<id>_<stage>/**/ops_perf_results_*.csv | head -1)
./python_env/bin/python python_env/lib/python3.10/site-packages/tt_perf_report/perf_report.py \
  "$CSV" --start-signpost start --end-signpost stop --no-color \
  --stacked-csv benchmarks/tracy/ltx_tpsp/<id>_<stage>_stacked.csv >/dev/null
./python_env/bin/python benchmarks/tracy/ltx_tpsp/bucket.py <id> <stage> \
  benchmarks/tracy/ltx_tpsp/<id>_<stage>_stacked.csv \
  benchmarks/tracy/ltx_tpsp/ltx_tpsp_results.json
```
Then regenerate plots:
```
./python_env/bin/python benchmarks/tracy/ltx_tpsp/plot.py \
  benchmarks/tracy/ltx_tpsp/ltx_tpsp_results.json benchmarks/tracy/ltx_tpsp
```

## Buckets (see bucket.py)
- `matmul_tp_ccl` = AllGatherMinimalMatmulAsyncOp + MinimalMatmulStridedReduceScatterAsync + MinimalMatmulDeviceOperation
- `ring_attention` = RingJointSDPADeviceOperation
- `overhead` = everything else. Buckets sum to the per-layer total (single traced forward,
  1 ring-SDPA instance = 1 layer — no /2, unlike the 2-layer-window anchor).

## Sanity gate before trusting the sweep
Reproduce the anchor first: `tp4_sp8` `stage_2` per-layer total should land near the
modeled **~10.8 ms** (stage_1 near **~4.3 ms**). If it's wildly off, the harness/signpost
window is wrong — fix that before running the rest. (Note the anchor summaries in
`ltx_4x8_ring_profiles` are a 2-layer window; divide by 2 to compare per-layer.)

## What to report back
1. Measured per-layer total + 3-bucket split for every runnable (id, stage).
2. Does the measured curve stay U-shaped with the floor at TP=4/SP=8 for stage_2? Where's
   the stage_1 floor (model says TP=2/SP=16, shipped TP=4/SP=8 +9%)?
3. Axis placement: `tp4_sp8` vs `tp4_sp8_altaxis` and `tp8_sp4` vs `tp8_sp4_altaxis` — does
   SP-on-len-8 (bandwidth) beat SP-on-len-4 (fewer hops)?
4. How far the modeled non-anchor points were from measured (the model's compute:comm split
   assumption is the least-certain part; the qualitative shape should hold).
