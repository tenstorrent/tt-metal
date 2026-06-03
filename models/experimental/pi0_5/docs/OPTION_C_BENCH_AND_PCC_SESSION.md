# pi0.5 Option C — smoke / bench / PCC session log (2026-06-03)

**Branch**: `sdawle/dvartanians/pi0.5_openpi_upstream_blaze`
**Host**: g11blx01 (32-chip Blackhole Galaxy)
**Checkpoint**: `/home/tt-admin/pi05_cache/pi05_libero_upstream`
**Session goal**:

1. Finish the Option C smoke-test suite begun in the previous session.
2. Add an Option B e2e benchmark that mirrors `test_option_c_benchmark.py`
   1:1 so both pipelines report identical per-stage timings on the same
   workload + real weights (apples-to-apples e2e perf).
3. Add an Option C end-to-end PCC test against the existing torch
   reference (`models.experimental.pi0_5.reference.torch_pi0_5_model.Pi0_5Model`)
   using the same shrunk depth the smoke/bench paths exercise.

## Layout reminders

- Option C lives under `models/experimental/pi0_5/tt/option_c/` — a 3-stage
  split (vision 4 chips / VLM prefill 18 chips / denoise 6 chips) on an 8×4
  parent mesh. See [`pi0-5-option-c-paired-l1`](../tt/option_c/) (auto-memory)
  and `tt/option_c/README.md`.
- Pipeline orchestrator: `Pi0_5PipelineC.run_inference()` returns
  `(clean_actions: ttnn.Tensor, StageTimingsC)`.
- All current dry runs use the **shrunk layout**
  `build_shrunk_layout(vlm_depth=2, expert_depth=1)` because the
  replicated upload path doesn't fit per-chip L1 at full 18-layer depth
  yet (layer-paired L1 is wired and tested but not the default).

## What ran in this session

### 1. Option C smoke (`tests/test_option_c_smoke.py`) — ✅ 11/11 PASS

```bash
python -m pytest -xvs models/experimental/pi0_5/tests/test_option_c_smoke.py
```

Coverage layered by strictness:

| # | Test                                              | Result |
|---|---------------------------------------------------|--------|
| 1 | test_default_layout_shape_c                       | PASS   |
| 2 | test_open_32_chip_mesh_partition_c                | PASS — 4/18/6 chips per stage confirmed |
| 3 | test_vlm_slice_forward_one_layer_c                | PASS   |
| 4 | test_expert_slice_forward_one_layer_c             | PASS   |
| 5 | test_inter_submesh_host_bounce_c                  | PASS   |
| 6 | test_e2e_vlm_to_expert_shrunk_c                   | PASS — synthetic prefix → 1-layer prefill → KV migrate → 1-layer expert step |
| 7 | test_full_pipeline_object_dry_run_c               | PASS — `Pi0_5PipelineC(...).initialize()` on real ckpt |
| 8 | test_vlm_slice_layer_paired_l1_two_layers         | PASS — 1 layer per chip × 2 chips, layer-paired L1 |
| 9 | test_stage_prefill_layer_paired_l1_dry_run        | PASS — same via `StagePrefill(layer_paired_l1=True)` |
| 10 | test_expert_slice_layer_paired_l1_two_chips      | PASS — 3 expert layers/chip × 2 chips |
| 11 | test_vision_slice_device_siglip_split_dry_run     | PASS — 3-chip SigLIP split + projector chip |

Log: `_bench_runs/option_c_smoke.log`. Process exit was 0; the trailing
`TT_THROW: MeshDevice cq ID 0 is in use by parent mesh ID 2 during close
of mesh ID 4` lines are the known parent-mesh-teardown ordering artifact
flagged in [`pi0-5-option-c-paired-l1`](../tt/option_c/) — they occur
*after* all assertions pass.

### 2. New Option B e2e benchmark (`tests/test_option_b_benchmark_e2e.py`)

The existing `test_option_b_benchmark.py` is a microbenchmark suite
(per-op breakdown, all-reduce isolation, dispatch floor) against
*random* weights. To get a side-by-side e2e number against Option C we
added a new file that mirrors `test_option_c_benchmark.py` exactly:

| Option C test                                | Option B counterpart                |
|----------------------------------------------|-------------------------------------|
| `test_oc_bench_e2e_staged_breakdown`         | `test_ob_bench_e2e_staged_breakdown` |
| `test_oc_bench_replicated_vs_layer_paired`   | `test_ob_bench_replicated_vs_tp`     |
| `test_oc_bench_prefill_seqlen_sweep`         | `test_ob_bench_prefill_seqlen_sweep` |

Shared knobs (env vars):

| Variable                | Default | Meaning                                |
|-------------------------|---------|----------------------------------------|
| `PI0_OB_E2E_BENCHMARK`  | unset   | Set to `1` to opt in (skipped otherwise) |
| `PI0_OB_E2E_WARMUP`     | 2       | warmup iters                           |
| `PI0_OB_E2E_ITERS`      | 5       | measured iters                         |
| `PI0_OB_DENOISE_STEPS`  | 10      | Euler integrator steps                 |
| `PI0_OB_CHECKPOINT`     | `/home/tt-admin/pi05_cache/pi05_libero_upstream` | real ckpt path |

The matched workload constants are identical to `test_option_c_benchmark.py`:

```
LANG_SEQ_LEN          = 256
PREFIX_LEN            = 512        # 256 image + 256 lang
ACTION_HORIZON        = 10         # upstream LIBERO ckpt
ACTION_HORIZON_PADDED = 32         # tile-aligned
NUM_DENOISE_STEPS     = 10
shrunk_layout         = vlm_depth=2, expert_depth=1
```

`test_ob_bench_e2e_staged_breakdown` drives the full Option B path
manually (`Pi0_5PipelineB.run_one_step` only fires a single expert step
— the full denoise loop isn't part of its orchestrator yet, see
`pipeline.py` line 14). We call each stage in order plus
`Stage3Expert.denoise` for the full Euler loop, so the per-stage
breakdown reads:

```
stage_0_vision_ms
transport_0_to_1_ms
stage_1_vlm_first_half_ms
transport_1_to_2_ms
stage_2_vlm_second_half_ms
kv_migration_ms
stage_3_denoise_ms
total_ms
+ per-Euler-step ms distribution
```

The Option C breakdown is the analogous 3-stage shape
(`stage_0_vision_ms`, `transport_0_to_1_ms`, `stage_1_prefill_ms`,
`kv_migration_ms`, `stage_2_denoise_ms`, `total_ms`, plus per-step).

### 3. New Option C end-to-end PCC test (`tests/pcc/test_pcc_option_c_vs_torch.py`)

Drives `Pi0_5PipelineC.run_inference()` and compares the final
clean-action tensor to `Pi0_5Model.forward_inference()` from the torch
reference — same approach as `test_pcc_pi05_model_vs_torch.py` (the
single-device PCC test). Both sides share the initial noise `x_0` (we
monkeypatch the torch reference's `denoising.sample_noise` to return our
`x_0.clone()` and pass the same padded `x_0` to the TT pipeline as
`noisy_actions`), so the flow-matching Euler trajectories converge from
identical starting points — same trick the single-device PCC test uses
(line 166 there).

Both sides run at the shrunk depth: the torch reference's
`Pi0_5ModelConfig` is created via `_make_shrunk_torch_config()` which
overrides `vlm_config.depth = VLM_DEPTH` and
`expert_config.depth = EXPERT_DEPTH` after the variant fills in the
widths. The TT side uses `build_shrunk_layout(vlm_depth=VLM_DEPTH,
expert_depth=EXPERT_DEPTH)`.

Knobs:

| Variable                       | Default | Meaning                          |
|--------------------------------|---------|----------------------------------|
| `PI0_OC_PCC`                   | unset   | `1` to opt in (skipped otherwise) |
| `PI0_OC_PCC_VLM_DEPTH`         | 2       | torch + TT shrunk vlm depth      |
| `PI0_OC_PCC_EXPERT_DEPTH`      | 1       | torch + TT shrunk expert depth   |
| `PI0_OC_PCC_STEPS`             | 10      | Euler integrator steps           |
| `PI0_OC_PCC_THRESHOLD`         | 0.90    | required min PCC                 |
| `PI0_OC_PCC_SEED`              | 42      | RNG seed for inputs + x_0        |
| `PI0_OC_PCC_CHECKPOINT`        | `/home/tt-admin/pi05_cache/pi05_libero_upstream` | real ckpt |

Test prints (in this order):

```
torch_actions.shape, then
tt_actions.shape, then
stage timings (ms), then
PCC(option_c vs torch),
then the first 3 rows × 6 cols side-by-side for inspection.
```

Asserts `pcc >= PI0_OC_PCC_THRESHOLD`.

## Repo state at session start

```
M  models/experimental/pi0_5/tt/ttnn_siglip.py
?? models/experimental/pi0_5/tests/pcc/test_pcc_vlm_stack_drilldown.py
?? models/experimental/pi0_5/tests/pcc/test_siglip_bs_attn_isolated.py
?? models/experimental/pi0_5/tests/pcc/test_siglip_bs_attn_substeps.py
?? models/experimental/pi0_5/tests/pcc/test_siglip_bs_drift_diag.py
?? models/experimental/pi0_5/tests/pcc/test_siglip_bs_layer0_split.py
?? models/experimental/pi0_5/tests/pcc/test_siglip_bs_per_layer_diag.py
?? models/experimental/pi0_5/tests/pcc/test_vlm_block_diag_in_pytest.py
?? models/experimental/pi0_5/tests/perf/test_perf_ttnn_e2e_host_breakdown.py
?? models/experimental/pi0_5/tests/perf/test_perf_ttnn_full_e2e_with_reports.py
?? models/experimental/pi0_5/tests/perf/test_siglip27_trace_profile.py
?? models/experimental/pi0_5/tests/perf/test_siglip27_traced.py
?? models/experimental/pi0_5/tests/test_option_b_benchmark.py        # microbench, pre-existing
?? models/experimental/pi0_5/tests/test_option_c_benchmark.py        # pre-existing
?? models/experimental/pi0_5/tests/test_option_c_smoke.py            # pre-existing
?? models/experimental/pi0_5/tt/option_c/                            # pre-existing
?? pi0_5_showcase.pptx
```

## New files added this session

```
models/experimental/pi0_5/tests/test_option_b_benchmark_e2e.py
models/experimental/pi0_5/tests/pcc/test_pcc_option_c_vs_torch.py
models/experimental/pi0_5/docs/OPTION_C_BENCH_AND_PCC_SESSION.md   # this file
```

(Removed mid-session: an earlier unified `tests/perf/test_option_b_vs_c_e2e.py`
combined-pipeline file — replaced with the mirrored-style approach above per
user feedback "we already had all the PCC test and perf test for Option B,
we can mirror Option C on that or use as a reference if needed".)

## What failed / blockers seen

1. **Device firmware timeout (Device 10) during the first e2e bench attempt.**
   `TT_THROW: Device 10: Timeout (10000 ms) waiting for physical cores to
   finish: 4-2, 11-2, 14-2, 13-2, 12-2, 10-2, 1-4, 3-3, 7-2, 2-4, 3-2,
   5-2, 6-2, 4-3.` Compiled ~30 kernels then hung. Likely a stale fabric /
   compile-launch race. Recovered partially by `tt-smi -r` board reset
   (user did manually).
2. **`get_cluster_type()` returns IndexError after the reset.** Second
   Option B run attempt (post-reset) failed at
   `ttnn.open_mesh_device → DispatchCoreConfig.__init__ →
   get_default_dispatch_core_type → cluster.get_cluster_type()` with:
   ```
   IndexError: unordered_map::at
     (ttnn/ttnn/device.py:92, in get_default_dispatch_core_type)
   ```
   This is not test-specific — running the same call in a clean
   subprocess (`python -c "import ttnn._ttnn.cluster as c;
   print(c.get_cluster_type())"`) reproduces. The cluster comes up
   ("Starting devices in cluster completed.") but then `get_cluster_type`
   maps the discovered type to something not in the C++
   `cluster_type_str_map` and `unordered_map::at` throws.

   `OPTION_B_STATUS.md` documented the same symptom on 2026-06-02:
   *"tt-metal rebuilt (build dir timestamp 2026-06-02 18:56).
   get_cluster_type() now returns ClusterType.BLACKHOLE_GALAXY. Both
   initial smoke tests pass; the build mismatch is resolved."*
   Suggests this state has happened before and was cleared by a rebuild
   (or by whatever side effect the rebuild produced — possibly a
   firmware re-flash / fresh cluster discovery state). **The earlier
   smoke run in this session DID work, so cluster type detection was
   healthy at session start; it broke after the firmware timeout
   crash even though the user ran `tt-smi -r` afterward.**

3. **Smoke teardown TT_THROWs** are cosmetic (parent-mesh close ordering)
   — already documented in `pi0-5-option-c-paired-l1` and don't affect
   test outcomes.

## Run commands (post board-reset)

```bash
# Option B e2e bench (real ckpt, matched params)
PI0_OB_E2E_BENCHMARK=1 PI0_OB_E2E_WARMUP=1 PI0_OB_E2E_ITERS=3 \
  python -m pytest -xvs \
  models/experimental/pi0_5/tests/test_option_b_benchmark_e2e.py::test_ob_bench_e2e_staged_breakdown

# Option C e2e bench (real ckpt, matched params)
PI0_OC_BENCHMARK=1 PI0_OC_BENCH_WARMUP=1 PI0_OC_BENCH_ITERS=3 \
  python -m pytest -xvs \
  models/experimental/pi0_5/tests/test_option_c_benchmark.py::test_oc_bench_e2e_staged_breakdown

# Option C end-to-end PCC vs torch
PI0_OC_PCC=1 \
  python -m pytest -xvs \
  models/experimental/pi0_5/tests/pcc/test_pcc_option_c_vs_torch.py
```

## Results table (populated as runs finish)

### Option B e2e staged breakdown

```
ATTEMPT 1 (pre-reset, 2026-06-03 03:02 UTC):
  FAILED — Device 10 firmware timeout (10000 ms) during compile.

ATTEMPT 2 (post `tt-smi -r`, 2026-06-03 03:11 UTC):
  FAILED — IndexError: unordered_map::at from get_cluster_type
  (Bench code never ran; failure inside ttnn.open_mesh_device fixture.)

ATTEMPT 3 (post `tt-smi -glx_reset_auto`, 2026-06-03 03:31 UTC):
  HUNG — cluster discovery + fabric topology succeeded (32 nodes, all
  degree 4), ~30 "Using pre-compiled firmware" lines fired, then
  silence for ~2 hours with no further progress. Process killed
  manually after 2h7m. No exception, no timeout — straight deadlock
  somewhere in parent-mesh / 4-submesh init.

  Cleanup turned ugly: SIGTERM (TaskStop) did NOT kill the python
  process (uninterruptible kernel sleep). `kill -9` produced a
  zombie (PID 25607, 27 min CPU consumed) that **init is not
  reaping**, and the tt driver still lists it in
  /proc/driver/tenstorrent/0/pids. New ttnn opens now fail with
  `Failed to allocate TLB window` — the leaked TLB resources from
  the zombie are blocking everything.
```

### Option C e2e staged breakdown

```
NOT RUN — chained after Option B; killed when Option B hung.
```

### Option C PCC vs torch reference

```
NOT RUN — chained after Option C bench; killed when Option B hung.
```

(All three logs will be at `_bench_runs/option_{b,c}_e2e.log` and
`_bench_runs/option_c_pcc.log`. This doc is updated in-place once each
run finishes.)

## Next steps

-1. **Reap zombie PID 25607 + clear leaked TLB windows.** Before any
    further test runs:
    ```bash
    # Confirm the zombie is still around
    ps -o pid,ppid,stat,etime,cmd -p 25607
    cat /proc/driver/tenstorrent/0/pids   # should still list 25607
    # Galaxy reset will reinit the driver and release the leaked TLBs.
    tt-smi -glx_reset_auto
    # If the zombie persists after the reset, a host reboot is the
    # only clean fix (init is the only thing that can reap an
    # orphaned zombie and it's refusing to here).
    # Confirm clean:
    python -c "import ttnn._ttnn.cluster as c; print(c.get_cluster_type())"
    # Expect: ClusterType.BLACKHOLE_GALAXY (and clean process exit)
    ```

    **CONFIRMED 2026-06-03 05:42 UTC: `tt-smi -glx_reset_auto` does
    NOT reap the zombie.** After the user's reset, PID 25607 was
    still `Zl` (zombie), `/proc/driver/tenstorrent/0/pids` still
    listed it twice, and the tt module's ref count was still 64.
    Writing an empty string to the proc pids file failed with
    `tee: Input/output error`. Opening a few `/dev/tenstorrent/*`
    handles did not change driver state. **Host reboot is required.**

-2. **Run Option C smoke FIRST as a hardware-health check.** Before
    re-running the bench, confirm the same 11/11 smoke that passed
    at session start still passes — that proves the recovery worked.
    ```bash
    python -m pytest -xvs models/experimental/pi0_5/tests/test_option_c_smoke.py
    ```
    If smoke fails the same way, the underlying issue isn't transient
    and we need the OPTION_B_STATUS.md historical recipe (rebuild tt-metal).

-3. **Then run Option C bench BEFORE Option B bench.** Both attempts
    in this session crashed on the Option B path; Option C smoke is
    the known-good path. Try the Option C bench standalone first to
    isolate whether the Option B 4-stage path has its own bug or it's
    just hardware state. Run them un-chained (`&&` chaining means a
    hang in step 1 kills the whole batch and leaks resources as we
    just experienced).
    ```bash
    PI0_OC_BENCHMARK=1 PI0_OC_BENCH_WARMUP=1 PI0_OC_BENCH_ITERS=3 \
      python -m pytest -xvs \
      models/experimental/pi0_5/tests/test_option_c_benchmark.py::test_oc_bench_e2e_staged_breakdown
    ```

0. **Recover cluster type detection.** First action next session — none
   of the bench / PCC numbers can land until `get_cluster_type()`
   stops throwing `IndexError: unordered_map::at`.

   **Root cause (from `tt_metal/llrt/tt_cluster.cpp:87`):
   `get_cluster_type_from_cluster_desc` walks every chip's
   `BoardType` and only returns `BLACKHOLE_GALAXY` when ALL 32 chips
   report `UBB_BLACKHOLE` (the source loops `cluster_desc->get_all_chips()`
   and falls through to `INVALID` if even one chip is unrecognized).
   The pybind wrapper for `ClusterType` doesn't bind `INVALID`, so
   converting it back to Python throws `IndexError: unordered_map::at`.
   This matches the symptom: tt-smi sees all 32 chips healthy
   (`board_type=Blackhole, series=tt-galaxy-bh`) but the inter-tray
   fabric state from the crashed Option B run is still poisoning the
   cluster descriptor that umd returns.**

   Recovery sequence to try in order (each step is safe to repeat):

   ```bash
   # 1. Galaxy-aware reset with retry (the right path for Blackhole
   #    Galaxy state issues — `tt-smi -r` alone is the generic reset
   #    and doesn't clear inter-tray fabric state).
   tt-smi -glx_reset_auto
   # then confirm:
   python -c "import ttnn._ttnn.cluster as c; print(c.get_cluster_type())"
   # expect: ClusterType.BLACKHOLE_GALAXY
   ```

   If `-glx_reset_auto` doesn't clear it:

   ```bash
   # 2. Single-shot galaxy reset (more aggressive — no retry, fails
   #    fast if the underlying issue is hardware-level).
   tt-smi -glx_reset

   # 3. Per-tray reset (target just the tray containing the crashed
   #    Device 10 first — Device 10 is in tray 2 per the tray map:
   #    Tray 2 → /dev/tenstorrent/8,9,10,11,12,13,14,15).
   tt-smi -glx_reset_tray 2
   ```

   If neither galaxy reset works:

   ```bash
   # 4. Rebuild tt-metal. OPTION_B_STATUS.md 2026-06-02 19:06 UTC note
   #    documents the same symptom getting cleared by a rebuild:
   #    "tt-metal rebuilt … get_cluster_type() now returns
   #    ClusterType.BLACKHOLE_GALAXY". A rebuild side-effects a
   #    fresh cluster discovery on first import.
   ./build_metal.sh
   ```

   Last resort: host reboot.

   In every case the confirmation oracle is the same one-liner —
   it should print `ClusterType.BLACKHOLE_GALAXY` and exit cleanly:
   ```
   python -c "import ttnn._ttnn.cluster as c; print(c.get_cluster_type())"
   ```
1. **Re-run the three commands** (Option B e2e bench → Option C e2e
   bench → Option C PCC) and **update this doc** with the per-stage
   medians and the PCC value.
2. **Decide on shrunk-vs-real-depth comparison.** Today both pipelines
   benchmark at `vlm_depth=2, expert_depth=1` because the replicated
   path doesn't fit full 18-layer depth in per-chip L1. The
   layer-paired L1 placement is wired + smoke-tested for Option C
   (tests #8–10 above) — once the same is in for Option B's TP=8 path,
   we can re-run both at full depth and the e2e totals become
   directly comparable to the analytical 8.90 ms target in
   `PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3.1.
3. **Wire Option C `Pi0_5PipelineC` with a `layer_paired_l1=True`
   flag** so `test_oc_bench_replicated_vs_layer_paired` can be driven
   through `Pi0_5PipelineC(...)` directly instead of constructing the
   `StagePrefill` / `StageDenoise` orchestrators by hand. The benchmark
   file already has a TODO comment to this effect.
4. **Add an Option B PCC test** to match the new Option C one — mirror
   `test_pcc_option_c_vs_torch.py` against `Pi0_5PipelineB` so the
   pipelines are validated head-to-head at the same shrunk depth.
5. **Resolve the parent-mesh teardown TT_THROW** so the smoke-test
   tail no longer prints critical-level lines after a clean pass.
   Likely needs an explicit close ordering pass before the parent
   fixture exits (see `_close_micro_submeshes` for the local pattern).
