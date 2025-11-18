[34 tools called]

## Triage Summary

- **vanilla_unet‑N150‑func** (job owner: Mohamed Bahnas/keerthana-r-mcw)
  - `pytest` exits immediately because `models/demos/wormhole/vanilla_unet/demo/demo.py` no longer exists; the demo lives under `models/demos/vanilla_unet/...`, so the test runner can’t even collect tests and fails with exit code 4.[^run4857]
  - The launch script still calls the old wormhole path:

    ```320:323:tests/scripts/single_card/run_single_card_demo_tests.sh
    run_vanilla_unet_demo() {
     # vanilla_unet demo
     pytest models/demos/wormhole/vanilla_unet/demo/demo.py
    }
    ```
  - **Fix:** point `run_vanilla_unet_demo` (and the N300 variant) at `models/demos/vanilla_unet/demo/demo.py`, or re-create the wormhole-specific wrapper if one is genuinely required.
  - **Extra contacts:** CODEOWNERS list `@dvartaniansTT` plus `@tenstorrent/cse-developer-ttnn` for this subtree, so loop them in alongside the job owner.[^codeowners]

- **vit‑N150‑func** (job owner: Ashai Reddy Ginuga)
  - The perf test completes, but `samples_per_sec` comes out *higher* (1458.8) than the hard-coded upper bound (1432.08), so the assertion fails even though the model is faster than expected.[^run4857]
  - The tolerance is derived from `expected_samples_per_sec = 1377` with `margin = 0.04` in the test:

    ```124:178:models/demos/wormhole/vit/tests/test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py
    if is_single_card_n300:
        expected_samples_per_sec = 1323
    else:  # n150
        expected_samples_per_sec = 1377
    ...
    margin = 0.04
    min_range = expected_samples_per_sec * (1 - margin)
    max_range = expected_samples_per_sec * (1 + margin)
    assert (
        samples_per_sec > min_range and samples_per_sec < max_range
    )
    ```
  - **Fix:** refresh the N150 baseline (set `expected_samples_per_sec` to the new observed value or learn it from historical data) and/or relax the upper bound so “better-than-expected” runs don’t fail. Consider checking only the lower bound if the goal is to catch regressions.
  - **Extra contacts:** CODEOWNERS for `models/demos/wormhole/vit` include `@arginugaTT`, `@uaydonat`, and `@tenstorrent/cse-developer-ttnn`; involve them if the perf target needs redefinition.[^codeowners]

- **yolov8s_world‑N150‑func and yolov8s_world‑N300‑func** (job owner: Dalar Vartanians)
  - Every test case in `models/demos/yolov8s_world/demo/demo.py` aborts with `TT_FATAL: Out of Memory: Not enough space to allocate 13107200 B L1 buffer across 64 banks … bank size is only 1 372 352 B`, triggered during a `Transpose/Permute` op inside TTNN’s data-movement path.[^run4857]
  - This indicates the intermediate tensor for the permute step needs ~13 MB of per-core L1, which now exceeds the hardware limits on both N150 and N300 runners.
  - **Fix ideas:**
    1. Update the model code to use a sharded or streaming memory config for the offending permute/transpose so the workspace stays in DRAM rather than L1.
    2. Split that layer into smaller spatial slices (or lower the batch/resolution for CI) so the per-core buffer falls under 1.37 MB.
    3. Profile CB/L1 usage via `TT_METAL_DUMP_L1_LAYOUT=1` to identify which buffer exploded after recent changes, then tune its `core_grid`, `tile` shape, or `cb_n_entries`.
  - **Extra contacts:** `models/demos/yolov8s_world` is owned by `@ssinghalTT` and the `@tenstorrent/cse-developer-ttnn` team per CODEOWNERS, so loop them in along with Dalar when reworking the memory layout.[^codeowners]

- **yolov5x‑N150‑func** (job owner: Dalar Vartanians)
  - Compilation fails repeatedly with `TT_THROW: Statically allocated circular buffers … clash with L1 buffers on core range [(x=0,y=0) - (x=7,y=7)]. L1 buffer allocated at 597984 and static circular buffer region ends at 763200`, meaning the compile-time circular buffers + runtime L1 allocations no longer fit simultaneously.[^run4857]
  - This typically happens when a recent kernel change increases CB size (larger tiles, more pipelines, extra accumulators) without re-balancing the `l1_small_size` budget.
  - **Fix ideas:**
    1. Reduce the number of static CB entries (e.g. lower `num_tiles_in_multi_buffer` in the yolov5x TTNN config).
    2. Reassign some buffers to DRAM (`MemoryConfig(buffer_semantics=DRAM)`) or break the program into multiple enqueues so fewer kernels coexist.
    3. Engage the conv kernel owners to revisit `Conv2dBlockConfig`/parallelization; the clash occurs in program IDs 25/455/885/1315, which are the large conv stages.
  - **Extra contacts:** Same CODEOWNERS as yolov8s_world (`@ssinghalTT`, `@tenstorrent/cse-developer-ttnn`) in addition to the job owner.[^codeowners]

## Additional Notes & Next Steps

- The Aggregate Workflow Data run you linked (`run 19470888989`) shows the same failing jobs and owner mapping; that dashboard is useful for tracking when fixes hit CI because it aggregates annotations over time.[^agg7035]
- Once the vanilla_unet path is corrected, re-run both N150 and N300 variants—they share the same script entrypoint.
- For the TTNN memory issues (yolov8s_world & yolov5x), capture `TT_METAL_DUMP_L1_LAYOUT` output or use the TTNN profiler to quantify CB/L1 usage before patching so you can validate improvements quantitatively.
- When adjusting performance thresholds (vit), update any downstream dashboards or documentation that reference the old numbers.
- After implementing fixes, re-trigger `(Single-card) Demo tests (no perf)` on `main` to confirm the five jobs clear.

[^run4857]: [(Single-card) Demo tests (no perf) · run 19467504051](https://github.com/tenstorrent/tt-metal/actions/runs/19467504051)
[^agg7035]: [Aggregate Workflow Data · run 19470888989](https://github.com/tenstorrent/tt-metal/actions/runs/19470888989)
[^codeowners]: `CODEOWNERS` in `.github/CODEOWNERS` (owners listed for each models/demos subtree)
