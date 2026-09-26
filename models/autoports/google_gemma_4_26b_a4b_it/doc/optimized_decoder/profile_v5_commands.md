The prepared [v5 profile driver](run_profiles_v5.py) reproduces the two successful v4 profile commands with v5 output paths, then generates summaries, advice-enabled reports, and same-run reconciliation. This preparation task did not execute the hardware commands. The exact prepared argument arrays are in [profile_v5_prepared_commands.json](profile_v5_prepared_commands.json).

From the repository root, the hardware owner can run:

```sh
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/run_profiles_v5.py --plan
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/run_profiles_v5.py
```

`--plan` only prints commands. The default invocation checks the frozen v5 runtime and completed validation summary, captures the inherited profiler/Watcher environment, and executes layer 0 followed by layer 5. It uses the same recorded 4096/128 fixtures, defaults, real weights, trace profiling, timing, and program-cache guard as v4. It writes `actual_optimized_v5_profile_commands.json`, `profile_actual_optimized_v5_layer{0,5}.json`, and `tracy/actual_optimized_v5_layer{0,5}/`. Existing evidence is protected from accidental overwrite.

Only after the two-command profile journal is complete does the driver generate both layers' summaries and reconciliations. This ensures layer 0's reconciliation hash binds the final journal. The summaries retain the single-ASIC common LoFi peak (`--peak-fidelity-cycles 1`) and source-derived native KV chunk (`--native-sdpa-read-chunk-size 128`). Actual v5 precision policies come from the matching runner. Decode report input contains one complete native replay between its signposts; `--active-experts 8` is applied only after all 256 indexed sparse records confirm eight active groups out of 128. Prefill receives no active-expert override. Both text and CSV reports have advice enabled. Reconciliation uses the matching v5 profile runner, final journal, and v5 unprofiled headline observation; the latter remains outside same-run gaps.

After both profile commands have already succeeded, CPU postprocessing alone can be repeated with:

```sh
python models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/run_profiles_v5.py --postprocess-only
```

The model-local [summarizer](../../tests/summarize_perf.py), lines 260–340, includes every `tt_dnn_device` record regardless of native operation name. Useful prefill FLOPs at lines 220–236 describe logical model work, so `MinimalMatmulDeviceOperation` needs no new name-specific FLOP formula. Generic operand traffic at lines 176–217 already uses actual CSV shapes/dtypes. Internal minimal-matmul K-block rounding remains excluded from useful work, with all operation time retained in the complete-layer denominator.

Each new profile directory also receives `prefill_projection_native_audit.json`. It requires the selected native minimal QKV shape/config/fidelity/dtypes, full native minimal output, and retained sliding regular output. It reads `M_block_size`, `K_block_size`, `N_block_size`, `subblock_h`, and `subblock_w` directly from raw attributes, retains all attributes/tensor/kernel-source fields, checks complete 4096-token projection coverage, and reconstructs useful QKV/output FLOPs from native logical shapes: 283,467,841,536 sliding and 401,579,442,176 full. Tied full K/V is counted once. Raw metrics remain pending until those new profiles exist.

The installed `tt-perf-report` 1.3.0 recognizes names containing `Matmul` for shape/FLOP analysis, but its config-advice parser (`perf_report.py`, lines 1369–1388) only recognizes `program_config`, `in0_block_w`, and `out_subblock_h/w`. Minimal matmul exposes `config`, `K_block_size`, and `subblock_h/w` instead. Consequently, a report may advise “No program_config specified” even when raw native attributes prove the selected config. The additional native audit records this limitation; it does not rewrite raw CSVs or installed tools. Native op names and config definitions are source-backed by `minimal_matmul_device_operation.hpp` and `minimal_matmul_device_operation_types.hpp`, and runtime policy is defined in `MinimalPrefillProjection` / `MinimalPrefillQKV` in the frozen v5 decoder.

CPU metadata controls exercised both native-name/config paths using synthetic substitutions on saved v4 shapes, including generic traffic and logical projection-FLOP equality. Those checks verify packaging only and are not v5 hardware or performance evidence.
