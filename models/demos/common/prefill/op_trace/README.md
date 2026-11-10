# Single-layer op traces: GLM-5.3 (DSA + MoE) and Gemma4 (sliding-window GQA)

Collects per-op device traces (Tracy `ops_perf_results_*.csv`) of ONE decoder layer of each model on a Blackhole
Galaxy (8x4), into one downloadable directory + tarball.

| name | what | script | test |
|---|---|---|---|
| `glm` | GLM-5.3-744B block at layer 6 (first full-indexer MoE layer: sparse-SDPA MLA + DSA indexer + 256-expert MoE), seq 5120, eager, warm-up forward then one signposted forward (`PROFILE_MEASURE_START/END`) | `models/demos/deepseek_v3_d_p/scripts/profile_glm_block.sh` | `deepseek_v3_d_p/tests/test_prefill_block.py::test_glm_prefill_block[...moe]` with `GLM_BLOCK_PROFILE=1` (hook added in that test) |
| `gemma` | Gemma4-31B first `sliding_attention` layer, 256k context, chunk 2048 and 4096, chunk index 1, traced replay between `gemma4-layer-local-sz<N>-chunk<C>-start/-stop` | `models/demos/gemma4_d_p/scripts/profile_sliding_layer.sh` (+ `check_layer_window.py`) | `gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk1-local-sz<N>-ctx_256k-8x4]` (existing CI test) |

## Run

```bash
cd $TT_METAL_HOME              # a built tt-metal with Tracy enabled, on a Blackhole Galaxy, fresh allocation
./models/demos/common/prefill/op_trace/run_all.sh
# subsets / knobs:
WHICH="gemma" SIZES="4096" ./models/demos/common/prefill/op_trace/run_all.sh
WHICH="glm" LAYER=dense ./models/demos/common/prefill/op_trace/run_all.sh
```

Output: `op_trace_bundle/<timestamp>/` (and `.tar.gz` next to it) with `SUMMARY.txt` (exit code, CSV count, ops per
device in the measured window, dropped-op count per run), flat `glm__<ts>__ops_perf_results_*.csv` /
`gemma__<ts>__ops_perf_results_*.csv`, and `glm/` + `gemma/` with each script's `script.log` and full Tracy output.
Download the tarball.

Weights: GLM uses real weights when `GLM53_HF_MODEL` + `TT_GLM53_PREFILL_TTNN_CACHE` point at a complete cache
(CI values are in `tests/pipeline_reorg/blaze_models_prefill_tests.yaml`, `glm_prefill_block`), else random weights
(same op timings). Gemma needs `HF_MODEL`/`HF_HOME`/`TT_CACHE_PATH` (defaults = CI values, offline).

## For Claude: running this and fixing it

**These scripts were written on a machine with no hardware and no ttnn.** Nothing in them has been executed.
Treat every failure as a probable script bug first. Loop: run -> read `<bundle>/<name>/script.log` -> fix -> rerun
only that part (`WHICH=...`). Do not weaken the checks to get green.

Done means: `SUMMARY.txt` shows for each run `exit=0 csvs>=1`, an op count per device, and
`ops with no device duration (dropped): 0`.

Things most likely to be wrong, in order:

1. **Gemma test id.** The id was derived by editing the CI id `chunkci-both-sz2048-ctx_256k-8x4`. Verify with
   `pytest --collect-only -q models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n | grep local-sz`
   and fix the bracket string in `profile_sliding_layer.sh`. Keep the node-id form (not `-k`): tracy re-joins the `-m` arg.
2. **GLM test id / skip.** `-k "seq5120 and torus-xy-8x4 and glm53 and moe"`; confirm with `--collect-only`. The test
   skips unless `is_blackhole()`; `requires_mesh_topology(8,4)` needs the right mesh graph / fabric on the host.
3. **CSV location / name.** Scripts look for `ops_perf_results_*.csv` under `<out>/reports/<ts>/`. If `tracy -o` lays it
   out differently, fix the `find`/`ls` in `profile_glm_block.sh`, `profile_sliding_layer.sh` and `run_all.sh`.
4. **Signpost window.** The GLM check reads `PROFILE_MEASURE_START/END`, Gemma's reads `...-start/-stop`, matching
   `signpost()` OP CODEs in the CSV. If the check says "signposts not found", print `df[df["OP TYPE"]=="signpost"]["OP CODE"].unique()`
   and align names. Column names (`DEVICE ID`, `HOST START TS`, `DEVICE KERNEL DURATION [ns]`) can differ by
   tt-metal version; check the CSV header.
5. **Dropped ops.** Expected counts are small (one layer, expected well under 1800 ops per device; unverified). If
   "ops with no device duration" is non-zero: raise `TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT` (GLM script, default 2000) /
   `--op-support-count` (Gemma, 20000), or add `ttnn.ReadDeviceProfiler(mesh_device)` calls inside the GLM block
   forward. A missing duration of 0 on some host-only ops may be a false positive; confirm before relaxing the check.
6. **GLM hook in `test_prefill_block.py`.** `GLM_BLOCK_PROFILE=1` runs two extra forwards before the normal one. If
   `block.forward` donates/deallocates its inputs, the second call will fail: re-create `tt_x` per forward.
7. **Tracy 32K source-location limit.** If a session fails with it, run one size per session
   (`SIZES=2048`, then `SIZES=4096`); the Gemma script already loops sessions per size.
8. **Device hangs between runs.** Reset the devices (`tt-smi -r`) between `glm` and `gemma`; CI notes nothing resets
   them between sections.

Keep fixes minimal, keep the scripts runnable standalone, and update this README if a step changes.
