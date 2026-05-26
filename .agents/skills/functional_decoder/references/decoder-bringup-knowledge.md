# Decoder Bringup Knowledge

Use this reference while bringing up a TTNN decoder layer. It folds in relevant local knowledge from personal skills and repo reading so the repo-local skill is portable.

## Repo Paths Worth Reading

- `models/common/modules/attention/attention_1d.py`: general config-first attention with prefill/decode split, paged KV cache, trace-friendly current-position tensors, Q/K norm hooks, and CCL integration.
- `models/common/modules/mlp/mlp_1d.py`, `models/common/modules/rmsnorm/rmsnorm_1d.py`, and `models/common/modules/rope/rope_1d.py`: common patterns for decode-vs-prefill memory layout, on-device RoPE embedding, and local/distributed norm paths.
- `models/common/modules/lazy_weight.py`: setup-time TTNN tensor conversion and deterministic cache-key pattern without runtime torch imports.
- `models/tt_transformers/tt/decoder.py`: standard transformer block composition and residual order.
- `models/tt_transformers/tt/generator.py` and `models/tt_transformers/tt/model.py`: production trace capture/replay, stable trace input tensors, decode preparation, and vLLM-facing contracts.
- `models/tt_transformers/tt/generator_vllm.py`: paged KV-cache allocation, per-layer page-table routing, and hybrid attention contracts.
- `models/demos/gpt_oss/tt/layer.py` and `models/demos/gpt_oss/tt/attention/`: model-specific standard attention with paged prefill/decode and MoE-adjacent tests.
- `models/demos/gemma4/tt/layer.py` and `models/demos/gemma4/tt/attention/`: strong examples for many norms, layer scales, sliding/full layer types, shared KV, and model-specific RoPE.
- `models/demos/deepseek_v3/tt/decoder_block/` and `models/demos/deepseek_v3/tt/mla/`: examples for dense-vs-MoE decoder kinds, MLA, row-sharded behavior, and specialized paged-cache updates.
- `tech_reports/LLMs/llms.md` and `tech_reports/LLMs/vLLM_integration.md`: TTNN LLM conventions for paged cache, prefill/decode shape differences, tracing, and vLLM integration.

## TTNN Correctness Defaults

- Start with `ttnn.bfloat16`, `ttnn.TILE_LAYOUT`, and `ttnn.DRAM_MEMORY_CONFIG` where that simplifies correctness; optimize memory layout after PCC is stable.
- Transpose 2D PyTorch linear weights before `ttnn.linear` unless the local TTNN helper already does it.
- Reshape norm weights so they broadcast over hidden width.
- Derive Q/K/V head reshapes from config, including GQA/MQA KV expansion, before optimizing.
- In `models/common` attention, Q/K weights may need HF-to-TTNN RoPE permutation handling such as `reverse_permute`; inspect the module contract and tests.
- For one-token decode scaffolding, sparse BF16 linear maps can encode RoPE or GQA reshapes, but this is not representative performance and should not be left unexplained in perf evidence.

## HF Reference And Synthetic Weights

- Prefer a layer-only HF reference over full causal LM loading.
- Inspect the decoder layer constructor, rotary/position embedding inputs, attention mask semantics, and cache API before writing the test.
- Use real weights to record tensor stats, not to run normal tests.
- For each tensor used by the TTNN layer, store at least name, shape, dtype, mean, and std.
- Generate synthetic weights deterministically from those stats in pytest.
- Synthetic input activations should approximate the distribution entering the decoder layer after embeddings/norms, not arbitrary huge random values.
- If full HF weight download is necessary to collect stats, record why partial shard loading was not practical.

## Paged KV Cache Checks

- Use paged prefill and paged decode as the final path. Non-paged cache is not worth spending time on for this bringup.
- Paged prefill usually uses `ttnn.experimental.paged_fill_cache`; decode uses `ttnn.experimental.paged_update_cache` and `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
- Decode should use tensor current positions when tracing; Python lists or scalar host state tend to break traceability.
- Use page-table permutations, nonzero slots, and random current positions to catch address and indexing bugs.
- For hybrid attention, inspect per-layer page-table routing and cache specs instead of assuming one page table shape covers every layer.
- Gemma-style shared-KV layers may skip K/V projection and cache update; tests must still prove the consumer layer reads the intended source cache.
- DeepSeek-style MLA cache updates can have aliasing hazards between prompt/speculative lanes; inspect masks and lane routing when validating cache correctness.

## Prefill And Decode Shapes

- Prefill commonly uses `(1, batch=1, seq_len, hidden)` style tensor shapes; decode commonly uses `(1, seq_len=1, batch, hidden)`.
- Decode tests must include the full supported context length unless measured KV-cache DRAM capacity forces a reduction. If reduced, record the attempted full-length or capacity-probe command, log, failure signature or byte calculation, and largest feasible tested context.
- Prefill tests must include the full supported sequence length unless measured L1/DRAM capacity forces a reduction. If reduced, record the attempted full-length or capacity-probe command, log, failure signature or byte calculation, and largest feasible tested sequence.
- Do not accept "tractability", profiling/watcher cost, runtime, or small synthetic proof scope as sequence-capacity evidence.
- Sliding-window models should include lengths around window boundaries, not only tiny smoke shapes.
- If a model has both full and sliding attention but identical decoder computation except mask/window configuration, use one parameterized implementation and tests for both modes.

## Trace And Profiling

- Open the device with a zero trace region, for example `trace_region_size=0`, TTNN can auto-detect the right size these days.
- Use the pattern: compile/warmup, synchronize, signpost start, execute warmed trace or warmed measured path, synchronize once, signpost end.
- Copy the newest Tracy ops CSV into the evidence directory and run `tt-perf-report` with start/end signposts for the measured window.
- `tt-perf-report --csv` may expose filtered kernel time as `Device Time` in microseconds rather than raw nanoseconds; convert accordingly and note the column used.
- Profiling should capture warmed prefill and warmed decode separately.
- Decode PCC should be measured from replay output, not only from an uncaptured forward pass.
- Tensor allocation after trace capture is a red flag. Trace capture should bake stable buffer addresses and replay should only update allowed input tensor contents before `execute_trace`.

## `tt-perf-report`

`tt-perf-report` is a Python CLI published separately from tt-metal. Install it in the active tt-metal Python environment, not in a random system Python:

```bash
python -m pip install tt-perf-report
tt-perf-report --help
```

The input is the post-processed Tracy ops CSV, usually named like `ops_perf_results_YYYY_MM_DD_HH_MM_SS.csv`. Generate that CSV with the normal Tracy profiler flow, for example:

```bash
python -m tracy -r -p -v -m pytest <test-path> -k "<selector>"
```

If Tracy collection fails, the LLM tech report notes a device-only fallback:

```bash
TT_METAL_DEVICE_PROFILER=1 pytest <test-path> -k "<selector>"
python tools/tracy/process_ops_logs.py --date
```

For functional-decoder artifacts, copy the relevant ops CSV into the artifact directory before running `tt-perf-report`:

```bash
export ARTIFACT_DIR="models/demos/<model>/doc/functional_decoder"
cp <ops_perf_results_*.csv> "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --csv "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.csv" \
  --no-advice \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.txt"
```

Use the same shape for prefill, with `PREFILL` signpost names and `prefill_*` filenames. If the installed `tt-perf-report` version does not support one of these flags, run `tt-perf-report --help`, use the closest supported flag set, and record the exact command in `commands.sh` and `manifest.json`.

When calculating latency from the report, prefer the filtered `tt-perf-report` CSV for the signposted window. Check the time column units. Some versions expose `Device Time` in microseconds; raw Tracy ops CSVs often expose `DEVICE KERNEL DURATION [ns]`. Record which column and unit you used in `functional_decoder.md`.

## Watcher Notes

- Watcher is fully supported on source builds; wheel-only environments may not have the same debug coverage.
- `TT_METAL_WATCHER=<n>` enables watcher and sets the polling interval in seconds unless the value has an `ms` suffix.
- `TT_METAL_LOGS_PATH=<dir>` redirects generated logs; watcher writes under `<dir>/generated/watcher/`.
- CI commonly uses `TT_METAL_WATCHER=2`, `TT_METAL_WATCHER_APPEND=1`, `TT_METAL_WATCHER_NOINLINE=1`, and `TT_METAL_WATCHER_DISABLE_ETH=1`.
- For decoder-layer artifact runs, prefer a unique `TT_METAL_LOGS_PATH` per command over appending unrelated runs into one log.
- Watcher and device profiler/debug-print should be separate runs because they use overlapping debug resources.
- Watcher detects asserts, invalid NOC coordinates or addresses, CB out-of-bounds NOC transactions, L1 address overflow, stack overflow, hardware faults on supported targets, waypoint state, and active kernel ids.
- A watcher-clean run still may contain normal attach, dump, kernel id, and detach lines. It must not contain fatal watcher exceptions or suspicious stack/L1/NOC/CB/sanitize messages.
- Good false-positive evidence includes the exact watcher log line, why the address/core/state is benign, the commit or issue proving it is known, and a separate correctness/profiling run showing the decoder path is otherwise sound.

## Fallback And Watcher Audit

- Runtime prefill/decode should not call torch, `ttnn.from_torch`, or `ttnn.to_torch` except at explicit boundaries.
- Setup-time conversion, input construction, final PCC comparison, and a named temporary prefill-to-decode boundary are allowed in the test harness.
- Host fallback hidden in helpers is still fallback; inspect wrappers as well as the layer file.
- `models/tt_transformers/tt/model.py` contains host-side last-token extraction patterns for batched prefill. Treat that as an existing production compromise, not as a pattern to copy silently.
- Run watcher-enabled tests. A clean watcher run is part of done; a suspected false positive needs evidence.
- Run TT hardware-facing commands one at a time: resets, `tt-smi`, import/open-device probes, Tracy/profile runs, and tests that touch the device.
- If wheel-installed `ttnn` hits SFPI/JIT mismatches, prefer a source-built tt-metal environment whose headers and compiler match.

## Code Quality Defaults

- Prefer explicit model contracts over permissive fallback logic.
- Do not silently infer or patch invalid config values in internal paths; fail directly.
- Do not add fake generality for one supported HF config shape.
- Keep debug-only fallback, logging, and compatibility shims out of final bringup code unless they are explicitly justified.
