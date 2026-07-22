---
name: functional-decoder
description: Bring up a functionally correct TTNN implementation of HuggingFace transformer decoder layers under the model-specific autoport directory. Use when producing a functional_decoder.py implementation, reading the HF decoder architecture, validating paged prefill/decode against the HF reference, and leaving compact correctness and performance evidence.
---

# Functional Decoder Bringup

## Mission Context

If this skill is used as part of `$model-bringup`, follow that skill's mission, workspace, and reporting contract. This stage turns the HF decoder layer (or multiple layers if there are different kinds in the HF model) into complete, working, tested TTNN code. It's called "functional" because it should be functionally complete and ready to ship - this is not a minimal prototype, it's the heart of everything else that follows. You should bring up the decoder on a single 1x1 device mesh. No need to use multiple devices at this stage.

## Your Part

Implement:

```text
models/autoports/<model>/tt/functional_decoder.py
```

Derive `<model>` from the HF model id: lowercase it and replace every non-alphanumeric character with an underscore, for example `org/Model-1.2B-Instruct` becomes `org_model_1_2b_instruct`. Downstream verification gates and tooling resolve the autoport directory from the HF model id, so do not add extra qualifiers such as hardware or experiment names.

with a model-appropriate `FunctionalDecoder` that subclasses `models.common.lightweightmodule.LightweightModule` and exposes a real weight-loading boundary:

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...
    def decode_forward(self, ...): ...
```

Implement forward signatures that fit the model. Keep them keyword-friendly. Record the signatures in the README.

## How To Approach It

Start by understanding the HF model. Load `AutoConfig`, inspect the installed `configuration_*.py` and `modeling_*.py`, and identify the decoder layer, attention, RoPE, cache API, MLP/MoE path, residual order, norm behavior, activation, bias flags, head shapes, and layer-kind differences. Trace through the model's execution line by line to make sure you understand it.

Before writing final tests, derive the model capability contract from the HF config and model code. For decoder-layer bringup this includes the target context/sequence length, cache and position semantics, layer kinds, mode switches such as sliding vs full attention, and the externally visible prefill/decode behavior later stages must preserve. Small shapes are useful for debugging, but stage completion requires evidence for the advertised contract or a hard physical device limit reason for the largest feasible reduction.

Support logical sequence lengths, not only friendly aligned lengths. If the implementation chunks, pads, or tiles prefill internally, public prefill inputs should still accept any valid `seq_len <= supported_context` unless the HF model itself has a real semantic restriction. Round or pad internally, mask padded tokens, and slice outputs back to the logical length. Do not leave checks such as `seq_len % 1024 == 0` on the public model path just because the current chunking helper wants that shape.

Implement correctness first. BF16, tile layout, and DRAM memory are fine while proving semantics. Move weight conversion, reshaping, dtype selection, `ttnn.as_tensor`, and cache construction into `from_state_dict` or setup helpers. Keep runtime prefill/decode free of hidden `torch`, `from_torch`, `to_torch`, or host fallback except explicit test boundaries.

For MoE models, validate the real router/gate and active experts end-to-end. Component tests for gate or experts are useful diagnostics, but the decoder result should include the gate-selected expert path a real model run would use. For non-Galaxy bringup, use the GPT-OSS active-expert pattern as the default: router/top-k scores as a sparsity tensor, `ttnn.sparse_matmul` for gate/up/down projections, score weighting, and a model-appropriate reduce over experts. Do not start from Galaxy-only fused MoE paths unless you have direct evidence that the target hardware and ops support them.

When PCC is low, debug it. Split the decoder into components, check HF parity, raise fidelity where useful, simplify the failing shape, and keep narrowing until the cause is understood. If a bug is tricky enough that ordinary narrowing stalls, use `$autofix`; it will run `$autodebug` if needed, then verify or refute each proposed bug before keeping any fix. If the cause is a tt-metal bug, make a reproducer or an on-branch workaround and record the evidence.

## Evidence To Leave

Use `PCC >= 0.995` as the default acceptance bar for prefill and decode unless the model gives a concrete reason to choose a different bar. Done means all of these are true and recorded:

- HF-vs-TTNN prefill and decode PCC and performance for each meaningful layer kind.
- Prefill and decode on-device performance measured from warmed runs (using traced execution for decode) - these should have perf report outputs to prove it.
- Paged KV-cache behavior, including page table and current-position handling.
- A short capability-contract evidence table: claim, evidence, and remaining risk for the advertised context/sequence length, cache behavior, layer kinds, and mode switches.
- Full advertised prefill/decode sequence length, or the largest feasible value when a hard physical device limit prevents the advertised value from fitting or running.
- Sequence-length coverage for valid non-aligned logical lengths: at least one small smoke length, one length exactly at a relevant tile/page/chunk boundary, one just across a boundary, and one long non-divisible length near the supported context. Use a reduced layer-level harness if the long test would otherwise be too slow, but do not treat aligned-only coverage as complete.
- `models/autoports/<model>/doc/context_contract.json` records the HF-advertised context, the current supported context, the largest context tested in prefill and decode, and any hard-physical-limit reduction with byte/probe evidence.
- Determinism for repeated identical inputs tested.
- Evidence there are no torch or host calls required within a single prefill or decode pass - everything runs and stays on device.
- Watcher-clean run with `TT_METAL_WATCHER=10`. If the command cannot run in the environment, record the exact environment failure and replacement evidence; do not skip asserts or weaken the test.


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
- `models/demos/gpt_oss/tt/experts/README.md`, `models/demos/gpt_oss/tt/experts/decode.py`, `models/demos/gpt_oss/tt/experts/prefill.py`, `models/demos/gpt_oss/tt/experts/weights.py`, `models/demos/gpt_oss/tt/experts/config.py`, and `models/demos/gpt_oss/tt/topk.py`: default MoE active-expert pattern using `ttnn.sparse_matmul`.
- `models/demos/gemma4/tt/layer.py` and `models/demos/gemma4/tt/attention/`: strong examples for many norms, layer scales, sliding/full layer types, shared KV, and model-specific RoPE.
- `models/demos/deepseek_v3/tt/decoder_block/` and `models/demos/deepseek_v3/tt/mla/`: examples for dense-vs-MoE decoder kinds, MLA, row-sharded behavior, and specialized paged-cache updates.
- `tech_reports/LLMs/llms.md` and `tech_reports/LLMs/vLLM_integration.md`: TTNN LLM conventions for paged cache, prefill/decode shape differences, tracing, and vLLM integration.

## TTNN Correctness Defaults

- Start with `ttnn.bfloat16`, `ttnn.TILE_LAYOUT`, and `ttnn.DRAM_MEMORY_CONFIG` where that simplifies correctness; optimize memory layout after PCC is stable.
- Transpose 2D PyTorch linear weights before `ttnn.linear` unless the local TTNN helper already does it.
- Reshape norm weights so they broadcast over hidden width.
- Derive Q/K/V head reshapes from config, including GQA/MQA KV expansion, before optimizing. When heads are tile-padded, slice back to the logical Q/KV head counts after RoPE so the GQA ratio is restored.
- In `models/common` attention, Q/K weights may need HF-to-TTNN RoPE permutation handling such as `reverse_permute`; inspect the module contract and tests. Verify the built RoPE tables against the HF model's own rotary-embedding class (max-abs-diff), not a default `inv_freq` — scaled variants like long-context RoPE are easy to miss.
- Keep distinct per-tensor memory placements distinct: do not homogenize one attention tensor's layout to match another's without a measurement, since a placement difference can be correctness-load-bearing, not just a perf choice.

## HF Reference And Synthetic Weights

- Prefer a layer-only HF reference over full causal LM loading.
- Inspect the decoder layer constructor, rotary/position embedding inputs, attention mask semantics, and cache API before writing the test.
- Use real weights to record tensor stats and to validate that `from_state_dict` can load a real checkpoint path when available. Treat the real HF state dict as the canonical key and shape contract. Normal CI tests should use synthetic weights generated from those stats so they do not require HF weight downloads.
- For each tensor used by the TTNN layer, store at least name, shape, dtype, mean, and std.
- Generate synthetic weights deterministically from those stats in pytest.
- Synthetic input activations should approximate the distribution entering the decoder layer after embeddings/norms, not arbitrary huge random values.
- Do not create smaller configs - replicate the real config of the reference.
- If full HF weight download is necessary to collect stats, record why partial shard loading was not practical.
- Always use real shapes even with synthetic weights.
- You must include a test with real weights and show it passes PCC >= 0.995 at least once at the end of your bringup.

## Paged KV Cache Checks

- Use paged prefill and paged decode as the final path. Non-paged cache is not worth spending time on for this bringup.
- Paged prefill usually uses `ttnn.experimental.paged_fill_cache`; decode uses `ttnn.experimental.paged_update_cache` and `ttnn.transformer.paged_scaled_dot_product_attention_decode`.
- Treat prefill fill-cache and decode update-cache as different dtype contracts. When the cache is lower precision such as `ttnn.bfloat8_b`, prefill should cast the K/V tensors passed to `paged_fill_cache` to the destination cache dtype before filling. Do not blindly apply the same cast to decode: `paged_update_cache` update tensors must remain a supported compute dtype such as BF16/FLOAT32, even when the destination cache is BFP8.
- Decode should use tensor current positions when tracing; Python lists or scalar host state tend to break traceability.
- Use page-table permutations, nonzero slots, and random current positions to catch address and indexing bugs.
- For hybrid attention, inspect per-layer page-table routing and cache specs instead of assuming one page table shape covers every layer.
- Gemma-style shared-KV layers may skip K/V projection and cache update; tests must still prove the consumer layer reads the intended source cache.
- DeepSeek-style MLA cache updates can have aliasing hazards between prompt/speculative lanes; inspect masks and lane routing when validating cache correctness.

## Prefill And Decode Shapes

- Prefill commonly uses `(1, batch, seq_len, hidden)` style tensor shapes; decode commonly uses `(1, seq_len=1, batch, hidden)`.
- Use batch 1 as the first smoke/performance target, but do not hard-code batch 1 into masks, page tables, current positions, cache indexing, residual shapes, or output handling.
- After batch-1 PCC passes, add at least one batch >1 prefill/decode correctness test. Test up to batch 32 when the target hardware, memory, and harness allow it. If batch 32 cannot run, record the largest tested batch and the hard limit.
- The target sequence/context length is the full value advertised by the HF config, usually `max_position_embeddings` or the model's equivalent field. Do not create a smaller model config. Do not hide a smaller implementation behind a smaller `max_model_len`.
- Do not reduce the advertised model capability to make bringup, tests, profiling, or serving easier. A reduction is acceptable only when a hard physical device limit prevents the advertised capability from fitting or running, such as device DRAM capacity for weights + KV/cache/state + required persistent buffers. If reduced, record the byte calculation or failed capacity probe, the largest feasible supported value, and the exact construction/serving setting that uses it.
- Decode tests must include the full supported context length unless measured KV-cache DRAM capacity forces a reduction. If reduced, record the attempted full-length or capacity-probe command, log, failure signature or byte calculation, and largest feasible tested context.
- Prefill tests must include the full supported sequence length unless measured L1/DRAM capacity forces a reduction. If reduced, record the attempted full-length or capacity-probe command, log, failure signature or byte calculation, and largest feasible tested sequence.
- Prefill and decode tests must also include valid logical lengths that are not divisible by the implementation's preferred chunk, tile, block, page, or trace length. A TTNN kernel may require padded physical shapes internally; the decoder API should own that padding and masking rather than rejecting the logical length.
- Do not accept "tractability", profiling/watcher cost, runtime, or small synthetic proof scope as sequence-capacity evidence.
- Sliding-window models should include lengths around window boundaries, not only tiny smoke shapes.
- If a model has both full and sliding attention but identical decoder computation except mask/window configuration, use one parameterized implementation and tests for both modes.

## Trace And Profiling

- The final decode pass must use traced TTNN execution. If trace capture, replay, stale inputs, event synchronization, or trace-safe input setup becomes tricky, use `$tt-enable-tracing` before accepting an eager-only decode path.
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
export ARTIFACT_DIR="models/autoports/<model>/doc/functional_decoder"
cp <ops_perf_results_*.csv> "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --csv "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.csv" \
  --no-advice \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.console.log"
tt-perf-report "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_ops.csv" \
  --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END \
  --no-summary \
  --no-advice \
  > "$ARTIFACT_DIR/tracy/<layer_kind_id>/decode_perf_report.txt"
```

Use the same shape for prefill, with `PREFILL` signpost names and `prefill_*` filenames. If the installed `tt-perf-report` version does not support one of these flags, run `tt-perf-report --help`, use the closest supported flag set, and record the exact command in the bringup log.

The `*_perf_report.txt` file is for the human-readable table. Do not redirect the stdout from a `--csv` run into that filename; `--csv` mode prints command/status boilerplate such as "Writing CSV output..." rather than the rendered report table. Keep that chatter in `*_perf_report.console.log` if it is useful for provenance.

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
- Setup-time conversion inside `from_state_dict`, input construction, final PCC comparison, and a named temporary prefill-to-decode boundary are allowed in the test harness.
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
