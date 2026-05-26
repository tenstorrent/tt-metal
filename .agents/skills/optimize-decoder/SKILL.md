---
name: optimize-decoder
description: Optimize a functional TTNN transformer decoder layer implementation under models/autoports/{model} after functional-decoder bringup. Use when producing models/autoports/{model}/tt/optimized_decoder.py with an OptimizedDecoder LightweightModule class matching the functional decoder interface, improving performance with L1-sharded activations, explicit memory/program configs, DRAM-sharded decode matmuls, lower precision weights/activations/KV cache, MoE active-expert execution, tt-perf-report advice, PCC preservation, stress, watcher-clean proof, and golden optimized-decoder artifacts.
---

# Optimize Decoder

## Goal

Take a decoder layer that already passed the `functional_decoder` step and optimize it to make the best practical use of TTNN and the target hardware. The target is hardware-limited execution: DRAM-bound ops should approach DRAM bandwidth limits, compute-bound ops should use the right math fidelity and core grids, and unnecessary data movement should disappear.

This is not done until the original functional bringup test contract still passes against the optimized implementation, optimized PCC and performance checks pass, the optimized stress test runs and passes, the production trace path runs where applicable, watcher-clean correctness evidence exists, and final `tt-perf-report` evidence shows what remains limiting.

A diagnostic failure packet is useful when an external blocker must be documented, but it is not goal completion. Do not mark the optimization goal complete with a `fail` manifest; a normal optimized-decoder run completes only with passing final artifacts, or with a true externally blocked state backed by evidence.

## Required Reference Reads

Before implementation, read these bundled references:

- `references/optimization-knowledge.md` for TTNN optimization patterns, code paths, precision defaults, and `tt-perf-report` usage.
- `references/artifact-formats.md` for exact artifact names, JSON schemas, status values, and markdown headings.

Also read the target model's `doc/functional_decoder/` artifacts before changing code. Those artifacts define the functional baseline, representative layer kinds, tested shapes, synthetic weights, PCC, trace behavior, and sequence limits. Parse `doc/functional_decoder/manifest.json` and require `"status": "pass"` before optimizing; if it is missing, failed, blocked, or skipped, repair or rerun functional bringup first unless the user explicitly requested diagnostic optimization analysis.

## Definition Of Done

- Start from `doc/functional_decoder/manifest.json` with `"status": "pass"` and preserve the same public test surface.
- Produce `models/autoports/<model>/tt/optimized_decoder.py` exporting `OptimizedDecoder(LightweightModule)` with the same `from_state_dict`, `prefill_forward`, and `decode_forward` interface contract as `FunctionalDecoder`.
- Rerun the functional decoder prefill, decode, PCC, KV-cache, determinism, stress, trace, and watcher checks against the optimized implementation. Optimized tests are additions to that contract, not replacements, unless the original test is cleanly parameterized to exercise the optimized backend with identical assertions.
- Run and pass the optimized stress test for every representative layer kind and exercised mode; final optimized artifacts must not mark stress as skipped.
- Pass optimized prefill and decode PCC with `PCC >= 0.995` for every representative layer kind and mode.
- Record PCC deltas against the functional decoder baseline; any material drop must be explained.
- Keep paged KV cache and traced decode semantics intact.
- Preserve the single-user prefill/decode target. For MoE layers, optimize the gate-selected active-expert path: keep end-to-end gate-plus-expert correctness, load only selected experts from DRAM where possible, and use TTNN sparse matmul or equivalent active-expert execution instead of densifying all experts except for debug baselines.
- Run a watcher-enabled optimized correctness pass and require it to be watcher-clean, or include evidence for a specific false positive.
- Report warmed per-layer prefill and decode latency with Tracy signposts and `tt-perf-report`.
- Use L1-sharded activations in decode wherever the op chain can keep them there.
- Define explicit memory configs and program configs for important ops instead of relying on defaults.
- Choose large core grids that cleanly fit activation/output shapes and avoid unnecessary resharding.
- Use DRAM-sharded matmuls for DRAM-bound matmuls, especially decode matmuls that read large weights.
- Start with BF16 activations and BFP8 weights, while keeping norm layers BF16.
- Try BFP8 KV cache and keep it if PCC and performance support it.
- Try BFP4 for MLP weights, especially FF1/FF3; test FF2/down-projection carefully because it is often more sensitive and may need to remain BFP8.
- Use compute-kernel configs appropriate to datatype and measured bottleneck.
- Install and run `tt-perf-report`; follow every actionable item until no useful advice remains or evidence shows a specific item does not improve performance.
- Avoid adding runtime torch, `ttnn.from_torch`, `ttnn.to_torch`, or host-device fallback inside prefill/decode.

## Autoport Output Contract

Create the optimized implementation under:

```text
models/autoports/<model>/tt/optimized_decoder.py
```

This file must export:

```python
class OptimizedDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...

    def decode_forward(self, ...): ...
```

`OptimizedDecoder` must subclass `models.common.lightweightmodule.LightweightModule`. `from_state_dict` is the optimized decoder's general weight-loading boundary: it must load real checkpoint tensors when provided and must also accept synthetic state dicts generated from the functional artifact stats for CI. Real HF state-dict keys and shapes remain canonical unless the optimized implementation records an explicit transform. Weight conversion, dtype selection, layout selection, cache construction, and `ttnn.as_tensor` calls belong in `from_state_dict` or setup helpers, not in `prefill_forward` or `decode_forward`.

The optimized class may reuse, wrap, or subclass the functional implementation only if the exported class still satisfies this contract and tests exercise the optimized path. Record the actual signatures, state-dict keys, input/output layout contract, and real/synthetic weight support in `implementation_contract.json`.

## Optimization Workflow

1. Read the functional artifacts under `models/autoports/<model>/doc/functional_decoder/` and confirm `manifest.json` has `"status": "pass"`.
2. Reproduce the functional decoder's final PCC and perf commands before optimizing, unless the prompt provides fresh baseline evidence.
3. Profile warmed prefill and warmed decode separately with signposts around only the measured windows.
4. Run `tt-perf-report` on each profile and classify ops as DRAM-bound, compute-bound, communication-bound, or data-movement-only.
5. Remove avoidable data movement first. Interleaved-to-sharded, sharded-to-interleaved, reshard, untilize, tilize, and host transfer ops should each have a reason.
6. Establish decode L1-sharded residual/activation flow and keep adjacent ops on compatible shard specs where possible.
7. Add explicit program configs for attention, MLP, norms, and any large matmul or SDPA op that shows up in the perf report.
8. Use DRAM-sharded matmul for DRAM-bound decode matmuls and ensure activation/output shard specs divide cleanly without padding surprises.
9. Tune precision and fidelity one group at a time: WQKV/WO, FF1/FF3, FF2, activation, KV cache, SDPA, and norms.
10. After each accepted optimization, rerun the focused PCC and perf check for the affected layer kind and mode.
11. Rerun the full functional decoder correctness suite in optimized mode before declaring the optimization done.
12. Run the optimized stress test and require a passing result.
13. At the end, rerun the complete optimized golden proof suite and write only that final evidence to `doc/optimized_decoder/`.

## Precision Defaults

Use this starting point unless the functional model or hardware makes it invalid:

- activations: BF16;
- norms and norm weights: BF16;
- attention weights WQKV and WO: BFP8;
- MLP weights: BFP8 initially;
- KV cache: try BFP8;
- FF1/FF3: try BFP4 after BFP8 passes;
- FF2/down-projection: try BFP4, but expect it to be more PCC-sensitive and fall back to BFP8 or BF16 when measured evidence requires it.

Use math fidelity to match datatype and sensitivity:

- BF16 weights or sensitive ops: HiFi4 or HiFi4 with FP32 accumulation when needed;
- BFP8 weights: HiFi2 is usually the starting point; try LoFi only with PCC evidence;
- BFP4 weights: LoFi is the expected starting point;
- SDPA: follow model and `tt-perf-report` evidence, but do not trade away PCC blindly.
- Attention often requires higher fidelity / less approximate math than MLP/MoE

## `tt-perf-report` Contract

Install in the active tt-metal Python environment:

```bash
python -m pip install tt-perf-report
tt-perf-report --help
```

Generate the ops CSV with Tracy or the device profiler fallback, then run `tt-perf-report` for prefill and decode separately using the final artifact filenames. Keep advice enabled for the required advice run; do not use `--no-advice` for the run that feeds `tt_perf_advice.json`. Record every command in `commands.sh`.

Treat `tt-perf-report` as required but not an oracle. Try its optimization advice. If advice does not improve performance, hurts PCC, or does not apply to the architecture, record the advice, trial result, and rejection reason in `tt_perf_advice.json`, then continue with the rest of the model.

## Artifact Layout

Final golden optimized-decoder evidence lives directly under:

```text
models/autoports/<model>/doc/optimized_decoder/
```

Do not save every debug attempt under `doc/optimized_decoder/`. Intermediate experiments belong in scratch space such as `generated/optimized_decoder/debug/`, `/tmp`, or an uncommitted local artifact directory. Once the optimized implementation is ready, rerun the final proof commands and copy only that golden evidence into `doc/optimized_decoder/`.

Required files:

```text
manifest.json
optimized_decoder.md
commands.sh
implementation_contract.json
baseline_summary.json
functional_regression_results.json
optimization_plan.json
precision_results.json
program_config_results.json
data_movement_audit.md
fallback_audit.md
tt_perf_advice.json
performance_results.json
results/pcc_results.json
results/kv_cache_results.json
results/determinism_results.json
results/stress_results.json
watcher/watcher_summary.json
pytest/<layer_kind_id>_prefill.log
pytest/<layer_kind_id>_decode.log
pytest/<layer_kind_id>_stress.log
pytest/<layer_kind_id>_watcher.log
watcher/<layer_kind_id>/generated/watcher/watcher.log
watcher/<layer_kind_id>/generated/watcher/kernel_names.txt
watcher/<layer_kind_id>/generated/watcher/kernel_elf_paths.txt
tracy/<layer_kind_id>/prefill_ops.csv
tracy/<layer_kind_id>/prefill_perf_report.csv
tracy/<layer_kind_id>/prefill_perf_report.txt
tracy/<layer_kind_id>/decode_ops.csv
tracy/<layer_kind_id>/decode_perf_report.csv
tracy/<layer_kind_id>/decode_perf_report.txt
```

If there are multiple layer kinds, repeat layer-kind-specific pytest and Tracy files for each `<layer_kind_id>`.

Do not store binary tensors, full model weights, TTNN program-cache directories, or giant profiler captures in the doc artifact tree.

Use `references/artifact-formats.md` for exact JSON keys, status enums, path conventions, and markdown headings.

## Knowledge Base

The required references are part of this skill's contract. Re-read the relevant sections before choosing program configs, changing precision, accepting or rejecting perf-report advice, or generating final artifacts.
