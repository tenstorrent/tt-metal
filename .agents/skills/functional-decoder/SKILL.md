---
name: functional-decoder
description: Bring up functionally complete TTNN implementations of HuggingFace transformer decoder layers under models/autoports/{model} in tt-metal. Use when implementing models/autoports/{model}/tt/functional_decoder.py with a FunctionalDecoder LightweightModule class, reading HF decoder architecture, targeting single-user paged prefill/decode, validating MoE gate-plus-active-expert behavior, supporting real or synthetic weight loading through from_state_dict, writing PCC pytests, tracing warmed decode, profiling with Tracy, and auditing watcher-clean execution with no runtime torch or host fallback except explicit boundaries.
---

# Functional Decoder Bringup

## Goal

Build working TTNN code and pytests for each unique HuggingFace decoder-layer kind in `models/autoports/<model>/`. "Functional" means functionally complete, not a smoke test: the decoder should have the semantics, cache behavior, sequence coverage, and evidence needed to roll it into a full model and use it as-is. This is not a planning-only skill: success means checked-in implementation, tests, and evidence.

The final tests must run without downloading model weights. Use real model weights to collect per-tensor statistics, then generate meaningful synthetic weights from those statistics for CI and local tests. The implementation itself must still load real weights when a real state dict is supplied, because the full model port will depend on the same decoder class.

Target single-user prefill and decode by default. Multi-user throughput is a later optimization unless the user explicitly asks for it. For MoE models, the primary path should run the real gate/router and then compute only the experts selected by that gate, aiming to use `ttnn.sparse_matmul` so DRAM traffic and compute follow active experts rather than dense all-expert matmuls.

Do not treat a near-threshold PCC miss as goal completion. A `fail` artifact is evidence for an attempted run, not success for the bringup goal, unless the user explicitly asked only for a diagnostic packet. For ordinary bringup, keep debugging until the decoder passes, the run is truly blocked by model access/hardware/tooling, or a repeated investigated failure has clear evidence and next steps.

Do not treat reduced sequence length as goal completion unless the reduction is forced by measured device capacity or explicitly approved by the user before the run. Convenience, "tractability", profiler runtime, watcher runtime, CI runtime, or a desire to finish a small proof are not valid reasons to skip the full sequence-length contract.

## Required Reference Reads

Before implementation, read these bundled references:

- `references/decoder-bringup-knowledge.md` for tt-metal code paths, watcher, trace, profiling, and fallback gotchas.
- `references/artifact-formats.md` for exact artifact names, JSON schemas, status values, and markdown headings.

Do not skip these reads. The `SKILL.md` body defines the workflow; the references define required details that keep runs consistent across models. tt-metal is a new framework, broad and deep background reading will set you up for success. Feel free to read around the codebase and truly familiarize yourself with how things work even down to the op and llk level as you see fit.

## Definition Of Done for the Decoder Bringup Goal

- Implement one parameterized TTNN decoder layer path per meaningful layer kind.
- Test the first representative layer index of each kind by default.
- Target single-user prefill/decode behavior by default; document any shape padding needed to satisfy tiled TTNN ops without treating it as multi-user coverage.
- For MoE layers, the full decoder PCC must include gate/router, expert selection, active experts, expert weighting/reduction, and the surrounding decoder residual/norm path end-to-end.
- Optimized TTNN ops used where appropriate, e.g. use sdpa_decode instead of hand-implementing attention in ttnn primitives (although if you need to implement something that ttnn does not have an op/transformers op/experimental op for or the existing op does not cover this model's case it is of then ok to write your own implementation out of primitives)
- Pass prefill and decode PCC against the HF reference decoder layer with `PCC >= 0.995`.
- Use paged KV cache only; non-paged attention is at best optional debug scaffolding, not a final path.
- Prove paged KV-cache behavior for prefill and decode, including page-table handling.
- Satisfy the sequence-length contract for every representative layer kind. Decode must test the full context length advertised by the reference model from a warmed trace replay path unless a measured device-capacity blocker prevents it. Prefill must test the full supported length unless L1/DRAM capacity prevents it.
- If full sequence length is not tested, prove the blocker with a capacity probe command, log, failure signature or byte calculation, and the largest feasible tested length. A final `pass` manifest may not cite reduced length without this evidence or explicit user-approved reduced scope.
- Decode PCC must be measured from a warmed trace replay path.
- Report warmed per-layer prefill and decode latency with Tracy signposts and `tt-perf-report`.
- Run a watcher-enabled correctness pass and require it to be watcher-clean, or include evidence for a specific false positive.
- Include a repeated-run determinism check with identical inputs and identical outputs.
- Include an optional stress mode that loops for about five minutes to catch hangs.
- No runtime torch, `ttnn.from_torch`, `ttnn.to_torch`, or host-device fallback inside prefill/decode logic except explicit input/output boundaries and any named temporary prefill-to-decode boundary.

## Autoport Output Contract

Create the implementation under:

```text
models/autoports/<model>/tt/functional_decoder.py
```

This file must export:

```python
class FunctionalDecoder(LightweightModule):
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs): ...

    def prefill_forward(self, ...): ...

    def decode_forward(self, ...): ...
```

`FunctionalDecoder` must subclass `models.common.lightweightmodule.LightweightModule`. `from_state_dict` is the general weight-loading boundary used by tests and by the later full-model port. It must load real checkpoint tensors when a real state dict is provided. It must also accept synthetic state dicts generated from `weight_stats.json` for CI. Setup-time torch work, tensor reshaping, dtype conversion, `ttnn.as_tensor`, and cache construction belong in `from_state_dict` or helper code it calls, not in `prefill_forward` or `decode_forward`.

Keep exact prefill/decode signatures model-appropriate and keyword-friendly. The tests and `implementation_contract.json` must document the actual method signatures, tensor layout contract, required runtime inputs, state-dict key contract, and whether one class instance handles all layer kinds or one representative layer kind.

## HF Reading Workflow

Fold HF model reading into the bringup. Before coding, write down the model facts needed to implement and test the layer:

1. Load `AutoConfig` first; avoid weights until needed.
2. Locate the installed HF implementation in `transformers/models/<family>/configuration_*.py` and `modeling_*.py`; use `inspect.getfile()` if the path is unclear.
3. Record hidden size, intermediate size, number of attention heads, KV heads, head dim, RoPE theta/scaling, norm type and eps, activation, bias flags, attention implementation, cache API, and `layer_types` or equivalent.
4. Read the decoder-layer forward path, attention path, RoPE application, cache update, MLP/MoE path, residual ordering, and any pre/post norm or layer-scale behavior.
5. For MoE models, record router/gate projection, top-k selection, routing-weight normalization, expert tensor layout, shared-expert behavior if any, and how HF applies selected experts back into the decoder output.
6. Map HF state-dict key names and shapes for the selected representative layer indices.
7. Compare with the closest existing tt-metal model or `models/common` primitive before inventing a new implementation.
8. For novel architectures, do a structural diff against a close HF reference model by comparing configs, classes, method signatures, and roles such as attention, MLP, MoE, RMSNorm, RoPE, decoder layer, and causal LM.

## Layer-Kind Selection

Use judgement, but treat distinct decoder computation as a distinct layer kind. Dense vs MoE layers qualify. Shared-KV vs normal layers qualify. MLA vs standard attention qualifies. Pure mask differences, such as sliding-window vs full attention, usually should not create two decoder implementations; parameterize the implementation and make tests cover both modes.

Default to the first layer index of each kind. If the target model has repeated layer types, do not bring up every repeated index unless the prompt asks for it or the weights/config reveal a real behavior difference.

## Weight Strategy

Prefer partial weight access:

- Fetch config and safetensors index first.
- Load only shard files that contain tensors for the selected representative layer indices when practical.
- Full model download is acceptable when HF makes partial loading hard, but record the reason.

For every real tensor used by the TTNN layer, record at least name, shape, dtype, mean, and standard deviation in a small checked-in stats artifact. Final pytests must synthesize weights from those stats and must not require HF weight downloads. Real HF state-dict keys and shapes are the canonical contract for `FunctionalDecoder.from_state_dict`; synthetic weights should use the same keys and shapes unless a documented key transform is unavoidable. Inputs should be drawn from a distribution that matches what the model's normalization path is expected to produce.

For MoE models, collect enough router and expert tensor statistics to synthesize the gate and the experts selected by that gate. Do not validate only a hand-picked expert path unless it is clearly labeled as debug-only and followed by the full gated decoder PCC.

## Implementation Guidance

- Prefer `models/common` modules when their contracts fit the target architecture.
- Write model-specific modules when the target needs MLA, unusual RoPE, shared KV, MoE routing, 2D/Galaxy mapping, custom masks, or shape/layout decisions that `models/common` does not represent cleanly.
- Keep setup-time conversion and weight loading separate from runtime forward paths. Real or synthetic weights should enter through `from_state_dict`; lazy or cached weight conversion is fine there. Hidden torch work in `prefill_forward` or `decode_forward` is not.
- Start correctness-first with BF16, tile layout, and DRAM memory where useful, then optimize layout/sharding after PCC is stable.
- Treat host movement between prefill and decode as an explicit boundary. It may be used in a layer bringup test, but the decoder implementation should not depend on host interaction once layers are stacked into a full model.
- For MoE, use the gate output to drive sparse expert execution. Prefer the GPT-OSS pattern in `models/demos/gpt_oss/tt/topk.py` plus `models/demos/gpt_oss/tt/experts/decode.py` and `models/demos/gpt_oss/tt/experts/prefill.py`: top-k router output feeds sparse expert matmuls instead of running every expert densely.
- Keep separate gate PCC and expert PCC tests useful for debugging, but do not let them substitute for a full decoder PCC where the gate selects the active experts end-to-end as in a real model run.

## Low-PCC Debug Ladder

If full-decoder PCC misses the threshold, do not stop after recording the failure. Run a short, evidence-producing debug ladder before declaring failure:

1. Split the decoder into component comparisons: input norm, QKV projection, RoPE, attention/SDPA output, output projection, shared MLP, router/gate, selected experts, expert reduction, post norms, residuals, and layer scales.
2. Re-run the worst component with higher fidelity: `ttnn.bfloat16` or `ttnn.float32` where supported, `MathFidelity.HiFi4`, `fp32_dest_acc_en=True`, `math_approx_mode=False`, exact GELU/activation modes, non-approx softmax/exp settings, and higher-precision accumulation for matmul/linear/SDPA program configs.
3. Check reference parity before blaming TTNN: HF dtype, activation variant, RoPE theta/scaling/partial rotation, attention scale, mask/window semantics, GQA/KV replication, K=V tying, RMSNorm epsilon/scale, router top-k ordering, and residual/layer-scale order.
4. Minimize the failing case: shorter sequence, single token, one head, one active expert, dense equivalent for sparse expert math, or one layer-kind representative. Preserve the minimized command and PCC in debug notes.
5. Prefer fixes that move the full decoder PCC, not only a component PCC. If a high-fidelity diagnostic passes but the production dtype path fails, record the delta and either keep the high-fidelity setting for functional correctness or justify the remaining optimized path separately.

Only write a final `fail` artifact after this ladder has been attempted or is impossible, and include the best PCC, tested knobs, component-localization evidence, and concrete next debugging steps.

## Required Tests

Write pytest coverage near the model bringup, usually under `models/autoports/<model>/tests/` or the local convention for that model.

Required test behavior:

- Instantiate the HF decoder layer directly when possible; use full `AutoModelForCausalLM` only when layer isolation is impractical.
- Normal test execution must not require real weights; instantiate `FunctionalDecoder` through `from_state_dict` with checked-in synthetic state dicts generated from recorded real-tensor stats.
- Compare TTNN vs HF for prefill and decode with `PCC >= 0.995`.
- For MoE layers, the required prefill/decode PCC compares the full decoder output after the TTNN gate selects experts and the TTNN expert path computes/weights/reduces those selected experts. Optional gate-only or expert-only PCC checks may be recorded as diagnostics.
- Exercise paged prefill, paged decode update, and paged SDPA decode.
- Use randomized or permuted page tables, nonzero user slots where applicable, and nontrivial current positions.
- Test decode at the full sequence length supported by the reference model unless KV-cache DRAM capacity prevents it. If it is prevented, run a capacity probe that attempts the full length or calculates an impossible allocation from actual model shapes and available device DRAM, then test the largest feasible length on the reserved hardware.
- Test prefill at the full supported length unless L1/DRAM capacity prevents it. If it is prevented, provide the same capacity evidence and largest feasible tested length.
- Explicitly report any max sequence reduction with command IDs, logs, failure signatures, and hardware/memory evidence. Do not use "tractability", profiling cost, watcher cost, runtime, or reduced synthetic sequence convenience as the reason for a passing reduced-length artifact.
- Run the same deterministic input multiple times and assert identical TTNN outputs.
- Gate the five-minute stress loop behind an opt-in marker or environment variable so regular CI stays focused.

## Trace, Profiling, And Watcher

The decode correctness test must use the production-shaped trace pattern:

1. Allocate stable device tensors and open the device with enough trace region.
2. Compile/warm up outside the measured window.
3. Capture decode with `ttnn.begin_trace_capture` and `ttnn.end_trace_capture`.
4. Update only allowed input tensors before replay, such as tokens, current positions, page tables, RoPE indices, or equivalent stable trace inputs.
5. Execute with `ttnn.execute_trace`.
6. Compare the trace replay output to HF for PCC.

Profile warmed prefill and warmed decode separately. Use Tracy signposts around only the measured windows, then run `tt-perf-report` for each window. If fixed-shape prefill trace is practical for the target, trace it too; otherwise still profile warmed prefill and record why prefill trace was not used.

Run watcher and profiling as separate passes. The device profiler and watcher both consume device-side debug resources and should not be enabled in the same command.

For the watcher pass, use source-build watcher support and capture the generated watcher logs under the run artifact directory. A typical local command shape is:

```bash
export TT_METAL_LOGS_PATH="$RUN_DIR/watcher/<layer_kind_id>"
export TT_METAL_WATCHER=2
export TT_METAL_WATCHER_NOINLINE=1
export TT_METAL_WATCHER_TEST_MODE=1
export TT_METAL_WATCHER_DISABLE_ETH=1
pytest <test-path> -k "<layer kind and mode selector>" -vv
```

`TT_METAL_WATCHER` is the polling interval in seconds unless the value uses an `ms` suffix. Use a longer interval for very long runs if watcher overhead distorts the test, but do not turn off assert, NOC sanitize, CB sanitize, stack usage, or waypoint features to make a run pass. `TT_METAL_WATCHER_DISABLE_ETH=1` is acceptable for single-card decoder-layer bringup unless the test intentionally uses Ethernet cores.

Watcher-clean means:

- the pytest command exits successfully without timeout or hang;
- watcher attaches and detaches cleanly;
- `generated/watcher/watcher.log` exists for the run;
- the watcher log and stderr contain no tripped assert, sanitize error, NOC error, circular-buffer overflow, L1 overflow, stack overflow, hardware fault, link error, pause left waiting, watcher exception, or watcher-server termination due to error;
- no test is skipped only because watcher is enabled unless the skip is explicitly recorded as a current limitation;
- if a watcher message is claimed to be a false positive, the evidence names the exact message, affected file/op, why it is false, and what separate run proves functional correctness.

Treat watcher findings, tensor allocation after trace capture, host reads in runtime, or `from_torch`/`to_torch` inside prefill/decode as red flags that must be fixed or explicitly justified.

## Golden Artifact Layout

Keep the functional decoder proof simple: the final golden artifacts live directly in the functional-decoder doc directory. Later phases should use sibling directories such as `doc/optimized_decoder/` or `doc/multichip_decoder/`.

```text
models/autoports/<model>/
  tt/
    functional_decoder.py
    optimized_decoder.py
    multichip_decoder.py
  tests/
  doc/
    functional_decoder/
    optimized_decoder/
    multichip_decoder/
```

This skill writes only the `functional_decoder` step:

```text
models/autoports/<model>/doc/functional_decoder/
```

Do not save every debug attempt under `models/autoports/<model>/doc/functional_decoder/`. Intermediate failing or exploratory runs belong in scratch space such as `generated/functional_decoder/debug/`, `/tmp`, or an uncommitted local artifact directory. Once the implementation is ready, rerun the final proof commands and copy only that golden set of logs, reports, and JSON summaries into `doc/functional_decoder/`.

`commands.sh` and `manifest.json` should record the final proof commands, not the entire trial-and-error history. Capacity probes that justify reduced sequence length are final proof commands, not discarded debug attempts; record their command IDs, logs, and failure signatures in `sequence_limits.json`. If a failed intermediate run exposed an important limitation, summarize it in `functional_decoder.md` or `fallback_audit.md` only when it explains a remaining limitation or a final design decision.

Required files:

```text
manifest.json
functional_decoder.md
commands.sh
model_facts.json
layer_kinds.json
implementation_contract.json
weight_stats.json
sequence_limits.json
fallback_audit.md
results/pcc_results.json
results/kv_cache_results.json
results/determinism_results.json
results/stress_results.json
watcher/watcher_summary.json
pytest/<layer_kind_id>_prefill.log
pytest/<layer_kind_id>_decode.log
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

If there are multiple layer kinds, repeat the layer-kind-specific pytest, watcher, and Tracy files for each `<layer_kind_id>`. Use lowercase hyphen-case ids such as `dense`, `moe`, `full-attention`, `sliding-attention`, `shared-kv`, or `mla`.

Do not store binary tensors, full model weights, TTNN program-cache directories, or giant profiler captures in the doc artifact tree.

All JSON files must be strict JSON, UTF-8, and schema-versioned with `"schema_version": 1`. Raw command logs are plain UTF-8 `.log` files. Tracy and `tt-perf-report` CSV files are copied unchanged from the tools.

Use `references/artifact-formats.md` for exact JSON keys, status enums, path conventions, and markdown headings. Do not invent alternate artifact names for a model-specific run.

Required JSON contents:

- `manifest.json`: step id, status, model id, model slug, git commit, dirty flag, host, hardware, arch, TTNN/tt-metal version evidence, start/end UTC, artifact schema version, and paths to all required artifacts.
- `model_facts.json`: HF config fields, decoder-layer class names, attention/MLP/MoE/RoPE/cache facts, and source files inspected.
- `layer_kinds.json`: one object per representative layer kind with layer kind id, representative layer index, reason it is unique, features, implementation files, pytest ids, expected max prefill length, and expected max decode context length.
- `implementation_contract.json`: exported class name `FunctionalDecoder`, implementation path `models/autoports/<model>/tt/functional_decoder.py`, `LightweightModule` subclass evidence, `from_state_dict` weight-loading contract, prefill/decode method signatures, real HF state-dict key mapping, runtime input contract, input/output layout contract, real-weight load evidence when available, and synthetic-weight CI evidence.
- `weight_stats.json`: one object per real tensor used to seed synthetic weights with layer kind id, layer index, tensor name, shape, dtype, mean, std, and deterministic synthetic seed.
- `sequence_limits.json`: requested reference max lengths, tested max lengths, pass/fail, command IDs, logs, and evidence for any reduction caused by KV-cache DRAM, L1, or other measured device limits. If any entry has `reduced: true`, it must prove the blocker with capacity evidence or name an explicit user-approved reduced scope; otherwise the manifest status must not be `pass`.
- `results/pcc_results.json`: one full-decoder object per layer kind and mode with PCC, threshold, pass/fail, output shape, dtype, input seed, whether trace replay was used, and the command id from `commands.sh`. For MoE, this full-decoder object must include gate and selected experts end-to-end; optional gate-only or expert-only component PCCs are diagnostics.
- `results/kv_cache_results.json`: cache shape, page-table shape, page-table seed, tested positions, tested user slots, comparison method, and pass/fail per layer kind.
- `results/determinism_results.json`: repeated-run count, bytewise or numeric equality method, pass/fail, and any nondeterministic tensors.
- `results/stress_results.json`: whether stress was run, duration seconds, iteration count, pass/fail, or reason it was skipped.
- `watcher/watcher_summary.json`: watcher env, command id, log paths, clean boolean, detected messages, false-positive justifications, and final status.

`functional_decoder.md` should be short and human-readable. It must summarize:

- model facts and selected representative layer indices;
- implementation contract and exported `FunctionalDecoder` class;
- real-tensor mean/std stats used for synthetic weights;
- pytest command lines and logs;
- PCC results for prefill and decode;
- watcher log or clean watcher evidence;
- determinism result and optional stress result;
- Tracy ops CSVs and `tt-perf-report` text/CSV for prefill and decode;
- fallback audit showing no runtime torch/host fallback in the relevant prefill/decode paths;
- any sequence-length reductions with hardware or memory evidence, including the attempted full-length or capacity-probe command and why the largest tested length is the maximum feasible length on that hardware.

## Knowledge Base

The required references are part of this skill's contract. Re-read the relevant sections before generating artifacts or running watcher/profiling commands, and treat them as checklists rather than substitutes for reading the target model code.
