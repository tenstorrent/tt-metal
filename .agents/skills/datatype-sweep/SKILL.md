---
name: datatype-sweep
description: Sweep and select TTNN model weight, activation, KV-cache, CCL datatype, and compute-fidelity policies for the fastest configuration that satisfies minimum full-model top-1 and top-5 accuracy. Use after optimized-full-model and before vLLM, or whenever a runnable TTNN full model needs evidence-backed precision selection.
---

# Datatype Sweep

## Mission Context

This skill starts from a working optimized TTNN full model and chooses the fastest practical precision configuration that still meets a stated full-model accuracy bar. It is normally used after optimized full model and before vLLM so the serving adapter inherits a settled weight, activation, CCL, KV-cache dtype, and compute-fidelity policy.

The selected precision artifact must be complete enough for later stages to consume mechanically. Include weight dtype groups, layer exceptions, compute fidelities, activation/residual dtype, CCL communication dtype, KV-cache dtype, logits/sampling dtype assumptions, and any loader/runtime flags needed to construct that exact policy. A selected config that only says "BFP8 weights" is incomplete.

Recorded policy is not enough. The runtime construction path must actually consume every selected dtype and compute-fidelity field. Prove this with a model summary, config propagation check, or profiler/perf-report rows from the measured candidate. If a field appears in `selected_precision_config.json` but the code path ignores it or hard-codes a different value, the sweep is incomplete.

The expensive source of truth is full-model top-1/top-5 accuracy. Decoder-layer PCC and component timing are useful only for ordering candidates and debugging surprises.

Datatype selection must preserve the capability and context contract. KV-cache dtype can change the largest feasible context, so recompute `models/autoports/<model>/doc/context_contract.json` for the selected config. If a candidate only passes by lowering context or another advertised capability, it fails unless a hard physical device limit prevents the advertised capability from fitting or running and evidence proves that the smaller value is the largest feasible one for that candidate.

Datatype selection must also preserve valid non-aligned prompt lengths. If a candidate changes KV-cache dtype, cache layout, trace buffers, or prefill chunking, rerun the non-aligned prompt/prefill check before selecting it. A faster candidate that only accepts aligned prompt lengths fails the capability contract.

If the user does not provide an accuracy bar, use:

- top-1 >= 90%;
- top-5 >= 98%.

Keep top-100 at the existing readiness expectation unless the user or model-specific evidence explicitly changes it.

## Preferred Outputs

Produce:

```text
models/autoports/<model>/doc/datatype_sweep/README.md
models/autoports/<model>/doc/datatype_sweep/work_log.md
models/autoports/<model>/doc/datatype_sweep/sweep_results.json
models/autoports/<model>/doc/datatype_sweep/sweep_results.csv
models/autoports/<model>/doc/datatype_sweep/selected_precision_config.json
models/autoports/<model>/doc/datatype_sweep/top1_perf_pareto.png
models/autoports/<model>/doc/datatype_sweep/top5_perf_pareto.png
```

The README should lead with the selected config, top-1, top-5, top-100, TTFT, trace-verified teacher-forcing decode t/s/u, post-selection token-out decode t/s/u when available, and the exact acceptance thresholds. Label each performance number with its measurement regime and workload shape.

## Baseline

Start from the completed optimized full model. Run and record the current baseline:

- readiness reference provenance, preferably AIME24 with chat template and `--gen-len 100`;
- `run_prefill_check` and `run_teacher_forcing` top-1/top-5/top-100;
- prefill TTFT and trace-verified teacher-forcing decode t/s/u from the readiness runner, generator metrics, or the model's benchmark harness;
- the active dtype policy for weights, norms, residual stream, CCL activations, KV cache, logits, sampling, and MoE routing.

If an existing reference has fewer than 100 generated tokens, regenerate the main readiness reference with 100 generated tokens unless a concrete model or memory limit prevents it.

Teacher-forcing decode performance is valid only when the measured path uses traced decode. Eager/untraced teacher-forcing results may be used for temporary correctness debugging, but they are not acceptable sweep evidence and must not be used to rank candidates, draw Pareto charts, or select the fastest passing config. If traced teacher forcing fails, fix tracing before continuing the sweep.

## Default Search

Use this coarse search first. It usually finds most of the available win with a small number of full-model runs.

1. Evaluate the closest canonical performance and accuracy policies, if any. When the model carries compiler-emitted per-tensor precision (e.g. from an IR/provenance seed), start each group from that mixed policy rather than a uniform blanket dtype.
2. Try BFP8 KV cache as a yes/no switch.
3. Try BFP8 CCL or residual-transfer activations as a yes/no switch.
4. Try BFP4 for eligible MLP/expert matmul groups, excluding the first and last layer by default unless the canonical policy says otherwise.
5. If full-model accuracy fails, restore the highest-risk groups first until top-1 and top-5 pass.
6. Once a passing inner-layer config is found, optionally try extending the surviving choices to the first and last layer. Keep that only if full-model accuracy still passes.

Restore-order example for dense MLP or MoE-style blocks:

1. FF2 / down projections / expert down projections.
2. QKV / attention input projections.
3. WO / attention output projection.
4. FF1+FF3 / gate+up projections / expert gate+up projections.

This order is a heuristic, not a law, with the typically most-sensitive parts of the model being tried first. Adjust it for the model architecture, profiler evidence, and observed failures. For MLA, shared experts, unusual norm placement, gated attention, or fused projections, map the groups to the nearest semantic operation and record the mapping.

When backing out a failed BFP4 trial, restore to BFP8 first unless BFP8 itself is known to be the failing precision for that tensor group.

For every material matmul group tested with BFP4 weights, include a LoFi compute-fidelity candidate for that same group. BFP4 with HiFi2 is allowed as a comparison or fallback, but it is not enough by itself. If a BFP4+LoFi candidate cannot be run, record the exact TTNN/runtime blocker and keep the stage open until the blocker is fixed or `$autofix` fails.

Sweep compute fidelity even when dtype is unchanged. For dominant decode projection groups, include BFP8+LoFi and BFP8+HiFi2 candidates when both are legal. Do not assume HiFi2 is fastest for BFP8, and do not assume LoFi is safe without full-model top-1/top-5 evidence. Record dtype and fidelity together in each candidate id so later stages can construct the exact policy.

Note sometimes different datatypes require small semantic changes to the code. KV cache is a common example of this - `paged_fill_cache` requires tensors that are the same datatype as the cache but `paged_update_cache` requires update tensors in BF16/FLOAT32 even when the destination cache is e.g. BFP8. So when changing datatypes first run a quick one-decoder smoketest to check it works correctly and get that right before using it or rejecting it in a full model pareto sweep. Also validate reduced-precision attention/KV at a representative long context and over a multi-step recurrent decode, not only short prompts: low-precision KV error accumulates with context, so a short-context PCC can pass while long-context degrades.

Sweep attention and MLP precision independently, and never let a synthetic/artificial-distribution PCC stress veto a candidate that clears the real-weight full-model accuracy bar; keep synthetic PCC as a diagnostic only.

## MoE Policy

Treat expert matmuls like MLP matmuls:

- expert gate/up projections follow the FF1+FF3 policy;
- expert down projections follow the FF2 policy;
- shared experts follow dense MLP policy unless evidence says otherwise;
- routed sparse expert matmuls should be swept with the active-expert path, not dense all-expert debug paths.

Router and weighting numerics are expected to be more sensitive. Do not include router logits, top-k selection, routing scores, gate weighting, or expert reduction weighting in the coarse BFP4 sweep, perhaps in BFP8 but with somewhat low priority as they are not typically expected to be a performance bottleneck. More generally, selection boundaries (router top-k, and the sampler's argmax/top-k) are precision-fragile — a near-tie flips under low precision — so gate any reduction there on a selection-stability real-weight PCC check rather than a fixed dtype floor.

For MoE sweeps, record route/top-k distribution or active expert count when model architecture allows this to vary, since the performance impact depends on how many experts actually run.

## Full-Model Evaluation

Every kept candidate must be validated with full-model accuracy. For each evaluated config record:

- config id and precision config path;
- weight dtype groups and layer ranges;
- activation, residual, CCL, logits/sampling, and KV-cache dtype choices;
- top-1, top-5, top-100, token count, and reference path;
- TTFT, trace-verified decode t/s/u, and the evidence that the teacher-forcing decode path was traced;
- compute fidelities used by each material matmul group and evidence that the measured runtime used them;
- direct comparisons for material same-dtype fidelity choices, especially BFP8+LoFi versus BFP8+HiFi2 on decode-bound projection groups;
- whether this config passed the user-specified accuracy bar;
- exact command, branch/commit, hardware, mesh, and environment notes.

Use `$qualitative-check` for prompt-based evaluation evidence: prompt-format metadata, rendered prompt artifacts or token ids, HF controls, and classification of raw-completion or chat-template behavior.

Always use trace-verified teacher-forcing decode t/s/u to rank datatype candidates - a non-traced path is not useful. If fully-traced decode is not working, use the $autofix skill until it is.

Teacher-forcing is the selection metric, not necessarily the serving headline. After selecting the winning config, run the same warmed token-out no-readback benchmark used by optimized full model, using the selected config through the normal construction path. Record this separately as post-selection token-out performance. If the model cannot run that benchmark, state why instead of presenting teacher-forcing performance as token-out performance.

If layer-level decode evidence, reduced token-out evidence, or a prior optimized full-model token-out artifact points to a different fastest passing policy than teacher-forcing does, resolve the conflict before selecting. Run the same token-out benchmark for the baseline and the top conflicting candidate through the normal construction path. If token-out cannot run because the generator or trace-feedback path fails, treat that as a bug to fix or a recorded blocker; do not silently use teacher-forcing throughput as the serving-policy answer.

## Plots

Use matplotlib/pyplot to generate two elegant, delightful Pareto charts:

- `top1_perf_pareto.png`: x-axis top-1 accuracy, y-axis performance.
- `top5_perf_pareto.png`: x-axis top-5 accuracy, y-axis performance.

Use trace-verified teacher-forcing decode t/s/u as the default performance metric. If only traced latency is available, convert to a higher-is-better performance metric or clearly label the axis. Do not plot eager/untraced decode performance as a Pareto candidate.

Plot every evaluated full-model config as a point. Fit or draw the non-dominated Pareto frontier through the evaluated points. Mark the selected config in red. Draw a vertical dotted line whose x-intercept is the minimum allowed accuracy level for that chart. Label or annotate enough points that the selected tradeoff and rejected candidates are clear without reading the raw JSON. Show these prominently in the README.md.

## Compute Fidelity

Compute fidelity is part of the selected precision policy, not only documentation.

For reduced-precision matmuls, sweep legal compute fidelities in the same candidate matrix as datatypes once the promising dtype groups are known. Use the `tt-perf-report` output from individual decoder or reduced full-model runs and full-model accuracy to guide this decision. BFP4 weights are expected to use LoFi unless measured evidence says otherwise. BFP8 weights should compare HiFi2 and LoFi for dominant decode projection groups instead of assuming one fidelity is best. HiFi4 is usually reserved for accuracy-sensitive operations with BF16 weights. FP32 destination accumulation is also an option.

Before selecting a config:

- prove the model code can construct the requested fidelity policy without hard-coded overrides;
- include at least one BFP4+LoFi candidate for every selected or seriously considered BFP4 matmul group;
- compare BFP4+LoFi against BFP4+HiFi2 when both are legal and the group is a meaningful decode cost;
- record a short table for each material matmul group: weight dtype, compute fidelity, traced decode speed, accuracy result, kept/rejected decision, and reason.

If the runtime does not currently expose fidelity as a policy field, add that plumbing before ranking candidates. Do not satisfy this requirement by writing `compute_fidelities` into JSON that the measured model ignores.

## Final Selection

Select the fastest config that satisfies the acceptance bar. If two configs are within measurement noise, prefer the simpler and safer one.

Before finishing:

- make the selected config the model's default construction path, or write a required config artifact that `build_generator`, full-model, and vLLM paths actually consume by default;
- keep a simple config change or override to return to the safe baseline setting;
- run a post-selection token-out no-readback performance check with the selected config and record the workload shape, trace status, TTFT, decode t/s/u, and runtime counters;
- use `$qualitative-check` to rerun the shared qualitative prompt suite on the selected config, with HF controls. If visible text quality selects or rejects a dtype/fidelity candidate, the verdict must come from this prompt-correct suite;
- add a short propagation check proving `build_generator` and the vLLM adapter load the same selected weight/activation/CCL/KV/fidelity policy used by the winning sweep result.
- update the context contract for the selected KV-cache dtype and prove later construction paths use a matching max context.

If no lower-precision config passes, keep the baseline and leave evidence that the sweep actually tested the likely wins.
