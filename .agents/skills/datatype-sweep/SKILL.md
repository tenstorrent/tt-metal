---
name: datatype-sweep
description: Sweep and select TTNN model weight, activation, KV-cache, and CCL datatypes for the fastest configuration that satisfies minimum full-model top-1 and top-5 accuracy. Use after optimized-full-model and before vLLM, or whenever a runnable TTNN full model needs evidence-backed precision selection.
---

# Datatype Sweep

## Mission Context

This skill starts from a working optimized TTNN full model and chooses the fastest practical precision configuration that still meets a stated full-model accuracy bar. It is normally used after optimized full model and before vLLM so the serving adapter inherits a settled weight, activation, CCL, and KV-cache dtype policy.

The expensive source of truth is full-model top-1/top-5 accuracy. Decoder-layer PCC and component timing are useful only for ordering candidates and debugging surprises.

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

The README should lead with the selected config, top-1, top-5, top-100, TTFT, decode t/s/u, and the exact acceptance thresholds.

## Baseline

Start from the completed optimized full model. Run and record the current baseline:

- readiness reference provenance, preferably AIME24 with chat template and `--gen-len 100`;
- `run_prefill_check` and `run_teacher_forcing` top-1/top-5/top-100;
- prefill TTFT and decode t/s/u from the readiness runner, generator metrics, or the model's benchmark harness;
- the active dtype policy for weights, norms, residual stream, CCL activations, KV cache, logits, sampling, and MoE routing.

If an existing reference has fewer than 100 generated tokens, regenerate the main readiness reference with 100 generated tokens unless a concrete model or memory limit prevents it.

## Default Search

Use this coarse search first. It usually finds most of the available win with a small number of full-model runs.

1. Try BFP8 KV cache as a yes/no switch.
2. Try BFP8 CCL or residual-transfer activations as a yes/no switch.
3. Try BFP4 for all eligible inner-layer BFP8 matmuls, excluding the first and last layer by default.
4. If full-model accuracy fails, restore the highest-risk groups first until top-1 and top-5 pass.
5. Once a passing inner-layer config is found, optionally try extending the surviving choices to the first and last layer. Keep that only if full-model accuracy still passes.

Restore-order example for dense MLP or MoE-style blocks:

1. FF2 / down projections / expert down projections.
2. QKV / attention input projections.
3. WO / attention output projection.
4. FF1+FF3 / gate+up projections / expert gate+up projections.

This order is a heuristic, not a law, with the typically most-sensitive parts of the model being tried first. Adjust it for the model architecture, profiler evidence, and observed failures. For MLA, shared experts, unusual norm placement, gated attention, or fused projections, map the groups to the nearest semantic operation and record the mapping.

When backing out a failed BFP4 trial, restore to BFP8 first unless BFP8 itself is known to be the failing precision for that tensor group.

## MoE Policy

Treat expert matmuls like MLP matmuls:

- expert gate/up projections follow the FF1+FF3 policy;
- expert down projections follow the FF2 policy;
- shared experts follow dense MLP policy unless evidence says otherwise;
- routed sparse expert matmuls should be swept with the active-expert path, not dense all-expert debug paths.

Router and weighting numerics are expected to be more sensitive. Do not include router logits, top-k selection, routing scores, gate weighting, or expert reduction weighting in the coarse BFP4 sweep, perhaps in BFP8 but with somewhat low priority as they are not typically expected to be a performance bottleneck.

For MoE sweeps, record route/top-k distribution or active expert count when model architecture allows this to vary, since the performance impact depends on how many experts actually run.

## Full-Model Evaluation

Every kept candidate must be validated with full-model accuracy. For each evaluated config record:

- config id and precision config path;
- weight dtype groups and layer ranges;
- activation, CCL, and KV-cache dtype choices;
- top-1, top-5, top-100, token count, and reference path;
- TTFT, decode t/s/u, and any fallback timing source;
- whether this config passed the user-specified accuracy bar;
- exact command, branch/commit, hardware, mesh, and environment notes.

## Plots

Use matplotlib/pyplot to generate two elegant, delightful Pareto charts:

- `top1_perf_pareto.png`: x-axis top-1 accuracy, y-axis performance.
- `top5_perf_pareto.png`: x-axis top-5 accuracy, y-axis performance.

Use decode t/s/u as the default performance metric. If only latency is available, convert to a higher-is-better performance metric or clearly label the axis.

Plot every evaluated full-model config as a point. Fit or draw the non-dominated Pareto frontier through the evaluated points. Mark the selected config in red. Draw a vertical dotted line whose x-intercept is the minimum allowed accuracy level for that chart. Label or annotate enough points that the selected tradeoff and rejected candidates are clear without reading the raw JSON. Show these prominently in the README.md.

## Final Selection

Select the fastest config that satisfies the acceptance bar. If two configs are within measurement noise, prefer the simpler and safer one.

Before finishing:

- confirm the selected config is the model's default with a simple config change to return to the safe baseline setting;
- run qualitative generation if the dtype changes are large or top-1 is close to the threshold - if this is bad then back off more changes until it is good;

If no lower-precision config passes, keep the baseline and leave evidence that the sweep actually tested the likely wins.
