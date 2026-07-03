---
name: datatype-sweep
description: Post-release sweep and selection of TTNN model weight, activation, KV-cache, CCL datatype, precision-bearing program/compute-kernel fields, and compute-fidelity policies. Use after the TTI release stage has passed, using a smaller TTI-release benchmark or eval as the screening gate, then rerun the full TTI-release benchmark/eval set for the fastest screening-passing config.
---

# Datatype Sweep

## Mission Context

Run this skill only after the TTI release stage has passed. Stages 01-09 keep
dtype and compute fidelity conservative, lowering them only when a byte
calculation or failed capacity probe shows that the advertised model capability
cannot fit otherwise. Use this stage for broad BFP4/LoFi, BFP8 KV-cache, BFP8
activation/CCL, precision-bearing matmul program/compute-kernel fields, and
compute-fidelity Pareto exploration.

Start from a TTI-release-passing autoport and read:

```text
models/autoports/<model>/doc/tti_release/post_release_sweep_benchmark.json
```

That handoff names the smaller TTI-release benchmark or eval metric used for
candidate screening, the pass threshold, the baseline conservative-policy
result, the performance metric to rank candidates, and the full benchmark/eval
set that must be rerun for the selected config.

Use the smaller TTI-release benchmark/eval named by the handoff as the
candidate selection gate. Decoder-layer PCC, readiness top-1/top-5, component
timing, and layer tests are smoke/debug evidence unless the TTI handoff selects
one of them as the screening metric.

## Preferred Outputs

Produce:

```text
models/autoports/<model>/doc/datatype_sweep/README.md
models/autoports/<model>/doc/datatype_sweep/work_log.md
models/autoports/<model>/doc/datatype_sweep/sweep_results.json
models/autoports/<model>/doc/datatype_sweep/sweep_results.csv
models/autoports/<model>/doc/datatype_sweep/selected_precision_config.json
models/autoports/<model>/doc/datatype_sweep/benchmark_perf_pareto.png
models/autoports/<model>/doc/datatype_sweep/post_selection_full_rerun.json
```

The README should lead with the selected config, screening benchmark score,
screening threshold, primary performance metric, post-selection full benchmark
verdict, TTFT/TPOT/decode t/s/u where applicable, and the exact workload shapes.

## Handoff Contract

The TTI release stage should write a JSON object with at least:

- `screening_benchmark`: human-readable benchmark/eval name.
- `screening_command` or `screening_workflow`: how to run the smaller screening
  item.
- `screening_metric`: quality/accuracy metric used as the pass gate.
- `baseline_score`: conservative-policy result from the release pass.
- `pass_threshold`: minimum acceptable screening score.
- `performance_metric`: higher-is-better metric used to rank passing configs,
  or a latency metric with direction clearly stated.
- `full_rerun`: commands, workflows, or artifacts for the full TTI-release rerun.

If the handoff is missing, inspect the TTI release artifacts and create the
smallest faithful handoff before sweeping. Do not invent a weak benchmark just
because it is cheap. The screening benchmark must be quality-sensitive enough
to catch meaningful dtype/fidelity regressions.

## Baseline

Refresh the conservative baseline before candidate work:

- active code path and selected branch/commit;
- current conservative dtype/fidelity policy;
- context contract and served `max_model_len`;
- TTI release status and copied report path;
- screening benchmark score and command;
- full benchmark/eval status from the TTI release stage;
- primary serving performance, normally vLLM single-user TTFT/TPOT/decode t/s/u.

If refreshing the baseline does not reproduce the TTI release pass, stop and
fix that regression before sweeping.

## Candidate Search

Use a coarse-to-fine search. Keep candidates complete enough to construct the
same policy later:

- weight dtype groups by semantic role and layer range;
- layer exceptions;
- compute fidelities for material matmul groups;
- precision-bearing program/compute-kernel fields, such as math fidelity or
  accumulator precision;
- activation/residual dtype;
- CCL payload dtype;
- KV-cache dtype and cache fill/update rules;
- logits/sampling dtype assumptions;
- loader/runtime flags required for the policy.

Good first candidates usually include:

1. Conservative baseline.
2. BFP8 KV cache, when it was not already needed for memory fit.
3. BFP8 CCL or residual-transfer activations.
4. BFP4 or lower-fidelity policies for dominant MLP/expert groups.
5. Attention projection and LM-head dtype/fidelity candidates when those rows
   are material.
6. First/last-layer exceptions after an inner-layer policy passes screening.

For every material BFP4 matmul group considered or selected, include a
BFP4+LoFi candidate for that same group, or record the exact TTNN/runtime
blocker plus `$autofix` evidence. Sweep compute fidelity as part of the policy,
not as a note written into JSON that the runtime ignores. If math fidelity or
another precision switch is housed in a matmul program or compute-kernel config,
include it in the stage-10 sweep policy.

## Evaluation

Every kept candidate must run the screening benchmark through the normal
construction path. For each candidate record:

- config id and precision config path;
- complete dtype/fidelity policy;
- precision-bearing program/compute-kernel fields;
- screening benchmark command/workflow, metric, score, pass/fail, and artifact;
- primary performance metric and workload shape;
- TTFT/TPOT/decode t/s/u when serving performance is the ranking metric;
- context length and any memory/capacity change;
- exact command, branch/commit, hardware, mesh, and environment notes;
- propagation evidence proving the measured runtime consumed the policy.

Do not rank candidates using eager/untraced decode if the production path is
traced serving. If the screening benchmark cannot run under a candidate because
of an implementation bug, fix the bug or keep the exact blocker; do not silently
drop the candidate from the report.

## Plots

Generate:

```text
benchmark_perf_pareto.png
```

Use the screening benchmark metric on one axis and the selected performance
metric on the other. Make the metric directions explicit. Plot every evaluated
candidate, draw the non-dominated Pareto frontier, mark the selected config in
red, and draw the pass threshold. Label enough points that the selected tradeoff
and rejected candidates are understandable without opening the raw JSON.

## Final Selection

Select the most performant evaluated config that passes the screening benchmark
and preserves the context/capability contract. If two configs are within
measurement noise, prefer the simpler and safer one.

Before finishing:

- write `selected_precision_config.json`;
- make the selected config the model's default construction path, or make it a
  required config artifact that `build_generator`, full-model, and vLLM paths
  consume by default;
- keep a simple override to return to the conservative baseline;
- prove every selected field is consumed by the measured runtime path using a
  model summary, config propagation check, or profiler/perf-report rows;
- update `doc/context_contract.json` for selected KV-cache or persistent-buffer
  changes;
- run the primary serving performance benchmark through the selected-config
  path and record TTFT/TPOT/decode t/s/u separately from the screening score.

If no lower-precision config passes, keep the conservative baseline and leave
evidence that the sweep actually tested the likely wins.

## Full Stage-10 Rerun

After selecting the fastest screening-passing config, rerun every other
benchmark/eval/test named by the TTI handoff through the selected-config
construction path. Record the result in `post_selection_full_rerun.json`.

If the selected config fails an unwaived full benchmark/eval row, try the next
fastest screening-passing config. If no candidate passes the full rerun, report
the conservative baseline as the only release-ready policy and classify the
failed lower-precision configs with exact evidence.
