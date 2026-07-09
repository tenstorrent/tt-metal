---
name: datatype-sweep
description: Sweep DiffusionGemma weight, activation, KV-cache, canvas-scratch, CCL, and decision-path precision using injected-noise diffusion-decision agreement versus traced denoise/block latency. Never use autoregressive teacher-forcing top-k as the selection gate.
---

# DiffusionGemma datatype sweep

Load `diffusion-gemma` first. This skill owns dg-07 precision selection for the
existing full DiffusionGemma path.

## Current completed result

The 2026-07-08 dg-07 sweep tested the dominant expert-weight reduction:

- baseline/selected experts: BF16;
- candidate: `DG_EXPERTS_BFP8=1`;
- BF8 saved 5.44 GiB/chip and improved traced @48 throughput
  18.18 → 19.83 t/s (+9.1%);
- BF8 failed all decision gates: committed agreement 0.227, entropy PCC 0.631,
  accept/renoise IoU 0.501 versus the BF16 trajectory.

Therefore BF16 remains selected. The BF8 knob is diagnostic/opt-in only and
must not become default until a new sweep proves sufficient fidelity headroom.
Do not rerun this exact candidate without a changed #48291 baseline or a new
technical reason.

Authoritative artifacts:

- `models/experimental/diffusion_gemma/doc/datatype_sweep/README.md`;
- `work_log.md`;
- `sweep_results.json` and `sweep_results.csv`;
- `selected_precision_config.json`;
- `pareto_argmax_vs_latency.png`;
- `pareto_accept_vs_latency.png`.

## Selection metric

Use one fixed initial canvas, injected Gumbel noise, and fixed renoise tokens
for baseline and candidate. Rank candidates by:

1. committed clean-argmax agreement;
2. per-step Gumbel/clean argmax agreement;
3. entropy PCC and maximum absolute error;
4. accept/renoise IoU;
5. canvas agreement and qualitative trajectory behavior;
6. traced denoise-step and per-block latency.

The BF16 baseline is the relative reference. Unless the user or dg-05 defines
a stronger bar, a reduced-precision candidate must satisfy all current dg-07
minimums:

- committed agreement >= 0.95;
- mean entropy PCC >= 0.95;
- mean accept/renoise IoU >= 0.90.

AIME24, teacher-forcing top-1/top-5/top-100, and autoregressive token-out t/s/u
are not DiffusionGemma selection metrics.

## Candidate policy

Keep the router, top-k, logits, probability, entropy, Gumbel-max, and
accept/renoise path at the highest validated precision. Sweep tensor groups
independently:

- expert gate/up/down weights;
- attention projections;
- shared MLP;
- embeddings and LM head;
- activation/residual and CCL payloads;
- KV cache;
- per-step canvas K/V scratch;
- compute fidelity and destination accumulation.

For each candidate, prove the requested dtype/fidelity reaches the runtime
path through cache names, model summary, memory view, or profiler-equivalent
evidence. A JSON policy that runtime ignores is not a candidate.

Use real weights and real recorded activations for fidelity decisions.
Synthetic tensors may find crashes or layout bugs but cannot approve or veto a
shipping precision policy by themselves.

The production MoE is `tt/sparse_moe.py` true-sparse token gather. Sweep that
path, not the retired dense-128 debug implementation.

## Context and capacity

If KV-cache, canvas-scratch, trace-buffer, mask, or persistent CCL dtype/layout
changes, recompute:

`models/experimental/diffusion_gemma/doc/context_contract.json`

Preserve the 262144 prompt+generated capacity unless byte math and a capacity
probe prove a hard physical limit. Include weights, frozen KV, canvas scratch,
non-causal masks, trace buffers, and fragmentation headroom.

## Measurement workflow

1. Reproduce the selected BF16 baseline through the normal model constructor.
2. Change one named precision/fidelity group.
3. Run a small component/runtime smoke.
4. Run the deterministic injected-noise decision trajectory.
5. Reject immediately if a mandatory decision bar fails.
6. For passing candidates, run the same warmed traced step/block benchmark.
7. Confirm non-aligned prompts, KV phase behavior, trace replay, and watcher
   remain valid.
8. Update the selected policy only if the candidate is faster and passes every
   gate.

Do not rank eager runs. Performance evidence is traced per-denoise-step,
per-block, and full-generation latency, with workload shape and step count.

## Outputs

Update:

```text
models/experimental/diffusion_gemma/doc/datatype_sweep/README.md
models/experimental/diffusion_gemma/doc/datatype_sweep/work_log.md
models/experimental/diffusion_gemma/doc/datatype_sweep/sweep_results.json
models/experimental/diffusion_gemma/doc/datatype_sweep/sweep_results.csv
models/experimental/diffusion_gemma/doc/datatype_sweep/selected_precision_config.json
models/experimental/diffusion_gemma/doc/datatype_sweep/pareto_argmax_vs_latency.png
models/experimental/diffusion_gemma/doc/datatype_sweep/pareto_accept_vs_latency.png
```

Each result row records exact dtype/fidelity groups, runtime-consumption proof,
DRAM used/chip, context impact, decision metrics, traced latency, command,
checkpoint revision, hardware, and kept/rejected reason.

Done means the selected configuration is consumed by the normal runtime,
passes the diffusion-decision floor, preserves the capability contract, wins
the traced comparison, and is recorded for vLLM/TTI consumers.
