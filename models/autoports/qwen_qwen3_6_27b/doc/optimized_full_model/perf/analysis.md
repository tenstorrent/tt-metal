# Reduced full-path profiling

The optimized path retains the selected decoder policy and split sampling. The
largest terminal cost is the unchanged vocabulary-sharded BF16/HiFi4 LM head
(1.55–1.57 ms/token). The complete sampler costs 0.453–0.473 ms of kernel work,
about 15% of this two-layer token-out probe and about 1.5% of its extrapolated
64-layer kernel work. It does not dominate the full-model path. No force argmax,
full-vocabulary collective or generic `TopKDeviceOperation` appears.

This is CPU-only analysis of `../profile.json` and the immutable raw CSV at
`../tracy/reports/2026_11_14_06_17_56/ops_perf_results_2026_11_14_06_17_56.csv`.
The profiler ran layers 0 and 3 with real weights, batch 1, prompt 128,
context allocation 2048, TP4 Blackhole. Each decode window has three steps.
It prepares only length 128 and is not allocation-safety acceptance or accuracy
validation. The profile records its commit, source hashes and environment.
`command_receipt.json` records all eight advice-enabled tt-perf-report 1.5.0
commands. Human tables are `prefill.txt`, `decode.txt`, `logits_only.txt`, and
`sampling.txt`; matching CSVs and tool summary CSV/PNGs are retained. Raw logs
stay local; all analysis artifacts are compact.

## Same-run timing reconciliation

| Window | Instrumented wall | Summed kernel work per rank | Boundary |
| --- | ---: | ---: | --- |
| Prefill | 5.678 ms TTFT | 4.272–4.297 ms | Reset/configuration, traced prefill, head, sample and first-token readback |
| Token-out | 3.134 ms/step; 319.10 t/s/u | 2.978–2.979 ms/step | Nonblocking model + sampler replay, no per-token readback, one final synchronization |
| Logits-only | 2.624 ms/step; 381.05 steps/s | 2.516–2.530 ms/step | Model trace including terminal, no sampling/readback; timing-only fixed token with advancing positions |
| Sampling | 0.538 ms/sample | 0.452–0.455 ms/sample | Same resident logits repeatedly sampled; no per-token readback |

These are instrumented reduced-model numbers, never substitutes for the final
uninstrumented all64 measurements. Logits-only is not teacher forcing. The
0.509 ms token-out minus logits-only difference is compatible with separately
measured 0.538 ms sampling, but differing timing boundaries prevent an exact
subtraction. The previous full-model-stage instrumented prefill was 8.133 ms;
this traced prefill is 5.678 ms (30.2% lower), diagnostic only. Final same-regime
all64 TTFT remains the acceptance metric.

`kernel_accounting.json` sums raw `DEVICE KERNEL DURATION [ns]` independently
per TP rank, converted to microseconds. Do not sum four ranks into latency.
The device kernel sums are additive work, not a measured critical-path interval.
Raw op-to-op gaps spanning profiler drains are invalid: prefill reports a
113.673 ms gap although the entire signpost span is only 6.226 ms. The selected
replay windows have much smaller gaps (maximum decode gap 34.021 us), but a
report percentage based on all windows or drain-crossing gaps remains invalid.
Do not accept the report's suggested savings from tracing as evidence: model,
sampling and this selected prefill bucket are already traced.

The modeled matmul bandwidth floor in this same reduced run is 1.528 ms per
step per rank; the measured additive kernels are 2.979 ms and host time is
3.134 ms. Their ratios are 51.3% and 48.8%, respectively, but these are only
ratios to the **modeled matmul subset**—not full-model DRAM utilization or a
complete roofline. The model omits custom kernels/collectives and reports
implausible >100% FLOPs for some small-M operations. No full-model utilization
percentage is claimed. The raw `PM COMPUTE` sum is 5.517 us; its shape/fidelity
model must not be substituted for observed compute time.

## Layer and terminal decomposition

`decompose_profile.py` validates the operation/shape boundaries before slicing
three repeats on each rank. `layer_decomposition.csv` retains the source raw
CSV line ranges; `layer_accounting.json` retains full values.

| Component | Kernel us/token across ranks |
| --- | ---: |
| Token embedding and RoPE table lookup/tilize | 13.08–13.15 |
| One GDN layer | 478.01–485.10 |
| One full-attention layer | 380.73–382.48 |
| Final hidden gather and normalization | 30.64–37.14 |
| LM head | 1549.46–1568.85 |
| Stable sampler-logit staging | 44.95–45.46 |
| Device current-position/RoPE advance | 1.24–1.29 |
| Split sampling within token-out | 453.23–472.61 |

The raw sequence carries hidden-sharded residuals (1280 per rank) between
layers; each next norm consumes that contract. The terminal gathers hidden
5120 once, then the vocabulary is sharded 62080 per rank. Sampling pads local
logits to 65536, selects physical local top32, and all-gathers 128 candidates,
not the vocabulary. The TopK large-indices family costs about 325 us including
prep/finish; it uses the 110-core Blackhole implementation. These observed
operations agree with the common sampler's semantic greedy k1/p0/temp1.
Device `PlusOne` updates position and RoPE; profile counters show zero
per-token position/RoPE refreshes and token reads in the measured decode loop.

Extrapolating **kernel work** as 48 GDN + 16 full-attention yields
29.064–29.376 ms for the layer stack. Terminal plus embeddings/RoPE/position
and sampling adds 2.113–2.119 ms, giving 31.183–31.490 ms. This is an estimate,
not an all64 measurement: layer weights repeat, context allocation is 2048,
and it omits inter-kernel dispatch/device gaps. The corresponding matmul-only
modeled bandwidth floor extrapolates to 13.592 ms and is not a full roofline.

The independent optimized-layer **wall-time** lower bound remains
48 × 0.509915878 + 16 × 0.411218298 = **31.055455 ms/token**, with a median-layer
bound of 31.074964 ms. Its provenance is copied into `layer_accounting.json`
from `../../full_model/layer_stack_lower_bound.json`; the underlying serial
`final29_pair_l{0,3}_p{0..4}_final.json` runs use prefix512. Adding the observed
terminal work suggests roughly 33.17 ms/token, but this mixes regimes and
must not be presented as an exact subtraction. Compare final all64 time both
to this wall bound and the same-run profile estimate, preserving these caveats.
The final all64 resident benchmark is32.258325ms/token,31.000t/s/u;
`../perf_summary.json` preserves the distinct regimes and source lineage.

## Runtime policy and program evidence

`runtime_matmul_policy.csv` preserves raw row IDs, global call count, dtypes,
math fidelity, memory configs, program configs and kernel source paths for
all nine decode matmuls plus prefill matmuls on rank0. This is executed-op
evidence, not a declaration in a JSON policy.

| Role | Executed weights/fidelity | Executed geometry |
| --- | --- | --- |
| GDN packed attention input/output | BFP8 / HiFi4 | DRAM-sharded; 2 readers/bank, in0_block_w2, N28/N20 |
| GDN packed gate/up and down | BFP8 / HiFi2 | DRAM-sharded; 2 readers/bank, block2, N34/N20 |
| Full-attention packed QKV/output | BFP8 / LoFi | DRAM-sharded; 2 readers/bank, block2, N14/N20 |
| Full-attention packed gate/up and down | BFP8 / LoFi | DRAM-sharded; 2 readers/bank, block2, N34/N20 |
| LM head | BF16 / HiFi4 | Interleaved 11×10 1D multicast, block2, output subblock1×3, N18 |

Raw `CORE COUNT` includes 70 for the DRAM-sharded projections; tt-perf-report's
rendered summary shows eight DRAM-bank-associated cores. Do not infer a new
8-core program from the rendered value; the raw program configuration and
kernel source are retained. GDN packed input outputs FP32; other selected
matmuls output BF16. CCL and residual activations remain BF16. No precision
frontier search or change is performed by this audit.

## Advice and opportunities

`advice_rows.csv` captures every nonempty per-op recommendation with its
report row ID. The following dispositions cover all unique recommendations.

| Advice/opportunity | Disposition |
| --- | --- |
| LM-head DRAM-sharded matmul | Already measured at unchanged BF16/HiFi4: split2_r2_k2 and whole_r3_k1 pass numerical/greedy checks but lose complete terminal time. `../../full_model/perf_terminal/lm_head_decision.json` and `../../full_model/lm_head_uninstrumented.json` retain current2.2032ms versus split2 2.3492ms and whole3 2.6892ms. The initial whole2 L1 failure was adapted; rejection is measured, not an API failure. |
| Increase GDN prefill projection grid | Preserve selected phase-specific packed/correction family; `../../optimized_multichip_decoder/final29_candidate_decisions.md` documents geometry/readers/block sweeps. Current profile is not a new geometry rejection. |
| Prefill input0 in L1 and larger output subblocks | Same frozen decoder evidence: `advice5_*.json`, `mmrs2_summary.csv`, `family3_summary.csv`, and geometry experiments under `../../optimized_multichip_decoder/`. Global GDN L1 MLP loses; full-attention L1 MLP is retained. Longer-prefill untried cases are hand-off candidates, not falsely measured rejections. |
| Minimal matmul needs explicit program config | Tool does not understand the selected specialized minimal-prefill family. Preserve its measured complete-family result; no replacement with generic unfused operations for cosmetic advice compliance. |
| GDN HiFi4→HiFi2 | Rejected by target HF contract; `../../optimized_multichip_decoder/AUTOFIX_contract29.md` and `final_validation29_contract_l0_tp4.json`. Current raw rows confirm HiFi4 execution. |
| Selected LoFi/HiFi2→higher fidelity | Generic accuracy advice does not override the tested selected policy. Preserve full-attention LoFi/GDN MLPHiFi2 and run final all64 readiness gates. |
| Trace/remove gaps | New selected prefill128, decode and sampling are traced. Drain-crossing gaps are not removable runtime overhead. Final short/long/chunk-boundary TTFT evidence remains parent-owned. |
| Sampling dominates | False for the full path: about0.45ms against about31ms extrapolated work, no generic TopK or full-vocab gather. Keep split greedy; force-argmax comparison/semantic ties remain in existing sampler evidence. |
| Remove terminal staging/tilize | At most about45us stable-logit staging and13us embedding/RoPE work here. Could help only if buffer identity, tile-padded batch and common-sampler contract survive; untried, no claimed gain. |

The raw profile shows no sampler contract defect requiring replacement. Final all64 performance/accuracy, context, larger-batch, Watcher/allocation
gates and stop-rule disposition are now recorded in `../README.md` and
`../hard_gates.json`; independent review is the remaining closure gate.

## Complete-model compulsory-traffic estimate

`roofline.json` and `estimate_roofline.py` now provide the full64 byte-based
estimate for the FUNCTIONAL comparison. Active executed matrices stream
7,899,709,440 bytes per device per token, including BFP8 exponent storage and
the BF16 LM head. Add GDN recurrent read/write traffic, SDPA's padded KV window,
conv state and selected embedding/RoPE rows: the estimate is **15.515526 ms/token**
at the prior stage5 local theoretical **512 GB/s per chip**, or
**64.4516 theoretical tokens/s/user**. Ten percent is **6.4452 t/s/u**;
the stricter endpoint of the accounting range is about **6.4493 t/s/u**.
This is a theoretical compulsory-traffic calculation, not measured bandwidth
or an assertion that the stage passed.

The range is 15.505568–15.515526 ms for the documented accounting conventions,
not a bound on wall time. Production context262144 uses a512-token SDPA read
window at positions128–254, whereas this reduced context2048 profile uses
128-token chunks and rounds those positions to256. The estimate counts the
512-token window, not all allocated KV. Per-device traffic divided by512GB/s
exactly equals aggregate four-device traffic divided by2048GB/s. It counts
only one selected embedding row per rank, never the full embedding table.
The raw `PM BANDWIDTH` subset above is a different generic performance model;
use the explicit physical-byte calculation and local stage5 constant for the
FUNCTIONAL threshold, not its 13.592ms extrapolation.
