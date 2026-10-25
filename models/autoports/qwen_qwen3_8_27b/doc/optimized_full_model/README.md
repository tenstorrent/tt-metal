# Optimized full model — stage07

|4096 input /128 output|Baseline warmed TTFT ms|Final warmed TTFT ms|Baseline token-out t/s/user|Final token-out t/s/user|
|---|---:|---:|---:|---:|
|B1,one request|433.812|431.361|35.186|35.252|
|B8|3577.570|3564.809|20.978|21.015|
|B16|7316.120|7302.092|13.701|13.725|
|B32|14760.040|14715.773|8.822|8.844|

B1 is the required headline workload, measured over three warmed all64-layer
requests in `benchmark_b1_perf.json`, with diagnostics disabled. Its127 decode
steps use nonblocking split traces, zero host input refreshes and one final output
read. The first-token read is included in TTFT. Baselines are stage06
`../full_model/benchmark_b*_perf.json`, same workload/hardware/precision. These are
end-to-end host timings, not device times or device rooflines.

Separate clean AIME teacher-forced token-out control:35.525t/s/user,114.056ms TTFT,
203input/100output, canonical sampling plus reference-token callbacks/readback.
This differs from the4096/128 autoregressive headline and is not a model-trace-only
measurement. Accuracy runs separately pass top1=99%,top5=100%,top100=100% over100
reference token positions for both prefill and teacher forcing.

Stage06 and stage07 AIME prefill and teacher-forcing results are unchanged: each
phase has top1=99%,top5=100%,top100=100% over the same100 reference token positions.
The HF reference identity is preserved in full_validation_provenance.json.

The initial paired candidate (`buffered_full_b1.json`) measured35.179 streamed versus
35.252 buffered t/s/user. Final results above come from the final default source;
source snapshots for the earlier candidate are retained under `sources/`.

EOS-aware default uses16-step output windows. On the exact47-token factual answer,
paired warmed total latencies are1739.87ms streamed,4064.64ms with128-step windows,
and1797.60ms with16-step windows. The selected policy costs57.72ms (3.32%) versus
streaming while retaining device feedback and avoiding per-token host sync; fixed
length throughput still uses one128-step window. Explicit streaming remains available.

The B32 agent workload (32768input/128output) measured144643.64ms TTFT and
7.441t/s/user after warmup (one measured request), versus147101.69ms and7.418 in
stage06. Full-stack mixed-slot replay and allocation audits pass; B32 peak accounted
DRAM is28.44GB/device against34.18GB physical capacity.

Target mesh is1x4 TP4 on four Blackhole p300c ASICs. Agent target remains32
conversations x32768 working-set tokens; supported decode batches1/8/16/32. Prior
stage06 measurements and capability evidence remain in `../full_model/`; they are
baseline evidence, not new stage07 results. Native262144 context is preserved.

The selected stage05/06 decoder precision, residual layouts, page/cache semantics
and rejection ledger are unchanged. See `topology_audit.md` for the full-path audit
and planned terminal/collective candidates. Real reduced B1/B32 buffered-output
probes pass exact streamed agreement for greedy and sampled modes with allocation
tracking including program-cache allocations. Final accuracy, qualitative, all-batch timings, profiles, watcher, external-cache,
mixed-slot and context/memory checks pass. Independent review is **clean-pass** in
[stage_review.md](stage_review.md); local checkpoint receipts are in work_log.md.

No vLLM work is included. The required local telemetry packet records
the measured final B1 headline and accuracy; device-time fields remain unknown. Commands and observations
are in `work_log.md`.

Selected changes and evidence:

- Default autoregression queues split model/sampler traces into a persistent output
  collector. `tt_out_tok`, device position/RoPE advance, changed-only page maps,
  top-k/top-p, fixed slots and inactive state remain intact. See runtime_audit.md
  and watcher_summary.json (14 exact tracker-enabled cases, all watcher checks on).
- Last-token prefill packs users before the terminal norm/head. Bitwise33/65/4097
  controls are in prefill_terminal_b32.json; the B32 reduced prefill envelope drops
  about48.96ms. The full-model TTFT table is the actual end-to-end measurement.
- B32 uses measured faster async candidate gathers; other measured batches retain
  the faster existing gather. Head and persistent-sampler alternatives were adapted
  and measured before rejection; see rejection_ledger.json and head_candidates.csv.
- Decoder dtype/fidelity/KV/CCL and residual contracts are preserved. Runtime
  attributes are checked in kernel_policy_audit.json. profiler_conclusions.md
  explains per-layer estimates, remaining gap, report advice and its limitations.
- External sparse-cache preparation, failure cleanup, continuation tails, page
  remapping and mixed/all-inactive preservation pass in external_prepare_reduced_b32.json
  and mixed_slots.json. Full64 mixed-slot reuse is also in each multi-user benchmark.

Native262144 capability evidence is inherited from the unchanged stage06 B1
cache/position/model path and explicitly linked in context_contract.json. This
stage measures native-size B1 allocation and fresh smaller-context/long-output
behavior; it does not claim a new native-context HF accuracy run. The32K agent
workload has one measured request after warmup; all4K headline rows have three.
EOS-aware16-step windows can finish up to15 extra device steps before returning a
short response. The1024-token repeated-text4096 quality case remains truncated,
exactly as in stage06; qualitative_verdict.md distinguishes that limitation from
its coherent nearby4095 and exact factual4096 controls. No dataset-wide accuracy
claim is made from the one-problem,100-position AIME check.

Reproduce inside the existing source-built container:

```bash
bash models/autoports/qwen_qwen3_8_27b/tests/optimized_multichip_env.sh python -m models.autoports.qwen_qwen3_8_27b.tests.run_full_validation --output-dir models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model --buffered-output --long-cases
bash models/autoports/qwen_qwen3_8_27b/tests/benchmark_optimized_full_model.sh
bash models/autoports/qwen_qwen3_8_27b/tests/validate_optimized_full_model.sh
```

Hardware jobs must stay serialized. Watcher and profiling are separate. The
initial watcher inline-code-size failure is resolved by the established NOINLINE
configuration, without disabling checks. CPU reconciliation and exact commands
are recorded in work_log.md. stage_checks.json reports degeneration/context pass
and six unchanged accounted weight-representation advisories. Raw local evidence
receipts are in artifact_manifest.json; reduced tt-perf-report tables/CSV/provenance
are under tracy/b1 and tracy/b32. The compact packet contains only matching all64
4096/128/B1 autoregressive results, accuracy, references and missing-measurement reasons.
