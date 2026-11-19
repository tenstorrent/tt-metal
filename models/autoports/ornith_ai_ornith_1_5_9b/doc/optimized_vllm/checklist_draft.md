# Optimized-vLLM checklist — Rule3 closure inventory

Optimization stopped under the explicit user cap at2026-11-14 05:07:49UTC; see [optimization_stop.json](optimization_stop.json). Remaining required guarded collection completed once and final audit passes. This self-check is not independent stage-review; review and local commits remain pending. Rule1/Rule2 are not claimed.

Selected contract: all32 layers, TP4/P300x2/mesh1×4, `bfp8_all_lofi`, context262144, integrated64 device split sampling, nonblocking decode traces, async scheduling, responsive1prefill:2decode, budget4096. Original/final labels: `stage10_final_before`/`stage10_guarded_after`; tracked=`stage10_guarded_tracked`. Primary128/128/1/concurrency1; CI100/100/32/concurrency unset.

Evidence: **G**=`selected_decoder_geometry_decision.json`; **H**=`selected_head_geometry.json`; **T**=`topology_audit.md`; **D**=accepted `../datatype_sweep/`; **X**=exact compact/prefill-refresh/attention-input full-stack controls, including4096/4105; **B**=`guarded_device/provenance.json`; **F**=`guarded_full_model_comparison.json`; **S**=`final_serving_audit.json`, guarded metrics/mixed, P4/P5 and qualitative review. README records shapes, values, scope and links.

| # | Optimize checklist line | Evidence-backed disposition |
|---|---|---|
| 1 | Warm traced real workloads; context/nonalignment | Context262144/nonalignment retained. X/B and final primary/CI, P5 and corrected mixed pass; exact workloads in README. |
| 2 | Topology audit and coherent collective families | Inherited coherent TP4 family retained (T). Final source/trace audit passes. Alternative collective families remain untried; no fresh ranking claimed. |
| 3 | Rank against best correct measured path; rerun default | Final matched128/128/B1 TTFT248.984→176.926ms (−28.94%); decode114.923→114.944t/s/u (effectively unchanged). CI100/100/32 throughput839.794→969.501tok/s (+15.45%). Guarded4096 selected. |
| 4 | vLLM no profiler; same-harness JSON | Serving profiler prohibition honored. Final benchmark JSON, source/config/route/input audit pass; no Tracy/tt-perf-report/device-profiler evidence used. |
| 5 | Shared qualitative suite | Twelve standalone outputs and final HTTP suite explicitly reviewed;67 sampling tests pass,75 skip. Existing cap/thermodynamics limitations retained. Owner blind story control0/10HF vs0/10HTTP,18/20capped; no population-equivalence claim. |
| 6 | Reduced-layer profiling; watcher separation | Serving profiling exception applies. Real layers0/3 are used for numerical/geometry controls without live-server profiling. No new profiler run is required. |
| 7 | Advice ledger and actionable trials | Inherited advice ledger retained; new trials address measured prefill interruption and host work. Final audited default paths retain exactness and improve TTFT/capacity as in line3. |
| 8 | Roofline/device/end-to-end accounting | Functional floor retained. Guarded128/128/B1 streamed full-model8.743032ms/token; served8.699881ms (ratio0.995065×). Resident126-step8.703898ms is separately scoped. |
| 9 | Short/long/chunk-boundary TTFT | Compact8192/head skip/input caching retained with X exactness. Guarded budgets512/2048/4096 worstITL994.8/972.8/947.6ms pass;4096 selected. P5 short/long/nonaligned timing passes. |
| 10 | Dominant matmul search tables | Keep production G/H winners: all ten selected-LoFi decoder roles,234rows/232passes/2pre-capture K3 constraints; selected-LoFi head production r3/K2 wins. Search is fixed-layout/readers1..4/decoderK≤16; larger grids/K are untried. |
| 11 | DRAM-sharded matmuls and reader controls | Keep production DRAM sharding/readers. G includes one-reader comparisons for decoder roles; H records one-reader L1 limits and slower legal alternatives. No blanket geometry optimality claim. |
| 12 | Packed versus split QKV | Keep existing packed projection based on inherited topology evidence and G selected-LoFi packed-input controls. No fresh split-family serving trial; not claimed rejected. |
| 13 | SDPA/KV contract; reduced KV dtype | D-selected BFP8 KV and paged SDPA/layout retained. B and final mixed/page-growth controls pass, including changed/unchanged/sparse mappings and permanent fail-closed page-copy handling. |
| 14 | Residual layout through norms/adds | BF16 local width-sharded residual contract retained (T). X/G/F and final source/decode audit pass; alternate layouts untried. |
| 15 | Logical batch versus tile padding | Logical1..32/physical32 decode retained; canonical16384/32768→8192 only when consumed rows fit. Nonaligned X/B/mixed/P5 pass;6144 rejected on exactness. CI/P4 pass at32users. |
| 16 | Cumulative contract/integrated sign-off | 160host tests, five guarded probes, exact X controls, F and final S pass. Stage gate has zero critical/errors and explicit degenerate pass; independent review/local commits remain required. |
| 17 | Row-parallel output decomposition | Keep inherited coherent-family winner in T; G measures selected-policy output projections. No new decomposition or collective family is claimed tested. |
| 18 | Persistent decode CCL buffers | Persistent CCL/trace inputs retained. Tracker positive control and B pass; final same-server mixed keeps six initial traces, zero recapture/retirement with chunked continuation and mode switches. |
| 19 | Packed/split gate-up | Keep separate decode gate/up and packed large-prefill family from inherited adapted comparisons. G adds selected-LoFi gate/up/down timing, not a new packed-decode ranking. |
| 20 | Phase-specific shards | Keep current phase placement and large-prefill numerical branch with X.6144 rejection proves sharing a branch alone is insufficient. Alternate shards remain untried. |
| 21 | Cross precision and geometry | Keep D-selected policy with current G/H reader/K controls. HiFi2 historical evidence is not substituted for selected LoFi. Alternative grids/repadding/decoderK>16 remain untried. |
| 22 | Group precision/fidelity; BFP4 | Keep accepted D real-model BF16/BFP8/BFP4 and fidelity trials. No new precision policy introduced. |
| 23 | Attention versus MLP precision | D separate-group decisions and G selected-LoFi attention/MLP controls retained. Final source and actual policy linkage pass. |
| 24 | Real weights/activations | Keep real target D and natural-text G/H/X controls. No synthetic-only veto underlies a new decision. |
| 25 | Selected policy reaches measured ops | D/G/H record actual policy; final audit matches all23 runtime files, copied/common/native sources, registration and actual default serving route. |
| 26 | Full-model head/sampling/feedback gap | Device split greedy sampling/persistent feedback/nonblocking replay/deferred read retained. B, borrowed-output/page-failure guards, F parity and actual per-window async/device route audit pass. Explicit host compatibility remains outside measured/load windows. |
| 27 | MoE sparse path | Not applicable: dense hybrid full/linear attention, no routed experts. |
| 28 | Evidence/final audit | Final guarded metrics, quality, raw P4/P5, forecast, limitations, corrected mixed and source/config/route audit pass. Gate advisories documented. Independent stage-review/local commit records pending. |
| 29 | Hardware safety/recovery | Serialized hardware ownership preserved; no profiling/reset. Final cleanup has no vLLM/EngineCore/device owners, plugin unchanged, no push. Local SHAs pending. |

## Stop rule and handoffs

Rule1 text: **“Every checklist line below has a measurement, a kept/rejected decision, or an exact blocker.”** This stage stops under **Rule3**, the four-hour wall cap from01:07:49 to05:07:49UTC, not a retrospective Rule1 claim. Rule2 is unsupported because candidate5’s mixed gains exceeded3%. User-authorized remaining gates ran once after the cap; no new performance candidates were launched.

Untried, with unknown incremental serving gains: alternate decoder grids/weight repadding/decoderK>16; new collective/residual families; more phase-specific shards; all-active sampler epilogue and compact slot-history construction; dense batched prefill; query-only full-attention padding reduction retaining projection/KV work. These are neither measured rejections nor promised speedups. Selected-LoFi head/decoder geometry searches were tried;6144 and responsive1:4 were rejected and are not handoffs.

Final source/client/workload audit passes with no errors/advisories. P4/P5, full-model parity, sampled/greedy quality, default async/device route and mixed trace retention are complete. Stage gate exit1 has zero critical/errors: historical512-token fixture mentions and optional denser-batching targets are explicitly advisory in `final_gate_disposition.json`. The15-request mixed stress test reaches1225.932ms maximum pause untracked; the separate31-decoder sweep’s1000ms bound is not a universal SLA. Review must assess these scopes and any nonblocking carry under the user’s cap instruction. Cleanup passes; independent review and stage-owned local SHAs remain pending, never push.
