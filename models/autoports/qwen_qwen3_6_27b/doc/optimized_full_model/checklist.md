# Optimization checklist and evidence status

All20 hard-gate jobs and the evidence collector pass. Independent stage review
returned clean-pass in round1; local checkpoint SHAs are in `commit_receipt.json`. Historical evidence
is labeled as inherited; completed package-mode probes retain their original
hashes through the exact one-line import equivalence proof, while readiness and
final mixed/host checks ran on the repaired source. See `hard_gates.json`,
`import_resolution_proof.json` and `runtime_audit.md`.

Paths below are relative to this directory. Detailed full-path shapes, collective
bytes and terminal alternatives are in [mesh_plan.md](mesh_plan.md).

| Optimize item | Verified starting evidence / decision | Stage7 disposition / remaining work |
| --- | --- | --- |
| Real warmed full path; context/nonaligned lengths | Stage6 all64 `../full_model/performance_final.json`, max context and mixed33/128/513 artifacts exist; bounded prefill chunks1..512 | `performance_final.json` passes P128/G128 and33/127/128/129/511/512/513/1025/32769; `readback_final.json` refreshes the original harness. Exact timing boundaries in README. |
| Operation topology audit | `mesh_plan.md`; `../optimized_multichip_decoder/topology_audit.md`, `final29_candidate_decisions.md`; no per-layer replication added | Complete resident and P128-trace paths measured. Terminal alternatives listed in mesh plan remain untried stop-rule handoffs |
| Rank against best correct default | Stage6 uninstrumented LM-head confirmation retains current head | Final resident30.999750t/s/u,86.850814ms TTFT; same-harness readback30.976371→30.956937t/s/u and87.031086→86.888559ms TTFT. Repeat ranges overlap; no material gain claim. |
| vLLM profiler exclusion | No vLLM work requested or performed | Not applicable; this is non-serving full-model profiling |
| Shared qualitative suite | Prior `../full_model/qualitative/{hf_suite,tt_suite,shared_suite_verdict}.json` | `qualitative_tt.json` and `qualitative_verdict.json`: all12 fresh TT histories equal stage6; all24 HF/TT texts read, matched template/settings/control reuse verified. |
| Reduced profiling, separate watcher | Prior layers[0,3] terminal reports under `../full_model/perf_terminal/` | `profile.json` passes layers[0,3]; `perf/{prefill,decode,logits_only,sampling}.{txt,csv}` and `command_receipt.json` retained. Separate `watcher.json` passes |
| Advice disposition | `../full_model/perf_terminal/profiling_analysis.md`; `../optimized_multichip_decoder/final29_advice.json` maps historical items | `perf/advice_rows.csv`, `perf/analysis.md` classify current rows; matmul/head/recurrence search remains stop-rule handoff where untried |
| Roofline/device/wall reconciliation | Prior per-rank kernel accounting excludes profiler-drain gaps; final29 known worker-count limitation retained | `perf_summary.json` reconciles reduced device projection and all64 wall regime with limitations;31.000t/s/u exceeds the6.4452 FUNCTIONAL floor. `perf/roofline.json` retains physical-byte derivation. |
| TTFT/chunk program table | Source keeps logical lengths and <=512 chunks; prior whole-model TTFT includes reset/first sample/read | `performance_final.json.prefill_rows` provides same-process controls through32769; `perf/prefill_kernel_selection.md` records chunk program selection. Only P128 uses new trace |
| Dominant matmul geometry / OPT-004 | `../optimized_multichip_decoder/candidate_search.csv`, `final29_candidate_decisions.md`, final29 profiles; stage6 head split/reader tests | Frozen decoder carried; new profile `perf/runtime_matmul_policy.csv` confirms runtime roles. Additional terminal geometry is untried stop-rule handoff |
| DRAM sharding / OPT-015 | Decoder eight storage cores/two readers; head one/two/three-reader designs and adapted L1 trials in `../full_model/perf_terminal/lm_head_probe_design.md`, `lm_head_decision.json` | Retain prior measured faster interleaved head; fresh profile shows1.55–1.57ms head. Broader head geometry untried stop-rule handoff, not family rejection |
| Packed QKV / OPT-001 | Selected packed QKVG and GDN primary/correction; `final29_candidate_decisions.md` and corresponding projection comparisons | Preserve; no split-projection fallback |
| SDPA/cache / OPT-002 | BFP8/page32/localKV1/head256; BF4 rejected in `../optimized_multichip_decoder/kv20_text_quantization.json`, `kv20_reference_provenance_correction.json` | Preserve exact dtype/layout and rounded-read-window allocation; no new KV frontier here |
| Residual layout / OPT-003 | `../optimized_multichip_decoder/final29_interlayer_contract.md`, source1280 hidden shard through stack | Fresh reduced `perf/layer_decomposition.csv` and `analysis.md` preserve1280-wide inter-layer shards; no replicated stream introduced |
| Logical batch/padding / OPT-005 | B1..32 low-level API, tile32 sampling; `../full_model/slots32_all64_pool_fixed.json`, mixed-prefill evidence | `slots32_tracking.json`, all64 `slots32.json`, `sampling_contract.json` and `mixed_all64.json` pass; inactive/remap/resident and mixed/nonaligned semantics preserved. |
| Cumulative signoff / OPT-006 | Stage6 full-model acceptance plus final29 decoder policy | `hard_gates.json` verifies all20 job exits and source lineage, AIME24 thresholds, context, runtime, trace, Watcher, all64 batch/mixed/resume and generated-text evidence. Independent review returned clean-pass (`stage_review_round1.md`). |
| Row-parallel families / OPT-008 | Selected RS/fused-MMRS thresholds; adapted sharded-next-consumer tests recorded in final29 ledger | Frozen decoder topology retained; terminal alternatives mapped in mesh plan are untried stop-rule handoffs where unmeasured |
| Persistent CCL / OPT-009 | `_collective` owns per-signature output and RS intermediate buffers; stage6 sample ownership repair clones borrowed output | `full_trace_tracking.json` all64 and `trace_contract.json` reduced full-shape battery pass with program-cache allocations included; persistent buffers remain enabled |
| Packed MLP / OPT-010 | Packed gate/up, fused packed SwiGLU in final29 default and ledger | Preserve selected family |
| Phase shards / OPT-011 | Accepted flat multi-row prefill and DRAM decode input configs; final29 geometry/flat evidence | Final full-path nonaligned lengths and reduced profile pass; extra long-prefill shard candidates are stop-rule handoffs |
| Precision x geometry / OPT-014 | Historical precision geometry in final29 decisions; current decoder policy fixed | Policy unchanged and verified in `perf/runtime_matmul_policy.csv`; further geometry/precision crosses deferred without rejection |
| Dtype/fidelity, OPT-007/012/013 | Runtime role configs visible in source; actual-HF HiFi4 repair; real-text BF4 KV rejection; profile dtype/fidelity rows | New runtime rows match preserved policy; no datatype frontier run. BFP4 Pareto/P1 selection remains datatype-sweep work |
| LM head, sampler, feedback | Stage6 split-greedy wins force-argmax, tie/changed-input checks; head alternatives slower; vocab masked before local32; persistent tt_out_tok | Fresh serial split0.512781ms vs argmax2.725168ms,200 iterations, both greedy-correct; `sampling_decision.md`. Final resident greedy/stochastic history/counters pass with no per-token host boundary |
| Decoder stack gap | `../full_model/layer_stack_lower_bound.json`31.055455ms, old full-model32.292ms with reads | New resident32.258325ms vs historical stack31.055455ms (~3.87% above stack alone). Reduced terminal+entry estimate2.113–2.119ms; no >10–15% unexplained gap. Workload caveats remain |
| MoE | Model is dense48 GDN +16 attention | Not applicable; no expert routing or EP plan |
| Full-path sharding/capacity | Mesh plan and unchanged context contract; prior all64 and B32 measured memory | Nonaligned32769 prefill/decode passes at C262144 with stable buffers and memory; all64 B32 context1024 regression passes. Prior maximum-context/pool proof retained; no dtype/layout/capability reduction. |
| Runtime fallback | Prior `../full_model/runtime_audit.md`; explicit host compatibility separate | `runtime_tracking.json` passes executed public-transfer sentinels and positive control; `runtime_audit.md` documents scope; final resident counters remain clean |
| Trace lifetime / tracker | Prior `../full_model/trace_contract_native.json`, positive control log | `full_trace_tracking.json`, `trace_contract.json`, `trace_positive_control.log` pass; four captures retained, no retirement. Final all64 before/after24456 programs and stable trace/input identities |
| Serving-ready state | Explicit cache/table/positions/batch; `../full_model/serving_contract.json`; recurrent-state prefix-cache exclusion | Existing serving contract preserved. `mixed_all64.json`, `prefill_resume.json`, `sampling_contract.json`, `slots32.json` and `host_compatibility.json` pass. No vLLM work. |
| AIME24 accuracy | Prior `../full_model/accuracy_native.json`: prefill100/100/100%, decode99/100/100% | `accuracy.json`: prefill100/100/100 and teacher forcing99/100/100 at top1/5/100, same as stage6; top5>=98%, top100=100%. TF31.04t/s/u has its own boundary label. |
| Autoregressive/degeneracy | Prior128-token readiness and shared suite outputs | `autoregressive_verdict.json`:128 TT tokens exactly equal HF and stage6. Shared suite passes observational review; `stage_gate.log` finds no degeneracy. |
| Final review and local commits | Prior stage6 clean-pass is only stage6 evidence | All hard gates pass; fresh independent round1 clean-pass. Both housekeeping notes closed; local stage-owned checkpoint recorded in `commit_receipt.json`, never pushed |

## Stop Rule2 and retained changes

The final same-process `performance_final.json` comparisons are:

| Retained change | Measured medians | Interpretation |
| --- | --- | --- |
| B1/P128 first-chunk trace | Eager86.855779 -> trace86.650346ms TTFT;0.2365% lower | Small observed change; three repetitions do not establish a robust speedup |
| Cache identical sampling parameters | Disabled86.730679 -> enabled86.746810ms;0.0186% higher | No measurable latency gain; keeps equal outputs and eliminates one parameter refresh per repeated request |
| No-readback low-level token-out |32.258325ms/step,30.999750t/s/u,127 steps | Required device-boundary contract fix; no claim that removal of reads creates a large gain |

The last two retained optional changes each yield less than3% improvement,
including the cache's neutral/slightly negative median. This satisfies the
Stop Rule2 effort bound; it does not establish two statistically
significant gains. Required correctness, accuracy and final performance reporting now pass.
Independent review returned clean-pass. No wall-cap or exhaustive-checklist
claim is needed to justify this stop decision.

At non128 lengths, disabling the helper selects exactly the same eager code.
Their paired timing variation (including513) therefore is not attributed to a
new kernel or claimed as a gain. Keep all samples in `performance_final.json`.

## Not tried (stop rule)

Expected whole-model gains below are unmeasured unless stated. These are handoffs,
not failures or empirical rejections; preserve the existing safe defaults.

- Terminal distributed norm followed by ordinary AG, and distributed norm plus
  fused AG-matmul: adapt actual BF16/H5120/N62080 head packing and persistent
  buffers before comparing. Final norm currently costs about31–37us/token;
  its standalone removal alone has a small full-model ceiling.
- Hidden-sharded terminal head with vocab reduce-scatter, or full-vocab partial
  all-reduce: modeled communication is much larger than the selected hidden AG
  plus candidate exchange; neither family was measured or rejected here.
- Further unchanged-policy LM-head geometry, larger legal K blocks, core/output
  subblocks and one/two/three-reader adaptations beyond the stage6 tested
  candidates. Stage6's specific slower/admission-failing candidates remain
  rejected; the entire geometry family does not.
- Reducing full-cache reset work, avoiding duplicate fresh GDN clearing, and
  direct state import/publication changes: preserve explicit reset semantics,
  stale-page safety and rounded attention windows before benchmarking.
- Additional traced prefill buckets, trace reuse across chunk starts, multi-row
  tracing, larger internal chunks and alternate long-prefill kernel geometries.
  Logical lengths remain unrestricted on the current eager fallback.
- An all-active B32 trace that omits inactive GDN state preservation, and native
  masked-state updates for mixed slots. Retain the tested inactive-row behavior.
- Additional terminal/sample buffer reuse to remove clone/layout staging while
  preserving common sampler ownership; no custom sampler or full-vocab gather.
- Earlier decoder handoffs remain untried: ephemeral/DiT fused-norm scratch for
  64 owners; phase-specific GDN HiFi4; lower-movement prefill norm; prediction,
  column and live-row recurrence combinations; BF16-DRAM joint combine/conv;
  long-prefill input-L1, row-cap, concat/window and geometry crosses under selected
  fidelity; QKV batched/fragment movement; SwiGLU bulk/tail configs; attention-gate
  whole-layer combinations; broader persistent-buffer signature capacity. Exact
  predecessor evidence/manifests are in
  `../optimized_multichip_decoder/final29_handoff.md` and `final29_candidate_decisions.md`.
- Broad precision/fidelity/KV/CCL Pareto exploration and BFP4/P1 selection belong
  to the explicitly deferred datatype-sweep stage, not a failed stage7 search.

Complete-model gates are **not** stop-rule handoffs: all now pass in
`hard_gates.json`. Independent review returned clean-pass; local checkpoint SHAs are in `commit_receipt.json`.
