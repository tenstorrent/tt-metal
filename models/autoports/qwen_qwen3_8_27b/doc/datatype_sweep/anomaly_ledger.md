# Investigated anomalies

| Observation | Evidence / investigation | Resolution |
| --- | --- | --- |
| BFP8 projections overlap live L1 | AUTODEBUG.md, AUTOFIX.md; exact triple-buffer budgets and real-weight alternative-reader checks | Dtype-dependent legal K/reader geometry; full64 LoFi and HiFi2 pass, rejected on speed |
| BF16 CCL exposes ordinary down/mix and later prefill mix overlaps | Preserved failures, four-layer/full-head reproduction, geometry_alternative_review.md | Only observed affected geometries adapted; boundaries/replay/full64 pass; rejected on speed |
| Very slow native BFP4-KV preparation | Interrupt traceback in reset/zeros_like; creation.cpp host packing path; kv_reset_review.md | BF4-only in-place device fill; exact production-shape all-rank zero/address proof, real replay and full64 pass; initial attempt has no ranked metric |
| Native BF16-KV initial SDPA overlap | AUTODEBUG_kv_bf16.md derives exact1083648 endpoint and18304-byte overlap | BF16-only K128, preserved native map/divisor guard;11 real-weight controls, native-end replay and full64 pass; rejected on measured speed/agreement |
| Cache PCC failures in two reduced comparisons | Failures preserve BFP8-geometry versus BFP4-control and BF16-KV partial-policy fallback-weight mismatch | Explicit same-weight/cache-precision controls pass unchanged.995 threshold; no output substitution or threshold relaxation |
| Small-context smoke did not expose native SDPA failure | Actual old capacities256/4352, corrected after source review | Native-sized maps and actual final position now tested; no claim that small smoke proved native admission |
| Initial reset primitive used equal-volume flattened shape | Initial1024x256 versus production8192x32 page rows | Reran exact production shape with explicit four-rank count; real generator/full64 always used production shape |
| HF CPU reference lacks optional fast linear-attention libraries | agent_long_hf.log; canonical HF torch fallback implementation | CPU-only reference completes exact factual retrieval; no HF timing copied into TT performance |
| CCL deprecation, packet-size and L1-semaphore warnings | Local candidate logs; inherited same build/collective setup | Calls execute successfully; allocation/tracing checks remain enabled. Numerical policies are compared with the same setup. Actual L1 failures above were investigated rather than dismissed as warnings |

All TT commands are serialized. Failed allocation attempts close the mesh normally;
no ARC/ERISC failure, hang, device reset or capability reduction occurred. The CPU
HF control overlapped only untimed preparation/reduced diagnostics, not any
accepted full-model throughput measurement. cpu_control_finished.json records
successful process exit before the subsequent measured candidates.

Qualitative output findings and their same-prompt controls are recorded separately
in qualitative_comparison.json and qualitative_verdict.md after selected validation.

## Free-running precision quality regressions

Observed anomaly: high AIME token agreement did not ensure a final answer within
the fixed qualitative budgets. Head BF4/LoFi with BF8 KV stayed in thinking on
the4096-token narrative through1024 output tokens. Both BF4-KV/head candidates
failed to provide final structured JSON within128. The BF4/HiFi2 head with BF8 KV
also remains in thinking through1024; BF8/LoFi head truncates its final JSON at128.

Evidence: `quality_candidates/` preserves each rejected policy's actual outputs
and verdict or `quality_gate_failure.json`; `selection_state.json` records reasons.
Affected path: full64-layer free-running generation, normal default precision loader.
Controls: exact same prompt tokens, HF and prior TT; shared structured HF128
produces correct4/3 JSON and baseline TT114 reaches EOS. Narrative HF512 and
previous TT1024 both reach final-answer text (budgets differ explicitly).
Likely subsystem: accumulated precision/fidelity effects on autoregressive choices.
Investigation: full-model readiness, exact prompt provenance, output inspection,
matched short budgets and fixed long budget. These are quality-based selection
rejections, not crash blockers or proofs of arithmetic errors.
Resolution: rejected policies; continue descending traced full-model speed with
restored precision/fidelity. head_bfp8_hifi2 passes the shared suite and all four long cases; see qualitative_verdict.md. Final token-out/memory/context gates pass; see performance.json and memory_capacity_plan.json.

## Generic allocation warning during clean timing

Observed anomaly: B32 timing emits "Allocating device buffers is potentially
unsafe due to the existence of an active trace" after preparation.
Evidence: selected/benchmark_b32_perf.log and
`tt_metal/impl/allocator/allocator.cpp:verify_safe_allocation`.
Affected path: clean timing with trace allocation tracking disabled.
Control or comparison: selected all64 tracker-enabled default validation, plus
stage07 runtime_audit.md and tracker-enabled B32 full/mixed/full controls.
Likely subsystem: generic allocator warning (once/thread), not a tracker finding.
Investigation: source emits on any allocation while allocations_unsafe_ is set,
except trace-storage context. Request eager-prefill temporaries are retired and
terminal logits copied before trace replay. Selected head weights are allocated
before capture; output dtype/shape, request buffers and lifetime ownership are
unchanged. No corruptible allocation exemptions or broad scopes were added.
Resolution: controlled by source/lifetime inspection and tracked validation;
selected B32 all-active/masked allocator audit passes with exact reconciliation and zero additional live bytes; see selected/benchmark_b32_perf.json.
