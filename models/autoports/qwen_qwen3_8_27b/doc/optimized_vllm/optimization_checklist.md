# Serving optimization checklist

Status: complete; independent stage_review.md returned clean-pass. This checklist is scoped to the serving change;
previous decoder/full-model optimization is not claimed as newly measured.

| Relevant optimize requirement | Evidence / disposition |
|---|---|
| Operation topology before local tuning | topology_audit.md; intermediate terminal work identified from actual adapter→generator call sequence |
| Selected precision reaches serving | Exact stage08 selected_precision_config.json consumed by Model; server AUTOPORT_ADAPTER and prepared event serialize policy. No dtype changes |
| Preserve optimized decoder residuals, projections, CCL, matmul geometry/readers, SDPA | No decoder changes. Inherited optimized_multichip_decoder and optimized_full_model topology/kernel audits plus datatype_sweep runtime evidence; serving profiles prohibited |
| Trace decode and canonical split greedy sampling | Existing generator decode/sample traces and tt_out_tok resident feedback retained; nonblocking execute_trace in generator.py |
| Async boundary and minimal read | Adapter returns device output when read_from_device=False; read_decode_output selects one token replica, asynchronous CPU transfer, plugin event wait then host formatting |
| No greedy host argmax/full-logit read in performance | Compatibility route requires explicit opt-in; performance manifests must omit it; shutdown/event counters must show zero host_logits_reads |
| Persistent state, changed-only refresh | Decode path unchanged; prior real allocator-growth controls plus current representative repeated trace control. Candidate only changes eager prefill terminal work |
| Non-aligned logical prompts/chunk boundaries | terminal_suppression_device.json:128+128+1,64+64+65,123+6,128+1; after_b32_lifecycle.json passes112 mixed requests |
| Allocation tracking/program-cache inclusion | terminal_suppression_device.json; tracking1 and skip_program_cache0; no new exemption/scope/buffer lifetime |
| Correctness and inactive-slot preservation | Reduced exact logits/token/3decode/cache comparisons,13 host contract tests;76 full sampling tests pass |
| Warmup/preparation | Existing generator-owned preparation of terminal-capable superset; no new compile signature or allocation introduced |
| Before/after primary and CI JSON | comparisons.json: matched primary_manifest_comparison.json and b32_manifest_comparison.json; all before/after counts verified. Headline4096/128/B1/concurrency1; CI100/100/32 secondary |
| Device/profile time and roofline | Intentionally null: no vLLM profiling permitted. No new Tracy, tt-perf-report, adapter profiler or ReadDeviceProfiler |
| Watcher | terminal_suppression_watcher_noeth.log passed compute watcher10; Ethernet kernel-capacity limitation documented |
| Qualitative shared suite and controls | qualitative_verdict.json:18 outputs reviewed,9 greedy exact full-model matches; explicit256-token cap/think-parser limitations |
| Batch1/8/16/32 configurations | batch_configurations.json tracks each final validation |
| Agentic serving and mixed-traffic trace lifecycle | after_b32_lifecycle.json passes; current doc/agentic_serving.json refreshed; rejected snapshot alternative recorded |
| Final default, cleanup, review and local commits | cleanup.json and stage_check_verdict.json complete; stage_review.md clean-pass; local commit receipts in work_log.md |

MoE sparse-matmul requirements are not applicable: this checkpoint is dense hybrid
linear/full attention. Device-level precision/topology sweeps belong to completed
stages05/07/08 and are unchanged; no serving profiler is used to repeat them.
