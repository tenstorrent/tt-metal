# Full-model multichip / optimization evidence map

|Applicable item|Implementation and evidence|
|---|---|
|Target topology and capacity|TP4, mesh1x4, four Blackhole p300c ASICs, ring/two links for decoder; topology_audit.md. Agent target32x32768, native context262144 remains explicit. Final allocator reconciliation is produced by summarize_optimized_full_model.py.|
|Embedding and full-stack ownership|model.py: BF16 embedding hidden-sharded1280/device, vocabulary-sharded BF16 head; all64 layers use the same selected decoder. Benchmark inventory includes all weights and request buffers.|
|Inter-layer residual layout|Unchanged multichip_decoder.py: B1-7 replicated L1, B8-32 fractured1280/device L1. No conversion inserted between layers. Stage05 residual_contract.md and stage06 rejection_ledger.json remain authoritative.|
|Decoder CCL / persistent buffers|Shared pooled workspace retained; native RS and accepted BFP8 transfer policy unchanged. Stage06 B1 prefill outK4 repair retained, downK17 unchanged. Current reduced profile attributes must match these values.|
|Norm / terminal layout|Final norm stays HiFi4/FP32 requested. Decode user packing retained; prefill last-token packing added. prefill_terminal_b32.json proves bitwise33/65/4097 and inactive-row logits. All-logits prefill preserves existing layout.|
|Matmul program geometry / memory / kernels|head_candidates.csv records19 real-hidden controls including DRAM-sharded and generic, reader/core/K/chunk/subblock variants. Large K adaptations executed after L1 failures; legal candidates lose end-to-end terminal latency. Existing BF16/HiFi4 generic head retained. No new kernel required. Decoder geometry/fidelity selection unchanged and checked in reduced runtime rows.|
|Greedy and sampled output|Canonical Sampling1D, sampler-ready sharded logits, power-of-two local padding,32 local candidates, tt_out_tok feedback. sampling_* controls compare semantic greedy against force-argmax and candidate-gather modes. B32 async selected; B1/8/16 baseline faster. No full-vocabulary all-gather selected.|
|Trace / host boundary|Persistent token/position/RoPE/page-table/collector inputs; nonblocking split model/sampler replay. buffered_final_b1.json, buffered_final_b32_transition.json and reduced_trace_contract.json include exact sequences, tracker/program-cache allocations and stable captures. runtime_audit.md maps intentional request/output boundaries.|
|Mixed prompts / fixed slots / page mapping|Reduced trace/mixed probes plus all64 benchmark mixed_trace_validation. external_prepare_reduced_b32.json verifies the changed trace set and failure cleanup. No scheduler-owned cache/table replaced.|
|Awkward lengths|Packed terminal33/65/4097 controls; full64 shared prompts, mixed33/65/1 and long4095; inherited native262141+3 and exact262144 capability evidence retained. No alignment-only public restriction.|
|Accuracy / qualitative|Final prefill_accuracy.json and teacher_forcing_accuracy.json: top5/top100100%. qualitative_comparison.json proves exact prior-stage outputs for all shared/long cases; HF controls separately scoped.|
|Performance / lower bound|Final benchmark_b*_perf.json, B1first then32/8/16;4096/128,3 warmed repeats. Teacher-forcing and EOS controls separate. layer_stack_comparison.json computes48linear+16full from optimized multichip layer medians; reduced profiler gives terminal attribution, not full-model device time.|
|Profiler / fallback / watcher|Reduced real layers0/3 plus full terminal only; profile_optimized_full_model.sh produces advice-enabled tt-perf-report tables, CSVs and provenance. Benchmark and collector probes throw on fallback. Watcher is a separate diagnostic, never timed/profiled together.|
|Precision / capability constraints|No decoder policy or capability reduction; no broad full-model datatype frontier, no vLLM work. rejection_ledger.json keeps controls and inherited rejection references.|

All referenced implementation/measurement gates are complete. Independent review
returns clean-pass in stage_review.md; local checkpoint receipts are in work_log.md.
