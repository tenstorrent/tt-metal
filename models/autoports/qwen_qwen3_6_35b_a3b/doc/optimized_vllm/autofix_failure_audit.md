# Independent AutoFix failure assessment

Reviewer: existing source auditor `grouped_floor_audit`, read-only follow-up, 2026-11-17. This is an AutoFix diagnosis audit, **not** the fresh independent stage-review acceptance gate. No edits or device execution.

The evidence supports **AutoFix failed to meet the functional floor; stage incomplete**. The elapsed cap alone does not establish that outcome.

- Latest matched128in/128out/1user async31.44498609ms versus21.63129875ms required leaves9.81368734ms, or31.2091% further reduction. Native31.50643ms differs only0.06144ms; no evidence supports a substantial remaining adapter penalty.
- The supported small repairs were implemented and discriminated: route suppression, solo attention, route plan, state commit, row transport, packed scores, valid prefix, native Q/K and scalar-clear replacement. Full40 control preserves11 vocabulary comparisons and three96-token trajectories.
- ND placement and mailbox compaction gave0.14% and0.70% gains and were rejected.4x8 violates the native geometry contract.
- Packed-projection reader, fused tiled combine/postreduce, row-aware generic-topk-equivalent selector and recurrence/tail redesigns have no demonstrated sufficient attribution or exact minimal repair yet. They remain untried redesign hypotheses, not proof of exhaustion.
- Reconcile the completed hypothesis13 evidence and current HTTP result; record cleanup and all remaining serving/eval/review gates as incomplete.

Evidence: `AUTOFIX.md`, `native_qk_zero_fill_metrics.json`, `full_qk_zero_fill_comparison.json`, `remaining_grouped_dataflow.md`, `remaining_attention_dataflow.md`.
