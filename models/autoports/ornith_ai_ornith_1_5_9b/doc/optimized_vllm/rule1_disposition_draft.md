# Optimize stop-rule disposition — Rule3

[optimization_stop.json](optimization_stop.json) records the explicit user cap at2026-11-14 05:07:49UTC, four hours after stage start01:07:49. Optimization stopped; only remaining required gates ran once. **Rule3 is the closure basis. Rule1 is not claimed. Rule2 is unsupported**: candidate5’s kept mixed gains exceeded3%, while rejected6144 and slower geometries are not kept candidates.

All29 checklist lines are reconciled in [checklist_draft.md](checklist_draft.md). Final guarded source-bound primary/CI/burst, full-model parity, sweep/P4/P5, sampling/quality and tracked/untracked mixed evidence pass; [final audit](final_serving_audit.json) has no errors/advisories. [Gate disposition](final_gate_disposition.json) has zero critical failures/errors, explicit degenerate-output pass, and carries historical fixture mentions plus optional stronger dense-batching targets as advisories. Independent stage-review and local stage-owned commit SHAs remain pending. The cap does not substitute for these requirements; nonblocking review carries require the user-authorized review disposition.

The measured path keeps the accepted precision and inherited TP4 topology. Final128/128/B1 TTFT improves28.94%, decode is effectively unchanged, served/full-model ratio0.995065×; CI100/100/32 output throughput improves15.45%. This supports the retained source, not global geometry/topology optimality. Corrected mixed trace reuse passes; its15-request untracked maximum pause1225.932ms is reported separately from the31-decoder budget-sweep bound.

## Untried handoffs — gains unknown

- Alternate decoder input/output grids, weight repadding and decoderK>16.
- New collective/residual families and additional phase-specific shards.
- All-active sampler epilogue and compact slot-scoped history construction.
- Dense batched prefill, subject to unchanged decode/trace retention and batched-equals-solo evidence.
- Query-only full-attention padding reduction with existing projection/KV work retained.

No speedup, failed trial or ranking is invented for these items. Selected-LoFi head and ten decoder-role reader/K searches were completed and production retained. Compact6144 failed exactness; responsive1:4 failed to improve latency and worsened32KTTFT. They are rejected trials, not untried handoffs. Full details and commands remain in the work log; cleanup confirms no active owners/processes and unchanged plugin.
