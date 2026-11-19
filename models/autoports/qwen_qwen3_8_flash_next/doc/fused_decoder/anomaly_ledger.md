# Final anomaly ledger

Historical logs and dated AutoFix entries retain their original status. This ledger records the current resolution; it does not rewrite raw failed gates.

| Observed anomaly | Control / investigation | Final resolution |
|---|---|---|
| Native RMSNorm changed routed expert membership and full-layer PCC | AUTODEBUG.md, norm_diagnosis_v1.json, original/partial-norm controls | Rejected native norm; selected graph preserves original norm arithmetic. Partial fusions measured and rejected when slower. |
| Native GDN convolution missed original BF16 boundaries | autofix_ci/AUTODEBUG.md; conv exact component and synthetic/HF controls | Fixed with exact SFPU product/pairwise reduction and explicit BF16 boundaries; final synthetic14/14. |
| Accurate QKV sum failed a batch32 request | autofix_batch/AUTODEBUG.md; exact_qkv component and batch controls | Fixed with original half-tree order in the fused product/reduction. Final QSA batch32 and integrated GDN batch32 pass. |
| Grouped and packed setup fragmented fully resident PLE memory | autofix_capacity and autofix_packed_capacity; matched functional4768 control and allocation ledger | Direct checkpoint grouping/unpack widening and early packed activation allocation restore4768/4767; integrated writer repeats pass. Capacity unchanged. |
| zeros_like(dtype=BF16) uploaded from host during capture | autofix_moe_cleanup/AUTODEBUG.md and live triage evidence | Replaced with device empty+fill. Repeated trace and Watcher gates pass; no reset needed. |
| Native trace-profile harness assumed old projection.block | autofix_trace_protocol/AUTODEBUG.md | Harness validates either original blocks or actual grouped weight shape; strict baseline/fused native report controls pass. Final per-op reports use measured kernels. |
| PLE34 row5 and QSA34/66 rows30/44 fall below0.995 | wrap_accuracy.json; matched literal, functional, packed-lookup HF and GR accumulation HF controls; autofix_ple_tail and autofix_qsa_tail | Controlled under owner whole-output gate: all whole outputs>=0.995, rows>=0.99. Original/new fused tensors byte-identical. Exact control scope recorded in eval_readiness/limitations.json. No token-logit claim at decoder-only stage. |
| General compact union/threshold3–4 slower | compact_tail_decision_v1.json and real decoder timing matrices | Rejected; selected threshold2 improves relevant tails, and does not change32-token/decode1 graph. |
| Profiler per-op interval totals differ from uninstrumented host replay | Final profile JSON identifies segmented prefill vs uninterrupted decode; final_performance.json records clean timing | Different measurement regimes explicitly separated. Segmented prefill gaps are not a latency claim; no70%roofline assertion. |

Host reports describe one visible chip on a P300 board and an unknown motherboard mapping. The configured P150 mesh, expected visible chip0, successful open/close, repeated tests and Watcher establish the intended single-chip regime. No device availability/corruption failure remains open.
