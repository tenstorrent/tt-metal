# Stage5 optimization evidence checklist

Final evidence collected; independent review and commit pending. This is not stage acceptance. Exact invocations, environments, source
hashes and exit statuses live in each `*_provenance.json` and `plans/*.json`.

| Requirement | Evidence / disposition |
|---|---|
| Target four-chip path; TP/EP ownership | All `mesh` jobs use1x4 p300_x2 ring; original accepted TP attention/EP4 active-expert partitions retained |
| Starting operation-topology audit | `operation_topology_audit.md`, before knob tuning; stage4 advice CSVs read directly |
| Fresh before timing, short/chunk boundary/long | `before_l*`, `before_mesh_l*_n32`, `before_mesh_l*_n1024`; real routing `real_mesh_before_l*` |
| Real inputs and largest supported batch | `real_prompt_fixture.json`;32 unique checkpoint embedding rows. Layer0 actual HF embedding input, later kinds routing stress, not full-stack activations |
| Exact sensitive projection optimization | Parallel indexer bit-exact BF16/FP32 Q1..32 six HF windows; `parallel_indexer_all_counts_v3`. Target-mesh whole-layer candidate passed |
| Shared same-input gate/up packing | `shared-*` native precision comparisons; DRAM packed/split geometry and phase-specific real-decoder tests |
| QKV packed/split | Already packed default; `dram_geometry_v3_qsa` and precision sweeps compare both on local real weights, including split costs |
| GDN Z/A/B packing | First DRAM group added costly intermediate padding/slicing. `dram_compact_gdn_l0` adapts packed weights to contiguous logical1560 output and removes compaction; real decoder passes |
| Async exact-order CCL | `async_l0`, `async-persistent_l0`: same PCC, slower than synchronous control |
| Persistent collective outputs | Async persistent explicit buffers/semaphores measured; fused native families prepare output buffers; not accepted solely from API availability |
| CCL precision | Global BFP8 async transport slower; real MoE/QSA subgroup trials passed whole-output PCC with less than0.2% gain; retain BF16 |
| Collective placement / output decomposition | `fused_family_HiFi4/LoFi`: native/fused RS and native/fused AG-MM with residual write + entire next GR |
| Lower-movement residual layout | `hidden_qsa_rs/agmm` carry four local640 stream slices through both GR blocks and MoE. Full QSA B32 slower by1.90%/1.77%, PCC passes. No immediate restore to old layout |
| Inter-layer contract | Retain one full BF16 stream/rank, local[1,1,T,2560], DRAM TILE; no boundary gather, reshard or reduction |
| DRAM-sharded decode / reader counts | `dram_geometry_v3_*`: five roles,8/16 storage shards,1/2/3 readers, legal blocks. Physical block40 L1 overflow logged; smaller blocks adapted and measured |
| Activation sharding | DRAM group includes actual L1 width sharding and output restore cost; configured independently for Q1 and Q2..32 |
| Precision crossed with geometry | `dram_precision_b{1,32}_*` plus real role trials completed. `measured_policy_rows.json` verifies42 selected policy rows on all4ranks: dominant DRAM HiFi4, remaining SDPA HiFi2 |
| Reduced KV dtype | BFP8 exact selected-row component and real full-layer trial pass whole-output PCC; B1 regresses12.09%, retain BF16 KV. No BFP8 maximum-context acceptance claimed |
| Routed experts / MoE share | Active gate-selected execution retained. Real fixture baseline B32 expert unions103/151/169; all32 routing patterns distinct. MoE timings are isolated traces of actual decoder intermediates |
| Nonaligned public sequences | Final tracked logical1..32 tails and33 prefill pass; final clean32/1024 and full public32769 smoke evidence collected |
| Final default / tracing / fallback / watcher | Final default selected; final_mesh_l{0,1,3} tracks changed B1/B32 inputs and exact output/state with runtime audit; separate watcher_mesh_l* logs clean on all4devices; positive control rejects deliberate live allocation |
| Context memory | BF16 KV/stream ownership preserved; final_context_memory_plan.json and long32769 target-pressure reservations recorded. Reservation is model/static/state pressure, not an instantiated full model or physical2GiB runtime envelope |
| Profiler advice | Final all-kind prefill/decode tables/CSV/raw ops/provenance complete. profiler_findings.md maps every advice clause; measured_policy_rows.json verifies actual fields. Headline timing remains separate clean traced runs |
| Independent stage review / commit | Pending; no stage acceptance or push |
| Final performance summary | real_mesh_final_l{0,1,3}.json and final_acceptance_summary.json contain same-harness final-default before/after PCC/perf and B1/B32 MoE shares; collector evidence_ready true, issues0 |
| Stop rule / untried handoff | Stop Rule2 recorded: last retained QSA refinements0.5865% and1.2193%. stop_rule_handoff.md plus exact final residual-role advice in profiler_findings.md; review still pending |

Full-model-only items (LM head, sampler,128in/128out generation throughput,
qualitative text suite, serving) are outside this decoder stage. Their nearest
applicable evidence is complete representative decoder prefill/decode at B1/B32,
accepted whole-output PCC, mutable-state parity, real routing and nonaligned
context smoke. No full-model FUNCTIONAL bandwidth or text-quality claim is made.


## Final evidence snapshot

Selected defaults: compact DRAM GDN and QSA packed projections, DRAM GR
injection, packed DRAM shared gate/up, DRAM WO, remaining QSA HiFi2, parallel
exact indexer and PLE lookup32. BF16 activation/KV and stream ownership remain.
Final collector reports `evidence_ready:true`, `issues:[]`, `stage_complete:false`.
Full-output0.995 remains the gate; per-row PCC is diagnostic. Historical GDN
LoFi failure was a legacy diagnostic-row check, not whole-output rejection.

The final profiles verify42 selected policy rows across all four ranks. Eager
prefill/decode summed device milliseconds are21.034/1.937 (L0),25.650/2.874
(L1),40.692/2.625 (L3); these are attribution sums, not traced headline latency.
All raw ops, advice-enabled human tables/CSVs and provenance exist.

Exact final-profile Stop Rule2 handoffs: real GR-only HiFi2; shared-down and
shared-route L1 input/subblock/fidelity; active-expert weighted-sum padding/K block,
L1/subblock/fidelity; PLE same-embedding key/value packing plus L1/subblock/fidelity;
additional prefill scheduling beyond eager drain gaps. Their row references,
bounds and prior related experiments are in profiler_findings.md. Do not call
these exact residual-role configurations measured merely because dominant roles
were swept. Further hidden PLE/GDN and packed-only expert extensions remain
performance handoffs, not failed gates.

Final tracked1..32 tails, stacks without boundary conversion, diverse B32
changed-input exact replay and fallback guards pass; separate all-device watcher
logs are clean. Long32769 PLE/GDN and QSA tests pass with selected-memory
reservation; the PLE case covers the shared GDN implementation. The deliberate
live-allocation positive control is rejected and replay succeeds after freeing it.

Independent stage-review, required findings/fixes/rereview and local stage-owned
commit SHAs remain pending. Collector readiness and this checklist do not mark
the stage complete. No push is authorized.

## Independent review closure

Round 1 returns `clean-pass`, no Required Work (`stage_review_round1.md`). Local
checkpoint metadata is recorded next; all technical gates are complete.

Stage completion: local implementation/evidence checkpoint `d103c86fc35f59f9d10f1e73daef969f92ffe258`; independent
review `clean-pass`. All mandatory stage gates complete; no push.
