# Optimize checklist assessment

Selected delivery is [delivery194](delivery194_manifest.json), derived from
candidate193 and frozen at SHA256
`315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
The optimization stop rule is met and all58 final-source suite jobs plus4
pytest cases pass (final_evidence_check.json). All times are one real text decoder layer on physical
Blackhole chip 0; no full-model claim is made. The initial topology audit is
the first table in `work_log.md`. Historical experiments below explain selection;
they do not substitute for the hash-bound `final194_*` gates.

| Requirement | Action and evidence | Decision |
|---|---|---|
| Real shapes, weights, batch and traced timing | Pinned revision in every result; integrated193_l0_{b1,b32,sequence}.json; final194 performance/correctness suites | B32 long decode is primary; B1 is regression coverage; complete final194 suite passed |
| Non-aligned lengths and context | final194_correctness_* covers contract, mixed positions, odd batches and prefix continuation; final194_context_context_l{0,3}_s{262143,262144}.json | Context suite has four zero-exit records; correctness suite passed; preserve 262144 |
| Runtime topology and host fallback | Initial audit; audit_runtime.py; run_contract.py replaces runtime from_torch/to_torch with errors | No host conversion in measured path; setup/reference IO is outside timing |
| Best correct candidate | integrated193_l0_sequence.json: minimum per-user PCC .9955427456 across 512 steps, traced median 1.597208 ms; delivery194_manifest.json | Frozen delivery194 selected; all final-source hard gates passed |
| Advice-enabled profiler | perf/profile195_bfp8_0 and perf/final194_profile_*/{prefill,decode}_perf_report.{txt,csv}, metrics.json and ops.csv.gz | Selection profile complete; all8 final194 profiles rendered and checked |
| Same-run performance accounting | account_decoder_perf.py; stop_rule_selection.json; perf/profile195_bfp8_0/accounting.json | 1.123648 ms roofline / 1.602493 ms same-run host = 70.1187465%; physical stored BFP bytes and necessary state/KV traffic at one-chip 512 GB/s |
| Prefill/TTFT short, long and boundary | final194_performance_*, final194_profile_* and final194_longprefill_*; earlier longprefill136 and primary181 controls | All final suites pass; optimization stopped, no ongoing tuning |
| Matmul program configurations | search_{hifi2,lofi,pipeline}_0.json, geometry_pipeline_*.json; geometry119/120/123 actual corrected-input pair sweeps | Per-role shapes, readers, core count, K blocks and fidelity measured |
| DRAM-sharded decode and Blackhole readers | One/two/three-reader initial controls; packed input C32/R4 then C40/R5; MLP pipeline R2 | Tuned role-specific values, not generic one-reader defaults |
| Packed versus separate attention | rowchunks_*; separate75 corrected QKV/Z/B/A wide32/narrow8 or4 | Corrected separate is slower and fails long PCC .994393; packed input retained |
| Packed versus separate MLP | rowchunks packed MLP slower at B1/B32; swiglu_pre_* phase-specific native composite | Separate decode gate/up; fused tile-pair SwiGLU prefill |
| Corrected input packing | pair99 and pair133/135 controls; narrow correction and zero-Z selection in integrated167/193 | Separate primary/residual projections retained; selected correction omits proven zero-Z residual columns and uses fused combine |
| Phase-specific activation shards | projection input geometry119/120/123; C64/shard96 repaired to legal C40/C80; explicit dimensions | C40/R5 input wins; output C32/R4; no logical shape restriction |
| Precision crossed with geometry | search_hifi2/search_lofi/geometry_pipeline and profile80 BFP8/BFP4 | Actual runtime dtypes/fidelity shown in CSV |
| Attention precision separate from MLP | precision_* and residual93; corrected input-only phase98/singleout105 | Two BFP8 input terms/HiFi4 required for long accuracy; single output BFP8 |
| MLP BFP4/LoFi candidate | precision_bfp4_* and perf/profile80_bfp4_0; real checkpoint weights and recorded activations | Default stays BFP8; stage8 reasoning-text probe decides final lower precision |
| MLP fidelity | Initial LoFi/HiFi2; mlp138 decode LoFi fails long .990742; phase139 prefill-only LoFi preserves long PCC but is not the selected default | Delivery194 GDN MLP HiFi2; full-attention MLP LoFi; no pending precision experiment |
| Residual/norm layout carry | norm159 full-attention projection handoff; norm179 L1 scratch/output partition; norm184 exact zero-mean probe; kernel_contracts.md | Compact residual; full-attention norm-to-projection L1; GDN exact-tree hidden norm with L1 scratch, 80 output cores and left-face inverse reads |
| SDPA and KV dtype | Native paged/standard SDPA; full113 B1 classic grid8x4, B32 paged grid8x8; BF16 KV legal8x4 repair vs BFP8 | BFP8 KV retained; full context actually validated |
| Optimized composite operations | delta168 exact recurrence; affine170 v3/core64; norm184; beta192 repaired row-only sigmoid/softplus; conv untilize pipeline; native prefill convolution/GDN and SDPA | Selected exact recurrence retains FP32 readout before BF16 persistence, packed Q/K reuse and balanced head ownership; fused beta/decay v4 and affine gate selected |
| Large prefill configs | prefill64/121/127 and primary181 controls; delivery194 DEFAULT_POLICIES | Explicit 11x10 configs, default K8 with GDN residual K10/subblock3; internal 640-row cap and public 512 chunk; primary181 K10 override not selected |
| Memory/program/compute configs | Frozen delivery194 DEFAULT_POLICIES; kernel_contracts.md; final194 profiler CSVs | Selected source bound by manifest; capacity_delivery194.json and final_measurements.md complete |
| Input L1 advice | Earlier regular input-L1 tests; swiglu137 direct MinimalSwiGLU input-L1 check |26.573/6.755ms vs26.462/6.749 DRAM; reject added transfer |
| Runtime movement | Compact [1,B,H], flat prefill correction/MLP, convolution untilize fusion, unflatten146 v2 own-input/whole-path probes | Packed prefill unflatten v2 selected; no pending movement experiment |
| Persistent GDN state L1 | AUTOFIX_l1_recurrent_state_placement.md and candidates/l1_state_budget_cpu.json are delivery118-era bounds | DRAM state retained; historical split-head bound 1,622,016 B/core exceeds physical 1,572,864 before reservations; any future placement must recompute final194 CB/owner budgets |
| Repeated replay and allocation tracking | final194_correctness_run_trace_reuse_l{0,3}.json and stress gates; tracking=1, skip_program_cache=0, positive controls | Both layer kinds pass; all correctness jobs track program-cache allocations |
| Watcher | final194_correctness_watcher_* from final_jobs_v2.py, separate from profiling | All10 watcher gates pass across both kinds |
| Batch capability and deterministic outputs | final194_correctness_stress_* B1/128 and B32/512; odd B3/S513 and B31/S33; real-text and repeated bit equality | Both layer kinds required; complete final194 suite passed; no public batch1 restriction |
| Multi-device collective families and persistent CCL | Single1x1 mesh, no CCL in decoder | Not applicable; multichip explicitly outside this goal |
| Row-parallel output decomposition | OPT-008 is tensor-parallel CCL algebra; this layer uses one device | Not applicable; no multichip work |
| MoE active experts | Pinned text config has dense MLP, no routed experts | Not applicable |
| Terminal norm/head/sampling/token feedback | Decoder-only scope; no generator/LM head | Not applicable; no full-model/vLLM work |
| Qualitative full-model suite | Decoder real-text activations and independent HF references retained | Decoder precision evidence only; full-model qualitative suite outside scope |
| Stage review and local commit | Required after complete final194 source-bound evidence | Round1 clean-pass: stage_review_round1.md; local checkpoint 2d86c2f57341; no push |

The owner-selected stop rule is satisfied by [stop_rule_selection.json](stop_rule_selection.json):
candidate193/profile195 achieved **70.1187465% B32 same-run host roofline**.
The last two kept warmed improvements were **0.384237%** (185→189) and
**0.393176%** (189→193), both below 3%. Their medians were 1.592330,
1.586212 and 1.579975 ms; these separate selection runs are not substituted
for the profile's 1.602493 ms host denominator. The eight-hour device-time
alternative was **not used**; neither B1 performance nor wall time closes this
rule. The stop rule ends required optimization; quality review remains required.

Final validation is complete. [final_evidence_check.json](final_evidence_check.json)
checks all52 gates from [final_jobs_v2.py](final_jobs_v2.py), six separate B32
long-prefill runs, and four pytest cases. All results match delivery194 SHA;
all eight prefill/decode report pairs and accounting artifacts exist.
Correctness uses allocation tracking including program-cache allocations,
with positive controls. Watcher and profiling run separately. No failed final
job is hidden by the summary. Independent review round1 returned clean-pass; local checkpoint 2d86c2f57341 is recorded in the manifest.

Deferred options after the satisfied stop rule (no new work performed):

| Option | Precise handoff and constraint |
| --- | --- |
| Further exact hidden-norm fusion | Untried: combine phases or use on-chip inverse publication/multicast. Preserve the five-chunk fold, dense physical mapping, COL→ROW reduction and final BF16 boundary; prove publication ordering and exact own-input bits before whole B32/512. The earlier changed-tree distributed norm failed despite small isolated error. Existing zero-mean/half-inverse reads are already selected, not deferred. |
| Recurrence NoC scheduling | Selected writer retains per-head remote-completion barriers. The departed172 variant was probed but is not selected; do not present it as untried. Further transaction-ID batching, explicit multicast or deeper rings are untried on delivery194 and need buffer-ownership/remote-completion proof, ring-wrap/trace tests and measured whole-path benefit. Existing input/output rings already hold two heads. |
| Different state/compute partition | Whole-head balance with Q/K triple reuse is selected; prior value partitions P2/P4 were slower. A new partition is untried on the final topology and needs a fresh critical-path hypothesis rather than repeating those controls. Preserve ordered FP32 reductions and disjoint persistent-state ownership. |
| Persistent L1 state | Coordinated decode/prefill placement remains unimplemented. Recover final matmul rectangle union, recompute CB and reserved-L1 budgets, separate state owners from conflicting compute, preserve the caller's state address through prefill/continuation, and validate capture/replay before timing. Smaller-batch feasibility is not allocator evidence; changing state traffic requires recomputing the roofline denominator. |
