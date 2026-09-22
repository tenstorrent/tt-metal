# Applicable optimization advice and evidence

This index maps the Stage 5 request and `$optimize` checklist to concrete
experiments. Final measurements and validation are identified by the README;
candidate reports retain their individual policy and source provenance.

| Applicable item | Trial and disposition | Evidence |
| --- | --- | --- |
| Operation topology first | Audited packed projections, internal reductions, helper conversions, residuals and prefill before tuning | First table in `work_log.md` |
| Repeated same-input projections | Packed attention beats padded/sliced separate attention; packed gate/up beats separate and fused SwiGLU | `topology_family_results.json`, `candidate_summary.csv` |
| Coherent multi-device families | Replicated AR, replicated RS/AG, carried hidden-1280 residual with distributed/fused norms, row/column AGMM and MMRS; downstream consumers remain adapted | `collective_contracts.md`, family matrices and profiles |
| Async CCL and placement | Native async AR wins; RS/AG, Ring one/two links and Linear topology measured | Topology table, `inter_layer_contract.md` |
| Activation/CCL dtype | BF16/BFP8 crossed with compatible residual and fused families; BF8 AR offers no material repeatable win | Family reports and precision matrices |
| Persistent buffers | Shared AR workspace retained; persistent on/off and fused gather/output/counter buffers measured | Collective contracts and topology policies |
| Lower-movement residual/norm | Forty-core L1 stream wins against 10/20/80-core and sharded hidden-1280 families; no conversion or collective added at the layer boundary | Residual matrices, `inter_layer_profile_audit.json` |
| DRAM matmul geometry | Real BFP4/LoFi roles sweep reader counts, input grids, large legal K divisors and derived output subblocks; native mesh/tail bugs repaired | `matmul_geometry_search.csv`, `projection_summary.csv`, AutoFix reports, native tests |
| Weight precision and fidelity | Separate attention/output/gate/down trials for BFP4/BFP8, LoFi/HiFi2 and accumulation; final BF16×BFP4/LoFi verified in runtime rows | Precision matrices, `final_matmul_rows.csv` |
| KV precision and SDPA | BFP4 KV fill/update adapted and measured; accepted BFP8 state parity retained. Explicit short/long SDPA grids/chunks measured, including non-aligned mapped capacity | KV/SDPA matrices, final length profiles |
| Native attention composites | Paged SDPA retained. Rank-4 monolithic GDN with explicit normalization passes but loses; serial scan/preparation controls lose or overlap | `gdn_*` reports, `gdn_multichip_probe.py` |
| Prefill programs and dispatch | Legal 2D/minimal/1D grids, blocks, DRAM/L1 and 4096 chunks measured. Initial L1 errors adapted. Caller-owned prefill traces validated and measured | Prefill matrices, `review_trace_prefill_*`, final profile tables |
| Unnecessary conversions | Review found four short-prefill identity casts per layer; guarded by dtype equality and reprofiled | `stage_review.md`, before/repaired profile comparison |
| Public batch/length/context | Logical lengths and users preserved; padding/slicing owned internally; 31 cases, B32 stress and five maximum-context probes | Final index, tests, `memory_capacity_plan.json`, `../context_contract.json` |
| Trace and fallback | Changed inputs/positions/page tables and state ownership checked; 100 queued evolving-state replays; forward forbids host/Torch fallback | Final stress and correctness JSON |
| Watcher | Full watcher checks retained; owned-NoC packet-tag teardown fixed, compiled and exercised | Watcher logs/exit markers and AutoTriage report |
| Roofline and host/device accounting | Stored bank/tile bytes, all four device spans and matching host interval; hardcoded profiler worker denominator corrected in supplemental CSV | `performance_accounting.json`, profiler provenance, host-gap AutoDebug/AutoFix |

MoE routing, LM head, sampling, token generation and serving apply to other
model/stage contracts. This is a dense decoder stage and implements none of
those paths. There is no deferred applicable decoder optimization in this
index; final signoff additionally requires the independent review's clean pass.
