# Optimization checklist and cumulative contract

Stage 3, single Blackhole chip0. Stop rule 2 ended exploratory tuning; required correctness and precision-matched confirmation continued. A deferred item is not claimed rejected or completed.

| Item | Action and evidence |
|---|---|
| Topology audit before knobs | Initial table in `work_log.md`; predecessor six per-op reports identified expert matmuls and exact GR reductions. |
| Canonical precision | `selected_decoder_policy.json`; BFP8 ordinary weights, LoFi experts, BF16/HiFi4 GDN z/a/b after real-weight failure isolation. Sensitive GR/router/indexer paths retained. |
| Dominant expert program geometry | `initial_sweep_v2.json`, `geometry_v1.json`, `memory_v2.json`; final precision-locked `final_geometry_lofi.json`. Gate/down independently vary cores, K blocks, subblocks and memory. |
| DRAM sharding / Blackhole readers | `dram_gdn_v1.json`, `dram_gdn_hifi4_v2.json`, `dram_shared_v1.json`, `dram_ple_v1.json`, `dram_qsa_v1.json`; 1/2/3 readers, 4/8/16 input cores, blocks 2/5/10/20. `dense_component_summary.json` includes transfer/split costs. Wide configurations that exhausted L1 were retried with smaller blocks/cores. |
| Packed versus separate projections | GDN final BF16/HiFi4 packed wins; shared and QSA packed win; PLE separate wins. `dense_component_summary.json`. Expert separate geometry and fused-SiLU controls: `split_experts_final*.json`. |
| Residual through consumers | `residual_v1.json` measures read/write/next-read with adapted consumer; integrated `rank_plan` compares DRAM, L1 and width-sharded L1. L1 kept; width-sharded does not win. |
| Phase-specific memory | Compact expert input/output L1; full physical-prefill intermediate DRAM. `memory_v2.json`, `rank_plan`, final profiler rows. Dense DRAM-sharded matmuls use width-sharded L1 input and DRAM weights. |
| SDPA/composites | Native paged SDPA retained; `sdpa_v1.json` varies K chunk and grid. Packed SiLU adapted to L1 rather than rejected at its original DRAM-only guard. Native fused-SiLU split expert comparison also measured. |
| Large prefill configs | Physical <=32 chunks use tuned 1D sparse and DRAM-sharded dense configs; long logical input still chunks internally. Wider exact-GR/recurrence implementation and large-M 2D kernels deferred under stop rule. No logical alignment restriction. |
| Runtime movement / host fallback | Optimized probe runtime audits forbid torch/from_torch/to_torch and fallback inside forwards. Setup packing and reference readback are outside measured calls. Remaining device row-major router/indexer conversions, DRAM sharding boundaries and public residual admission/return are required by current consumers and included in comparisons. |
| Active MoE | Device routing, exact top10 indexed decode, two-row union and variable prefill union preserved; no dense-all-experts execution. `nnz` is incompatible with indexed mode in this implementation, so unset. Final reports show active=10/512 decode. |
| BFP4 candidates | Real checkpoint gate/up and down separately: `precision_hf_plan`, `gate4_perf_v1.json`, `down4_perf_v1.json`, `bfp4_geometry_policies.json`; final LoFi rows in `tracy/final_ops_{gate4,down4}/report`. BFP8 remains user-required default; stage8 decides final precision. |
| Attention precision | Independent GDN/QSA/PLE groups and real-weight HF probes. BF16 GDN necessary on real weights; BFP8/HiFi2 fails batch and continuation. BFP4 attention and additional fidelity/geometry combinations deferred under stop rule, not rejected by synthetic evidence. |
| Logical batch/padding | Tuple batching retained; GDN/QSA32 and resident PLE5 requests pass final probes. Physical padding never changes logical positions. |
| Context | Unchanged BF16 paged KV layout/capacity262144. Final32768 component-prefix smokes for all kinds plus full public terminal decode. Full PLE4767 strict and4768 whole-output-contract validation preserve predecessor prepared envelope. Exact scopes in `../context_contract.json`. |
| Integrated accuracy / trace | `final_accuracy_summary.json`, `final_test_suite.xml`, `final_v3_real_ci0.json`, final continuation/batch artifacts. Program-cache allocation tracking enabled for correctness; final stress covers128input/128decode. |
| Determinism / watcher | Changing-input traced replay, state/reset equality in final probes; separate `final_v3_watcher_continuation{0,1}` and `final_watcher_continuation3` jobs. No watcher/profiler overlap. |
| Before/after final default | `final_perf_{fused,optimized}_{0,1,3}_{32,33,129,1024}.json`; final policy restored explicitly by variant `default`. Clean traced decode and warmed eager prefill; no profiler or allocation tracker in timing. |
| Per-op advice | Six final representative reports plus two BFP4 reports, `final_per_op_summary.json`; conclusions and deferred low-impact advice in README/work log. |
| Multi-device/CCL, LM head, sampling, full-model TTFT, vLLM | Not present in this decoder stage; no work or claims in these areas. Full-model performance accounting/qualitative generation gates apply later. |

The 128-in/128-out decoder stress passes. Per-op bandwidth is reported with active weight traffic, not all resident experts/PLE weights, and padded32-row FLOPs are not useful-token throughput. No full-model theoretical-throughput percentage is inferred from this isolated layer. The predecessor's advertised capacities and quality bar are preserved.

Review round1 missing physical-two-row Watcher branch: covered for GDN/PLE/QSA by `review_watcher_tail2_summary.json`; raw PLE row5 diagnostic assessed against the explicit whole-output contract and matched fused predecessor.
