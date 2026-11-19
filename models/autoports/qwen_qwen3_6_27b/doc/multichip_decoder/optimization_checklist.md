# Optimization decisions

The selected default uses BF16 activations/CCL, BFP8 weights/cache, LoFi
attention, GDN MLP HiFi2 and full-attention MLP LoFi. Candidate and final
measurements are serial on the same four chips. Stop rule1 is the intended
closure rule: decisions below plus final profiler/accounting and hard gates.
No claim of70% roofline or8h optimization time is made.

| Skill item | Evidence / decision |
|---|---|
| Actual shapes/weights; trace; context | Real checkpoint weights throughout; recorded target activations for fidelity/BFP4 selection; delivery_stress, delivery_context, delivery_perf. |
| Topology before tuning | Preimplementation mesh_plan; candidate_v1/v2/v3; topology probes. Hidden-sharded residual carried through the next consumer. |
| Final default reproduces selection | delivery_perf and delivery correctness/stress/context; source hashes and selected_policy in results. |
| Profiler and advice | Initial GDN/full profiles identified batch padding, conv and head movement; compact implementations remove those costs. Delivery-source profiles and accounting are separate from watcher; unsafe original native MMRS was rejected after watcher failures. |
| Prefill/TTFT | Layer prefill reported at short,512,513,1025 and maximum context. Full-model TTFT is not claimed in this decoder stage. Per-shape kernel tables in final profiles. |
| Matmul geometry | final_readers repeats final narrow+L1 K2/4/8/16, four storage workers and one-reader controls; geometry_v1 reader/core/block controls; geometry_precision_v1 HiFi2 K4/8/16; lofi_geometry_v1 selected-policy K2/8/16 and four workers. K2/8workers/2readers retained. Larger native pipeline K is legal but slower. Safe prefill sweep and coupled integration select11x10/K8, recording the0.094ms GDN B1 tradeoff for larger-shape gains. |
| DRAM readers | Two Blackhole readers vs one-reader/nonpipeline controls in geometry_v1, repeated under final fidelity in final_readers. |
| Packed QKV | Retain optimized baseline packing; its work_log.md:681 records the slower separate corrected QKV/Z/B/A control, and full66_packed measures prefill packing. Local Q6/K1/V1 stages one64-tile stream. This stage reuses that decision; no new TP4 split-Q/K/V timing is claimed. Compare classic local-head assembly vs compact packed staging:1.515→0.718ms B32. |
| Reduced KV dtype | BFP8 inherited from validated optimized baseline and tested against that baseline at full context, page remap and changed positions. Local one-KV-head cache ownership is explicit. |
| Residual/norm layout | Sharded/replicated whole-layer controls; phase-specific gather-norm controls in candidate_v3. Public[B,S,1280] retained. |
| Logical vs tile batch | Decode compaction plus exact local QKV/conv/recurrence tests across awkward batches; crops native concat physical32 to logical B. |
| Cumulative contract | kernel_contracts.md; integrated_v1 then final defaults; mixed-layer stack/resource tests. |
| Row output decomposition | Separate matmul+RS vs fused MMRS through next residual/norm:0.500857→0.406864ms in topology probe. Original unpartitioned prefill fusion fails watcher. Safe native-prefill comparisons select separate row matmul/RS through512 and above8192 flattened rows, with128-row native pieces for513..8192; decode stays separate. |
| Persistent CCL buffers | Captured fixed trace buffers; allocation tracking includes program cache. Fused AG probe has persistent scratch and admission fence;50queued replays pass. Final packed-SwiGLU comparison:0.753429ms fused vs0.496683ms separate, PCC0.99999267; rejected. |
| Narrow GDN correction | narrow_v1 four-way wide/narrow × DRAM/L1 experiment at B1/B32; narrow+L1 selected, saves401,080,320 bytes/device across48 layers. |
| Packed MLP | candidate_v3 separate vs packed gate/up decode; tile-pair SwiGLU prefill; safe native sweeps select88-core M2/K8/N26 for33..512 rows, existing110-core config otherwise; copies included in capacity. |
| Phase-specific shards | Interleaved11x10/K8 ordinary prefill; eight width-sharded activation storage cores and DRAM weights decode; compact batch row path. |
| Precision x geometry | LoFi geometry sweep under final BFP8 policy; HiFi2 sweep separate. BFP4 candidates retained for stage8 rather than used to reject BFP8 geometry. |
| Fidelity by tensor group | fidelity_v1: GDN attention LoFi/HiFi2/HiFi4; MLP LoFi control; full attention alreadyLoFi. geometry_precision_v1 extends real activation length254. |
| Attention precision separate | bfp4_v2 attention and MLP separate. Full-attention BFP4 fails real cache PCC0.995; GDN and MLP BFP4 candidates recorded for datatype stage. |
| Real precision evidence | Activation file SHA in results points to prior-stage recorded checkpoint activations; no random-weight veto. |
| Runtime dtype/fidelity | Final profiler rows plus matrix_allocations/selected_policy metadata. |
| Full-model/LM head/sampling | Not implemented by this decoder-only goal; BF16 embedding/head memory estimated conservatively, bounded-token logits contract recorded. |
| MoE | Model is dense; no router, expert execution or EP decision applies. |
| Hardware/recovery | Owner-authorized idle/own-hung resets; live triage before AGminimal hang termination; reset/mesh-smoke evidence. No T3K/Galaxy platform path on these Blackhole chips. |
| Final evidence/audit | README evidence index, host/runtime audits, trace tracking positive controls, stress, watcher, stage review and local commit log. |

Delivery-source profile advice and resource-test findings are reconciled in
`profile_accounting.md`, `gate_summary.json` and the resource artifacts. Stop
rule1 is satisfied by this checklist and the completed final evidence. The primary opportunity beyond this
stage is reducing the remaining local kernels/CCL overhead and evaluating
BFP4 with full-model reasoning accuracy in the dedicated datatype stage.
