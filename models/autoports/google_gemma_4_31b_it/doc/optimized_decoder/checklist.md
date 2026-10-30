# Optimize checklist

All27 optimization items are completed or explicitly inapplicable below.
Final delivered validation is in final_acceptance.json; independent stage_review.md returns clean-pass.

| # | Requirement | Evidence / applicability |
|---|---|---|
| 1 | Fully traced decoder; no host fallback | run_optimized_decoder.py runtime conversion guards; final_runtime; optimized_trace.py |
| 2 | L1 width-sharded decode boundaries | optimized_decoder.py residual/norm/MLP; full Q DRAM exception measured in full_refined |
| 3 | Large prefill program configs | minimal_geometry and prefill_combined; explicit MinimalMatmul M4/K8/N16, full-N 2D adaptation |
| 4 | Initial topology audit | work_log.md initial audit and cumulative experiments; final topology table in README |
| 5 | Coherent multi-device topology families | N/A: single-device stage; no collectives |
| 6 | Lower-movement distributed residual families | N/A: single-device stage; local carried residual tested in combined_0 and prefill_l1_adapted |
| 7 | Strongest correct baseline and candidates | fused baseline; candidate_ledger; finalist_combinations; real_input_final_ranking; final_decode_k32 |
| 8 | Selected default reproduced | real_input_final_ranking sliding default and final_runtime full default; tracy_final/sliding_attention and tracy_final_k32/full_attention profiles |
| 9 | Runtime dtype/fidelity verified | final profile performance.json material prefill/decode matmul rows; summarizer asserts BFLOAT4_B/LoFi |
| 10 | SDPA and composite ops | paged scaled_dot_product_attention, decode SDPA, fused cache update, RMSNorm, RoPE, fused GELU multiply |
| 11 | Same-input projection packing compared | attention_packing_0, packed_refined, full_refined, finalist_combinations, real_input_final_ranking; packed attention wins, separate gate/up wins |
| 12 | Explicit important op configs | optimized_decoder.py per-role geometry, memory, norm, SDPA and compute kernel configs |
| 13 | Material matmul geometry sweeps | geometry_0_bfp4, combined_0, full_geometry, full_refined, output_padding, output_padding6144, minimal_geometry; candidate_ledger |
| 14 | Decode fidelity sweep | precision/manifest.json BFP8 LoFi versus HiFi2; role-separated BFP4 LoFi trials |
| 15 | Independent attention precision | precision and full_refined; recorded HF activations supersede synthetic-only veto |
| 16 | MLP gate/up/down BFP4 LoFi | precision, combined_0, full_geometry; measured runtime rows verify selected policy |
| 17 | Legal shard/core geometries | sender cores and reader workers distinguished; batch_grid_fix and output padding; per-role sweeps |
| 18 | DRAM-sharded decode; legal reader counts | reader_microbench layer0/5 exact selected geometry one/two/three where legal; output_padding6144 adapted three-reader family; whole-layer integration |
| 19 | Collective/layout movement minimized | No collectives; carried residual/shared MLP input, L1 heads; final topology audit identifies remaining required split/slice conversions |
| 20 | Fused matmul/CCL | N/A: no collective operations in single-device decoder |
| 21 | Persistent CCL buffers | N/A: no collective operations |
| 22 | MoE active experts | N/A: pinned 31B model is dense; config validation rejects unsupported MoE |
| 23 | LM head/sampling/token feedback | N/A: explicitly excluded single-layer stage; no LM head or sampling |
| 24 | LM head DRAM sharding | N/A: no LM head in decoder module |
| 25 | Real-weight/input reduced precision | record_decoder_activations.py pinned HF; precision and final correctness; whole-model datatype frontier remains separate stage |
| 26 | Performance accounting | final profile performance.json whole-layer FW windows + same-run host_timing.json; roofline assumptions and peak_basis.json; report drains external to layer |
| 27 | Batch capability | final_runtime distinct inputs through sliding56/full36; delivered pytest1/8/16/32; watcher B1/B32; final profile trees all configured batches |

Correctness, PCC, paged KV, repeated replay, fallback guards and watcher are
recorded in final_runtime and pytest_delivered_final.log. Maximum context
acceptance is final_context_v2/sliding and final_runtime/full. No full-model qualitative or vLLM serving claim
is applicable to this decoder-only stage. No serving profiler was run.
