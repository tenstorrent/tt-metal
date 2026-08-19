# Optimized Decoder

This directory records the optimized-decoder state for `Qwen/Qwen3.6-35B-A3B` under `models/autoports/qwen_qwen3_6_35b_a3b`. Scope is single-device TTNN decoder-layer optimization only. No multichip-decoder, full-model, or vLLM work is included.

## Implementation

`tt/optimized_decoder.py` implements `OptimizedDecoder(FusedDecoder)` with the fused decoder public contract and an optimized runtime path. The measured prefill/decode path constructs `OptimizedDecoder` directly and enters `_OptimizedQwenMoe` plus `_OptimizedFullAttention`; tests audit the measured functions for functional fallback calls, Torch conversion calls, and host fallback patterns.

Final policy:

| Role | Final setting |
| --- | --- |
| Dense full-attention weights | `ttnn.bfloat8_b` |
| Dense linear-attention weights | `ttnn.bfloat8_b` |
| Shared MoE weights | `ttnn.bfloat8_b` |
| Routed MoE weights | `ttnn.bfloat8_b` for linear layers, `ttnn.bfloat4_b` for full-attention layers |
| Sparse matmul output dtype | `ttnn.bfloat16` |
| Sparse matmul memory | DRAM |
| Sparse matmul program knob | `in0_block_w=4` |
| Decode sparse active experts | exact `nnz=tokens * top_k` (`8` for measured batch-1 decode) |
| Full-attention decode SDPA | explicit `SDPAProgramConfig`, `q_chunk_size=32`, `k_chunk_size=64`, `max_cores_per_head_batch=16` |

The optimized path preserves fused semantics: prefill/decode, full-attention paged KV-cache update/use, linear-attention recurrent state, batch-1 and batch-2 coverage, repeated decode determinism, and valid non-aligned logical sequence lengths. Routed-MoE prefill pads internally to tile shape and slices back to the logical length, so there is no public `seq_len % chunk_size == 0` requirement.

## Context Contract

No update was made to `doc/context_contract.json`. The optimization does not reduce advertised context capacity, public shapes, cache block size, cache dtype, KV-cache layout contract, or linear-attention logical-length support. The final changes are weight dtype policy, sparse matmul program config, exact decode sparse `nnz`, full-attention decode SDPA program config, and internal MoE padding only.

## Topology Audit

The pass started from measured fused-decoder topology. Fused had already packed same-input dense projections and MoE gate/up projections; `tt-perf-report` showed the remaining dominant rows were routed-MoE sparse matmuls, followed by smaller packed dense projections, TTNN routing/scatter/reduce rows, and small device-side layout bridges.

| Opportunity | Action | Evidence |
| --- | --- | --- |
| Repeated same-input matmuls | Kept packed projections for full `q/k/v`, linear `qkv/z/b/a`, shared gate/up, and routed gate/up. No repeated same-input projection remains in the measured optimized path. | `OptimizedDecoder.graph_summary`; final report rows include packed projection widths `9216` and `12352`. |
| Canonical precision/fidelity | Kept dense/shared BFP8, BF16 state/cache outputs, and layer-kind routed policy. Full-attention routed MoE uses BFP4; linear routed MoE stays BFP8. | `logs/candidate_exact_nnz_combo_correctness.log`; final PCC table below. |
| BFP4/LoFi material matmuls | Tried routed BFP4, all-MoE BFP4, and DRAM-sharded LoFi dense projection candidates using real weights. | All-MoE BFP4 failed real linear prefill PCC (`0.9890764118945616`); routed BFP4 did not beat final sparsew4/exact-`nnz`; DRAM-sharded LoFi projections were slower. |
| Sharded layouts and DRAM-sharded decode matmuls | Adapted the dense decode projection candidate after the first TTNN API issue and timed the legal candidate. | `logs/candidate_dense_dram_sharded_adapted.log`: linear projection `0.233 ms` baseline vs `0.274 ms` candidate; full projection `0.139 ms` baseline vs `0.263 ms` candidate. Rejected. |
| Sparse matmul program config | Accepted `in0_block_w=4`; rejected `in0_block_w=2` final-geometry baseline and the 16-core/output-subblock variant. | Final walls `20.414/8.973/1.537/1.213 ms`; final device totals `12.123/7.669/1.445/1.126 ms`. |
| Decode MoE active-expert execution | Accepted exact decode sparse `nnz`. | `decode_exact_nnz` improved full decode from old sparsew2 `1.813 ms` wall to `1.350 ms`; sparsew4 plus exact `nnz` improves final full decode to `1.213 ms`. |
| Sparse input L1 advice | Tried explicit L1 inputs for routed decode and prefill. | `decode_l1_sparse_inputs_exact_nnz` was slower (`1.693/1.373 ms` decode); `prefill_l1_sparse_inputs_exact_nnz` did not beat final prefill and tied/trailed decode. Rejected. |
| SDPA/composite ops | Kept dedicated paged SDPA and the K64 full-decode program config. | `logs/candidate_sdpa_k64_correctness.log`; final full decode uses SDPA K64 and remains fastest. |
| Output subblock advice | Tried a legal smaller-core/output-subblock sparse variant. | `sparse_cores16_out2_exact_nnz` was slower (`23.942/12.933/1.648/1.332 ms` wall). Rejected. |
| Runtime data movement | Rejected explicit reshards/L1 moves when timed. Final source and report audit show no host fallback, `torch`, `from_torch`, `to_torch`, or reshard rows in measured prefill/decode. | `logs/runtime_fallback_audit_final_sparsew4_exactnnz.log`; final report movement rows are small device-side layout bridges. |

No remaining stage-local decoder optimization candidate beat the final Blackhole path. Remaining speedups would require TTNN kernel/op implementation changes or later pipeline stages outside this goal.

## Correctness

Acceptance bar: PCC >= `0.995`. Final watcher-clean artifact: `logs/watcher_correctness_final_sparsew4_exactnnz.log` (`14 passed, 32 deselected`). The watcher used `TT_METAL_WATCHER_DISABLE_ETH=1` because this is a single-device decoder-layer run with no Ethernet/fabric data path; watcher still attached to all four local devices and checked worker progress.

| Case | Prefill PCC | Traced decode PCC |
| --- | ---: | ---: |
| synthetic linear layer 0, seq 5 | 0.9995203064456892 | 0.9994662304973918 |
| synthetic full layer 3, seq 33 | 0.9995777703980826 | 0.9993898329949268 |
| synthetic batch-2 linear, seq 5 | 0.9994317553119714 | 0.9992362429708346 |
| synthetic batch-2 full, seq 33 | 0.9995916971087269 | 0.9992974376522705 |
| synthetic linear non-aligned seq 65 | 0.9975411512233607 | 0.9994416658202645 |
| synthetic full non-aligned seq 33 | 0.9995777703980826 | 0.9993898329949268 |
| real linear layer 0, seq 1 | 0.9991292848497009 | 0.9987610738665994 |
| real full layer 3, seq 1 | 0.9996961087208918 | 0.9995498281934312 |
| real linear layer 0, seq 5 | 0.9996973622269505 | 0.9986944187478957 |
| real full layer 3, seq 5 | 0.9996233758352069 | 0.9994969304756183 |

Repeated decode determinism PCC is `1.0` for linear and full attention. The dynamic fallback audit with `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'` passed the real-weight optimized path (`logs/runtime_fallback_audit_final_sparsew4_exactnnz.log`).

## Performance

Performance was captured with warmed prefill and warmed traced decode on local Blackhole p300c hardware. `tt-perf-report` tables were regenerated from Blackhole-normalized profiler CSVs with `DEVICE ARCH=blackhole` and `AVAILABLE WORKER CORE COUNT=110`.

| Case | Fused wall ms | Final wall ms | Wall delta | Fused device ms | Final device ms | Device delta | Final table |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linear prefill, seq 5 | 33.628 | 20.414 | -39.3% | 25.289 | 12.123 | -52.1% | `tracy/final/linear_attention/prefill_perf_report.csv` |
| full prefill, seq 33 | 23.038 | 8.973 | -61.1% | 21.684 | 7.669 | -64.6% | `tracy/final/full_attention/prefill_perf_report.csv` |
| linear traced decode after seq 5 | 2.463 | 1.537 | -37.6% | 2.368 | 1.445 | -39.0% | `tracy/final/linear_attention/decode_perf_report.csv` |
| full traced decode after seq 33 | 2.121 | 1.213 | -42.8% | 2.036 | 1.126 | -44.7% | `tracy/final/full_attention/decode_perf_report.csv` |

Final dominant rows:

| Window | Rows | Device ms | Device movement ms | Dominant rows |
| --- | ---: | ---: | ---: | --- |
| linear prefill | 492 | 12.123 | 0.172 | all-expert sparse gate/up `4.107 ms`, sparse down `1.255 ms` |
| full prefill | 100 | 7.669 | 0.074 | all-expert sparse gate/up `4.106 ms`, sparse down `1.251 ms` |
| linear decode | 93 | 1.445 | 0.038 | active sparse gate/up `0.133 ms`, unary/transpose rows dominate remaining cost |
| full decode | 73 | 1.126 | 0.023 | active sparse gate/up `0.133 ms`, unary/transpose rows dominate remaining cost |

The final optimized runtime is faster than the best correct fused traced-decode baseline and faster than all correct optimized candidates measured in this stage.

## Candidate Matrix

All rows below used real weights and final sparse-geometry-era tests. Correctness for exact-`nnz` combinations passed in `logs/candidate_exact_nnz_combo_correctness.log` (`12 passed, 34 deselected`).

| Policy | Linear prefill wall | Full prefill wall | Linear decode wall | Full decode wall | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| old sparsew2 default | 24.846 | 13.907 | 2.136 | 1.813 | Superseded |
| `decode_exact_nnz` | 24.445 | 13.457 | 1.688 | 1.350 | Kept as part of final |
| `routed_bfp4_exact_nnz` | 24.943 | 13.416 | 1.683 | 1.350 | Rejected; no win |
| `decode_l1_sparse_inputs_exact_nnz` | 24.808 | 13.454 | 1.693 | 1.373 | Rejected; slower |
| `prefill_l1_sparse_inputs_exact_nnz` | 24.781 | 13.370 | 1.675 | 1.349 | Rejected; no prefill win |
| `sparse_in0_block_w4_exact_nnz` | 20.976 | 9.060 | 1.537 | 1.220 | Kept |
| `sparse_cores16_out2_exact_nnz` | 23.942 | 12.933 | 1.648 | 1.332 | Rejected; slower |
| final no-env default | 20.414 | 8.973 | 1.537 | 1.213 | Accepted |

`tt-perf-report` advice that was actioned and rejected with evidence:

| Advice | Action | Result |
| --- | --- | --- |
| Increase sparse `in0_block_w` | Tried `2` and `4`; selected `4`. | Large sparse rows dropped from old sparsew2 `7.403/2.295 ms` to final `4.107/1.255 ms`. |
| Place sparse input 0 in L1 | Tried decode-L1 and prefill-L1 candidates. | Slower or tied; explicit movement outweighed locality benefit. |
| Increase output subblock area | Tried legal 16-core/output-subblock sparse variant. | Slower; rejected. |
| Use DRAM-sharded dense decode matmul / LoFi | Adapted after initial API issue and reran. | Slower after required layout conversion; rejected. |
| Use higher fidelity for BF16 accuracy | PCC is above acceptance bar with final policy; HiFi/LoFi alternatives did not produce a faster correct runtime. | Kept final fidelity policy. |

## Artifacts

- Code: `tt/optimized_decoder.py`
- Tests: `tests/test_optimized_decoder.py`
- Final correctness and watcher: `logs/watcher_correctness_final_sparsew4_exactnnz.log`
- Dynamic fallback audit: `logs/runtime_fallback_audit_final_sparsew4_exactnnz.log`
- Static and collection: `logs/py_compile_final_exactnnz_sparsew4.log`, `logs/pytest_collect_final_exactnnz_sparsew4.log`
- Final perf tee: `logs/tracy_perf_final_sparsew4_exactnnz_summary.log`
- Final Blackhole report input: `tracy/final_sparsew4_exactnnz_raw/reports/2026_08_19_02_29_34/ops_perf_results_2026_08_19_02_29_34_blackhole.csv.parts/`
- Final report tables and summaries: `tracy/final/*/*_perf_report.{csv,txt}`, `tracy/final/*/*_summary.csv.csv`
- Candidate evidence: `logs/candidate_exact_nnz_combo_correctness.log`, `logs/candidate_sparse_geometry_correctness.log`, `logs/candidate_dense_dram_sharded_adapted.log`, `tracy/candidate_*_exact_nnz/*/*_perf_report.csv`
- Hardware snapshots: `logs/tt_smi_initial.log`, `logs/tt_smi_before_candidates.log`, `logs/tt_smi_final_sparsew4_exactnnz.log`
