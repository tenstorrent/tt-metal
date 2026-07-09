# Optimized Decoder

Target: `meta-llama/Llama-3.1-8B-Instruct`

This stage adds a repo-local `tt/optimized_decoder.py` for the single-device
autoport pipeline. The optimized decoder is independent of `FunctionalDecoder`;
tests exercise the optimized prefill and paged decode paths directly.

## Runtime Contract

- `OptimizedDecoder.from_state_dict(state_dict, *, hf_config, layer_idx, mesh_device, batch=32, page_block_size=32, max_num_blocks=128, policy=None)`
- `prefill_forward(hidden_states, *, position_cos, position_sin, attn_mask=None, kv_cache=None, page_table=None)`
- `decode_forward(hidden_states, *, current_pos, position_cos, position_sin, kv_cache, page_table)`
- Prefill input/output: `[1, batch, seq, 4096]`.
- Decode input/output: `[1, 1, batch, 4096]`.
- Paged KV cache: `[max_num_blocks, 8, page_block_size, 128]`.
- Tested page config: `page_block_size=32`, `max_num_blocks=128`, batch 32.
- The functional context contract is preserved and `doc/context_contract.json` now records the optimized paged-decode validation scope, BFP8 cache dtype, page size, and tested cache limits.
- Runtime tests guard against calling `FunctionalDecoder` and against `torch`, `ttnn.from_torch`, or `ttnn.to_torch` inside forward methods.

## Final Policy

| Area | Policy |
| --- | --- |
| Activations/norm outputs | BF16 |
| Attention weights | BFP8 |
| MLP weights | BFP4 |
| KV cache | BFP8 |
| Attention matmul / SDPA | HiFi2 |
| MLP matmul | LoFi, explicit 1D program configs with `in0_block_w=32` |
| Norms | HiFi4 |

## Operation Topology

| Path | Final ops | Candidate/replacement evidence |
| --- | --- | --- |
| Prefill QKV | One packed `[Q,K,V]` matmul, then TTNN slice/reshape/permute. | `nlp_create_qkv_heads` rejected: current rank/shape did not satisfy helper contract (`input_shape[1] == 1`). Manual TTNN ops are correct for non-aligned seq 5. |
| Decode QKV | One packed `[Q,K,V]` matmul, WIDTH_SHARDED output, `nlp_create_qkv_heads_decode`. | Applies the forge WIDTH_SHARDED output recommendation after legalizing the emitted 11x9 grid to the local 8x8 grid. Separate Q/K/V not kept because packed path is correct and avoids repeated same-input matmuls. |
| Decode RoPE | `rotary_embedding_llama` for Q and K with sharded cos/sin and transformation matrix. | Fused QK RoPE was not kept; separate Q/K RoPE is required by the current helper/output layout sequence and remains a small fraction of decode time. |
| Paged KV | Separate `paged_update_cache` for K and V, BFP8 cache. | Fused update was considered, but the measured path spends ~32 us total in both update ops and decode is MLP/SDPA dominated. No fused update was kept without a measured win. |
| Decode SDPA | `paged_scaled_dot_product_attention_decode` with explicit `SDPAProgramConfig`. | Correct with paged/non-contiguous page table and repeated decode across page boundary. |
| Decode o_proj/residual | o_proj consumes L1 WIDTH_SHARDED concat-head output, then writes DRAM for the residual add. | Forge WIDTH_SHARDED residual output was not kept because it would add a reshard on the residual path; `tt-perf-report` now reports `in0_block_w=4` and output subblock 1x4 as good for the legal o_proj row. |
| MLP | Separate gate/up/down matmuls; gate matmul fuses SiLU in the program config. | BFP4/LoFi MLP beats BFP8/HiFi2 decode while preserving real-weight PCC. The dominant MLP sweep covered `in0_block_w=1/2/4/8/16/32/64`, L1-input variants, and forge DRAM-sharded `w=2`; final decode uses explicit `w=32` because it has the best same-process traced replay evidence. |
| DRAM-sharded decode matmuls | Not kept. | QKV DRAM-sharded candidate hit `bad optional access`; MLP DRAM-sharded program candidates are preserved in `decode_geometry_sweep.json` and fail current legal shard divisibility for `in0_block_w=4/8`. Earlier corrected MLP-only DRAM-sharded full decode ran at 2.842 ms, slower than final. |

## Forge Sharding Disposition

The optimization pass used `doc/functional_decoder/forge_sharding_recommendations.json` as a seed and legalized the emitted 11x grid to this 8x8 Wormhole device.

| Forge role | Action |
| --- | --- |
| QKV WIDTH_SHARDED output | Kept in decode. |
| RoPE cos/sin sharding | Kept as sharded decode RoPE inputs and transformation matrix. |
| KV update / SDPA decode | Kept with BFP8 paged cache and explicit SDPA program config. |
| o_proj WIDTH_SHARDED input/output | Kept WIDTH_SHARDED input from concat-heads; output remains DRAM for residual add. |
| MLP WIDTH_SHARDED / DRAM-sharded weights | Precision-locked sweep kept BFP4/LoFi and promoted explicit `in0_block_w=32`; L1 input and DRAM-sharded variants are rejected in `decode_geometry_sweep.md/json`. |
| Residual WIDTH_SHARDED path | Rejected because the final residual/norm sequence is DRAM based and avoiding extra reshard is faster in measured full decode. |

## Correctness And Perf

Primary command:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tt/optimized_decoder.py \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py
timeout 1200 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  --tb=short -s
```

Result: 9 passed.

Watcher command:

```bash
timeout 1200 env TT_METAL_WATCHER=10 pytest -q \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_optimized_decoder.py \
  -k 'not perf' --tb=short -s
```

Result: 6 passed, 3 deselected; watcher detached cleanly.

| Measurement | Final value |
| --- | ---: |
| Prefill real-weight seq 5 PCC | 0.999991 |
| Decode real-weight prefix 5 PCC | 0.999991 |
| Repeated decode pos 31 PCC | 0.999992 |
| Repeated decode pos 32 PCC | 0.999992 |
| Repeated decode pos 33 PCC | 0.999991 |
| Trace replay PCC threshold | >= 0.9999 |
| Isolated warmed prefill seq 5 wall time | 48.659 ms |
| Latest full-suite warmed decode prefix 31 wall time | 1.941 ms |
| Latest full-suite traced warmed decode prefix 31 wall time | 1.755 ms |
| Standalone final traced decode prefix 31 wall time | 1.750 ms |
| Same-process traced replay `w=32` best/median | 1.285 / 1.492 ms |
| Same-script traced baseline BFP8/HiFi2 decode | 1.921 ms |
| Same-script traced final BFP4/LoFi decode | 1.629 ms |

Earlier correct traced-decode baseline before the BFP4/LoFi MLP change was
measured with the same trace replay script at 1.921 ms. Eager decode in the
same test harness measured baseline BFP8/HiFi2 at 2.385 ms, BFP4/LoFi MLP at
2.316 ms, and all-HiFi4 at 2.581 ms. A follow-up MLP geometry sweep found
explicit `in0_block_w=32` as the fastest MLP-local candidate. A same-process
traced decode comparison then selected `w=32` over `w=8` on best replay time
with equal median replay time.

## Perf Artifacts

Final Tracy and `tt-perf-report` artifacts:

- `doc/optimized_decoder/tracy/final_prefill/reports/final_prefill/2026_07_09_09_29_00/ops_perf_results_final_prefill_2026_07_09_09_29_00.csv`
- `doc/optimized_decoder/tracy/final_prefill/prefill_perf_report.csv`
- `doc/optimized_decoder/tracy/final_prefill/prefill_perf_report.txt`
- `doc/optimized_decoder/tracy/final_decode/reports/final_decode/2026_07_09_09_54_13/ops_perf_results_final_decode_2026_07_09_09_54_13.csv`
- `doc/optimized_decoder/tracy/final_decode/decode_perf_report.csv`
- `doc/optimized_decoder/tracy/final_decode/decode_perf_report.txt`
- `doc/optimized_decoder/decode_geometry_sweep.json`
- `doc/optimized_decoder/decode_geometry_sweep.md`
- `doc/optimized_decoder/trace_block_sweep.json`
- `doc/optimized_decoder/trace_block_sweep.log`
- `doc/optimized_decoder/final_trace_decode.log`

Final decode report highlights:

- QKV: `127.941 us`, HiFi2 BF16 x BFP8.
- o_proj: `92.584 us`, HiFi2 BF16 x BFP8, L1 WIDTH_SHARDED input.
- SDPA decode: `24.715 us`.
- Gate MLP: `187.929 us`, LoFi BF16 x BFP4, `in0_block_w=32`.
- Up MLP: `190.544 us`, LoFi BF16 x BFP4, `in0_block_w=32`.
- Down MLP: `173.793 us`, LoFi BF16 x BFP4, `in0_block_w=32`.

Remaining `tt-perf-report` advice about placing MLP input0 in L1 or using
DRAM-sharded program configs was tried materially. L1 WIDTH_SHARDED input
requires moving SiLU into the program config; the legal explicit config path was
then measured. Legal L1-input variants were slower than the final DRAM-input
explicit configs, larger L1 block widths were blocked by shard divisibility,
and the forge-seeded DRAM-sharded `w=2` candidate blocks because Python-created
weights are interleaved while the DRAM-sharded matmul requires WIDTH_SHARDED
input B. The earlier corrected MLP-only DRAM-sharded full decode was slower, so
DRAM-sharded MLP was rejected.
