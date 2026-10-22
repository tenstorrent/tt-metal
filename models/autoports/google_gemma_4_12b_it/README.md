# google/gemma-4-12B-it on Blackhole QB2 (P300x2, TP4)

The text decoder of [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) on the four Blackhole
chips of a QB2, as a 1x4 mesh with tensor parallel 4. This is the output of the tt-model-bringup pipeline, all 11
stages. Tracking issue: tenstorrent/tt-agentic-bringup-qb2#93.

- **Model:** dense, 48 layers: 40 sliding-window layers (window 1024) and 8 full-attention layers.
  - Hidden size 3840; 16 query heads.
  - KV heads: 8 on sliding layers (head dim 256), 1 on full-attention layers (head dim 512).
  - Tied 262K vocabulary, final logit soft-cap 30, 262,144-token native context.
- **Not included:** the image and audio encoders.
- **Served context:** the full 262,144 tokens, with a vLLM-owned paged KV cache.

## Layout
- `tt/model.py`: weights, config and the decoder stack.
- `tt/generator.py`: prefill/decode API with traces and on-device sampling.
- `tt/generator_vllm.py`: the vLLM adapter (`Gemma4AutoportForCausalLM`).
- `tt/precision.py`: precision policies.
  - The default is `doc/datatype_sweep/selected_precision_config.json` (`b0_lmh4_lofi`): BFP8 weights, a BFP4 LoFi
    LM head, and a BFP8 KV cache.
  - `GEMMA4_PRECISION_CONFIG=baseline` rolls back to the stage-7 policy.
- `tt/sampling.py`: the device sampler. It covers top_k 1–32; anything else samples on the host.
- `tt/*_decoder.py`: the decoder layers at each bring-up stage. `optimized_decoder.py` is the one served.
- `tests/`: accuracy tests against HF, full-context tests, perf tests and vLLM adapter tests.
- `doc/context_contract.json`: the served-context contract.

## Requirements
- The opt-in SDPA flag `fp32_output_accumulation` (tenstorrent/tt-metal PR stacked below this one).
  - The 8 full-attention layers turn it on for every chunked-prefill call.
  - Without it, full-attention prefill PCC drops to 0.9865 at 131K tokens and 0.97 at 262K.
- vllm-tt-plugin `main`, unmodified. The adapter is registered through
  `TT_MODEL_CLASS_OVERRIDES=Gemma4UnifiedForConditionalGeneration=models.autoports.google_gemma_4_12b_it.tt.generator_vllm:Gemma4AutoportForCausalLM`.

## Serving (tt-inference-server)
- Spec `id_gemma4-12b-qb2_gemma-4-12B-it_p300x2`, selected with `--impl gemma4-12b-qb2`.
- `MESH_DEVICE=P300x2`, 32 slots, block size 64, 8192-token chunked prefill, `FABRIC_1D_RING`.
- The server default is top_k 32, so requests stay on the device sampler.
- `reasoning-parser gemma4`.

## Results
CI on `bh-qb-ge` (tenstorrent/tt-agentic-bringup-qb2#93):

| Point | TTFT | TPOT | Throughput |
|---|---:|---:|---:|
| 1 user, 128/128 | 40 ms | 14.5 ms (69 t/s/u) | |
| 32 users, 128/128 | 0.91 s | 16.8 ms | 1,345 tok/s |
| 1 user, 4096/128 | 214 ms | 15.4 ms | |
| 32 users, 4096/128 | 2.24 s | 54.5 ms | 446 tok/s |
| 1 user, 32768/128 | 2.59 s | 15.9 ms | |
| 1 user, 131072/128 | 21.8 s | 17.8 ms | |

| Eval (thinking on, T=1.0, top_p 0.95, top_k 20) | n | TT | Card |
|---|---:|---:|---:|
| GPQA Diamond | 198 | 66.7 | 78.8 |
| AIME 2026, no tools | 30 × 4 | 71.7 | 77.5 |
| MMLU-Pro, 16K output cap | 849 | 73.1 (65 empty answers; 79.2 without them) | 77.2 |

Accuracy against HF on the AIME24 chat reference (100 tokens): top-1 95–96%, top-5 / top-100 100%.

## Known limitations
- **Paged decode SDPA can read one page-table entry past a row.** This happens when `k_chunk_size` does not divide
  `row_width * block_size`; it is an op bug.
  - Decode uses 128-token K chunks at batch 1 and one block at batch 32.
  - `_table_for_k_chunk` (`tt/optimized_decoder.py`) pads the row with block 0 up to a whole number of K chunks.
    The padded keys lie past every query's causal range, and nothing is padded when the width already divides.
  - `tests/test_optimized_decoder.py::test_decode_odd_width_page_table` pins this contract.
- **Prefix caching is off.**
