# google/gemma-4-12B-it on Blackhole QB2 (P300x2)

TTNN implementation of the [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) text decoder (dense,
48 layers, 5 sliding-window (1024) : 1 full attention, 16 query heads, KV heads 8 x 256 on sliding and 1 x 512 on full
layers, tied 262K vocabulary, final logit soft-cap 30) for the QB2 1x4 Blackhole mesh, TP4, served through vLLM with
the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). Native 262,144-token context. The image and audio
encoders are not ported. Produced by the tt-model-bringup agentic pipeline; tracking:
tenstorrent/tt-agentic-bringup-qb2#93.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 48-layer model: embedding, decoder stack, final norm, LM head, logit soft-cap |
| `tt/multichip_decoder.py` | TP4 decoder layer: QKV / gate / up column-parallel, o_proj / down row-parallel; the full layers' single KV head is replicated on every chip |
| `tt/optimized_decoder.py`, `tt/fused_decoder.py`, `tt/functional_decoder.py` | Single-chip layer programs, policies and constants the TP4 layer builds on |
| `tt/generator.py`, `tt/sampling.py` | Traced decode, chunked prefill, on-device sampling (top_k 1..32) |
| `tt/generator_vllm.py` | vLLM adapter (`Gemma4_12BQB2ForCausalLM`), vLLM-owned hybrid KV cache, decode-input reload contract v1 |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision: BFP8 weights, BFP4 LoFi LM head, BFP8 KV cache. `GEMMA4_PRECISION_CONFIG=baseline` restores the pre-sweep policy (`doc/datatype_sweep/configs/baseline_stage7.json`) |
| `vllm_metadata.json` | Architecture registration for `EXTRA_MODELS_DIR` |
| `tests/test_vllm_adapter.py` | Device test of the adapter: resident-input decode vs reload, chunked prefill, seeded rows |

## Requirements

- tenstorrent/tt-metal#61986 (stacked below): the 8 full-attention layers pass its opt-in SDPA flag
  `fp32_output_accumulation` on every chunked-prefill call. On one chip with this model's settings it raises the
  attention PCC at a 262K prefix from 0.9991 to 0.9997, for about 4% more time in that attention.
- vllm-tt-plugin `main`, unmodified, with vLLM 0.30.

## Serving

The adapter is found through `EXTRA_MODELS_DIR=models/demos` and `vllm_metadata.json`, and selected with
`--hf-overrides '{"architectures": ["TTGemma4_12BQB2ForCausalLM"]}'`. The tt-inference-server spec
`--impl gemma4-12b-qb2` carries the full recipe; the essential settings are `--block-size 64 --max-num-seqs 32
--max-model-len 262144 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching
--reasoning-parser gemma4 --override-generation-config '{"top_k": 32}'` and the TT config
`sample_on_device_mode=all`, `fabric_config=FABRIC_1D_RING`, `trace_region_size=0`, `prefill_chunk_alignment=8192`.

The device sampler covers top_k 1..32, so the server caps the checkpoint's top_k 64 at 32. Requests it cannot serve
(penalties, logprobs, top_k > 32) and prefill steps with an intermediate chunk sample on the host.

## Results (CI release run, 32-slot server, vllm-tt-plugin main)

| Point | TTFT | Decode |
|---|---|---|
| 1 user, 128/128 | 36 ms | 12.3 ms/token |
| 32 users, 128/128 | 0.89 s | 1,494 tok/s total |
| 1 user, 4096/128 | 210 ms | 13.2 ms/token |
| 32 users, 4096/128 | 2.23 s | 466 tok/s total |
| 1 user, 32768/128 | 2.37 s | 13.7 ms/token |
| 1 user, 131072/128 | 20.4 s | 15.6 ms/token |

GPQA Diamond 66.7 (198 questions, thinking on; card 78.8, TT reference 65). Earlier CI runs on the same model code:
AIME 2026 71.7 (30 x 4; card 77.5), MMLU-Pro 73.1 (849 questions, 16K output cap; card 77.2). Accuracy against HF on
the AIME24 chat reference (100 tokens): top-1 95-96%, top-5 / top-100 100%.

## Known limitations

- **Paged decode SDPA can read one page-table entry past a row** when `k_chunk_size` does not divide
  `row_width * block_size`. `_table_for_k_chunk` (`tt/optimized_decoder.py`) pads the row with block 0 up to a whole
  number of K chunks; the padded keys lie past every query's causal range, and nothing is padded when the width
  already divides (vLLM's rows).
- **Prefix caching is off.**
- **Decode attention uses the batch-32 core split at every batch** (`sdpa_decode_max_cores`, `tt/optimized_decoder.py`)
  so a seeded sampled request gives the same text whatever the server's load. One user decodes slower at long
  context than the op default would allow (measured locally: 15.9 / 22.3 / 44.1 ms/token at 4K / 32K / 128K, vs
  13.5 / 14.6 / 18.7); 32 users are unchanged. The user-split full-layer attention (8+ users past
  `decode_user_split_min_position`) still has its own reduction order.

## Test

```bash
pytest models/demos/gemma4_12b_qb2/tests/test_vllm_adapter.py
```
