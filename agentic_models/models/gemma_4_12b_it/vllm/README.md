# google/gemma-4-12B-it on Blackhole QB2 (P300x2)

TTNN implementation of the [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) text decoder (dense,
48 layers, 5 sliding-window (1024) : 1 full attention, 16 query heads, KV heads 8 x 256 on sliding and 1 x 512 on full
layers, tied 262K vocabulary, final logit soft-cap 30) for the QB2 1x4 Blackhole mesh, TP4, served through vLLM with
the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). Native 262,144-token context. The image and audio
encoders are not ported. Produced by the tt-model-bringup agentic pipeline; tracking:
tenstorrent/tt-agentic-bringup-qb2#93. Moved here from tt-metal `models/demos/gemma4_12b_qb2` (tt-metal#61987);
`provenance/upstream.json` records the source revision and the files changed since.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 48-layer model: embedding, decoder stack, final norm, LM head, logit soft-cap |
| `tt/multichip_decoder.py` | TP4 decoder layer: QKV / gate / up column-parallel, o_proj / down row-parallel; the full layers' single KV head is replicated on every chip |
| `tt/optimized_decoder.py`, `tt/fused_decoder.py`, `tt/functional_decoder.py` | Single-chip layer programs, policies and constants the TP4 layer builds on |
| `tt/generator.py`, `tt/sampling.py` | Traced decode, chunked prefill, on-device sampling (top_k 1..32) |
| `tt/generator_vllm.py` | vLLM adapter (`Gemma4_12BQB2ForCausalLM`), vLLM-owned hybrid KV cache, decode-input reload contract v1 |
| `tt/precision.py`, `tt/precision_config.json` | Selected precision: BFP8 weights on every layer, the tanh-approximated GELU (gelu_pytorch_tanh, as the checkpoint defines it), BFP4 LoFi LM head, BFP8 KV cache; the config's `selection` block records how it was chosen. `GEMMA4_PRECISION_CONFIG=<path>` loads another config |
| `vllm_metadata.json` | Architecture registration for the plugin's `EXTRA_MODELS_DIR` |
| `model/entry.py` | `register()`: registers the architecture with vLLM (used by `agentic-models check`) |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`; the metadata names the adapter |
| `model/tests/test_vllm_adapter.py` | Device test of the adapter: resident-input decode vs reload, chunked prefill, seeded rows |

## Requirements

- The root `pins.json` versions: tt-metal `tt_metal` with the `patches.json` patches (the model needs none of its own),
  vllm-tt-plugin `vllm_tt_plugin` (unmodified) and vLLM `vllm`, with transformers 5.16 as vLLM 0.32 requires.
  Long-prefix full attention runs the stock chunked SDPA (attention PCC 0.9991 at a 262K prefix on one chip); the
  SDPA precision recipes (tt-metal #59522, #60197) are the planned follow-up.
- Install this repository into the same Python environment (`pip install -e .`).

## Serving

The adapter is found through `EXTRA_MODELS_DIR=<repo>/src/agentic_models/models/gemma_4_12b_it` (the plugin reads
`vllm/vllm_metadata.json`), and selected with
`--hf-overrides '{"architectures": ["TTGemma4_12BQB2ForCausalLM"]}'`. The tt-inference-server spec
`--impl gemma4-12b-qb2` carries the full recipe; the essential settings are `--block-size 64 --max-num-seqs 32
--max-model-len 262144 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching
--reasoning-parser gemma4 --override-generation-config '{"top_k": 32}'` and the TT config
`sample_on_device_mode=all`, `fabric_config=FABRIC_1D_RING`, `trace_region_size=0`, `prefill_chunk_alignment=8192`.

The device sampler covers top_k 1..32, so the server caps the checkpoint's top_k 64 at 32. Requests it cannot serve
(penalties, logprobs, top_k > 32) and prefill steps with an intermediate chunk sample on the host.

## Results at the pinned versions (manual QB2 run, 2026-11-23)

tt-metal 58afa857 with the `patches.json` patches, vLLM 0.32.0, vllm-tt-plugin f6fe76b, the TTI dev-spec server
arguments, on one QB2 host:

- GPQA Diamond (198 questions, seed 42, thinking on, T 1.0 / top-k 20 / top-p 0.95): **144/198 (72.7)**; pass mark
  135, same-protocol GPU reference 147 (74.24).
- tt-inference-server spec tests: 22/22 in 2 of 2 runs, including `test_penalties` 9/9.
- 1 user, 128/128: TTFT 35.6 ms, 13.2 ms/token. 32 users: 1,425 tok/s at 128/128, 1,773 tok/s at 128/1024.
- Device adapter test: 6 passed.

## Earlier results (CI, 32-slot server, vllm-tt-plugin main)

All numbers are for the shipped precision (`tt/precision_config.json`). Run ids are tenstorrent/tt-agentic-bringup-qb2
Actions runs. Release run 41887592398 ran on runner p05t05, the slowest QB2 runner (about 15% slower decode than the
others); release run 42036961299 measured perf within 1-2% of it.

Performance, release run 42036961299 (tt-metal 3901a3ff: early-November main plus this model, stock SDPA):

| Point | TTFT | Decode |
|---|---|---|
| 1 user, 128/128 | 38 ms | 15.9 ms/token |
| 32 users, 128/128 | 0.94 s | 1,272 tok/s total |
| 1 user, 4096/128 | 220 ms | 19.1 ms/token |
| 32 users, 4096/128 | 2.30 s | 433 tok/s total |
| 1 user, 32768/128 | 2.44 s | 25.3 ms/token |
| 1 user, 131072/128 | 20.4 s | 46.2 ms/token |

Accuracy (thinking on, T 1.0 / top-k 20 / top-p 0.95):
- GPQA Diamond (198 questions): **72.7**, the mean of seeds 42 / 43 / 44 = 69.7 / 75.3 / 73.2 (runs 41887592398 /
  41927441152 / 41927449480, tt-metal 774ba9b3, which still passed the since-dropped SDPA `fp32_output_accumulation`
  flag on full-attention prefill). On the stock SDPA, release run 42036961299 scored 71.7 at seed 42. Same-protocol
  GPU reference 74.24, card 78.8.
- LiveCodeBench: **72.5** (probe run 42033697767; GPU reference 73.28, card 72).

Against HF bf16 on CPU (local), teacher-forced on GPQA reasoning traces: top-1 agreement 97.6%,
KL(HF || TT) x 1000 = 7.7.

## Known limitations

- **Paged decode SDPA can read one page-table entry past a row** when `k_chunk_size` does not divide
  `row_width * block_size`. `_table_for_k_chunk` (`tt/optimized_decoder.py`) pads the row with block 0 up to a whole
  number of K chunks; the padded keys lie past every query's causal range, and nothing is padded when the width
  already divides (vLLM's rows).
- **Prefix caching is off.**
- **Decode attention uses the batch-32 core split at every batch** (`sdpa_decode_max_cores`, `tt/optimized_decoder.py`)
  so a seeded sampled request gives the same text whatever the server's load. One user decodes slower at long
  context than with the op default's per-batch split (19.1 / 25.3 / 46.2 ms/token at 4K / 32K / 128K in the CI table
  above; the gap to the op default has not been re-measured on the shipped precision). 32 users were unchanged when
  the split was introduced. The user-split full-layer attention (8+ users past
  `decode_user_split_min_position`) still has its own reduction order.

## Test

```bash
agentic-models check --model gemma_4_12b_it --phase vllm     # imports the adapter, registers the architecture
agentic-models test --model gemma_4_12b_it --phase vllm      # CPU package test + device adapter test (QB2)
```
