# IFM/K2-Horizon-MoVA-36B-A4B on Blackhole QB2 (P300x2)

TTNN implementation of [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) (48 layers,
hidden 2560, 32 query / 8 KV heads of 128, MoE with 100 routed experts top-8 plus one shared expert, MoVA value
experts 64 top-4, 250K vocabulary, 36B total / ~4B active parameters) for the QB2 1x4 Blackhole mesh, served through
vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). Native 524,288-token context.
Produced by the tt-model-bringup agentic pipeline; tracking: tenstorrent/tt-agentic-bringup-qb2#96.
Moved here from tt-metal `models/demos/k2_horizon_mova_36b_a4b_qb2` (tt-metal#64652); `provenance/upstream.json`
records the source revision and the files changed since.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 48-layer model: embedding, decoder stack, final norm, LM head; checkpoint loading |
| `tt/multichip_decoder.py` | TP4 attention (8 Q / 2 KV heads per chip, head-sharded KV cache) and the MoE / MoVA layer: decode with 1-8 live rows runs the routed experts tensor-parallel, fuller decode batches and prefill run them expert-parallel (25 experts per chip) |
| `tt/optimized_decoder.py` | Single-chip layer programs, policies and constants the TP4 layer builds on |
| `tt/generator.py` | Traced decode, chunked prefill, on-device sampling (request top_k 1..64, sampled with k <= 32), prefix-cache resumes |
| `tt/generator_vllm.py` | vLLM adapter (`K2HorizonForCausalLM`), vLLM-owned paged KV cache, decode-input reload contract v1 |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision (below). `K2_PRECISION_CONFIG=<path>` loads another config |
| `tt/readiness_contract.py` | The `Generator` base class, vendored from the tt-model-bringup runtime |
| `vllm_metadata.json` | Architecture registration for the plugin's `EXTRA_MODELS_DIR` |
| `model/entry.py` | `register()`: registers the architecture with vLLM (used by `agentic-models check`) |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`; the metadata names the adapter |
| `model/tests/test_vllm_adapter.py` | Device test of the adapter on two real layers: KV-only intermediate prefill rows, eager vs traced prefill head, resident-input decode vs explicit reload over 160 steps |
| `model/tests/test_sampling_params.py` | Host-only test of the sampling parameters sent to the device: top_k 1..64 accepted, 33..64 sent as 32, greedy and padding rows, prefill and decode-reload uploads, plugin parameter conversion |

## Precision

`dense8_hifi2_bfp4experts`: BFP8 weights with HiFi2 for attention, the dense MLP and the value
experts; routed experts BFP4 (HiFi2); BF16 HiFi4 LM head; BFP8 KV cache; BF16 residual. Over a 3000-token GPQA
reasoning trace, teacher-forced against HF bf16, top-1 agreement is 96.5%. The bring-up's first selection (BFP4 LoFi
almost everywhere, chosen on 100-token teacher forcing) reached 85.2% there and scored 38.4 on GPQA Diamond.

## Requirements

- The root `pins.json` versions: tt-metal `tt_metal` with the `patches.json` patches, including
  `tt-metal-sampling-topk-logprobs-4-device.patch` (the top-K log-probs path on 4-device meshes), vllm-tt-plugin
  `vllm_tt_plugin` and vLLM `vllm`, with transformers 5.16 as vLLM 0.32 requires.
- Optional: vllm-tt-plugin#282 (exact host top-p from the top-1024 logits) speeds up host-sampled decode about 1.5x at
  32 users; nothing depends on it.
- Install this repository into the same Python environment (`pip install -e .`).

## Serving

The adapter is found through `EXTRA_MODELS_DIR=<repo>/src/agentic_models/models/k2_horizon_mova_36b_a4b` (the plugin
reads `vllm/vllm_metadata.json`), and selected with
`--hf-overrides '{"architectures": ["TTK2HorizonMoVAQB2ForCausalLM"]}'` (K2-Horizon-7B registers the checkpoint's
own `K2HorizonForCausalLM`). The tt-inference-server spec `--impl k2_horizon_mova_qb2` carries the full recipe; the
essential settings are `--block-size 128 --max-num-seqs 32 --max-model-len 524288 --max-num-batched-tokens 4096
--enable-prefix-caching --trust-remote-code --reasoning-parser k2_horizon --tool-call-parser k2_horizon` and the TT
config `sample_on_device_mode=all`, `trace_mode=all`, `trace_region_size=0`, `fabric_config=FABRIC_1D_RING`.

Requests with top_k 1..64 sample on device (`serving_capabilities["max_device_top_k"] = 64`, so the plugin keeps
them there). The device sampler, `ttnn.sampling`, takes k <= 32 only, and the shared `format_sampling_params` caps k
at 32, so a request top_k above 32 samples with k=32: GPQA's top_k 40 runs as 32.

`K2_VLLM_HOST_COMPAT=1` lets requests the device sampler cannot serve (top_k unset or above 64 with temperature > 0)
and prefill steps that contain an intermediate chunk sample on the host. Without it the adapter rejects them.

Prefix caching needs 128-token blocks: with 64-token blocks a resumed prefill does not reproduce the uncached logits
bit for bit.

## Results at the pinned versions (manual QB2 run, 2026-11-24)

tt-metal 58afa857 with the `patches.json` patches, vLLM 0.32.0, the TTI dev-spec server arguments, on one QB2 host:

- GPQA Diamond (198 questions, seed 42, T 1.0 / top-p 0.95 / top-k 40 requested): **81.31 (161/198)**, against the CI
  80.81 and the card 80.8; 1 h 57 m. Every decode step sampled on device (k=32). This run used vllm-tt-plugin f6fe76b
  plus #282.
- 1 user, 128/128: TTFT 100.7 ms, 34.5 ms/token (29.0 t/s/u). 32 users, 128/1024: 675 tok/s total.
- Host test `model/tests/test_sampling_params.py`: 34 passed. Device adapter test: 1 passed. tt-metal
  `models/common/tests/test_sampling.py -k log_probs`: 4 passed, 5 skipped (need 8 or 32 devices).

## Earlier results

Benchmarks: CI run [42017156688](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/42017156688),
32-slot server.

| Point | TTFT | Decode |
|---|---|---|
| 1 user, 128/128 | 104 ms | 34.6 ms/token (28.9 t/s/u) |
| 32 users, 128/128 | 3.0 s | 43.3 ms/token |
| 32 users, 128/1024 | 3.0 s | 669 tok/s total |
| 1 user, 8K-128K prompt | 4.8-150 s | 35-141 ms/token |

Accuracy, CI (thinking on, T 1.0 / top-p 0.95 / top-k 40 requested, sampled on device with k=32):
- GPQA Diamond: **80.81** (198 questions; card 80.8), [41957483979](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/41957483979), 2 h 03 m.
- Terminal-Bench 2.1, 5 tasks: **3/5**, [41986491344](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/41986491344).

## Known limitations

- **Single-user decode runs through the 32-row decode trace.** The model alone decodes about 2x faster at batch 1;
  fixed per-layer costs (single-core routing, two all-reduces per layer, the BF16 LM head) dominate at one user.
- **Prompts are prefilled one user at a time**, so 32 simultaneous new prompts wait about 3 s for their first token.
- **The KV pool is 524,288 tokens shared by all users.** Many long concurrent requests queue instead of running at once.
- **A prompt tail that is not a whole 128-token page prefills one token per step.** Prefill chunks are 128-token
  pages (power-of-two multiples) or single tokens, so up to 127 tail tokens add about 16 ms each: a 127-token prompt
  takes about 2.1 s to its first token, a 128-token prompt about 90 ms. Benchmarks with page-multiple prompt lengths
  do not show it.

## Test

```bash
agentic-models check --model k2_horizon_mova_36b_a4b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model k2_horizon_mova_36b_a4b --phase vllm    # CPU package and sampling-params tests + QB2 device test
```
