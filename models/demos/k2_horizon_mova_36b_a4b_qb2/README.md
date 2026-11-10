# IFM/K2-Horizon-MoVA-36B-A4B on Blackhole QB2 (P300x2)

TTNN implementation of [K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) (48 layers,
hidden 2560, 32 query / 8 KV heads of 128, MoE with 100 routed experts top-8 plus one shared expert, MoVA value
experts 64 top-4, 250K vocabulary, 36B total / ~4B active parameters) for the QB2 1x4 Blackhole mesh, served through
vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). Native 524,288-token context.
Produced by the tt-model-bringup agentic pipeline; tracking: tenstorrent/tt-agentic-bringup-qb2#96.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 48-layer model: embedding, decoder stack, final norm, LM head; checkpoint loading |
| `tt/multichip_decoder.py` | TP4 attention (8 Q / 2 KV heads per chip, head-sharded KV cache) and the MoE / MoVA layer: decode with 1-8 live rows runs the routed experts tensor-parallel, fuller decode batches and prefill run them expert-parallel (25 experts per chip) |
| `tt/optimized_decoder.py` | Single-chip layer programs, policies and constants the TP4 layer builds on |
| `tt/generator.py` | Traced decode, chunked prefill, on-device sampling (top_k 1..64), prefix-cache resumes |
| `tt/generator_vllm.py` | vLLM adapter (`K2HorizonForCausalLM`), vLLM-owned paged KV cache, decode-input reload contract v1 |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision (below). `K2_PRECISION_CONFIG=<path>` loads another config |
| `tt/readiness_contract.py` | The `Generator` base class, vendored from the tt-model-bringup runtime |
| `vllm_metadata.json` | Architecture registration for `EXTRA_MODELS_DIR` |
| `tests/test_vllm_adapter.py` | Device test of the adapter on two real layers: KV-only intermediate prefill rows, eager vs traced prefill head, resident-input decode vs explicit reload over 160 steps |

## Precision

`dense8_hifi2_bfp4experts`: BFP8 weights with HiFi2 for attention, the dense MLP and the value
experts; routed experts BFP4 (HiFi2); BF16 HiFi4 LM head; BFP8 KV cache; BF16 residual. Over a 3000-token GPQA
reasoning trace, teacher-forced against HF bf16, top-1 agreement is 96.5%. The bring-up's first selection (BFP4 LoFi
almost everywhere, chosen on 100-token teacher forcing) reached 85.2% there and scored 38.4 on GPQA Diamond.

## Requirements

- tt-metal `main` with the `models/common/sampling` change from the same PR: the top-K logprob path runs on 4-device
  meshes.
- vllm-tt-plugin `main` with vLLM 0.30.

## Serving

The adapter is found through `EXTRA_MODELS_DIR=models/demos` and `vllm_metadata.json`, and selected with
`--hf-overrides '{"architectures": ["TTK2HorizonMoVAQB2ForCausalLM"]}'` (K2-Horizon-7B registers the checkpoint's
own `K2HorizonForCausalLM`). The tt-inference-server spec `--impl k2_horizon_mova_qb2` carries the full recipe; the
essential settings are `--block-size 128 --max-num-seqs 32 --max-model-len 524288 --max-num-batched-tokens 4096
--enable-prefix-caching --trust-remote-code --reasoning-parser k2_horizon --tool-call-parser k2_horizon` and the TT
config `sample_on_device_mode=all`, `trace_mode=all`, `trace_region_size=0`, `fabric_config=FABRIC_1D_RING`.

`K2_VLLM_HOST_COMPAT=1` lets requests the device sampler cannot serve (top_k unset or above 64 with temperature > 0)
and prefill steps that contain an intermediate chunk sample on the host. Without it the adapter rejects them.

Prefix caching needs 128-token blocks: with 64-token blocks a resumed prefill does not reproduce the uncached logits
bit for bit.

## Results

Benchmarks: CI run [42017156688](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/42017156688),
32-slot server.

| Point | TTFT | Decode |
|---|---|---|
| 1 user, 128/128 | 104 ms | 34.6 ms/token (28.9 t/s/u) |
| 32 users, 128/128 | 3.0 s | 43.3 ms/token |
| 32 users, 128/1024 | 3.0 s | 669 tok/s total |
| 1 user, 8K-128K prompt | 4.8-150 s | 35-141 ms/token |

Accuracy, CI (thinking on, T 1.0 / top-p 0.95 / top-k 40):
- GPQA Diamond: **80.81** (198 questions; card 80.8), [41957483979](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/41957483979), 2 h 03 m.
- Terminal-Bench 2.1, 5 tasks: **3/5**, [41986491344](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/41986491344).

## Known limitations

- **Single-user decode runs through the 32-row decode trace.** The model alone decodes about 2x faster at batch 1;
  fixed per-layer costs (single-core routing, two all-reduces per layer, the BF16 LM head) dominate at one user.
- **Prompts are prefilled one user at a time**, so 32 simultaneous new prompts wait about 3 s for their first token.
- **The KV pool is 524,288 tokens shared by all users.** Many long concurrent requests queue instead of running at once.

## Test

```bash
pytest models/demos/k2_horizon_mova_36b_a4b_qb2/tests/test_vllm_adapter.py
```
