# Aleph-Alpha/Kolibri-1 on Blackhole QB2 (P300x2)

TTNN implementation of [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1-BF16)
(78B MoE, 3.46B active: 384 routed experts top-6 plus one shared expert, 50 layers,
sliding-window and global attention) for the QB2 1x4 Blackhole mesh, TP4, served
through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin).
Native 262,144-token context. Produced by the tt-model-bringup agentic pipeline;
tracking: tenstorrent/tt-agentic-bringup-qb2#73.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 50-layer model: embedding, decoder stack, final norm, LM head |
| `tt/multichip_decoder.py`, `tt/optimized_decoder.py` | TP4 decoder layer: attention, MoE (indexed single-token path, token-grouped dispatch / fused SwiGLU / combine path for prefill and batched decode, expert-parallel weights), collectives |
| `tt/generator.py` | Traced decode, bucketed traced prefill, on-device sampling |
| `tt/generator_vllm.py` | vLLM adapter (`KolibriForCausalLM`) |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision: BFP4/LoFi decoder weights, BFP8/HiFi2 LM head, BF16 KV cache and logits |
| `tt/checkpoint.py` | Checkpoint lookup: `MODEL_WEIGHTS_DIR`/`HF_MODEL`, else the HF cache, pinned revision `7a8f290e` |
| `tt/readiness_contract.py` | Vendored generator contract from the bring-up runtime |
| `tests/test_vllm_adapter.py` | Device test of the vLLM adapter's decode-input reload contract |

## Serving

Build tt-metal and activate its Python environment, then install the pinned tt-transformers
package, which provides the sampler and LM head:
`uv pip install --no-deps -r models/autoports/aleph_alpha_kolibri_1_bf16/requirements.txt`. `--no-deps` keeps
the ttnn built from this tree; a resolving install would replace it.

Serve with vLLM and the vllm-tt-plugin (it registers the `kolibri1` config type, the
architecture, and the `kolibri1` reasoning and tool-call parsers). The tt-inference-server
catalog entry `kolibri_1_qb2` carries the full recipe; the essential settings are
`--block-size 32 --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 2048
--enable-chunked-prefill --reasoning-parser kolibri1 --enable-auto-tool-choice
--tool-call-parser kolibri1` and the TT config `sample_on_device_mode=all`,
`fabric_config=FABRIC_1D`, `trace_mode=all`. The device sampler covers top_k 1..32. Requests it cannot serve (top_k above 32, as in
the model card's eval sampling, penalties, logprobs, structured outputs) sample on the
host; `KOLIBRI_VLLM_HOST_COMPAT=0` refuses them instead.

Batched decode uses the token-grouped expert path with expert-parallel weights by default
(tt-agentic-bringup-qb2#75); `KOLIBRI_GROUPED_DECODE=0` and `KOLIBRI_EP_EXPERTS=0` restore
the dense sparse-matmul path.

## Results (32-slot server, CI)

| Point | TTFT | Decode |
|---|---|---|
| 1 user, 128/128 | 233 ms | 57 ms/token |
| 32 users, 128/128 | 3.2 s | 250 tok/s total |
| 32 users, 4096/128 | 5.5 s | 100 tok/s total |

GPQA Diamond 85.0 (80 questions, 64K thinking budget; card 84.8), AIME 2025 27/30,
MMLU-Pro 70.3 (card 80.1). Release run:
https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/38432395585

## Test

```bash
pytest models/autoports/aleph_alpha_kolibri_1_bf16/tests/test_vllm_adapter.py
```
