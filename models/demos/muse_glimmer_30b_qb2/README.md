# meta-models/Muse-Glimmer-30B on Blackhole QB2 (P300x2)

TTNN implementation of the [Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
text decoder (dense, 52 layers, 3 sliding-window (2048) : 1 global attention, GQA 32 / 2,
gated attention, RoPE on sliding layers only) for the QB2 1x4 Blackhole mesh, TP4, served
through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin).
Native 131,072-token context. The perception encoder (image input) is not ported. Produced
by the tt-model-bringup agentic pipeline; tracking: tenstorrent/tt-agentic-bringup-qb2#87.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 52-layer model: embedding, decoder stack, final norm, LM head, logit softcap |
| `tt/multichip_decoder.py`, `tt/optimized_decoder.py` | TP4 decoder layer. Each chip owns one KV head (ownership `[0,0,1,1]`) so GQA grouping survives TP 4 > 2 KV heads |
| `tt/generator.py`, `tt/prepared_prefill.py` | Traced decode, chunked prefill, on-device sampling (top_k 1..32) |
| `tt/generator_vllm.py` | vLLM adapter (`MuseGlimmerForConditionalGeneration`), decode-input reload contract v1 |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision: BFP4/LoFi projections and LM head, BFP8 KV cache, BF16 residual and logits. `MUSE_PRECISION_CONFIG=baseline` restores the pre-sweep policy |
| `tt/readiness_contract.py` | Vendored generator contract from the bring-up runtime |
| `tt/benchmark_timing.py` | Optional phase timing (`MUSE_PHASE_EVENTS`) |
| `vllm_metadata.json` | Architecture registration for `EXTRA_MODELS_DIR` |
| `requirements.txt` | Pinned [tt-transformers](https://github.com/tenstorrent/tt-transformers) (sampler, LM head, lazy weights) |
| `tests/test_vllm_adapter.py` | Device test of the adapter: resident-input decode vs reload, sync/async readback, host-sampled prefill |

## Setup

Build tt-metal and activate its Python environment, then install the pinned tt-transformers
package, which provides the sampler and LM head:
`uv pip install --no-deps -r models/demos/muse_glimmer_30b_qb2/requirements.txt`. `--no-deps` keeps
the ttnn built from this tree; a resolving install would replace it.

## Serving

vLLM (0.30) ships the `muse_glimmer` HF config and reasoning/tool parsers, so the plugin
needs no model registration; the adapter is found through `EXTRA_MODELS_DIR` and
`vllm_metadata.json`. The tt-inference-server catalog entry `muse_glimmer_30b_qb2` carries
the full recipe; the essential settings are `--block-size 32 --max-num-seqs 32
--max-model-len 131072 --max-num-batched-tokens 512 --enable-chunked-prefill
--no-enable-prefix-caching --reasoning-parser muse_glimmer --enable-auto-tool-choice
--tool-call-parser muse_glimmer` and the TT config `sample_on_device_mode=all`,
`fabric_config=FABRIC_1D_RING`, `trace_region_size=0`.

The device sampler covers top_k 1..32. Requests it cannot serve (the model card's top_k 64,
penalties, logprobs, structured outputs) and prefill steps with an intermediate chunk sample
on the host; `MUSE_VLLM_ALLOW_HOST_SAMPLING=0` refuses them instead.

## Results (CI release run, 32-slot server, vllm-tt-plugin main)

| Point | TTFT | Decode |
|---|---|---|
| 1 user, 128/128 | 85 ms | 17.2 ms/token |
| 32 users, 128/128 | 0.94 s | 996 tok/s total |
| 1 user, 4096/128 | 1.4 s | 19.2 ms/token |
| 1 user, 32768/128 | 22.6 s | 25.5 ms/token |
| 1 user, 65536/128 | 70.2 s | 32.6 ms/token |

AIME 2026 93.3 (30 problems, 64K budget; card 94.7), GPQA Diamond 83.75 (80 questions;
card 83.5, GPU reference 82.83), SWE-bench Verified 6/10, Terminal-Bench 2.1 2/10.
Long-prompt prefill (~1.3-1.5K tok/s, prefix caching off) is the main performance gap.

## Test

```bash
pytest models/demos/muse_glimmer_30b_qb2/tests/test_vllm_adapter.py
```
