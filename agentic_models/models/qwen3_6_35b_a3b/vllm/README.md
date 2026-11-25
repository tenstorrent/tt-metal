# Qwen/Qwen3.6-35B-A3B on Blackhole QB2 (P300x2)

TTNN implementation of [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (40 layers: 30 Gated DeltaNet
linear-attention layers and 10 full-attention layers; MoE with 256 routed experts top-8 plus a shared expert; hidden
2048; MTP head; 262,144-token context; 35B total / ~3B active parameters) for the QB2 1x4 Blackhole mesh, served
text-only through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). Produced by the
tt-model-bringup agentic pipeline on a QB2 host; the bring-up repository is not public, so
`provenance/upstream.json` records its host path, commit (`32f487a9`, branch `perf/multiuser`), the hash of every
vendored file and the files changed here.

## Layout

| Path | Content |
|---|---|
| `tt/generator_vllm.py` | vLLM adapter (`QwenForCausalLM`): capabilities, vLLM-owned paged KV cache for the 10 attention layers, decode-input reload contract v1, MTP spec-decode hooks |
| `tt/fast_generator.py`, `tt/fast_gdn.py`, `tt/fast_attn.py` | The served multi-user path (`QWEN_FAST=1`): 32-row decode traces per 8/16/24/32-row extent, chunked GDN prefill, paged SDPA |
| `tt/fast_spec.py`, `tt/fast_mtp.py` | MTP speculative decoding (`QWEN_SPEC=K`): device drafter, verify through the decode graph |
| `tt/prefix_cache.py`, `tt/cache_contract.py` | GDN state snapshots at 32-token block prefixes, so vLLM prefix-cache hits restore recurrent state |
| `tt/model.py`, `tt/multichip_decoder.py`, `tt/fused_decoder.py`, `tt/optimized_decoder.py`, `tt/mesh_*.py`, ... | Model, TP4 attention and EP4 / TP4 routed MoE decoder, checkpoint loading |
| `tt/generator.py`, `tt/serving*.py`, `tt/slot_state.py`, `tt/prefill*.py` | Base generator, serving command layer, on-device sampling, chunked prefill |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision (below) |
| `tt/readiness_contract.py` | The `Generator` base class, vendored from the tt-model-bringup 0.1.24 runtime |
| `vllm_metadata.json` | The checkpoint architecture and the adapter class |
| `model/entry.py` | `register()`: registers the architecture with vLLM, overwriting any earlier entry (used by `agentic-models check`) |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`; no bring-up package path left; metadata and registration |
| `model/tests/test_vllm_adapter.py` | CPU: capabilities, plugin hook keywords, config-time hooks, and the vendored edits listed below |

## Precision

`baseline_bfp8`: BFP8 weights for every group; LoFi for the routed experts, HiFi2 for attention, the shared expert
and the LM head; BF16 KV cache, activations and residual; FP32 recurrent state with ordered FP32 linear-attention
arithmetic; prefill experts at HiFi4 (`QWEN_PREFILL_EXPERT_FID`). Bring-up readiness: prefill top-1/5/100
0.98/1.00/1.00, teacher forcing 1.00, 32K-token KL against HF fp32 <= 0.0024, flat 3K-token decode drift ~0.0044.

## Requirements

- The root `pins.json` versions: tt-metal `tt_metal` 58afa857, vllm-tt-plugin `vllm_tt_plugin` f6fe76b and vLLM
  0.32.0 (transformers 5.16), with these `patches.json` patches:
  - tt-metal, shared: `tt-metal-bringup-moe-fused-swiglu.patch` and `tt-metal-bringup-post-combine-reduce.patch`
    (`ttnn.bringup.moe_fused_swiglu` / `ttnn.bringup.post_combine_reduce`: the tt-metal 8c48b4e4 versions of the
    DeepSeek-prefill MoE ops this model was brought up on, plus `stochastic_rounding` and `valid_token_rows`; the
    pinned ops stay unchanged for their other users).
  - tt-metal, this model: `qwen3_6_35b_a3b-chunk-gdn-exact-state.patch` and
    `qwen3_6_35b_a3b-fused-recurrent-gdn-exact-state.patch` (`exact_state=True` on
    `ttnn.transformer.chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule`; default off, this model passes
    True). With the default kernels the bring-up's decode drifted (3K-token KL 0.085 instead of 0.0044).
  - vllm-tt-plugin, shared: `vllm-tt-plugin-prefill-state-slot-continuation.patch` (a continuing chunked prefill
    keeps its state slot; the adapter declares `preserve_prefill_state_slots`).
  - vllm-tt-plugin, this model: `qwen3_6_35b_a3b-plugin-hybrid-align-chunked-mm.patch` (prefix caching in vLLM's
    mamba "align" mode), `qwen3_6_35b_a3b-plugin-decode-slot-order.patch` (`TT_DECODE_SLOT_ORDER=1`) and
    `qwen3_6_35b_a3b-plugin-prefill-wait-for-grid.patch` (`TT_PREFILL_WAIT_FOR_GRID=1`); both env switches default off.
- Install this repository into the same Python environment (`pip install -e .`).
- The checkpoint at revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7`. When `--model` is a repo id, the model reads
  `MODEL_WEIGHTS_DIR` (or `HF_MODEL`) when it is a local checkpoint directory (tt-inference-server exports it), else
  `$HF_HOME/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<revision>`.

## Serving

The adapter is selected with `TT_MODEL_CLASS_OVERRIDES`, and the checkpoint keeps its own
`Qwen3_5MoeForConditionalGeneration` architecture (no `--hf-overrides`, no `EXTRA_MODELS_DIR`), so vLLM's hybrid-model
handling is the validated one. The server log must show `Applied TT_MODEL_CLASS_OVERRIDES`.

```bash
export TT_MODEL_CLASS_OVERRIDES='TTQwen3_5MoeForConditionalGeneration=agentic_models.models.qwen3_6_35b_a3b.vllm.tt.generator_vllm:QwenForCausalLM'
export MESH_DEVICE=P300x2 VLLM_RPC_TIMEOUT=300000
export QWEN_FAST=1 QWEN_KV_POOL_TOKENS=655360 QWEN_PREFILL_ROWS=2048 \
       QWEN_SPEC=3 QWEN_SPEC_MAX_ROWS=4 QWEN_PREFIX_CACHE=1 QWEN_PREFIX_SNAPSHOTS=32
export TT_DECODE_SLOT_ORDER=1 TT_PREFILL_WAIT_FOR_GRID=1
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-35B-A3B \
  --revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7 --tokenizer-revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7 \
  --block-size 32 --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 2048 \
  --async-scheduling --enable-prefix-caching --max-logprobs -1 --language-model-only \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 3}' \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 805306368, "fabric_config": "FABRIC_1D", "trace_mode": "decode_only", "prefill_chunk_alignment": 256}}'
```

- `QWEN_FAST=1` is required: without it the adapter serves the original, much slower bring-up path. The capabilities are
  computed from `QWEN_FAST`, `QWEN_SPEC` and `QWEN_PREFIX_CACHE` when the adapter is imported, so set them before the
  server starts.
- `--async-scheduling` is what the bring-up launcher's `QWEN_VLLM_ASYNC=1` selects.
- `QWEN_KV_POOL_TOKENS` is the KV pool shared by all requests (default 262,144; ~10 KiB per token per chip).
- Prefix caching runs in vLLM's mamba "align" mode; the model restores GDN state from its own snapshots
  (`QWEN_PREFIX_SNAPSHOTS`).
- `trace_mode` must be `decode_only` (prefill runs prepared eager programs).
- Cold start is ~5-8 minutes with a warm kernel cache; the first start also compiles ~67 prefill shapes.

## Results

Not yet validated at the pinned versions (`manifest.json` validation `pending`). Bring-up references (tt-metal
8c48b4e4, vllm-tt-plugin 042a9486 with its local patches, vLLM 0.31), which are not the bar at the pins:

- GPQA Diamond (198, thinking on, card sampling T 1.0 / top-p 0.95 / top-k 20 / presence penalty 1.5, full-context
  budget): 168/198 = 84.8 (card 86.0).
- vllm bench serve, random, greedy: 128/128 1 user 47.2 tok/s/user, TTFT 0.18 s; 1024/1024 16 users 19.5 tok/s/user,
  294 tok/s total.

## Known limitations

- **Device sampling takes top_k <= 32** (`max_device_top_k` 32, the `ttnn.sampling` limit at the pin). The plugin
  samples requests with a larger or unset top_k on the host. The bring-up sampled top_k up to 64 on device with a
  tt-metal change that is not carried here; card sampling (top_k 20) is unaffected. Sampled tokens are not
  bit-identical to the bring-up's (the pinned sampler draws differently).
- **MTP drafts only while at most `QWEN_SPEC_MAX_ROWS` (4) requests are live**, and not for requests with penalties.
  Draft acceptance has been seen to drop late in a long-lived server.
- **Text only.** Image inputs are rejected.
- **Resumed prefill is not bit-identical to a single prefill**; the PCC difference reproduces in HF and was accepted
  at bring-up.
- **Long prompts at 16 users**: 4K-token prompts reach a p90 time to first token of ~11 s at the default 2048-token
  batch budget.
- **Memory headroom is small**: the 655,360-token pool with MTP left ~0.3 GiB per DRAM bank free at bring-up. If the
  server runs out of device memory at the pins, lower `QWEN_KV_POOL_TOKENS`.

## Test

```bash
agentic-models check --model qwen3_6_35b_a3b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model qwen3_6_35b_a3b --phase vllm    # CPU package and adapter tests
```
