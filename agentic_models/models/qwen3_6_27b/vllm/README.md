# Qwen/Qwen3.6-27B on Blackhole QB2 (P300x2)

TTNN implementation of [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) for the QB2 1x4 Blackhole mesh, served
through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). The model is the hybrid
Qwen3.5 text stack: 64 layers, of which 48 are Gated DeltaNet (linear attention) and 16 are full attention. It has
hidden 5120, 24 query / 4 KV heads of 256, a 248,320-token vocabulary and one MTP head. Native 262,144-token context.
The port is tensor-parallel over 4 chips. vLLM owns the paged KV cache of the 16 full-attention layers; the model owns
the GDN recurrent state per slot, plus optional recurrent-state snapshots for prefix caching. The MTP head drafts
speculative tokens.

Produced by the tt-model-bringup agentic pipeline on a QB2 host (tt-metal `models/autoports/qwen_qwen3_6_27b`,
branch `perf/multiuser` at c4912f0d). The bring-up repository is not published. `provenance/upstream.json` records
the host path, the commit, the sha256 of every vendored source file and of every file edited since, and what each
edit does. This phase sits next to the Galaxy `reload` phase of the same model and shares no code with it.

## Layout

| Path | Content |
|---|---|
| `tt/generator_vllm.py` | vLLM adapter `QwenVLLM`: capabilities, the KV-cache spec (block 32, full-attention layers only), prefill/decode, MTP verify and propose, async decode readback |
| `tt/generator.py` | `QwenGenerator`: traced decode buckets, chunked prefill, on-device sampling and penalties, slot-owned GDN state, prefix-cache snapshots, MTP |
| `tt/model.py`, `tt/multichip_decoder.py`, `tt/optimized_decoder.py` | Model, checkpoint loading (HF revision pinned in `tt/model.py`), TP4 decoder layers |
| `tt/*_candidate.py`, `tt/flat_prefill.py`, `tt/bounded_prefill.py`, ... | The kernels and programs the decoder selects (the import closure of `generator_vllm`). Bring-up modules outside that closure are not vendored |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Precision policy. The shipped config is the bring-up's config P, `bfp8_baseline` |
| `tt/sampling/` | `models.common.sampling` `generator.py` / `tt_sampling.py` from the tt-metal pin, with the bring-up's `max_top_k` changes. The model builds both samplers with width 32 |
| `tt/kernels/qkv_causal_conv1d_silu.cpp` | The tt-metal pin's kda conv compute kernel source. `tt/tiled_prefill_conv_candidate.py` rewrites it into a generic-op kernel |
| `tt/readiness_contract.py` | The `Generator` base class, vendored from the tt-model-bringup 0.1.24 runtime |
| `vllm_metadata.json` | HF architecture and adapter class |
| `model/entry.py` | `register()`: registers `TTQwen3_5ForConditionalGeneration` with vLLM. It overwrites any earlier registration, so the plugin builtin is replaced (used by `agentic-models check`) |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`, no bring-up module paths remain, the metadata names the adapter, register() overwrites |
| `model/tests/test_vllm_adapter.py` | CPU: capabilities, the prefill kwargs handed to the generator, device sampler width, MMRS NoC routing, the kda kernel text, every module imports without a device |

## Precision

`bfp8_baseline` (config P): every weight is BFP8. Fidelity is HiFi2 for the GDN-layer MLPs, HiFi4 for GDN attention,
and LoFi for the full-attention layers. The LM head is BF16 HiFi4, the KV cache is BFP8, and activations, residual
and CCL are BF16. The GDN recurrent state persists in FP32 between steps (`QWEN_PERF_STATE_FP32=1`, the default).
Against fp32 HF, the 32K-token decode KL is 0.00079, with top-1 agreement of 99.1-99.5%. The bring-up's stage-8
selection (`gdn_attn8_head8_mlp4_lofi`, BFP4 inner MLPs) is not shipped. `QWEN_VLLM_PRECISION_CONFIG=<path>` loads
another config.

## Requirements

- tt-metal at the root `pins.json` `tt_metal` with the `patches.json` patches, including:
  - `tt-metal-op-matmul-reduce-scatter-prefill.patch`: `ttnn.experimental.matmul_reduce_scatter_prefill`, which is
    missing at the pin. It is the fused prefill row-parallel projection (GDN out-proj and MLP down-proj) for prefill
    chunks of more than 256 rows.
  - `qwen3_6_27b-matmul-reduce-scatter-prefill-dedicated-read-noc.patch`: the `dedicated_read_noc` option of that op,
    default off. This model passes `True`, which is the NoC routing it was validated with.
  - `qwen3_6_27b-reshape-tiled-compact-mapping.patch`: compact reshape mapping pages, active only with
    `TTNN_RESHAPE_TILED_COMPACT_MAPPING=1`. Without it, 32-slot preparation runs out of device memory (14.7 GB/device
    of cached reshape mapping tensors).
- vllm-tt-plugin at `pins.json` `vllm_tt_plugin` with the `patches.json` `vllm_tt_plugin` patches. Each is gated by
  a capability this adapter declares:
  - `preserve_prefill_state_slots` (P1): a continuing chunked prefill keeps its GDN state slot.
  - `supports_prefill_sampling_mask` (P2): chunked-prefill batches keep device sampling, with
    `prefill_sampling_mask` naming the final rows.
  - `supports_original_prompt_lens` (P3): device penalties tell prompt tokens from replayed output.
  - `prefix_cache_recurrent_state_in_model` (P4): prefix caching with model-owned recurrent state. vLLM serves plain
    attention APC (`mamba_cache_mode` none) and the model restores GDN snapshots. Without P4, engine init fails when
    `--enable-prefix-caching` is set.
- vLLM `pins.json` `vllm` (0.32.0), with the transformers release vLLM 0.32 installs (5.16; `tt/model.py` imports
  `transformers.models.qwen3_5`).
- HF weights at the revision in `manifest.json` (`8f33b98937c6031bb49e713dee3e8e185aafd26f`), already in the HF
  cache. The model loads them with `local_files_only=True`, or from `MODEL_WEIGHTS_DIR` (or `HF_MODEL`) when it is a
  local checkpoint directory (tt-inference-server exports it).
- This repository installed into the same Python environment (`pip install -e .`).

## Serving

The checkpoint's architecture `Qwen3_5ForConditionalGeneration` resolves to `TTQwen3_5ForConditionalGeneration`,
which is also a plugin builtin (the tt-metal `models/demos/blackhole/qwen36` port). Select this adapter with
`TT_MODEL_CLASS_OVERRIDES`. The plugin applies it first and unconditionally. Neither `EXTRA_MODELS_DIR` nor
`--hf-overrides` is used. A correct start logs `Applied TT_MODEL_CLASS_OVERRIDES: TTQwen3_5ForConditionalGeneration
-> agentic_models.models.qwen3_6_27b.vllm.tt.generator_vllm:QwenVLLM` and, after warm-up, a `QWEN_VLLM_PRECISION`
line with `"config_id": "bfp8_baseline"`.

```bash
export TT_MODEL_CLASS_OVERRIDES=TTQwen3_5ForConditionalGeneration=agentic_models.models.qwen3_6_27b.vllm.tt.generator_vllm:QwenVLLM
export QWEN_VLLM_KV_POOL_TOKENS=1048576 QWEN_VLLM_PREFIX_CACHE=32 QWEN_VLLM_SPEC_K=3
export QWEN_VLLM_HOST_COMPATIBILITY=1 QWEN_VLLM_TELEMETRY=0 TT_METAL_TRACE_ALLOC_TRACKING=0
export TTNN_RESHAPE_TILED_COMPACT_MAPPING=1
export VLLM_RPC_TIMEOUT=900000            # cold start compiles 64 layers, 512 prefill shapes and MTP
vllm serve Qwen/Qwen3.6-27B --block-size 32 --max-num-seqs 32 --max-model-len 262144 --dtype bfloat16 \
  --max-num-batched-tokens 512 --enable-chunked-prefill --enable-prefix-caching --language-model-only \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"enable_thinking": false}' --async-scheduling \
  --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter", "num_speculative_tokens": 3}' \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 400000000, "fabric_config": "FABRIC_1D", "trace_mode": "all", "seeded_sampling_policy": "auto"}}'
```

Environment variables:

- `QWEN_VLLM_KV_POOL_TOKENS`: the attention KV pool shared by all users (default 262,144). The spec value, 1,048,576,
  keeps concurrent long reasoning requests from being preempted. Preemption with model-owned GDN state has not been
  validated.
- `QWEN_VLLM_PREFIX_CACHE=N`: keeps N recurrent-state snapshots and enables prefix caching. 0 turns it off, in which
  case also drop `--enable-prefix-caching`.
- `QWEN_VLLM_SPEC_K=K`: MTP speculative decoding with 1-3 tokens. It must match `num_speculative_tokens`; 0 turns it
  off, in which case also drop `--speculative-config`.
- `QWEN_VLLM_HOST_COMPATIBILITY=1`: lets requests the device sampler cannot serve run on the host. These are top_k
  above 32, top_k unset with temperature > 0, and log-probs. Without it the adapter rejects them.
- `TTNN_RESHAPE_TILED_COMPACT_MAPPING=1`: required for 32 slots; see Requirements.

For a server health check, allow at least 3600 s: a warm start takes about 6 minutes, a cold one up to about
33 minutes.

## Results

Not yet validated at the repository pins (`manifest.json` `validation.status` is `pending`). The bring-up reference
is tt-metal v0.83.0-dev20261108 + the bring-up commits, vllm-tt-plugin 042a9486 + its patches, vLLM 0.31.0, with the
serving command above. Pin measurements will replace these numbers.

- GPQA Diamond (198 questions, card sampling T 1.0 / top-p 0.95 / top-k 20, thinking on, max_tokens 131072,
  concurrency 16): **85.9 (170/198)**, 2 truncated; card 87.8. With an 81,920-token budget the score drops to about
  80, because of truncation.
- SWE-bench Verified, gold-validated 50-task subset: 76.0 (38/50). Terminal-Bench 2.0 (89 tasks, 3 h cap): 44.9.
- Perf (cold, greedy, MTP K=3, prefix cache on):

  | ISL/OSL | c1 tok/s/user | c1 TTFT | c16 tok/s/user | c16 output tok/s |
  |---|---|---|---|---|
  | 128/128 | 58.6 | 137 ms | 27.7 | 352.2 |
  | 1024/1024 | 73.8 | 317 ms | 34.3 | 506.5 |
  | 4096/512 | 58.8 | 1189 ms | 18.0 | 226.7 |

## Known limitations

- **Device sampling is limited to top_k <= 32.** `ttnn.sampling` at the tt-metal pin takes k <= 32, and the model
  declares `max_device_top_k` 32. Requests with a larger or unset top_k sample on the host, which needs
  `QWEN_VLLM_HOST_COMPATIBILITY=1`. The card's top_k 20 stays on device. The bring-up ran with a 64-wide device
  sampler, so top_k 33..64 behaves differently from the bring-up.
- **MTP drafts only for up to 16 live rows.** With K=3 it drafts for up to 8 rows and with K=1 for up to 16. Above
  16 live rows the drafter declines and decode runs without speculation.
- **Text only.** The adapter serves `--language-model-only` and requires zero mRoPE deltas.
- **block_size must be 32** (the selected KV layout); `get_kv_cache_spec` rejects other sizes.
- **Long cold start**, up to about 33 minutes, so TTI health and `VLLM_RPC_TIMEOUT` must allow for it.
- **Bring-up numbers are not pin numbers.** The pin has 658 commits of op drift from the bring-up base: reduce-scatter,
  all-gather, SDPA and matmul, plus the 16-bit sampling RNG instead of 24-bit. Perf and accuracy are re-measured at
  the pins before the validation block is filled.
- Terminal-Bench 2.0 at 44.9 is below the GPU reference (53.9).

## Test

```bash
agentic-models check --model qwen3_6_27b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model qwen3_6_27b --phase vllm    # CPU package and adapter tests
```
