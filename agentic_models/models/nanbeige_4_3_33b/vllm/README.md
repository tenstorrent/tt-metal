# Nanbeige/Nanbeige4.3-33B on Blackhole QB2 (P300x2)

TTNN implementation of [Nanbeige4.3-33B](https://huggingface.co/Nanbeige/Nanbeige4.3-33B) for the QB2 1x4 Blackhole
mesh, served through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). The checkpoint is a
dense Llama (`LlamaForCausalLM`): 52 layers, hidden 7168, 56 query / 8 KV heads of 128, MLP 22016, 166,912-token
vocabulary, untied LM head (31.9B decode parameters, 33.1B with the embedding table). Tensor-parallel over the 4 chips.

Produced by the tt-model-bringup pipeline (0.2.0) on QB2 host m5, Nov 24-26 2026. The code is vendored from the
bring-up checkout (tt-metal `main` at 2f0c96ea, directory `nanbeige_nanbeige4_3_33b`; a root snapshot of tt-metal
v0.83.0-dev20261123 plus the pipeline commits), which is not published. `provenance/upstream.json` records the
source, the hash of every vendored file and the two files edited since.

## Layout

| Path | Content |
|---|---|
| `tt/model.py`, `tt/decoder.py` | 52-layer TP4 dense decoder, LM head, checkpoint loading, DRAM-sharded BFP8 projections |
| `tt/cache.py`, `tt/kernels/` | Model-owned `ttnn.generic_op` programs (partial paged KV fill, position masks, slot remaps, seed advance) and their kernels, compiled at run time against the pinned tt-metal headers |
| `tt/prefill_plan.py` | Shared-prefix page ownership and producer-before-reader ordering for batched prefill |
| `tt/generator.py` | Paged prefill (one row, or up to 16 rows per step), traced decode buckets 1/2/4/8/16, on-device sampling and penalties (`models.common.sampling` at the pin) |
| `tt/generator_vllm.py` | vLLM adapter `NanbeigeVLLM`; checkpoint resolution; the tt-metal opt-in switches |
| `tt/precision.py`, `config/precision.json` | Selected precision `stage06_all_linear_bfp8_kv_bfp8`: BFP8 weights, HiFi4 with FP32 accumulation, BFP8 KV cache, 589,824-token KV pool. `NANBEIGE_PRECISION_CONFIG=<path>` loads another policy |
| `vllm_metadata.json`, `model/entry.py` | Architecture registration (`register()` is used by `agentic-models check`) |
| `model/tests/` | CPU: vendored files match provenance, checkpoint source, the adapter's plugin contract |

## Requirements

- The root `pins.json` versions: tt-metal `tt_metal` with the `patches.json` patches, vllm-tt-plugin `vllm_tt_plugin`
  and vLLM `vllm` (0.32.0).
- tt-metal patches, all opt-in (the adapter sets the switches in its own process at import):
  - `nanbeige_4_3_33b-sampling-top64.patch` (`TTNN_NANBEIGE_SAMPLING_TOP64=1`, forced by the adapter): `ttnn.sampling`
    compiles 64 candidates. **Required**: the adapter declares `max_device_top_k = 64` and samples unset top-k on device
    from the top 64.
  - `nanbeige_4_3_33b-chunked-sdpa-per-row-starts.patch` (a `[batch]` `chunk_start_idx` tensor, which the pinned op
    rejects): one batched paged SDPA for multi-row prefill. **Required** for concurrent prefill.
  - `nanbeige_4_3_33b-fp32-precision.patch` (`TTNN_NANBEIGE_PRECISION=1`, default on; `0` turns it off): the bring-up's
    Blackhole FP32-preserving paths in RoPE, RMSNorm, line reduce-scatter, chunked SDPA prefill, paged SDPA decode and
    the BF16 -> BFP8 paged KV-cache update.
- vllm-tt-plugin patches, enabled only by capabilities this adapter declares: `vllm-tt-plugin-prefill-sampling-mask.patch`
  (`supports_prefill_sampling_mask`), `vllm-tt-plugin-original-prompt-lens.patch` (`supports_original_prompt_lens`) and
  `nanbeige_4_3_33b-plugin-sampling-routing.patch` (`supports_device_top_k_unset`, `reconcile_async_grammar_rows`).
- HF weights at the manifest revision: `MODEL_WEIGHTS_DIR` (or `HF_MODEL`) when it holds `config.json` and the
  safetensors (tt-inference-server), else `$HF_HOME` (`snapshot_download(local_files_only=True)`).
- Install this repository into the same Python environment (`pip install -e .`).

## Serving

The checkpoint's own architecture is kept; the adapter replaces the plugin's builtin `TTLlamaForCausalLM`
(tt-transformers Llama) through the class overrides:

```bash
export TT_MODEL_CLASS_OVERRIDES='TTLlamaForCausalLM=agentic_models.models.nanbeige_4_3_33b.vllm.tt.generator_vllm:NanbeigeVLLM'
export MESH_DEVICE=P300x2
```

Check that the server log shows `Applied TT_MODEL_CLASS_OVERRIDES` and `NANBEIGE_ADAPTER`. Arguments (the bring-up's
production server, `serve.sh`):

```bash
--model Nanbeige/Nanbeige4.3-33B --revision df919e1e2cf077a22204c78b6c826bf3db18bc08 \
  --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 896 --block-size 32 \
  --enable-prefix-caching --enable-chunked-prefill --async-scheduling --dtype bfloat16 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template-content-format string \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"tool_call_format":"xml"}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":-1}' \
  --additional-config '{"tt":{"trace_mode":"decode_only","enable_model_warmup":true,"sample_on_device_mode":"all",
    "seeded_sampling_policy":"auto","trace_region_size":536870912,"fabric_config":"FABRIC_1D"}}'
```

Slots must be 1, 2, 4, 8 or 16. Requests with top_k 1..64 or unset sample on device (unset = top 64, then top-p);
min_p, logit_bias, bad_words, allowed tokens and log-probs sample on the host.

## Results

Validation at the pinned versions is **pending** (`manifest.json`). Bring-up references (m5, 2026-11-26, tt-metal
v0.83.0-dev20261123 + bring-up commits, plugin f6fe76b + 6 commits, vLLM 5df63a7 + 2 commits):

- Accuracy: prefill KL vs fp32 0.0037 / 0.0027 / 0.0039 at 2K / 8K / 32K; top-1 agreement 98.9% on an agent transcript.
- GPQA Diamond (time-boxed partial run): 101/111 answered = 91.0% (card 89.5); the first 40 questions 36/40.
- SWE-bench Verified (10-task set): 9/10. Terminal-Bench 2.0 (10-task set): 4 of the 5 tasks that finished.
- Perf, 16-slot server:

| ISL/OSL | Users | TTFT mean (ms) | tok/s/user | Output tok/s |
|---|---:|---:|---:|---:|
| 128/128 | 1 / 16 | 389 / 908 | 26.2 / 23.0 | 24.5 / 318 |
| 1024/1024 | 1 / 16 | 777 / 4996 | 25.2 / 20.4 | 24.8 / 296 |
| 4096/512 | 1 / 16 | 2039 / 12047 | 23.9 / 10.5 | 21.8 / 134 |
| 32768/512 | 1 / 16 | 22766 / 120692 | 20.0 / 1.6 | 10.6 / 18.3 |

## Known limitations

- **Not yet validated at the pins.** The bring-up ran on tt-metal v0.83.0-dev20261123, about 1,245 commits after the
  pin. The FP32 precision paths were ported onto the pin's kernels (gated) and compile there, but their numerics are
  checked only by the pinned-version serving run.
- **Decode adds the all-reduce result as a separate op.** The bring-up fused the residual add into a one-shot
  `all_reduce_async(residual_tensor=...)`, which is newer than the pinned tt-metal (and its FP32 reduction patch is not
  carried). Decode keeps the composite `ttnn.all_reduce` + add that prefill already uses: slower per step.
- **vLLM changes not carried.** The bring-up's vLLM had two commits that the pinned vLLM 0.32.0 lacks: `reasoning_content`
  accepted as an alias of `reasoning` in chat history, and bad-word bans extended to equivalent single-token aliases.
  Clients that send prior reasoning as `reasoning_content` lose it from the prompt; `bad_words` bans only the tokens
  vLLM's tokenization finds.
- **No speculative decoding** (the checkpoint has no MTP or draft head). At most 16 sequences, 131,072-token context,
  one 589,824-token KV pool shared by all slots.

## Test

```bash
agentic-models check --model nanbeige_4_3_33b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model nanbeige_4_3_33b --phase vllm    # CPU package and adapter-contract tests
```
