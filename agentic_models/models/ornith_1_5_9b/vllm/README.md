# ornith-ai/Ornith-1.5-9B on Blackhole QB2 (P300x2)

TTNN implementation of [Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) for the QB2 1x4 Blackhole
mesh, served through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). The checkpoint is
a Qwen3.5 hybrid (`Qwen3_5ForConditionalGeneration`; served text-only): 32 layers, hidden 4096, 24 Gated DeltaNet
layers and 8 full-attention layers (16 query / 4 KV heads of 256, every fourth layer), 248K vocabulary, native
262,144-token context. Tensor-parallel over the 4 chips.

Produced by the tt-model-bringup agentic pipeline on a QB2 host. The code is vendored from the bring-up checkout
(tt-metal branch `metrics/gpqa-gap` at b43970c8d, directory `ornith_ai_ornith_1_5_9b`), which is not published.
`provenance/upstream.json` records the source, the hash of every vendored file and the files edited since.

## Layout

| Path | Content |
|---|---|
| `tt/model.py`, `tt/multichip_decoder.py` | 32-layer TP4 text model: GDN and full-attention decoder layers, LM head, checkpoint loading |
| `tt/conv_gates.py`, `tt/flat_ordered_conv_gates.py`, `tt/partial_norm_rope.py`, `tt/history_row_write.py`, `tt/inactive_state_copy.py`, `tt/kernels/` | Model-owned `ttnn.generic_op` programs and their kernels (compiled at run time against the pinned tt-metal headers) |
| `tt/prefill_*.py` | Chunked prefill planning, staging and operators |
| `tt/generator.py`, `tt/batch_state.py`, `tt/sampling_state.py`, `tt/decode_contract.py`, `tt/cache_contract.py`, `tt/generator_contract.py` | Traced decode, fixed 32 slots, slot-owned recurrent state, on-device sampling and penalties, KV pool sizing |
| `tt/generator_vllm.py` | vLLM adapter `OrnithForCausalLM` (and the opt-in MTP variant `OrnithMTPForCausalLM`, `tt/mtp_spec.py`, `tt/spec_chain.py`) |
| `tt/prefix_snapshots.py` | GDN state snapshots at prompt ends that make vLLM prefix caching exact for this hybrid model |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision `bfp8_all_hifi2`: BFP8 weights with HiFi2 matmuls everywhere, BFP8 KV cache, FP32 recurrent state. `ORNITH_PRECISION_CONFIG=<path>` loads another config |
| `tt/sampling/` | `models/common/sampling` `generator.py` and `tt_sampling.py` from the pinned tt-metal, plus the bring-up's `max_top_k` plumbing. Vendored because the model uses the sampler's internals, and so tt-metal patches to the shared sampler cannot change this model |
| `tt/_tt_transformers/` | `LazyWeight` and `LMHead1D` (the DRAM decode head) and their helpers from tt-transformers 25ebcdb, which agentic-models does not install |
| `vllm_metadata.json`, `model/entry.py` | Architecture registration (`register()` is used by `agentic-models check`) |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`, no bring-up import paths remain, the metadata names the adapter |
| `model/tests/test_vllm_adapter.py` | CPU: the capabilities and kwargs the plugin relies on, device top-k 32, vendored sampler, KV spec and pool, registration |

## Requirements

- The root `pins.json` versions: tt-metal `tt_metal` with the `patches.json` patches, vllm-tt-plugin `vllm_tt_plugin`
  and vLLM `vllm` (0.32.0), with the transformers version vLLM 0.32 resolves.
- Two vllm-tt-plugin patches, both enabled only by capabilities this adapter declares:
  - `vllm-tt-plugin-original-prompt-lens.patch` (capability `supports_original_prompt_lens`): the plugin passes each
    request's full prompt length to `prefill_forward`. **Required**: without it the adapter refuses every prefill.
  - `vllm-tt-plugin-prefill-state-slot-continuation.patch` (capability `preserve_prefill_state_slots`): a continuing
    chunked prefill keeps its recurrent-state slot. Without it, concurrent chunked prefills (prompts above
    `--max-num-batched-tokens` arriving together) can move a request to another slot and silently corrupt its GDN
    state.
- No model-specific tt-metal patch. `ttnn.experimental.matmul_reduce_scatter_prefill` is called only by the
  `fused_prefill` path of `tt/multichip_decoder.py`, which nothing enables.
- HF weights at the manifest revision in `$HF_HOME` (`snapshot_download(local_files_only=True)`), or
  `ORNITH_CHECKPOINT=<dir>`.
- Install this repository into the same Python environment (`pip install -e .`).

## Serving

The checkpoint's own architecture is kept, and the adapter replaces the plugin's builtin
`TTQwen3_5ForConditionalGeneration` (the in-tree Qwen3.5 port) through the plugin's class overrides:

```bash
export TT_MODEL_CLASS_OVERRIDES='TTQwen3_5ForConditionalGeneration=agentic_models.models.ornith_1_5_9b.vllm.tt.generator_vllm:OrnithForCausalLM'
```

Neither `EXTRA_MODELS_DIR` nor `--hf-overrides` is used. Check that the server log shows the plugin's
`Applied TT_MODEL_CLASS_OVERRIDES` line: without it the builtin port serves this checkpoint.

Environment: `MESH_DEVICE=P300x2`, `VLLM_RPC_TIMEOUT=7200000`, `TT_METAL_TRACE_ALLOC_TRACKING=0`,
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.

Arguments (the tt-inference-server spec, branch `vkovacevic/ornith-1.5-9b-ci` d67d443f7, plus the final bring-up
serving changes):

```bash
--model ornith-ai/Ornith-1.5-9B --revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 \
  --tokenizer-revision 489cb97981b8654bcfcf30ce1f94ed1b62e07b53 \
  --block-size 32 --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 4096 \
  --enable-prefix-caching --enable-prompt-tokens-details --language-model-only --async-scheduling \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' --override-generation-config '{"top_k": 20}' \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_mode": "decode_only",
    "trace_region_size": 200000000, "fabric_config": "FABRIC_1D",
    "decode_interleave_prefill_steps": 1, "decode_interleave_decode_steps": 2}}'
```

Thinking is off by default; GPQA turns it on per request and needs a 131,072-token generation budget (`max_gen_toks`)
to reach the reference: with 32K it lands near 81.

Model settings (environment, production defaults in brackets): `ORNITH_PREFIX_CACHE` [1],
`ORNITH_SNAPSHOT_MIN_TOKENS` [2048], `ORNITH_PREFIX_SNAPSHOTS` [64], `ORNITH_DECODE_BUCKETS` [1,2,4,8,16; 32 always],
`ORNITH_KV_POOL_TOKENS` [1,048,576 BFP8 tokens shared by all slots], `ORNITH_HISTORY_SKIP` [1],
`ORNITH_PRECISION_CONFIG` [unset: the selected config].

Requests with top_k 1..32 sample on device (`max_device_top_k = 32`, the `ttnn.sampling` limit at the pinned
tt-metal), including the card's top_k 20. Larger top_k, and host-only parameters (`min_p`, `logit_bias`,
`bad_words`, log-probs), sample on the host, which is 3-12x slower per step.

## Results

Validation at the pinned versions is **pending** (`manifest.json`). Bring-up references, measured on tt-metal
8c48b4e4 + the bring-up branch, vllm-tt-plugin 042a9486 + 3 commits and vLLM 0.31.0, with a 64-wide device sampler:

- GPQA Diamond (198 questions, thinking on, T 1.0 / top-p 0.95 / top-k 20 / presence 1.5, 131,072-token budget):
  89.39 / 86.36 / 86.87, mean **87.54** (card 86.4). Measured before the bring-up's prefill GDN exact-state change,
  which is not carried here.
- Perf (round 6 final, b43970c8d):

| ISL/OSL | Users | TTFT mean (ms) | tok/s/user | Output tok/s |
|---|---:|---:|---:|---:|
| 128/128 | 1 / 16 | 64 / 817 | 102.4 / 88.2 | 98 / 907 |
| 1024/1024 | 1 / 16 | 112 / 1001 | 101.4 / 77.6 | 100 / 1152 |
| 4096/512 | 1 / 16 | 273 / 2596 | 100.4 / 42.6 | 95 / 557 |
| 32768/512 | 1 / 8 | 2401 / 8605 | 95.3 / 20.4 | 66 / 121 |

## Known limitations

- **Not yet validated at the pins.** The bring-up ran 658 tt-metal commits later, with plugin 042a9486 and vLLM
  0.31. Kernel drift between the two tt-metal versions (LLK changes in the GDN gate math, the sampler's 16-bit random
  threshold) can move numerics and perf; the KL, GPQA and perf gates are re-run at the pins before the manifest is
  marked passed.
- **Prefill uses the pinned `chunk_gated_delta_rule`.** The bring-up's later exact FP32 prefill state (prefill KL
  0.0044 to 0.0017) edited that shared op for every caller and is not carried. If the KL gate regresses, the model can
  opt into a flag-gated version of the op change.
- **Prefix caching patches vLLM.** `tt/prefix_snapshots.py` wraps `KVCacheManager.get_computed_blocks` (installed in
  `initialize_vllm_model`) to trim hits to the last GDN snapshot, and forces vLLM's hybrid `mamba_cache_mode` to
  `none`. `ORNITH_PREFIX_CACHE=0` (and no `--enable-prefix-caching`) turns it off at a TTFT cost on agent traffic.
- **Decode-only traces, 32-token pages.** `trace_mode=all` and other block sizes are rejected. At most 32 sequences.
- **One shared KV pool.** 1,048,576 tokens (about 3.5 GiB per chip) for all slots; 2M does not fit. Many long
  concurrent requests queue.
- **MTP is opt-in and unvalidated here.** `OrnithMTPForCausalLM` (select it in `TT_MODEL_CLASS_OVERRIDES`,
  `ORNITH_MTP=K`, at most 16 sequences, no async scheduling) was not part of the production configuration.
- Serving longer than about 10 h hit a SIGBUS in the tt-metal realtime profiler during the bring-up; split very long
  eval runs.

## Test

```bash
agentic-models check --model ornith_1_5_9b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model ornith_1_5_9b --phase vllm    # CPU package and adapter-contract tests
```
