# ornith-ai/Ornith-1.5-35B-A3B on Blackhole QB2 (P300x2)

TTNN implementation of [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (Qwen3.5-MoE
architecture: 40 layers mixing Gated DeltaNet linear attention with full attention, hidden 2048, 256 routed experts
top-8 plus a shared expert, ~3B active parameters) for the QB2 1x4 Blackhole mesh (TP4 attention, EP4 experts), served
through vLLM with the [vllm-tt-plugin](https://github.com/tenstorrent/vllm-tt-plugin). Text only: the vision encoder
and the MTP head are not served. Logical context 262,144 tokens per request, 32 fixed request slots.
Produced by the tt-model-bringup agentic pipeline on a QB2 bring-up host; `provenance/upstream.json` records the
source checkout, commit and every file changed since.

## Layout

| Path | Content |
|---|---|
| `tt/model.py` | 40-layer model: embedding, decoder stack, final norm, LM head; checkpoint loading; the measured KV-capacity gate |
| `tt/multichip_decoder.py` | TP4/EP4 layer: attention / Gated DeltaNet on 4 chips, MoE dispatch, experts and combine |
| `tt/optimized_decoder.py` | Single-chip layer programs, policies and constants the TP4/EP4 layer builds on |
| `tt/generator.py` | Traced decode over 32 fixed slots, chunked prefill, on-device sampling and penalties, recurrent-state slots |
| `tt/generator_vllm.py` | vLLM adapter (`OrnithForCausalLM`), vLLM-owned paged KV cache, decode-input reload contract v1 |
| `tt/serving_observer.py`, `tt/benchmark_timing.py` | Opt-in host telemetry (`ORNITH_VLLM_TRACE_EVENTS=1`, `ORNITH_BENCHMARK_TIMING_DIR`) |
| `tt/precision.py`, `doc/datatype_sweep/selected_precision_config.json` | Selected precision (below) |
| `doc/full_model/kv_capacity.json`, `doc/optimized_vllm/native_tail_capacity/` | Measured KV pool size and the evidence files the gate requires |
| `tt/readiness_contract.py` | The `Generator` base class, vendored from the tt-model-bringup 0.1.24 runtime |
| `vllm_metadata.json` | The architecture (`Qwen3_5MoeForConditionalGeneration`) and the adapter class |
| `model/entry.py` | `register()`: registers the adapter for `TTQwen3_5MoeForConditionalGeneration`, replacing any earlier entry |
| `model/tests/test_package.py` | CPU: vendored files match `provenance/upstream.json`; the capacity gate names the vendored sources; no bring-up paths |
| `model/tests/test_vllm_adapter.py` | CPU: capabilities, device top-k limit 32, plugin keyword arguments, KV budget, forked-op call sites, registration |

## Precision

`bfp8_attention_lofi_bf16kv`: BFP8 weights for attention, routed and shared experts and the LM head; LoFi for
attention decode and the experts, HiFi2 LM head, HiFi4 attention prefill and router; BF16 activations, residual and CCL; **BF16 KV cache** (BFP8 KV failed the bring-up's per-user prefix check);
FP32 recurrent state. The bring-up measured at least 91.99% top-1 against HF over AIME24-chat and real-text references.

## Requirements

- tt-metal `tt_metal` from the root `pins.json` with the `patches.json` patches, including these, which this model
  calls:
  - `tt-metal-op-nomic-moe-grouped-matmul-routing.patch`: `ttnn.experimental.nomic_embed_text_v2_moe.routed_combine`.
  - `tt-metal-op-kda-indexed-entry-copy.patch`: `ttnn.experimental.kda.indexed_entry_copy`.
  - `tt-metal-bringup-moe-fused-swiglu.patch`: `ttnn.bringup.moe_fused_swiglu`, a fork of the newer
    `deepseek_prefill` op with an end-of-reader write/atomic drain for traced serving.
  - `tt-metal-bringup-post-combine-reduce.patch`: `ttnn.bringup.post_combine_reduce`, a fork of the newer
    `deepseek_prefill` op with per-token weights and `combine_row_index`.
- vllm-tt-plugin `vllm_tt_plugin` with the `patches.json` `vllm_tt_plugin` patches:
  - `vllm-tt-plugin-prefill-state-slot-continuation.patch` (capability `preserve_prefill_state_slots`): a continuing
    chunked prefill keeps its recurrent-state slot.
  - `vllm-tt-plugin-prefill-sampling-mask.patch` (capability `supports_prefill_sampling_mask`): intermediate
    chunked-prefill rows stay on the device-sampling route and receive `prefill_sampling_mask`.
- vLLM `vllm` (0.32.0) with transformers 5.16; this repository installed in the same environment (`pip install -e .`).
- The checkpoint at revision `10fbf86fed7ecee4a061f8b499a618f46001cac1` in `$HF_HOME`, or in `MODEL_WEIGHTS_DIR` (or
  `HF_MODEL`) when it is a local checkpoint directory (tt-inference-server exports it). The model never downloads.

## Serving

The checkpoint's architecture, `Qwen3_5MoeForConditionalGeneration`, is also mapped by the plugin to its in-tree
Qwen3.5 demo, so this adapter is selected with the plugin's class override, not `EXTRA_MODELS_DIR`/`--hf-overrides`:

```bash
export TT_MODEL_CLASS_OVERRIDES=TTQwen3_5MoeForConditionalGeneration=agentic_models.models.ornith_1_5_35b_a3b.vllm.tt.generator_vllm:OrnithForCausalLM
export MESH_DEVICE=P300x2 VLLM_RPC_TIMEOUT=1800000 ORNITH_VLLM_TRACE_EVENTS=0
python -m vllm.entrypoints.openai.api_server --model ornith-ai/Ornith-1.5-35B-A3B \
  --revision 10fbf86fed7ecee4a061f8b499a618f46001cac1 --tokenizer-revision 10fbf86fed7ecee4a061f8b499a618f46001cac1 \
  --served-model-name ornith-ai/Ornith-1.5-35B-A3B ornith \
  --block-size 32 --max-num-seqs 32 --max-model-len 262144 --max-num-batched-tokens 2048 \
  --additional-config '{"tt": {"sample_on_device_mode": "all", "trace_region_size": 134217728, "fabric_config": "FABRIC_1D"}}' \
  --async-scheduling --language-model-only --max-logprobs -1 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- Check the startup log for `Applied TT_MODEL_CLASS_OVERRIDES` and `ORNITH_ADAPTER ...generator_vllm:OrnithForCausalLM`;
  without the override the plugin's builtin demo serves silently.
- Prefix caching stays off (the model declares `supports_prefix_caching: False`; KV pages alone do not restore the
  recurrent state).
- The KV pool comes from `doc/full_model/kv_capacity.json` (60,620 blocks, 1,938,816 schedulable tokens). The model
  refuses to start when the hashed runtime sources differ from that record. A cold start, including trace capture,
  can take tens of minutes; allow a health timeout of at least 3600 s.
- Card sampling: thinking on, temperature 1.0, top-p 0.95, top-k 20. GPQA needs `max_tokens` up to 262,144 minus the
  prompt (a 32K cap truncates about 26 of 198 answers).

## Results

At the pinned versions: **pending** (`manifest.json` `validation.status`). Bring-up reference (tt-metal at the bring-up
base, vllm-tt-plugin 042a9486 + c018beb, vLLM 0.31.0, 32 slots):

- GPQA Diamond: **93.9 (186/198)**, card 89.2; thinking on, T 1.0 / top-p 0.95 / top-k 20, no truncations, 7.75 h.
- 1 user, 128/128: TTFT 94 ms, 66.7 ms/token (15.0 t/s/u). 32 users, 128/128: 11.9 t/s/u, 295 tok/s total.
  4K prompt TTFT 643 ms; 32K prompt TTFT 5.9 s.
- KL(HF||TT) on reasoning positions: prefill 0.034 (top-1 0.969), decode B1 0.045, decode B32 0.039.

## Known limitations

- **Decode always runs the fixed 32-row trace**, about 67 ms/step even for one user.
- **Prompts are prefilled one user at a time**; 32 simultaneous long prompts queue for their first token.
- **Device sampling takes top_k 1..32.** The bring-up sampled up to top_k 64 on device with a widened
  `ttnn.sampling`, which is not shipped; requests with top_k 33..64 (and log-prob requests) sample on the host.
- **The KV-capacity record was measured at the bring-up base**, not at the pinned tt-metal. Its source hashes were
  updated to the vendored files (`measured_source_sha256` keeps the measured ones); re-measure at the pins before
  relying on the full 60,620-block pool, and lower `--max-num-seqs` if the pool does not fit.
- **The bring-up ran 658 tt-metal commits newer than the pin.** Two of its op changes are not carried (the
  multicore `ttnn.topk` for the router, a perf-only change, and the separate expert down-projection format, unused by
  the selected BFP8 policy); kernels such as softplus and the sampler RNG differ at the pin. Accuracy and perf at the
  pins are re-validated before `validation.status` changes.
- **Seeded sampling is reproducible only within an identical batch layout.**

## Test

```bash
agentic-models check --model ornith_1_5_35b_a3b --phase vllm   # imports the adapter, registers the architecture
agentic-models test --model ornith_1_5_35b_a3b --phase vllm    # CPU package and adapter-contract tests
```
