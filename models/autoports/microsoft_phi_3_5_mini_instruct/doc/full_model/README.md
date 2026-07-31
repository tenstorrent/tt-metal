# Full Model

This stage adds the full HF autoregressive path for `microsoft/Phi-3.5-mini-instruct` on the optimized 1x8 multichip decoder. It implements token embedding, the 32-layer optimized decoder stack, final RMSNorm, vocab-sharded LM head, paged KV cache ownership, and the standard Metal readiness generator contract in:

- `models/autoports/microsoft_phi_3_5_mini_instruct/tt/model.py`
- `models/autoports/microsoft_phi_3_5_mini_instruct/tt/generator.py`

## Strategy

- Mesh: 1x8 T3K ring, `FABRIC_1D_RING`.
- Decoder stack: 32 `MultichipDecoder` layers, preserving the optimized multichip decoder policy.
- Inter-layer residual: replicated BF16 `[1, 1, T, 3072]` between decoder layers.
- In-layer tensor parallelism: width sharding only inside a layer, with collectives after attention `o_proj` and MLP `down_proj`.
- Decode math: LoFi decode kernels, BF8 CCL payloads with BF16 casts, paged KV cache, and optimized local matmul policy from the decoder stage.
- Prefill math: BF16/HiFi2 prefill path.
- LM head: vocab-sharded output across the 8 devices for split sampling; no full-vocab all-gather for token-out decode.
- Sampling: `SamplingGenerator` with force-argmax disabled. Greedy uses the same top-k/top-p-capable path with `temperature=0.0`, `top_k=1`, `top_p=0.0`.

## Reference

Fresh AIME24 chat-template reference:

```bash
python -m models.common.readiness_check.generate \
  --hf-model microsoft/Phi-3.5-mini-instruct \
  --revision 2fe192450127e6a83f7441aef6e3ca586c338b77 \
  --no-model-trust-remote-code \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt
```

The reference metadata records the HF model id, revision, tokenizer, native HF model class, prompt source, chat-template flag, generation length, top-k, and command. The pinned remote Phi CausalLM code is incompatible with this checkout's Transformers cache API and produced invalid degenerate logits, so the reference uses the native Transformers `transformers.models.phi3.modeling_phi3.Phi3ForCausalLM` implementation with the same HF revision and tokenizer.

Generated reference summary:

- Artifact: `models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt`
- Prompt: AIME24 index 0, chat-template tokenized length 161.
- Continuation length: 100 tokens.
- Continuation starts: `Let's break down the problem step by step:`

## Accuracy

Prefill command:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --reference models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result:

```text
entry[0]   top1=0.960 (96/100)  top5=1.000 (100/100)  top100=1.000 (100/100)
AGGREGATE  top1=0.960 (96/100)  top5=1.000 (100/100)  top100=1.000 (100/100)
```

Teacher-forcing command:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --reference models/autoports/microsoft_phi_3_5_mini_instruct/readiness_aime24_chat_100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result:

```text
entry[0]   top1=0.910 (91/100)  top5=1.000 (100/100)  top100=1.000 (100/100)  TTFT=221.54ms  decode=36.88 t/s/u  e2e=34.42 t/s/u
AGGREGATE  top1=0.910 (91/100)  top5=1.000 (100/100)  top100=1.000 (100/100)  TTFT=221.54ms  decode=36.88 t/s/u  e2e=34.42 t/s/u
```

## Autoregressive

Command:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/microsoft_phi_3_5_mini_instruct \
  --hf-model microsoft/Phi-3.5-mini-instruct \
  --hf-revision 2fe192450127e6a83f7441aef6e3ca586c338b77 \
  --no-model-trust-remote-code \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 100 \
  --output-dir models/autoports/microsoft_phi_3_5_mini_instruct/readiness_autoregressive
```

Artifacts:

- `readiness_autoregressive/hf_completion.txt`
- `readiness_autoregressive/tt_completion.txt`
- `readiness_autoregressive/autoregressive_meta.json`
- `readiness_autoregressive/degenerate_report.json`

Qualitative verdict: HF and TT both produce coherent English story continuations. TT diverges early in content but continues the prompt naturally, with no repetition, wrong-language drift, or early EOS. The degeneracy checker reports `No degenerate output detected`.

Token-out TT-only perf on the same 100-token prompt:

```text
TTFT=265.98ms
decode_tokens=99
decode_elapsed_s=2.5802
decode_t/s/u=38.37
e2e_t/s/u=35.14
```

## Trace Evidence

Trace evidence command was a TT-only script that ran greedy decode, same-page trace reuse, changed-page recapture, and top-k/top-p sampling on the same generator.

Key counters after 100-token token-out greedy:

```text
model_trace_captures=1
model_trace_replays=99
sampling_trace_captures=1
sampled_token_readbacks=99
full_logits_decode_readbacks=0
device_position_advances=99
device_token_feedbacks=99
page_table_changed_refreshes=0
```

Changed page-table evidence from the trace probe:

```text
same-page reuse: model_trace_captures=1, sampling_trace_captures=1, page_table_changed_refreshes=0
new-page recapture: model_trace_captures=2, sampling_trace_captures=2, page_table_changed_refreshes=1
top-k/top-p sample: '! How are you'
```

The first traced decode bug found during bring-up was that the sampler capture call returned the unchanged output tensor on its capture step. The generator now pre-captures the sampler trace during trace setup, resets the token input, and uses only sampler trace replay for real decoded tokens.

## Runtime Fallback Audit

- Model path: full model uses all 32 optimized `MultichipDecoder` layers. No single-chip or replicated decoder fallback is present.
- Embedding path: token embedding is hidden-dim sharded and gathered once to the replicated inter-layer residual layout required by the optimized decoder.
- LM head path: prefill uses vocab-sharded logits with host readback only for readiness scoring; decode leaves logits vocab-sharded for split sampling.
- Decode host boundary: traced decode performs no host argmax and no full-logit readback. The only per-token host readback is the sampled token id required by the readiness generator return contract.
- Sampling: force-argmax is disabled, `SAMPLING_AG_CONFIG` is not set, and no `TopKDeviceOperation` symbol is used in the autoport. Token-out greedy uses canonical split sampling with local top-k candidates and gathered top-k values/indices, not full-vocab all-gather.
- Token feedback: sampler writes the sampled token into the persistent traced token tensor. There is no Python token-feedback loop and no per-token host rebuild of token inputs, RoPE position state, masks, or page tables.
- Position feedback: the model trace advances the persistent `current_pos` tensor with `ttnn.plus_one`; counters show one device position advance per decode token.
- Page table: unchanged page tables reuse the trace. Changed page-table tensor identity releases and recaptures the model and sampler traces.
- Cache ownership: the generator owns the paged KV cache. `reset()` fills cache tensors in place and resets sampling params without rebuilding the model path.

## Limitations

- The native Transformers Phi3 implementation is used for HF full-model reference generation because the pinned remote CausalLM code is incompatible with the current Transformers cache API in this checkout.
- `run_prefill_check` and `run_teacher_forcing` emit nanobind leak diagnostics on process exit; the checks complete and close the mesh.
- Creating a brand-new page-table tensor while a trace is still active can trigger a TTNN allocation warning in ad hoc probes. The generator handles a supplied changed page table by releasing and recapturing traces.
