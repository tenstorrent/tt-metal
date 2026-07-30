# google/gemma-4-12B Full Model

Batch-1 T3K result: TTFT 117.8 ms for a 149-token AIME24 prompt; decode throughput 4.49 tokens/s/user over 31 timed decode tokens.

This stage adds the repo-local full autoregressive TTNN path in `tt/model.py` and `tt/generator.py`. It uses the optimized TP8 multichip decoder stack from `tt/multichip_decoder.py` on `FABRIC_1D_RING`, loads real `google/gemma-4-12B` weights, applies tied embeddings as the LM head, exposes paged KV cache helpers, and implements readiness-compatible prefill/decode/generate boundaries.

## Accuracy

Main reference: `readiness_aime24_plain.refpt`, generated from DeepSeek AIME24 prompt 0 with 32 HF greedy continuation tokens. The HF tokenizer has `chat_template = None`, so the required chat-template AIME24 path is not available for this base checkpoint; no Tale-of-Two-Cities reference is used as the main gate.

| Check | Reference | Result |
| --- | --- | --- |
| `run_prefill_check` | `readiness_aime24_plain.refpt` | top1 31/32, top5 32/32, top100 32/32 |
| `run_teacher_forcing` | `readiness_aime24_plain.refpt` | top1 30/32, top5 32/32, top100 32/32 |
| `run_autoregressive` | AIME24 prompt 0 plain tokenization | HF and TT 32-token continuations are identical |

Autoregressive verdict: coherent, no repetition, no wrong-language drift, and no early divergence after suppressing non-EOS tokenizer special tokens during text generation.

## Performance

Measured with model and tensor caches loaded, after one warm generation. The measurement includes cache reset, prefill, final norm, LM head, CPU logits readback, host argmax, special-token suppression, and repeated decode.

```text
Prompt tokens: 149
TTFT: 117.764 ms
Timed decode tokens: 31
Decode wall time: 6.910 s
Decode throughput: 4.486 tokens/s/user
```

Full-wrapper decode is still eager and host-logit based. Traced decode evidence currently comes from the inherited optimized multichip decoder layer reports, not from a complete token-to-token full-wrapper trace. Relevant reduced-layer reports:

- `../optimized_multichip_decoder/tracy/sliding/decode_perf_report.txt`: sliding traced decode 679.406 us device time.
- `../optimized_multichip_decoder/tracy/full/decode_perf_report.txt`: full-attention traced decode 881.606 us device time.
- `../optimized_multichip_decoder/tracy/sliding/prefill_perf_report.txt`: sliding prefill 1487.611 us device time.
- `../optimized_multichip_decoder/tracy/full/prefill_perf_report.txt`: full-attention prefill 1713.979 us device time.

## Runtime Audit

Optimized path: T3K uses `MultichipDecoder` for all 48 layers with TP8 and Ring CCL. The `OptimizedDecoder fallback` path only appears when `tp == 1`; it was not used in the full-model T3K validation runs.

Host boundaries:

- Input IDs, decode positions, page table, RoPE caches, and weights are created with `ttnn.from_torch`.
- Readiness logits and generated tokens are returned with `ttnn.to_torch`.
- High-level generation defaults to CPU logits plus masked host argmax because raw TT top-1 selected Gemma's `<image|>` token at one decode step; the target token stayed top-5/top-100.
- Low-level `decode_forward(..., sample_on_device=True)` remains exposed for later serving work, but it is not the default text-generation path.

Other limitations:

- Decode final norm output is moved to DRAM before the LM head.
- Full-wrapper trace capture/replay is not implemented yet.
- Cache reset is device-side `ttnn.mul(..., output_tensor=tensor)` over owned KV tensors; external callers can pass their own cache/page table through low-level methods.

## Artifacts

- Reference: `../../readiness_aime24_plain.refpt`
- Prompt: `artifacts/aime24_prompt_0_plain.txt`
- Prefill log: `artifacts/run_prefill_check_aime24_plain.log`
- Teacher-forcing log: `artifacts/run_teacher_forcing_aime24_plain_masked_default.log`
- Autoregressive log: `artifacts/run_autoregressive_aime24_plain_masked.log`
- Autoregressive completions/meta: `artifacts/autoregressive_aime24_plain_masked/`
- Performance log: `artifacts/perf_batch1_aime24_plain_masked_host_argmax.log`
- AutoFix probes: `artifacts/autofix_host_argmax_probe.log`, `artifacts/autofix_special_mask_probe.log`
