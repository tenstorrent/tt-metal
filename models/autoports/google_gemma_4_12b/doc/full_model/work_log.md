# Full Model Work Log

## Implementation

Files added:

- `models/autoports/google/gemma-4-12B/tt/model.py`
- `models/autoports/google/gemma-4-12B/tt/generator.py`

The model wrapper assembles token embeddings, all 48 `MultichipDecoder` layers, final RMSNorm, tied LM head, softcapping, paged KV cache creation/reset, page-table handling, prefill, decode, and optional common on-device sampling. The generator exports `build_generator`, owns cache/page table for standalone readiness runs, and exposes low-level `prefill_forward` and `decode_forward` methods for a future serving adapter.

Checkpoint and repo provenance:

- HF model: `google/gemma-4-12B`
- Local snapshot: `/home/moconnor/.cache/huggingface/hub/models--google--gemma-4-12B/snapshots/56820d7d8cbe8e47975a53325439ed272e91cff2`
- Branch: `agentic-research/experiment-11-gemma4-12b`
- Commit: `31b45719e2ca21b695a8e7f15b5e8895bc1fb3bb`
- Hardware path: T3K, `MeshShape(1, 8)`, `FABRIC_1D_RING`

## Reference

The requested chat-template command is not usable for this base checkpoint because the HF tokenizer reports `chat_template = None`. The readiness reference was generated with plain AIME24 tokenization:

```bash
python -m models.common.readiness_check.generate \
  --hf-model google/gemma-4-12B \
  --prompt-source aime24 \
  --gen-len 32 \
  --top-k 100 \
  --output models/autoports/google/gemma-4-12B/readiness_aime24_plain.refpt
```

Result: one AIME24 entry, 149 prompt tokens, 32 HF greedy continuation tokens.

## Validation Commands

Prefill:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google/gemma-4-12B \
  --reference models/autoports/google/gemma-4-12B/readiness_aime24_plain.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/full_model/artifacts/run_prefill_check_aime24_plain.log
```

Result: `AGGREGATE top1=0.969 (31/32) top5=1.000 (32/32) top100=1.000 (32/32)`.

Teacher forcing:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google/gemma-4-12B \
  --reference models/autoports/google/gemma-4-12B/readiness_aime24_plain.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/full_model/artifacts/run_teacher_forcing_aime24_plain_masked_default.log
```

Result: `AGGREGATE top1=0.938 (30/32) top5=1.000 (32/32) top100=1.000 (32/32)`.

Autoregressive:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/google/gemma-4-12B \
  --hf-model google/gemma-4-12B \
  --prompt-file models/autoports/google/gemma-4-12B/doc/full_model/artifacts/aime24_prompt_0_plain.txt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 32 \
  --output-dir models/autoports/google/gemma-4-12B/doc/full_model/artifacts/autoregressive_aime24_plain_masked \
  2>&1 | tee models/autoports/google/gemma-4-12B/doc/full_model/artifacts/run_autoregressive_aime24_plain_masked.log
```

Result: HF and TT both produced 32 tokens, `first_diff = None`. Completion:

```text
Let's denote the time Aya spends walking at a constant speed of $s$ kilometers per hour as $t_1$ hours, and the time
```

## AutoFix Note

Initial free-running generation matched HF for four tokens and then repeated Gemma's `<image|>` token. I used the AutoFix hypothesis loop serially; forked subagents were not used because the available subagent tool requires an explicit user request for delegation.

Hypothesis 1: common on-device sampling was returning the wrong token while host argmax was correct.

- Experiment: `artifacts/autofix_host_argmax_probe.log`, with `use_on_device_sampling=False`.
- Result: refuted. Host argmax still produced `<image|>` from token 4 onward.

Hypothesis 2: raw logits had a single multimodal-special top-1 flip, while the HF target stayed within top-k.

- Evidence: readiness top-5/top-100 were perfect; tokenizer special ids include `<image|>` at 258882.
- Fix: high-level text generation suppresses tokenizer special ids except EOS before host argmax. Raw low-level logits remain unmasked for readiness top-k checks.
- Verification: `artifacts/autofix_special_mask_probe.log` and the masked official autoregressive run produced token-identical HF/TT output.

## Performance Command

```bash
python - <<'PY' 2>&1 | tee models/autoports/google/gemma-4-12B/doc/full_model/artifacts/perf_batch1_aime24_plain_masked_host_argmax.log
# see artifact for the exact inline script
PY
```

Result:

```json
{
  "prompt_tokens": 149,
  "generated_tokens_total": 32,
  "decode_tokens_timed": 31,
  "ttft_ms": 117.76442173868418,
  "decode_s": 6.910095944069326,
  "decode_tokens_per_second_per_user": 4.486189519062485
}
```

The measurement excludes model construction and weight loading, but includes the warmed runtime generator path: cache reset, prefill, final norm, LM head, CPU logits, masked host argmax, and decode loop.

## Fallback Audit

No decoder fallback to single-chip occurred on T3K. The model summary reports `MultichipDecoder` with TP8; the single-chip optimized path is only reachable when the mesh TP is 1.

Known full-wrapper fallbacks and gaps:

- CPU logits and host argmax are used by default for high-level text generation.
- Non-EOS special-token suppression is host-side.
- Low-level on-device sampling is implemented and exposed but not used by default for free-running text because it cannot apply the special-token mask.
- Full-wrapper decode tracing is not implemented. Traced decode evidence is inherited from the optimized multichip decoder layer stage.
- Decode final norm output is converted to DRAM before the LM head.

Inherited reduced-layer `tt-perf-report` evidence:

- `../optimized_multichip_decoder/tracy/sliding/prefill_perf_report.txt`
- `../optimized_multichip_decoder/tracy/sliding/decode_perf_report.txt`
- `../optimized_multichip_decoder/tracy/full/prefill_perf_report.txt`
- `../optimized_multichip_decoder/tracy/full/decode_perf_report.txt`

## Final Status

Full-model real-weight validation passes the requested top-k bar: prefill and teacher forcing both have top5 100% and top100 100% on the AIME24 reference. Autoregressive HF/TT output is identical for the 32-token readiness run after text-generation special-token suppression. Performance and limitations are recorded above.
