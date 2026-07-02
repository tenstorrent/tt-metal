# Qwen3-4B Full Model Work Log

## 2026-07-02

Selected skills:

- `$full-model` for full autoregressive model/generator bringup.
- `$tt-device-usage` for mesh use and serialized hardware validation.
- `$tt-enable-tracing` for traced decode/split-sampling checks.
- `$autofix` for decode correctness and sampling trace repair.
- `$stage-review` remains required before completion.

Implemented:

- `tt/model.py`: 36-layer `Qwen3FullModel`, replicated BF16 embedding/final norm, BFP4 TP4 LM head, shared RoPE, full-context paged KV.
- `tt/generator.py`: readiness generator, explicit cache/page-table API, traced free-running token-out, teacher-forcing compatibility, host-sampling compatibility.
- `tests/test_full_model_contract.py`.
- Readiness fixes for Qwen chat-template mapping output and P150_X4 mesh label.

Preserved optimized multichip policy:

- replicated inter-layer residual layout `[1, 1, M, 2560]`;
- TP4 internal projections/KV cache;
- BF16 paged KV, BFP4 weights, LoFi matmuls;
- persistent decode all-reduce resources;
- no single-chip or host-side decoder fallback.

Context contract:

- TTNN paged ops required 32-token KV blocks. Updated default from `2560 x 16` to `1280 x 32`, preserving `40960` context.
- Recomputed per-device KV bytes: `1,509,949,440` for 36 layers.
- `python .agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b` passed.

Important fixes:

- Removed full `40960 x 40960` eager causal-mask allocation; prefill relies on causal SDPA.
- Avoided per-layer 40960 RoPE duplication by using full-model shared RoPE tables.
- Shared persistent all-reduce scratch/index across layers to avoid L1 collisions.
- Added decode-side input RMSNorm to optimized and multichip decoder `_decode_qkv`; one-layer AIME boundary cosine improved to `0.993237`.
- Materialized captured model trace once after capture so first sampled logits are real traced logits.
- Rejected `Sampling1D` force-argmax for this sharded LM-head layout after the clean sampler decision probe returned `158` for a logits argmax of `198`.
- Selected `Sampling1D` top-k=1 (`p=0`, `temp=1`) as canonical greedy split sampling; first 20 traced teacher-forcing tokens matched host argmax exactly.

AIME24 reference:

```bash
python -m models.common.readiness_check.generate \
  --hf-model Qwen/Qwen3-4B \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt
```

Result: prompt `[1, 158]`, generated `[1, 100]`, top-k `[100, 100]`.

Validation:

```bash
python -m py_compile models/autoports/qwen_qwen3_4b/tt/model.py models/autoports/qwen_qwen3_4b/tt/generator.py models/autoports/qwen_qwen3_4b/tt/optimized_decoder.py models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py models/common/readiness_check/generate.py models/common/readiness_check/mesh_device.py
pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short
```

Result: `4 passed`.

Decoder regression after changing paged KV geometry to 32-token pages:

```bash
pytest -q models/autoports/qwen_qwen3_4b/tests/test_optimized_decoder.py models/autoports/qwen_qwen3_4b/tests/test_multichip_decoder.py --tb=short
```

Result: `20 passed, 4 skipped`.

Prefill readiness:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/qwen_qwen3_4b \
  --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING
```

Result: `top1=0.930 (93/100)`, `top5=1.000 (100/100)`, `top100=1.000 (100/100)`.

Teacher-forcing traced decode:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/qwen_qwen3_4b \
  --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING
```

Result: `top1=0.930 (93/100)`, `top5=1.000 (100/100)`, `top100=1.000 (100/100)`, `TTFT=846.76ms`, `decode=36.66 t/s/u`.

Batch-2 full-model mixed prompt smoke:

```text
PREFILL_ROW 0 prompt_len 16 first_token 151667
PREFILL_ROW 1 prompt_len 20 first_token 198
BATCH2_PREFILL_SHAPE (2, 1, 151936)
BATCH2_DECODE_SHAPE (2, 151936)
BATCH2_DECODE_FINITE True
BATCH2_DECODE_START_POS [16, 20]
BATCH2_DECODE_TOP1 [198, 32313]
```

Free-running qualitative:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/qwen_qwen3_4b \
  --hf-model Qwen/Qwen3-4B \
  --prompt-file models/common/readiness_check/autoregressive_prompt.txt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D_RING \
  --output-dir models/autoports/qwen_qwen3_4b/doc/full_model/autoregressive \
  --max-new-tokens 100
```

Result: HF and TT each produced 100 tokens. TT stayed coherent in English, with no repetition loop or wrong-language drift.

Prompt-correct qualitative suite:

- six prompts from `models/common/readiness_check/vllm_prompts.txt`;
- tokenizer class `Qwen2Tokenizer`, chat template present, prompt mode `chat`;
- prompts rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=True)`;
- HF greedy controls and TT greedy outputs saved in `qualitative/vllm_qualitative_outputs.json`;
- all six prompt verdicts pass in `qualitative/qualitative_verdict.json`.

Degenerate-output check:

```bash
python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/qwen_qwen3_4b --scope all --missing-artifacts advisory --json models/autoports/qwen_qwen3_4b/doc/full_model/degenerate_output_report.json
```

Result: exit `0`, no findings.

Token-out trace timing:

```text
TOKEN_OUT_TIMINGS {'ttft_ms': 332.5899839401245, 'decode_t/s/u': 37.693503211804, 'e2e_t/s/u': 33.79477518178094}
TOKEN_OUT_TRACE_COUNTERS {'trace_replays': 98, 'token_host_refreshes': 0, 'position_host_refreshes': 2, 'page_table_host_refreshes': 0, 'syncs': 2, 'readbacks': 99}
```

Artifacts:

- `run_prefill_check.log`
- `run_teacher_forcing.log`
- `run_autoregressive.log`
- `token_out_timing_trace.log`
- `teacher_forcing_trace_vs_host_20.log`
- `self_consistency_aime_boundary_after_norm.log`
- `sampler_decision_clean.log`
- `batch2_mixed_decode.log`
- `qualitative_chat_suite.log`
- `check_degenerate_output.log`

Remaining:

- independent `$stage-review clean-pass`;
- stage-owned commit after review fixes.
