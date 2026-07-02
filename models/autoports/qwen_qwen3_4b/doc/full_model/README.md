# Qwen3-4B Full Model

Stage: full-model
Model: Qwen/Qwen3-4B
Implementation: `models/autoports/qwen_qwen3_4b/tt/model.py`, `models/autoports/qwen_qwen3_4b/tt/generator.py`
Target: 1x4 Blackhole p150b ring, TP4, `FABRIC_1D_RING`
Status: full-model readiness gates passing; stage-review still required before final handoff

## Performance Summary

| Path | TTFT | Decode throughput | E2E throughput | Evidence |
| --- | ---: | ---: | ---: | --- |
| Teacher-forcing traced decode | `846.76 ms` | `36.66 t/s/u` | `28.19 t/s/u` | `run_teacher_forcing.log` |
| Free-running traced token-out | `332.59 ms` | `37.69 t/s/u` | `33.79 t/s/u` | `token_out_timing_trace.log` |

Both paths use the optimized multichip decoder stack. Free-running token-out uses traced on-device top-k=1 split sampling with no per-token host token feedback.

## Implementation

The full-model wrapper preserves the optimized `MultichipDecoder` strategy:

- replicated BF16 hidden state at each layer boundary, `[1, 1, logical_seq_or_batch, 2560]`;
- TP4 internal QKV, attention heads, gate/up/down, and local KV heads;
- BFP4 projection weights, BF16 activations, BF16 paged KV cache, LoFi matmuls;
- shared persistent decode all-reduce scratch across layers;
- paged KV geometry `max_num_blocks=1280`, `block_size=32`, preserving context `40960`;
- no single-chip, host-side decoder, replicated fallback, or less-optimized decoder path.

Full-model additions:

- BF16 replicated token embedding and final RMSNorm;
- 36-layer stack by default;
- tied LM head loaded from embeddings when `lm_head.weight` is absent;
- BFP4 TP4 vocab-sharded LM head, per-device vocab shard `37984`;
- public generator-owned padding/slicing for non-aligned prefill lengths.

## Readiness Contract

`tt/generator.py` exports `build_generator(model_dir, mesh_device, **kwargs)`.

`Qwen3Generator` implements:

- `prefill_forward(tokens, *, page_table, kv_cache, prompt_lens, return_all_logits=False, **kwargs)`
- `decode_forward(tokens, start_pos, *, page_table, kv_cache, enable_trace=True, return_device_logits=False, **kwargs)`
- `generate(prompt_token_ids, max_new_tokens, *, next_input=None, enable_trace=True, host_sampling_compat=None, stop_on_eos=False, **kwargs)`
- `reset()`

Standalone generation owns KV cache and page table for the batch-1 latency path. Low-level prefill/decode accept explicit cache, page table, prompt-length, user-row, batch, and position state. `host_sampling_compat=True` remains available for tests that need host argmax; measured token-out uses on-device traced sampling.

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

This uses a shared batch-2 page table, row-indexed cache fill, and mixed decode positions.

## Accuracy

Fresh AIME24 HF-tokenizer chat-template reference:

```bash
python -m models.common.readiness_check.generate \
  --hf-model Qwen/Qwen3-4B \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt
```

Reference metadata: prompt `[1, 158]`, generated `[1, 100]`, top-k `[100, 100]`, `eos_id=151645`, `pad_id=151643`.

| Gate | top-1 | top-5 | top-100 | Evidence |
| --- | ---: | ---: | ---: | --- |
| Prefill | `93/100` | `100/100` | `100/100` | `run_prefill_check.log` |
| Decode teacher forcing | `93/100` | `100/100` | `100/100` | `run_teacher_forcing.log` |

Full-stack prefill/decode self-consistency at the AIME boundary after fixing decode RMSNorm:

```text
ref_top [198, 319, 271, 510, 201, 624, 1406, 84169, 11394, 280]
dec_top [198, 319, 271, 201, 624, 510, 1406, 5267, 280, 4753]
cos 0.966571033000946
```

## Sampling Decision

Compared both common sampler paths on the traced greedy token-out contract:

- `Sampling1D` top-k=1, `p=0`, `temp=1`: selected. It returns the semantically greedy token and writes a rank-4 token output compatible with the decode feedback buffer.
- `Sampling1D` force-argmax: rejected for this LM-head layout. In the clean sampler decision probe, it returned `158` instead of the host-composed argmax `198`.

No custom sampler was added.

## Trace Evidence

Free-running 100-token token-out:

```text
TOKEN_OUT_TIMINGS {'ttft_ms': 332.5899839401245, 'decode_t/s/u': 37.693503211804, 'e2e_t/s/u': 33.79477518178094}
TOKEN_OUT_TRACE_COUNTERS {'trace_replays': 98, 'token_host_refreshes': 0, 'position_host_refreshes': 2, 'page_table_host_refreshes': 0, 'syncs': 2, 'readbacks': 99}
```

Teacher-forcing traced comparison over the first 20 AIME tokens matched host argmax exactly. Host token refreshes occur only in teacher-forcing compatibility mode to inject ground-truth next inputs; free-running token-out has `token_host_refreshes=0`.

## Qualitative Check

Prompt-correct qualitative suite:

- prompt source: `models/common/readiness_check/vllm_prompts.txt`
- prompt mode: chat, because Qwen tokenizer has a chat template
- rendering: `tokenizer.apply_chat_template(..., add_generation_prompt=True)`
- artifacts: `qualitative/qualitative_prompt_format.json`, `qualitative/qualitative_rendered_prompts.json`, `qualitative/vllm_qualitative_outputs.json`, `qualitative/qualitative_verdict.json`
- verdict: all six prompts passed against HF greedy controls; no empty output, wrong-language drift, early repetition, or control-token corruption observed.

`check_degenerate_output.py` result: `No degenerate output detected`, with zero findings across the six qualitative TT outputs and the autoregressive completion.

`run_autoregressive` additionally generated 100 HF tokens and 100 TT tokens from the shared readiness continuation prompt. Both outputs continue the Elena/light-pattern story coherently in English. They diverge naturally after the first sentence, with no observed repetition loop, wrong-language drift, or early collapse.

Artifacts:

- `autoregressive/hf_completion.txt`
- `autoregressive/tt_completion.txt`
- `autoregressive/autoregressive_meta.json`

## Runtime Fallback Audit

- Model path: TTNN embedding, decoder stack, final norm, LM head; no torch conversion inside model prefill/decode.
- Cache ownership: generator owns page table/KV cache for standalone generation; low-level callers can pass explicit state.
- Host-logit boundary: readiness prefill/decode can read logits for checks; optimized token-out does not read full logits per token.
- Sampling: measured greedy token-out uses traced `Sampling1D` top-k=1 on device.
- Reset behavior: `reset()` reallocates KV cache/page table and clears sampling trace state.

## Validation Commands

```bash
python -m py_compile models/autoports/qwen_qwen3_4b/tt/model.py models/autoports/qwen_qwen3_4b/tt/generator.py
pytest -q models/autoports/qwen_qwen3_4b/tests/test_full_model_contract.py --tb=short
python .agents/scripts/check_context_contract.py --model-dir models/autoports/qwen_qwen3_4b
python -m models.common.readiness_check.run_prefill_check --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING
python -m models.common.readiness_check.run_teacher_forcing --model-dir models/autoports/qwen_qwen3_4b --reference models/autoports/qwen_qwen3_4b/doc/full_model/readiness_aime24_chat.refpt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING
python -m models.common.readiness_check.run_autoregressive --model-dir models/autoports/qwen_qwen3_4b --hf-model Qwen/Qwen3-4B --prompt-file models/common/readiness_check/autoregressive_prompt.txt --mesh-device P150_X4 --fabric-config FABRIC_1D_RING --output-dir models/autoports/qwen_qwen3_4b/doc/full_model/autoregressive --max-new-tokens 100
python models/common/readiness_check/check_degenerate_output.py --model-dir models/autoports/qwen_qwen3_4b --scope all --missing-artifacts advisory --json models/autoports/qwen_qwen3_4b/doc/full_model/degenerate_output_report.json
```

## Limitations

- Public standalone `generate()` currently measures the batch-1 latency path. Low-level full-model prefill/decode has batch-2 mixed-position evidence; broader serving scheduler coverage belongs to the next integration stage.
- `$stage-review` is still required before declaring the stage complete.
