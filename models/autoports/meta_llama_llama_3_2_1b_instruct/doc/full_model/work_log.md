# Full Model Work Log

## Scope

Built the TTNN full autoregressive wrapper and readiness generator for
`meta-llama/Llama-3.2-1B-Instruct` from the optimized 1x8 multichip decoder.
No vLLM work was started.

## Commands And Results

Fresh AIME24 HF-tokenizer chat-template reference:

```bash
python -m models.common.readiness_check.generate \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --prompt-source aime24 \
  --chat-template \
  --gen-len 100 \
  --top-k 100 \
  --output models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt
```

Reference metadata: `readiness_v1`, model id
`meta-llama/Llama-3.2-1B-Instruct`, AIME24 prompt source, chat-template prompt,
prompt length 184, generated length 100, `k=100`, BOS 128000, EOS/PAD 128009.

Compile check:

```bash
python -m py_compile \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/model.py \
  models/autoports/meta_llama_llama_3_2_1b_instruct/tt/generator.py
```

Prefill readiness:

```bash
python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 0.880 (88/100), top5 1.000 (100/100), top100 1.000 (100/100).

Teacher-forcing readiness:

```bash
python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --reference models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/aime24_chat_template_gen100_top100.refpt \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING
```

Result: top1 0.860 (86/100), top5 1.000 (100/100), top100 1.000 (100/100),
TTFT 252.12 ms, decode 52.55 t/s/u, e2e 46.68 t/s/u.

Autoregressive readiness:

```bash
python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/meta_llama_llama_3_2_1b_instruct \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --mesh-device T3K \
  --fabric-config FABRIC_1D_RING \
  --max-new-tokens 128 \
  --output-dir models/autoports/meta_llama_llama_3_2_1b_instruct/doc/full_model/artifacts/autoregressive_default_128
```

Qualitative verdict: HF opens a hidden-box story branch. TT opens a silver-dust
story branch. TT diverges early after the common prompt but remains coherent,
English, non-repetitive, and on-topic through 128 generated tokens.

Degeneracy check:

```bash
python models/common/readiness_check/check_degenerate_output.py \
  --hf-model meta-llama/Llama-3.2-1B-Instruct \
  --root models/autoports \
  --scope autoregressive \
  --missing-artifacts critical
```

Result: no degenerate output detected. TT adjacent duplication 0.0, trigram
loop fraction 0.0561. HF/TT token agreement is informational only: 4/128.

Focused split-greedy comparison:

- Compared traced split-greedy token output against host-composed greedy argmax
  for every AIME24 decode step.
- Result: no mismatches across 99 decode-token outputs.
- Accuracy using sampled predictions: top1 0.860, top5 1.000, top100 1.000.

Trace evidence script:

- Greedy free-running, default readiness prompt, 128 generated tokens.
- Top-k/top-p smoke, default readiness prompt, 16 generated tokens,
  `top_k=16`, `top_p=0.9`, `temperature=0.8`.
- Page-table focused replay with unchanged page table followed by changed
  inactive row 31.

Artifacts written under `artifacts/trace_evidence/`.

## Rejection Ledger

Rejected paths:

- single-chip fallback;
- replicated host-side decode;
- decode host argmax;
- full-vocab all-gather for greedy;
- force-argmax greedy;
- generic `ttnn.sampling(k=1)` for semantic greedy.

Observed and fixed:

- Generic `ttnn.sampling(k=1)` returned non-greedy tokens on wide gathered
  candidate buffers.
- Eager split-greedy over repeated decode steps was corrupted when using the
  older all-gather path. Switching candidate gathers to
  `ttnn.experimental.all_gather_async` made eager and traced split-greedy
  semantically greedy across the AIME24 reference.
- Sampler trace replay was returned to nonblocking after correctness was proven.

Kept path:

- local shard `topk(k=32)` for candidate extraction;
- async candidate all-gather only;
- global argmax over gathered candidates;
- device write into the persistent decode token input.

Carried decoder policy:

- BFP8 attention weights and paged KV cache;
- BFP4 MLP weights;
- BF16 activations, residual, norms, and terminal path;
- one local KV head per chip with replicated page table;
- prefill K/V fill tensors cast to the BFP8 cache dtype;
- BFP8 residual all-gather and reduce-scatter payloads;
- replicated full-hidden inter-layer residual boundary from the optimized
  multichip decoder.

## Runtime Audit

- Model and generator reject external KV caches unless they are the generator's
  owned cache handle.
- `generate(enable_trace=False)` raises for readiness.
- Decode token-out counters from greedy audit: host argmax 0, full-logits
  readbacks 0, token refreshes 1, current-position/RoPE refreshes 1,
  page-table refreshes 1, device feedback steps 127.
- Page-table audit: unchanged replay used no host refresh; changed replay
  refreshed page table only.
- `reset()` clears KV cache, sampling output state, timing metadata, generation
  metadata, and counters while preserving reusable traces and weights.

## Exact Artifacts

- `artifacts/aime24_chat_template_gen100_top100.refpt`
- `artifacts/autoregressive_default_128/hf_completion.txt`
- `artifacts/autoregressive_default_128/tt_completion.txt`
- `artifacts/autoregressive_default_128/autoregressive_meta.json`
- `artifacts/trace_evidence/greedy_default_128_completion.txt`
- `artifacts/trace_evidence/topk_topp_smoke_completion.txt`
- `artifacts/trace_evidence/trace_audit_greedy_default_128.json`
- `artifacts/trace_evidence/trace_audit_topk_topp_smoke.json`
- `artifacts/trace_evidence/trace_audit_page_table_refresh.json`

## Limitations

- High-level readiness path is batch-1 on a batch-32 decode tensor shape.
- Max sequence is 4096 with batch 32 due to the 128K total KV-token budget in
  the attention module.
- Full-model low-level profiler CSVs were not collected in this stage; reported
  performance is from readiness harness timing and trace audit timing.
- No vLLM adapter exists yet.
