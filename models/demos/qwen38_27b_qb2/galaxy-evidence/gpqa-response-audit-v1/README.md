# GPQA truncation audit, Oct 9 2026 UTC

The full optimized 64K-output result remains **163/198 (82.32%)**. All 198
questions were scored. Fifteen responses reached their output budget without
emitting final-answer content, and all fifteen scored zero. They were not
excluded from the denominator and did not run to a natural conclusion.

The [numeric audit](audit.json) matches every saved private response against
the scored receipt's final-answer and reasoning SHA-256 hashes, finish reason
and token counts. The original receipts and raw responses are unchanged.
No prompts, reference answers or generated text are exported by the audit.

| Group | Questions | Correct | Empty final-answer content |
|---|---:|---:|---:|
| All scored questions | 198 | 163 | 16 |
| Natural stop | 183 | 163 | 1 |
| Output budget reached | 15 | 0 | 15 |

Every length-limited response contains exactly **65,536 output tokens**.
Their prompts range from 163 to 542 tokens. The original
[deployment receipt](../qualification-overnight-v1/deployment-after-stop.json)
records a 262,144-token context, so these were output-budget cutoffs, not
exhaustion of the model's context window. Reasoning consumes the same output
budget as the final answer in this benchmark.

The natural-stop subset is 163/183 (89.07%). That is a diagnostic subset,
**not a replacement GPQA score**. With all other answers unchanged, fourteen
of the fifteen currently truncated cases would have to become correct to
reach the existing 177/198 qualification gate. More output alone does not
establish that they will do so.

## Reasoning diagnosis

Exact repeated 20-word sequences account for at most 0.246% of the last 4,096
whitespace-delimited words of any truncated response (median zero). A private
review of five truncated response tails found continuing calculations and
reconsideration of candidate conclusions, rather than an obvious exact-copy
loop. This does not establish semantic correctness or rule out unproductive
reconsideration. One separate wrong response with a natural stop and no final
answer has 69.3% duplicate tail sequences; it is not one of the fifteen cutoffs.

The pinned checkpoint's model card describes `xhigh` as the default reasoning
effort and recommends large reasoning budgets for agentic tasks. It does not
state a complete matching GPQA evaluation protocol. That recommendation does
not by itself authorize treating incomplete benchmark answers as correct.

The inspected Transformers 5.12.1 Qwen3.5 recurrent reference uses
`rsqrt(sum(x*x) + 1e-6)`, as does the custom kernel. The suspected mismatch
between adding epsilon and clamping the norm is not present in that source.
This source check does not establish full-model numerical equivalence.

The live native-recurrence control retains the same 64K output budget so its
comparison remains useful. A higher-budget full run would be a separately
reported experiment, retaining all questions and the unchanged score gate.

## Reproduce without hardware

Run on the allocated host, where private responses remain access-controlled:

```bash
python -m models.demos.qwen38_27b_qb2.tests.gpqa_response_audit \
  --receipts /home/ttuser/qwen38-artifacts-20261007/overnight-v1/candidate/evaluation/gpqa/gpqa-responses.jsonl \
  --private-responses /home/ttuser/qwen38-artifacts-20261007/overnight-v1/candidate/evaluation/gpqa/private-responses \
  --max-output-tokens 65536 --max-model-len 262144 \
  --output /path/to/new-audit.json
```

Use the task's Python environment and the committed model source on
`PYTHONPATH`. The output path must be new. [Manifest](manifest.json) records
the executed commands and exact audit source hashes. Six isolated CPU tests
passed, including rejection of missing/duplicated questions, altered raw
outputs, mismatched finish reasons and incompatible declared limits. No live
model, hardware queue, checkpoint or precision setting was changed.
