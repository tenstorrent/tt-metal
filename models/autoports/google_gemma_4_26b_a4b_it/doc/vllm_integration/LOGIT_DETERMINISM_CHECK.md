# Numeric logit reproducibility check

`tests/check_vllm_logit_determinism.py` compares the standalone generator and direct serving adapter using the selected precision policy. It defaults to reduced layers0/5; `--full-model` selects all30. Device execution is pending the main agent's serialized run after serving shutdown.

The fixture uses two distinct 31/63-token prompts, full-vocabulary last-prefill logits, and three traced decode steps. Every subsequent case consumes the same teacher tokens selected from the initial standalone baseline, so a numeric difference cannot change later inputs and confound the comparison. Each standalone and isolated adapter run repeats; paired runs use both `[A,B]` and `[B,A]`, repeat each order, and explicitly compare each request's row0/row1 logits. Adapter calls use32-row wire padding, compact logical decode rows, distinct attention-group page IDs, a shared hybrid cache pool, and nonidentity prefill state slots.

Every comparison requires exact equality with finite values. No PCC or allclose threshold is guessed. JSON records maximum/mean absolute difference, differing element count, PCC, per-step maximum difference, and top1 agreement; a companion `.pt` file preserves complete numeric tensors even on assertion failure. If a case fails, assess those artifacts before proposing a tolerance. The script reports runtime precision and selected layer indices.

Run from `/workspace/tt-metal`, after exclusive device ownership is available:

```bash
TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/runtime_logs \
python -m models.autoports.google_gemma_4_26b_a4b_it.tests.check_vllm_logit_determinism \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/logit_determinism_reduced.json
```

For full-model evidence, use the same command with `--full-model` and a separate output such as `logit_determinism_full.json`. Optional `--layers` overrides the reduced layer list. `--repeats` defaults to2 and cannot be lower; `--decode-steps` defaults to3 and cannot be lower than2.

The script explicitly enables the adapter's optional host-logits compatibility mode. It retains the ordinary generator's `host_sampling=False`; it does not substitute host argmax for canonical device sampling. Host argmax only selects fixed diagnostic teacher tokens. The result covers numerical model/adapter logits, not HTTP admission, scheduler behavior, stochastic sampling, or serving performance.

Trace allocation tracking can be combined by setting `TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` before launching Python. This checker captures model logits traces only. It does **not** exercise the separate common-sampler trace and does not replace the canonical split-sampling tracker check in `AUTODEBUG_trace_allocations.md`. No profiler is used.

Host preparation checks: Python compilation, CLI `--help`, synthetic numeric-comparison edge cases (equal, shifted, shape mismatch, nonfinite, constant), and disjoint hybrid page-table layout checks. These checks opened no device and do not constitute a numerical model pass.
