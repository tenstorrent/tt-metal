# vLLM integration work log

## Implementation

- Added the thin `Falcon3ForCausalLM` adapter and registered
  `TTFalcon3ForCausalLM` in the TT vLLM platform. Falcon3 checkpoints whose HF
  architecture is the generic `LlamaForCausalLM` are mapped to the TT adapter.
- Reused full-model low-level prefill/decode and canonical split-sampling traces.
  Added explicit host compatibility only when vLLM requests unsupported host
  features; performance remains device sampled and traced.
- Removed the standalone full-reservation assumption for externally sized paged
  caches and accepted vLLM's masked zero-padded SDPA tail while retaining strict
  live-page ownership checks.
- Updated the shared runner for `P300x2`, current `--additional-config`, and an
  exact 37-token non-aligned request artifact.
- Passed the declared prompt-plus-output horizon through the shared runner, so
  Falcon3 grows RoPE before decode trace capture without reserving a 32K table
  for every short request.

## Repair loop

- Initial greedy requests arrived with vLLM's unrestricted top-k sentinel;
  zero-temperature rows are normalized to top-1.
- Host-only logprobs/min-p/penalty paths required explicit full-logits adapter
  mode and `[B,1,V]` decode shape.
- A 32-sequence server initially failed because the model demanded 32 separate
  full-context cache reservations. vLLM's 4,128-block cache is now authoritative.
- First multi-request host sampling exposed vLLM's zero-padded, causally masked
  page-table tail; ownership validation now examines logical pages only.
- Stochastic top-k beyond 32 and non-default penalties are explicitly routed to
  host compatibility. `$autofix` independently confirmed the last three penalty
  failures were invalid decoded-text assumptions, not stale penalty history;
  model-independent tests were corrected. Component penalty/sampling tests pass.
- Investigation artifacts: `AUTODEBUG.md` and `AUTOFIX.md`.
- Stage review identified position-256 corruption and missing overlap,
  degeneracy, determinism, and alias evidence. `$autofix` traced the corruption
  to the initial 256-row RoPE table captured by the vLLM decode trace. The
  horizon fix and requested focused evidence were added before rereview.

## Verification

- Adapter contract: `pytest -q .../tests/test_generator_vllm_contract.py` ->
  **6 passed**.
- Plugin registration/shared-runner unit tests: **14 passed** across
  `test_platform_falcon3.py` and `test_sample_time_grammar.py`.
- Plugin merged penalty/sampling metadata:
  `pytest -q plugins/vllm-tt-plugin/tests/test_lane_input_batch.py -k 'penalt or sampling'`
  -> **4 passed**.
- Final shared command: see `README.md`; exit 0.
- Full plugin sampling: **72 passed, 1 skipped**.
- Non-aligned request: exact 37-token input, 4 completion tokens, pass.
- Qualitative: 6 prompts x greedy/sampled, all outputs read; base-model request
  contamination documented against exact HF 256-token controls; degeneracy
  checker passed.
- Async focus: isolated and churned 300-token outputs match exactly; repeated
  and cross-batch-position top-logprob signatures match; vLLM first token is HF
  BF16 rank 1. The exact isolated/overlap strings also passed the shared
  degeneracy checker with no finding.
- Primary 128/128/1: TTFT P50/P99 182.5/182.5 ms; TPOT mean/P99 15.9/15.9 ms;
  ITL P50/P99 14.6/15.1 ms; output throughput 58.2 tok/s; TPOT-derived 62.9
  t/s/u.
- CI burst 100/100/32: 32/32 complete; TTFT P50/P99 414.4/415.6 ms; TPOT
  mean/P99 16.9/18.3 ms; ITL P50/P99 15.1/74.7 ms; output throughput 1,539.7
  tok/s; TPOT-derived 59.3 t/s/u (secondary only).
- Runtime cleanup: final server terminated cleanly, no remaining vLLM/EngineCore
  process, TT snapshot readable. Repeated-lifecycle ERISC recovery procedure and
  limitation are recorded in `README.md`.

## Commits

- tt-metal stage commit: `PENDING`
- vLLM TT-plugin stage commit: `PENDING`
- Pushes: none.
