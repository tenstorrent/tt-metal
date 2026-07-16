# Stage 09 anomaly ledger

## vLLM context reduction

- Observed anomaly: full-depth vLLM cannot advertise the standalone 262144-token context.
- Evidence: `evidence/full_262144_capacity_failed_server.log`,
  `evidence/full_157696_capacity_failed_server.log`, and
  `evidence/context_capacity_audit.json`.
- Affected path: vLLM-owned hybrid KV-cache pool on four P150b devices.
- Control or comparison: standalone full-model context remains 262144; vLLM
  shares ten physical K/V workspaces across fifty sliding and ten global layers.
- Likely subsystem: per-bank DRAM capacity after the HMA cache pool.
- Investigation performed: two full-depth allocation failures, source-derived
  prompt-live-set accounting, real maximum-total-length requests, and autofix
  of dead prompt-sized attention and MLP normalization lifetimes. The reviews
  rejected both the earlier arbitrary reserve and startup-plus-short-prompt as
  capacity proof. The request checker was also fixed to reject HTTP-200 error
  envelopes.
- Resolution: controlled hard physical limit after full-layer streaming.
  Attention output projection/all-reduce, post-attention norm/residual, and the
  long-prompt MLP residual path are chunked; global layers use a zero-copy HMA
  read view with layer geometry. The served limit is `113280`: `9740` pool
  blocks leave `483372736` post-KV bytes/bank, its mandatory peak is
  `480509952`, sixty aligned page tables use `1704960`, and the margin is
  `1157824`. Adjacent aligned `113344` is source-proven short `148800`
  bytes/bank. A real `113279`-input plus one-output completion passed, so this
  is not an input-alignment restriction.

## Hybrid-cache decode geometry

- Observed anomaly: the first reduced hybrid run failed when a global-attention
  layer reused a physical cache workspace with sliding-attention geometry.
- Evidence: `evidence/reduced_hma_view_failed_server.log`; the repaired
  geometry contract is covered directly in
  `tests/test_vllm_adapter_contract.py` and `tt/multichip_decoder.py`.
- Affected path: decode paged K/V update for cross-group shared HMA storage.
- Control or comparison: equal-volume cache workspaces have different effective
  block size and local KV-head geometry.
- Likely subsystem: fused paged-cache update geometry assumptions.
- Investigation performed: source/API inspection proved that a storage reshape
  would allocate/copy, while the non-fused update accepts the effective block
  size and local-head overrides.
- Resolution: fixed. Native geometry retains the fused update; shared
  cross-geometry uses the supported non-fused pair of updates.

## Split-trace allocator warning

- Observed anomaly: TT Metal emits `Allocating device buffers is unsafe due to
  the existence of an active trace` once when the canonical sampler trace is
  established.
- Evidence: final `readiness_vllm/server.log`, passing reduced log
  `evidence/reduced_dynamic_batch_passing_server.log`, and the inherited Stage
  06/07 control in `../optimized_full_model/anomaly_ledger.md`.
- Affected path: first split model/sampler trace setup, not steady replay or a
  host/prefill transition.
- Control or comparison: the adapter synchronizes and releases both traces
  before batch-shaped page-table allocation or host eager decode. Token output,
  local/gathered sampler pairs, logits, position/RoPE inputs, parameters, and
  page tables are persistent; regular all-gather targets the preallocated
  gathered-pair tensor. Repeated replay, reset/recreate, changed-page-table,
  watcher, profiler, and serving correctness controls pass.
- Likely subsystem: allocator registration of the required second cooperating
  trace while the model trace is already registered.
- Investigation performed: `$autofix` separated transition-time trace lifetime
  from second-trace setup and added log brackets containing active batch and
  both trace IDs. The allocator does observe a real buffer request, so the
  warning is not dismissed as text-only.
- Resolution: controlled inherited canonical split-trace setup anomaly. Reopen
  if a warning occurs outside the trace-prepare/trace-ready bracket, near trace
  release or prefill, or with corruption/correctness failure.

## Dynamic decode batch and masked execute failure

- Observed anomaly: the first dynamic-batch reduced smoke crashed with an empty
  pending-sample deque.
- Evidence: `evidence/reduced_dynamic_pending_queue_failed_server.log` and
  `evidence/reduced_dynamic_pending_queue_failed_sampling.log`.
- Affected path: mixed host/device sampling immediately after dynamic-batch
  integration.
- Control or comparison: vLLM's GPU runner returns `None` when execute state was
  not published so EngineCore resolves the captured execute future.
- Likely subsystem: TT sample deferral plus the adapter's optional host decode.
- Investigation performed: `$autofix` proved the empty deque masked an earlier
  `execute_model` exception. Surfacing that exception identified an undefined
  host-branch page-table variable introduced by the dynamic-batch refactor.
- Resolution: fixed. TT now exposes the original execute error, host decode
  converts page tables after synchronized trace release, focused CPU contracts
  pass, and the reduced mixed sampling smoke passes.

## Base-checkpoint continuation behavior

- Observed anomaly: several qualitative outputs continue request-list corpora,
  repeat phrases, or fail to follow the apparent instruction.
- Evidence: `readiness_vllm/vllm_qualitative_outputs.json`,
  `readiness_vllm/qualitative_verdict.md`, and Stage 08 controls under
  `../datatype_sweep/qualitative/`.
- Affected path: user-visible raw text continuation.
- Control or comparison: the exact tokenizer has `chat_template=None`; Hugging
  Face controls reproduce the supervised-learning/thermodynamics question
  loops and translation-exercise continuation.
- Likely subsystem: base-checkpoint training distribution and prompt format.
- Investigation performed: every greedy and sampled output was read for
  coherence, topic, repetition, gibberish, language drift, and request
  contamination; the scoped mechanical-degeneracy checker passes.
- Resolution: controlled checkpoint/prompt-format limitation. No gibberish,
  wrong-language corruption, cross-request state leakage, or stale-token
  duplication is present. Request-list contamination within individual
  continuations and phrase-level loops are explicitly documented;
  instruction-following quality is not overstated.
