# Runtime fallback audit

Verdict: pass. The canonical full-model path has no single-chip, host-logit,
replicated-weight, host-argmax, or Python token-feedback fallback.

## Model path

- Construction rejects any mesh other than P300 shape 1x4 and retains TP=4
  Q8/KV2 ownership. Embedding and LM-head weights are tensor-parallel sharded;
  replicated tensors are limited to small shared RoPE/control state and are not
  replicated model-compute fallbacks.
- Every decoder layer is `MultiChipDecoder`; no runtime branch instantiates its
  single-chip baseline. QKV/gate/up and O/down retain their accepted column/row
  ownership, two-link ring collectives, BF16 residual/CCL state, and 16-core L1
  inter-layer layout.
- The LM head remains four 8192-column/device splits with BFP8/HiFi2 weights.
  Invalid padded vocabulary columns are masked before sampling.

## Generator and cache ownership

- Callers may supply cache and page table explicitly to low-level prefill and
  decode. The high-level generator owns allocation only when those arguments
  are absent. KV heads remain sharded by TP rank in BFP8 or explicit BF16.
- Non-aligned prompt padding, logical masks, chunk page tables, positions, and
  output slicing remain inside the generator; no host model substitutes a
  chunk or a terminal layer.
- The public allocation horizon is `prompt_length + max_new_tokens - 1` because
  prefill emits the first requested token. This accepts an exact 131,072-token
  prompt for one output without inventing a 131,073rd cache position, while a
  real 131,071-token non-divisible prompt fills and inspects the last of all
  2,048 physical pages on hardware.
- Cache identity or page-table shape changes release both traces before
  recapture. Same-shape page-table contents are copied only when changed.
- `reset()` fills every device cache with zero in place, performs one explicit
  synchronization, clears host page/request seed state, and deliberately
  preserves compatible prefill/decode trace IDs and bound cache addresses.
  `teardown()`, host compatibility, cache identity changes, page-table shape
  changes, and sampling-topology changes are the explicit release boundaries.

## Logit and sampling boundaries

- Canonical `sampling_mode="device"` prefill retains final-token logits on
  device, runs split Sampling1D, and reads only the sampled token ID for the
  caller. The sampled tensor is also written into persistent `tt_out_tok`; the
  host ID is never copied back as normal feedback.
- Canonical decode executes a model trace followed by a sampling trace. Current
  position and RoPE position increment on device. There is no per-token page
  table, position, full-logits, or seed rebuild.
- Full-vocabulary force-argmax exists only in the explicit comparison gate.
  The canonical trace uses exact local argmax and one packed FP32
  rank-candidate gather; it never gathers the full logits tensor.
- `_local_logits_to_torch` is confined to readiness prefill/teacher checks and
  explicit `sampling_mode="host"`. Host compatibility releases active traces
  before allocating eager buffers. It is never selected implicitly.
- A `next_input` callback is the explicit teacher-forcing boundary. Only the
  forced token is copied; positions and page tables remain device-owned.

## Evidence

`logs/runtime_boundary_source_audit_review_fixes.log` records the final
source search. `artifacts/full_model_perf.json` proves 127 token-out replays
with 128 caller token reads but only one request-boundary token H2D copy.
`artifacts/full_model_evidence.json`
proves token feedback equality, coherent positions, and page-table copy deltas.
`full_model_contract_coverage.json` proves the real near-context cache fill and
exact maximum-prompt boundary. The six-prompt shared suite under
`artifacts/qualitative_suite/` also repeats prompt 0 bit-identically across
`reset()` with unchanged trace IDs and no unchanged-page-table copy.
All authoritative hardware runs set `throw_exception_on_fallback=true`.
The final metric-bearing proof is explicit in
`logs/full_model_perf_fallback_guard.log`,
`logs/full_model_evidence_fallback_guard.log`, and
`logs/test_reduced_full_model_fallback_guard.log`; each header records the
guard as true and each command exits successfully.
