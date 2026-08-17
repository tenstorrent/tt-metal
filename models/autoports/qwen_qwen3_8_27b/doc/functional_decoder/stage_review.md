# Independent stage review

Date: 2026-08-17 UTC

Verdict: **clean-pass**

Required work: none. No stage-blocking gaps or other blocking concerns were
found after AutoFix and fresh rereview.

Verified closure:

- Both meaningful target layer kinds implement documented exact public APIs.
- Real-weight HF-vs-TTNN PCC exceeds 0.995 for paged prefill and traced decode.
- B=2 uses the public paged prefill/decode flow with distinct 64/96 lengths,
  positions, disjoint/permuted pages, and per-row real-weight PCC.
- Near-limit non-aligned prefill reaches 262,143 tokens with 127-token DeltaNet
  and 2,047-token full-attention tails, a 63-token partial page, then decode at
  position 262,143. Advertised 262,144-token capability is preserved.
- Four bounded profiler captures have clean console logs, source/filtered CSVs,
  human-readable tables, paired signposts, and provenance.
- Runtime audit includes the target-local overrides and finds no host fallback.
- All 25 representative consumed real checkpoint tensors have statistics and
  exact checkpoint provenance.
- The expanded suite reports 28 passed; reset-state determinism is bitwise.
- Post-fix watcher evidence covers B=1, B=2 routing, and non-aligned maximum
  context and ends with clean detach of devices 0-3.
- Scope is confined to this functional-decoder autoport directory.

Controlled anomalies:

- Nanobind shutdown warnings were reproduced identically by an unrelated,
  pre-existing CPU-only test and are not stage-owned or a device-close failure.
- The original combined profiler overflow was replaced by four clean bounded
  captures.
- Shared batch-one paged prefill was corrected in the autoport-local path and
  validated against independent real-weight row oracles.

Residual risk: B>1 full-attention prefill is row-serial because the shared cache
fill primitive lacks a batch-index tensor. This is a later optimization concern,
not a functional-stage correctness gap.
