# Owner wrap-up hand-off

The owner wall-cap instruction stops candidate tuning on 2026-11-13. The selected
configuration enables the exact residual writer and a compact-expert threshold
of two physical rows. This note makes no 70% roofline claim. Required closure
uses `wrap_final_plan.json`; pending jobs are not described here as passes.

## Not tried (stop rule)

- Further matmul geometry, DRAM reader/shard and fidelity combinations: leave to
  the separate optimized-decoder stage. No measured gain is claimed here.
- General dynamic-size compact expert unions beyond two physical rows: fixed
  padded unions are already measured slower at full tiles, and thresholds 3/4
  regress some complete public lengths. A new indexed-kernel contract that skips
  padded slots could help; the current indexed sparse API computes each slot.
- Further residual-read fusion and direct output geometry for precise projection
  reductions: potential dispatch/movement savings are unmeasured. Read-side
  stream grouping changes order and cannot simply be deleted.
- Additional QSA GR accumulation-order controls and higher-precision PLE table
  formats: the existing HF controls and literal failures remain on disk. Further
  precision experiments are not needed solely to force every row above 0.995
  under the owner's whole-output/aggregate gate. Do not erase the original
  diagnostic results or present changed-arithmetic controls as literal HF.
- Additional max-context and capacity sweeps: not run under owner direction.
  KV dtype/layout and context262144 are unchanged; existing maximum-context
  evidence is retained and final-source closure uses 32K smokes.
- Additional trace/per-op profiling variants: final selected-path reports are
  required once; new candidate-specific profiling is deferred.

## Acceptance and evidence policy

Whole-output (or per-layer-kind aggregate) PCC must be at least0.995. Per-row,
per-user and per-step PCC is diagnostic, with floor0.99 and top-1 agreement.
Matched HF-reproduced row misses are recorded as known limitations. This policy
is an explicit owner update; historical strict-row probe outcomes remain intact.
This decoder stage has no LM head or generated-token contract, so any top-1
metric must identify its output domain rather than call hidden-state argmax a
predicted token.

Keep logs and one final manifest. Do not create new per-artifact archives or
repeat historical archival work. Final review, local stage-owned commit and
commit-SHA logging remain required. No push is authorized.
