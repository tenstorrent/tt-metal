# Functional Decoder Stage Review — Round 2

Verdict: **MORE WORK NEEDED**

Review date: 2026-07-25 UTC

## Required Work

- **P2: Correct the README's expert-parallel sharding description**

  **Evidence:** `README.md:44-48` describes gate/up tensors as
  column-parallel and expert-down tensors as row-parallel. The selected emit
  instead stores eight complete experts per TP rank: layer-12 gate/up is
  `[8, 2880, 5760]` and down is `[8, 2880, 2880]` in
  `g0_prefill/main.py:7240-7247`, versus 32 experts at full width. The runtime
  repeats the normalized activation over eight local experts and all-reduces
  the locally routed expert sum (`g0_prefill/main.py:3974-3977,4072-4081`);
  neither expert matrix is split over an input/output feature axis. The
  structured handoff correctly records both weights as `parallel_kind:
  "expert"` with tensor axis 0 in
  `multichip_provenance.json:252-287`, so the required README currently
  contradicts the source and the authoritative structured provenance. Q/K/V
  are column-parallel and O is row-parallel; all expert weights and biases are
  expert-parallel.

  **Why this matters:** The README and provenance are the multichip-stage
  handoff. Calling full per-rank expert matrices column/row parallel can lead a
  downstream implementation to shard each expert along the wrong dimension
  instead of assigning complete experts to ranks.

  **Required next step:** Rewrite the two bullets to identify only Q/K/V as
  column-parallel, only O as row-parallel, and gate/up/down expert tensors as
  expert-parallel over the 32-to-8 expert axis. Keep the existing local-expert
  output all-reduce description.

## Other Concerns

- None beyond the required documentation correction. Round-1's three required
  findings are otherwise closed:
  - the custom compute-kernel configuration now occurs only on the two
    RMSNorm calls, while projections, attention, router, and expert matmuls
    use framework defaults;
  - the real-weight test now covers position-256 sliding decode, including
    cache-row, post-attention, and full-layer PCC controls above 0.9994;
  - the provenance now includes 19 graph-derived sharded transient groups in
    addition to parameters, caches/constants, boundaries, and all 16
    representative-layer collectives. Every recorded transient source alias
    resolves inside the documented layer-12 ranges.

## Hard-Check Gaps

- This independent review did not open a TT device, run pytest, rerun a
  capacity probe, reset hardware, or start a server. The persisted JUnit XML
  parses and records 6 tests, 0 failures, 0 errors, and 0 skips; the numerical
  PCC and capacity details remain in the README/work log rather than JUnit
  `system-out`.
- The official-weight test supplies already dequantized dense expert weights,
  so `_dense_expert_weight`'s raw packed MXFP4 blocks/scales branch remains
  statically inspected rather than directly covered. This does not invalidate
  the required real-weight dense-math control.

## Anomaly Ledger

- **Observed anomaly:** Required prose classifies expert weights using feature
  column/row parallelism, while the source and structured provenance classify
  them as expert-axis parallel.
- **Evidence:** `README.md:44-48`;
  `g0_prefill/main.py:7240-7247,3974-3977,4072-4081`;
  `multichip_provenance.json:252-287`.
- **Affected path:** Multichip provenance handoff; the current dense
  functional runtime is not affected.
- **Control or comparison:** Source local shapes are full feature-width
  matrices for 8 of 32 experts, and the local expert sum is followed by a TP
  all-reduce. The JSON independently records tensor axis 0 and
  `parallel_kind: "expert"`.
- **Likely subsystem:** Documentation of the TP4 collapse.
- **Investigation performed:** Reconciled the README bullets, representative
  layer parameter shapes, expert runtime sequence, all-reduce placement, and
  structured tensor inventory.
- **Resolution:** More work needed; correct the two README bullets.

## Scope Inspected

- **Goal/skill paths:** `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/forge-functional-decoder/SKILL.md`,
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- **Artifact paths:** `doc/context_contract.json`; functional-decoder README,
  work log, round-1 review, AutoDebug report, JUnit XML, and
  `multichip_provenance.json`.
- **Code paths:** `tt/functional_decoder.py`, both package initializers,
  `tests/test_functional_decoder.py`, and
  `tests/functional_decoder_capacity_probe.py`.
- **Source paths:** Layer-12 prefill range 3879-4089 and decode range
  3318-3488, layer-12 weight creation, and relevant consteval transformations
  under `/home/mvasiljevic/emit-gptoss/g0_prefill` and `g1_decode`.
- **Repository state:** Branch `mvasiljevic/gpt-oss-pipeline-progress` at
  `dd34ac32928d704bf0aff87fd25f047c5fbb6af0`; live stage-owned changes are
  confined to `models/autoports/openai_gpt_oss_20b/`.
- **Commands run:** Read-only source/artifact inspection; AST runtime-token
  audit; source-hash verification; JSON/XML parsing; transient-alias
  reconciliation; `git status`, `git diff --check`; and both exact and
  HF-aware strict invocations of
  `.agents/scripts/check_context_contract.py`. Both context checks passed with
  target 131072, supported 21248, and the DRAM-limited classification.

## Residual Risk

- After the README correction, the only material residual risk visible in
  this review is the unexercised packed-weight loading branch. The current
  dense runtime math, emitted/default compute policy, sliding decode behavior,
  context disclosure, TP collapse, runtime fallback audit, and structured
  provenance otherwise satisfy the functional-decoder stage contract.
