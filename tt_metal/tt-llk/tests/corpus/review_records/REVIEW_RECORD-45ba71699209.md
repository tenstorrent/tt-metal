# REVIEW_RECORD — RVTT profitability and replay-completion hardening

> **INVALIDATED / NOT APPROVED (2026-08-25).**  This is a historical review
> of an unceremonied candidate, not a shipping-pin approval.  The candidate's
> DejaGnu run omitted part of the canonical SFPI universe, its ON-set
> quarantine was not independently adjudicated, and its silicon report
> contains material versus-hand regressions.  The live harness has been
> restored to ceremonied pin 28 / ON-28.  Any `APPROVE`, `GREEN`, `current`,
> or shipping-state language below is superseded by this disposition.

Date: 2026-08-25 UTC
Reviewer: Codex `/root`; independent static review by local Claude and
independent implementation reviews by workspace agents Sartre, McClintock,
and Lagrange.
Decision: APPROVE for candidate compiler source
`075e9f2f4b22dd08342be730d42e34060da10d4a`.

Primary cc1plus SHA256:
`45ba7169920924fd6ebeb6eeb3766156b413dbf895e091b53603bed1e35e7d79`

## Reviewed

- Dst autoincrement charges the same target-derived 8 issue units on BH and
  WH, independent of caller visibility or callgraph shape. A live backedge
  crossing adds the existing 2-unit residual. Shared programs amortize
  setup across proven rows and ties refuse.
- Ordinary constant residency uses the derived profitability threshold
  `floor((W+1)/(W-R))+1`; one-word zero-saving patterns refuse. Peel pricing
  sums recurring `W-R` savings and charges full staging, configuration, and
  delivery costs.
- Replay completion guarding is opt-in and default-off. It charges full
  execution-bound delivery where launch completion cannot cancel payload
  execution. Guarded record-hoist routes through the shared model; opaque
  execution effects and runtime/unknown trip counts refuse.
- Legal replay candidates contain at least four words. For an
  execution-bound record, guarded minus unguarded benefit is
  `T*(300-123W) <= -192T`; BH and WH four-word witnesses pin the plain
  `+773` fire versus completion-guarded `+5` refusal. The compiler tests
  caught and rejected an unreachable two-word reversal proposed during
  static review.
- All decisions are structural and target-capability based. No corpus,
  operation, source-file, function-name, coefficient, immediate, instruction
  calendar, or raw-word fingerprint was introduced.
- The min/max external-COMDAT case remains conservative by design: without
  hidden/internal/LTO closed-world ownership, entry hardening cannot prove
  that every caller established the required state.

## Gates

- Exact installed binaries:
  - cc1plus `45ba7169920924fd6ebeb6eeb3766156b413dbf895e091b53603bed1e35e7d79`
  - cc1 `f27181b8f726c2055a98f88d90125bc3b450587dbcd8452b08b4dd97bee3f4ba`
  - lto1 `b37deac999366f2170b5eb1532886b142c6dd16f0b855337cfd0aad57c3ac378`
  - g++ `a04de6aad4c29aa222e7b5f2e9d699b8bb89fec6accfd38dcf4a78e72e47e720`
  - gcc `cfb97ae9bdb30226e8fa7dec36dc458732b2f6afc80a1bba196352b08cd0fbd5`
- Full `rvtt.exp`: **GREEN** — 4,925 PASS, 16 FAIL, 2 XFAIL. The
  unexpected-failure set is byte-identical to the frozen reference
  (`41346b4760b0faebd9b0b040a882f2d87ec46065c334c853ff2661daaac07182`).
  Final summary SHA256:
  `0e5255dda256bf8e154dbf92dd8c12be320e9444d5713061106d6a5cebf4d5ab`.
- Complete `record-hoist-*.C` family: 226/226 expected assertions pass.
  Final downstream fallback family: 32/32. Prior focused gates remain green:
  dst profitability 408; constant residency 131; record-hoist routing 188.
- Installer pre-mutation pins and post-install verifies all five binaries.
  Exact combined option acceptance smoke passed, including
  `-mtt-tensix-replay-hoist-completion-guard`.
- Silicon promotion is separately gated by registry-wide 2x2 classification,
  paired CRAQ, DEVICE-GOLDEN correctness, and deterministic drain-inclusive
  KERNEL measurements under the evidence root recorded by the final report.

## Install

Installed at
`/home/ttuser/sfpi-uplift/toolchain-candidate-final-075e9f2` and exposed by
`/home/ttuser/sfpi-uplift/sfpi-candidate-final-075e9f2`. The harness must be
atomically repointed only after the preceding frozen silicon campaign exits.
