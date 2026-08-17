# REVIEW_RECORD-7e9480e7371b — silicon authorization for pin 8 (wave-6 build)

Pin: cc1plus sha256 `7e9480e7371bf823cb5f981ba32223eb8d162803512ebc92465323ef1e6c8005`
(driver sha256 `9caf5b3f839e0f6a77f381f3700629daff714a50b1c606b87a836aaa04c2f274`)
Built from: sfpi-gcc `104e8dbbc32` (wave-6 merge stack) via the sfpi superproject build at
`~/sfpi-uplift/sfpi/build/sfpi` (tt-quietbox-0).
Date: 2026-08-17
Reviewer: the SFPI-uplift orchestrator session (Lane AW writing this record retroactively as
part of the enforcement-layer work; the underlying wave-6 reviews were run by review lanes
spawned by the SAME orchestrator that merged the work).
Independence: **NOT independent.** This is a retroactive record, written honestly rather than
as absolution: no reviewer outside the orchestrator session has reviewed the pin-8 compiler
stack, and HANDOFF §1(4) requires independent review of compiler mutations before silicon.
This record exists so that (a) the FIRST pin-8 silicon runs against a written, falsifiable
statement of what was and was not reviewed, and (b) the new preflight gate has a truthful
record to bind to instead of nothing.  An owner/independent re-review superseding this record
is the intended next step; until then every pin-8 silicon result inherits this caveat.

## Reviewed commits/branches

Compiler (sfpi-gcc, all on nkapre/sfpi at `104e8dbbc32`):
- `104e8dbbc32` — merge agent/exp-parity-d3m3: exp-class latency audit + interlock fill (D3),
  programmable constant-register allocation + typed-region markers (M3/D2 compiler half).
- agent/multiresult-effects (`ad35244b6da` merge) — multi-result typed effects + epoch-scoped
  replay soundness (TopK captures engage).
- Wave-6 lanes AJ (WH bank-base conviction fix: dst-autoincr owns ONE slot), AP (WH
  macro-tables mirror — derived calendars form on WH), AO (MOP-form, default-off, priced by
  the corrected model), AM (exp D3 latency audit).
- Reviews that DID run (wave-6, ledger item 10): three hostile review lanes over the ~36-commit
  overnight wave; verdicts CONFIRMED GOOD on the engineering (timing facts rest on the in-tree
  BlackholeA0 SFPLOADMACRO.md spec; derive-core reproduces 77/77+542/542+17/17+11/11; WP10
  emits the handwritten 0x770 protocol; Track B dst-ownership clean; pin promotions honored the
  conf prose rule) and VIOLATIONS_FOUND on the measurement/review discipline (V1 stale/lying
  baseline header, V2 §1(4) breaches ×2, V3 unvalidated −44.8% headline) — the violations that
  this enforcement layer converts into gates.

tt-metal (nkapre/sfpi lineage): `12f90fc3ac` (pin-8 promotion; WH sweep legs re-enabled;
mop-form joins ON set + knobs) and `3f462a5997` (corpus compile-selector remap).

## Gates checked

- Paired CRAQ oracle for pin 8: the CORRECTED craq-sim `9f324140` sims
  (bh `32489dda4fd6…`, wh `8f0079a9a16c…`), now PINNED in sweep_2x2.conf and verified by
  preflight and every phase entry.
- Removed exact-calendar flags error on use at this cc1plus (sweep preflight probe).
- OFF/ON flag sets accepted by the pinned cc1plus (sweep preflight probe); the pin-8 ON set
  adds `-mtt-tensix-optimize-mop-form` (self-refusing profitability gate per its lane review).
- conf-lint: pin values ↔ conf prose ↔ PIN HISTORY (CURRENT) ↔ baseline header anchors agree
  (mechanical, corpus/conf_lint.sh).
- NOT checked for pin 8 (honestly): no independent DejaGnu byte-parity run of the pin-8 build
  is recorded in-repo (the weekly dejagnu gate covers this on its next run); no pin-8 silicon
  cells exist yet — the first sweep under this pin re-measures everything per the drift gates,
  which is the designed backstop.

## Limitations

Self-review: authored, merged, and reviewed within one orchestrator session; the wave-6 review
lanes were adversarial but not organizationally independent.  This record does NOT discharge
the ledger's §1(4) findings for the silicon that ALREADY ran under pins 5–7 minutes after
their promotions — those breaches stand as recorded in the ledger (HANDOFF §5 items 9(ii),
10 V2); it prevents the next one from happening silently.  Supersede this file with an
independent review record for pin 8 (same filename, honest diff) when one exists.
