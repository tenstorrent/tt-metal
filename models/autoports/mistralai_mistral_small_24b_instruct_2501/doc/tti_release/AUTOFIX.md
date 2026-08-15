# AutoFix: Mistral Small 24B v10 release failures

## Outcome

AutoFix repaired both operational EngineCore failures without changing the
model, implementation, serving context, acceptance thresholds, or eval scores.
The corrected server completed the exact hardware controls and resumed native
benchmark sweep without page- or slot-allocation failures.

The overall release remains `release-workflow-fail` and `readiness-fail` because
the mandatory quality gates remain below threshold:

- IFEval: 72.55740423987976% versus 78.755% required.
- GPQA flexible extract: 38.88888888888889% versus 43.035% required.

There are no `known_issues` masks or waivers. Operational repair is not used to
reinterpret either valid quality result.

## Proven repairs

### Full async KV lookahead

The first failure was an exact page-boundary under-allocation. Host state was at
token 2,429 while device-authoritative async state reached position 2,432, the
first position of a new 32-token page. Commit
`971ee6cfcdd97a36a98e26f96ff7dda08441d219` reserves the full three-token TT
async pipeline depth. Its exact regression and a 2,399-input/96-output P300x2
control passed; the resumed GPQA run completed 90/90 operationally.

### Pressure-only stale state-slot reclamation

The second failure followed an idle transition between benchmark points. A
scheduler-only final cleanup never reached the TT worker, leaving 32 historical
host slot owners before 31 new prefills. Commit
`aab6d846caf95c5e9cf8038f3338650a9132c383` snapshots the immediately prior
persistent-batch owners and, only under impossible slot pressure, reclaims older
off-batch owners while preserving potentially live recent state.

The focused regression recreates the 32-old/31-new allocation. The exact
production-server lifecycle control ran concurrency 1 for 8 requests immediately
followed by concurrency 32 for 256 requests; both completed with zero failures.

## Verification

- Focused host regressions: 10 passed across state slots, async lookahead, and
  async decode preemption.
- Boundary hardware control: 1/1 completed, 0 failed.
- Idle-transition control: 8/8 followed by 256/256 completed, 0 failed.
- Native release GPQA after lookahead repair: 90/90 completed operationally.
- Native benchmark after both repairs: see `benchmark_report_v10_slotfix.md`.
- Native spec/API conformance after both repairs: see
  `spec_report_v10_slotfix.md`.

Fresh-context failure evidence and source-level reasoning are recorded in
`AUTODEBUG_V10.md`; compact control metrics are in
`hardware_controls_v10.json`.
