# <repo>#<N> — <issue title>      (or: <findings file> — <N> open items)

Started: <YYYY-MM-DD HH:MM> on <hostname> (<board>, <arch>)
Source: <parent issue url | findings file path> · sync: <git upstream | --sync-cmd | local-only> · last sync: <HH:MM PUSH_OK>
Options: --ci-budget sanity=5,nightly=5 · --group-max 8 · --only-open? · --unattended run-name <name>
Resume: read this file top to bottom, `findings.py reconcile <file>` (file source), then `gh pr list --author <me> --state open`.

## Items

| # | key / summary | group | class | status | sub-issue | PR | branch / worktree | head SHA |
|---|---|---|---|---|---|---|---|---|
| 1 | | | | triage | | | | |

Status values: triage · repro · fix · verify · pr · bot-review(rN) · ci-queue(<since HH:MM>) · ci · done · parked(<reason>) · frozen(<who asked>)

## Groups (one PR each)

| group | items | files / function | body size (chars, must be ≤ 52,000) | PR |
|---|---|---|---|---|

## CI queue (PRs whose bots are quiet, waiting for a budget slot — nearest-to-done first)

| PR | SHA | needs (sanity / nightly / e2e) | queued since | why this run (first / after fix / after rebase) |
|---|---|---|---|---|

## Pending external events (what the scheduler is waiting on)

| what | run id / url | armed how (bg watch / monitor) | started | SHA | for item |
|---|---|---|---|---|---|

## Retry counters

| PR | bot re-dispatches (per SHA) | CI reruns (per cause) | rebase+retrigger (≤3) | code-change CI cycles (≤4) | main-wait started |
|---|---|---|---|---|---|

## Perf A/B (§3b — one row per item × arch on a measured path)

| item | arch | perf module / -k / --op | base sha | verdict (exit) | median Δ% | worst variant Δ% | evidence dir |
|---|---|---|---|---|---|---|---|

## CI verdicts

| PR | workflow | run | job | verdict | evidence |
|---|---|---|---|---|---|

## Log (append-only, timestamped)

- <HH:MM> ...
