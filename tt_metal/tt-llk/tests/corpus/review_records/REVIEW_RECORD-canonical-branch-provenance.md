# Canonical-branch provenance record (answers wave-10 owner-question 1)

Date: 2026-08-20. Author: swarm orchestrator session (nkapre-authorized).

## Question 1 — who moved tt-metal `nkapre/sfpi`?

The fast-forward `a89594fe2738 → e6375987677f` (2026-08-20, this repo) and the sfpi-repo
fast-forward `4f0ebd4 → cd5b59a` were executed by the swarm orchestrator **on an explicit
owner order**, not unilaterally. Sequence, as recorded in the session ledger before execution:

1. Lane CP (wave-9 chronic remediation) found the frozen-branch root cause and raised the
   owner-ask: "fast-forward tt-metal nkapre/sfpi → work tip (verified pure ff) or amend
   HANDOFF §2."
2. The ask was put to the owner verbatim as decision item 4; the owner replied "sure ok do it"
   (2026-08-20, session transcript; ledger entry "OWNER DECISIONS EXECUTED" written at
   execution time).
3. The orchestrator verified pure fast-forward (`git merge-base --is-ancestor`) and pushed.

Standing owner directive since (2026-08-20): **"keep everything in nkapre/sfpi"** — the
canonical branch is kept fast-forwarded to the integration tip at every merge from now on;
this record's own merge commit is the first executed under that rule.

## Question 2 — the CRAQ ruling (straight-silicon default)

RATIFIED by the owner 2026-08-20 ("ratify straight-silicon and amend the handoff").
Executed same day: HANDOFF §1(3) amended in both copies (owner's master and the sfpi repo's
docs/handoff-20260817/HANDOFF.md) — per-cell device-golden correctness legs gate every sweep
perf cell; CRAQ on the pinned sims remains the lane-level mechanism-validation and debug
oracle; every `--skip-craq-gate` run carries an explicit taint line (gate merged at
3a83b1ccde). Both sweep wrappers now carry the ratification rationale inline above the flag.

## Addendum (2026-08-20, owner order): `work/nkapre-sfpi` RETIRED

Owner ruling on the frozen-vs-live root cause: the dual-branch arrangement is dead.
`work/nkapre-sfpi` was deleted on origin at `f23af11348` (verified byte-identical to
`nkapre/sfpi` at deletion). `nkapre/sfpi` is the single canonical and integration branch;
all lane merges land there directly. Reviews and provenance cite one branch from now on —
the divergence class that generated three waves of false chronic findings cannot recur.
