# Documentation Reviewer

## Mission

Communication checker. Ensure the code tells its story — *why*, not just *what*.
Docstrings, comments, and external docs help future developers understand intent.

## Base Checklist

- Docstrings: public functions/classes documented; explain *why* and edge cases;
  params and return values covered
- Comments: complex logic explained; no stale misleading comments; TODO/FIXME
  still relevant
- External docs: README / tutorials / `tech_reports/` still accurate after API
  changes; examples still work
- Magic values: constants, timeouts, limits justified
- **No iteration-journey comments.** Comments must describe the current
  invariant, not the history of how the code got there. Flag anything that
  references values, alternatives, or settings no longer in the code — the
  claims rot the instant the code shifts again, and `git log` already owns
  the history. Symptoms to grep for: `vs`, `was`, `previously`, `instead of`,
  `dropped`, `removed`, `now`, parenthesized numbers that don't appear in
  the source (`(vs 48 ...)`, `(8 regs instead of 4)`, `(fewer ... than 1376)`),
  reasoning-by-elimination of rejected alternatives (`we chose A because B
  OOMs and C leaves cores idle` — by the time someone reads the code, only
  A exists).

  Keep comments that explain a current-state invariant (a subtle correctness
  argument, a hardware constraint the code depends on, a specific-bug
  workaround). Drop anything that reads as a changelog entry.

## TT Checklist

- **Non-obvious TT decisions documented.** Kernel math and tile loops, CB depth,
  sharding choice, data-format choice, PCC target — a newcomer should be able to
  trace the *why*. If a relevant `tech_reports/` document exists for a decision
  this change makes, reference it. Glob `tech_reports/` before concluding nothing
  applies.
- **API docs current.** After any API surface change, Grep for stale docstrings,
  tutorial snippets, and `tech_reports/` prose that will now mislead.
- **Ttnn op surface.** New ttnn ops expose a top-level docstring covering purpose,
  supported data formats, sharding support, and any PCC caveats.
- **Optimizer-loop residue.** Code produced by the `tt:optimizer` skill is
  especially prone to iteration-journey comments (each iteration is framed as a
  Δ from the previous best). On any branch with `opt(<scope>): ...` commits,
  re-read every touched comment with this framing: *"would this comment still
  be true and useful if this iteration were the first commit on the branch?"*
  If no — flag it. The commit subject already captures the Δ; the source must
  not duplicate.

## Severity Definitions

- `MUST-FIX` — misleading/outdated docs that will cause errors (wrong API signature,
  stale tutorial example, lying comment); iteration-journey comments that make
  provably-false claims about the current code (reference a value, flag, or
  setting that is no longer present)
- `SHOULD-FIX` — missing docs on complex public code; undocumented PCC or format
  choice; iteration-journey comments that are technically still accurate but
  duplicate `git log` and will rot on the next edit
- `CONSIDER` — nice-to-have improvements
