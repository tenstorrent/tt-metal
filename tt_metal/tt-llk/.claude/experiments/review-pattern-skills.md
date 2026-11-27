# Experiment: skills distilled from review patterns

Status: experimental, not proposed for merge yet.

## Hypothesis

Most review comments on a feature stack repeat a small set of patterns. If
those patterns are written into skills, an agent that writes or reviews the
next PR of the same kind catches them before a human reviewer has to.

## Method

The sample was one reviewer's comments (about 170) on a stack of seven
performance-counter PRs in tt-metal:
[#65806](https://github.com/tenstorrent/tt-metal/pull/65806),
[#66112](https://github.com/tenstorrent/tt-metal/pull/66112),
[#66113](https://github.com/tenstorrent/tt-metal/pull/66113),
[#66148](https://github.com/tenstorrent/tt-metal/pull/66148),
[#66149](https://github.com/tenstorrent/tt-metal/pull/66149),
[#66150](https://github.com/tenstorrent/tt-metal/pull/66150) and
[#66306](https://github.com/tenstorrent/tt-metal/pull/66306).

The comments were grouped by theme:

| Pattern | Share |
|---------|-------|
| Duplication: per-arch tables, redeclared registers, copied helpers, layouts described in several places, host constants mirrored by hand | about 40 |
| Counter correctness: wrap, window bounds, buffer reservation, silent option conflicts, missing device keys, directional metrics | about 30 |
| Docs written as PR narrative: measurements, costs, reviewer notes, copied tables, repeated sections | about 25 |
| Conflicts or gaps with the public ISA docs | about 13 |
| Scope: decoders with no caller, captures no test uses | about 12 |
| Oversharing: RTL names, design, firmware and profiler internals | about 10 |
| Weak tests: tests that only restate constants | about 10 |

The PRs are named here only as the source sample. The skill text is generic
and does not cite them.

## Changes

- `skills/code-comments`: rules and examples for PR evidence in comments,
  linking the ISA instead of "measured", no RTL or internal names, and one
  rationale in one place; four checklist items.
- `skills/docs-writing` (new): reference instead of narrative, one source of
  truth per table or layout, ISA alignment, oversharing, readability.
- `skills/perf-counter-family` (new): reuse of existing register headers and
  shared capture steps, host parsing of device constants, correctness checks
  for counters, perturbation, scope, ISA cross-check.
- `CLAUDE.md`: two skill-table rows and a pointer under Documentation.

The weak-test pattern is not encoded in this experiment; `test-development`
is unchanged.

## How to evaluate

On the next counter or documentation PR:

1. Run the author-side agent with these skills loaded, or ask it to self-review
   against their checklists before review.
2. Compare the human review comments with the checklists.
3. Record, per pattern: caught before review, missed, and false positives
   (guidance that pushed a wrong change).

## Exit

Keep the skills if repeat comments drop noticeably. Trim the rules that
produce false positives or never fire. Drop the experiment if neither happens.
