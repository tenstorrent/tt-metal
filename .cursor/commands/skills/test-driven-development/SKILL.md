---
name: test-driven-development
description: >-
  Guides test-first workflows for bugs and features: write a minimal failing
  test that encodes contracts and clear failure messages, confirm the gap is
  not already covered, implement and document the fix, then verify green.
  Use when fixing bugs with tests, adding features via TDD, writing regression
  tests, or when the user asks for red-green-refactor or test-driven
  development.
---

# Test-Driven Development (TDD)

Use this workflow when implementing a fix or feature **test-first**, so the
change is provable and regressions stay visible.

## Core loop

Work in this order unless the user’s repo has a mandated variant (e.g. always
run full CI locally).

```
Task progress:
- [ ] Minimal failing test(s) for the bug or gap
- [ ] Stub or placeholder implementation + doc of the gap (if useful before green)
- [ ] Run new test — expect failure with a clear message
- [ ] Confirm existing tests did not already encode this behavior
- [ ] Implement the real fix or feature
- [ ] Re-run tests — new test passes; no regressions
```

---

## 1. Minimal reproducible test(s)

- **Scope**: One focused scenario (or a tiny set) that reproduces the bug or
  encodes the missing contract. Avoid testing unrelated paths in the same
  change unless necessary.
- **Contracts**: When behavior is specified (APIs, invariants, error
  semantics), assert those explicitly—inputs, outputs, edge cases, and error
  types/messages where the project cares.
- **Failure messages**: Prefer assertions and matchers that print **actionable**
  context on failure (expected vs actual, relevant inputs, identifiers). Avoid
  bare `assert x` without a message when the framework allows a custom string
  or rich matcher.
- **Location**: Place tests where the project already groups similar cases
  (same file/directory patterns, naming conventions).

---

## 2. Implementation sketch and documentation

- **Code**: Add the smallest change that can eventually satisfy the test—often
  starting with a failing state is enough before step 3; then replace with the
  real implementation after the red phase is validated.
- **Documentation**: In code (docstring or comment near the fix) or in the
  test name/docstring, briefly state **what was wrong or missing** and **what
  behavior is now guaranteed**. Link to issue/ticket if the project does that.

---

## 3. Run the new test (expect red)

- Run **only** the new test(s) first for speed, then widen as needed.
- Confirm the failure explains the gap (message, stack, diff)—not a flaky or
  unrelated error (imports, env, timeouts).

---

## 4. Validate existing tests do not cover this case

Before treating the new test as redundant:

- Search for overlapping cases (test names, similar assertions).
- If an existing test *should* have failed but did not, the bug may be in test
  setup, feature flags, or the scenario is not actually exercised—adjust before
  duplicating.
- Goal: **one clear test** that would have caught the bug if it had existed
  before the fix, without useless duplication.

---

## 5. Apply the fix or enhancement

- Implement the minimal correct behavior to satisfy the test and real
  requirements.
- Avoid widening scope (refactors, drive-by cleanups) unless required for the
  fix or requested by the user.

---

## 6. Re-run and verify green

- Run the new test(s) again — **must pass**.
- Run the **relevant suite** (module/package/project standard) to catch
  regressions.
- If CI differs from local commands, align with project docs or existing
  scripts.

---

## Quick reference: red-green-refactor

| Phase   | Action |
|--------|--------|
| **Red**    | New test fails for the right reason |
| **Green**  | Smallest change that passes |
| **Refactor** | Clean up with tests still green (only if in scope) |

---

## Optional deep dive

For framework-specific assertion patterns (e.g. pytest, Jest, Go `testing`),
see [reference.md](reference.md).
