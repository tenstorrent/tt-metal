---
name: code-comments
description: Write, edit, or review concise comments in tt-llk C++, Python, tests, kernels, and infrastructure. Use when adding explanatory comments, shortening verbose comments, removing stale or duplicated notes, documenting hardware constraints, or reviewing comment quality.
user_invocable: true
---

# Code Comments

## Goal

Leave the smallest durable explanation that a future reader cannot recover
from the code itself. Comments explain **why**, a non-obvious constraint, or
the evidence behind a choice. They do not narrate syntax.

## Before adding a comment

1. Read the surrounding function and nearby comments. Do not repeat them.
2. Prefer a clear name, helper, type, assertion, or simpler control flow when
   it can make the code self-explanatory.
3. Identify the exact fact the reader needs:
   - hardware or architecture constraint;
   - ordering, synchronization, or lifetime invariant;
   - intentionally unusual test variant;
   - reason an apparently simpler implementation is wrong;
   - external source of truth or issue that must be consulted.
4. If none applies, omit the comment.

## Style

- Keep a local comment to one or two lines. Put longer design rationale in a
  skill, reference document, issue, or PR description and link to it.
- Lead with the reason or invariant, not a history of edits.
- Describe current behavior. Avoid "now", "previously", "removed", and other
  wording that becomes stale after the next change.
- Place the comment immediately above the smallest code region it governs.
- Use the repository's normal sentence style and punctuation. Do not add
  decorative headings or separator art for one statement.
- Do not copy the same explanation to multiple branches. Extract a helper or
  place one comment above their shared decision.
- Delete or update a comment in the same change that invalidates it.

## Hardware and performance claims

A code comment records intent; it is not authority for hardware semantics,
instruction timing, ordering, or errata.

- Ground the claim in the applicable live source: LLK implementation for what
  code issues, ISA or hardware documentation for semantics, and generated
  code/disassembly or measurements for performance.
- Name the mechanism precisely (`Dest` section release clears the packed half)
  instead of asserting a vague outcome (`needed for perf`).
- Link a stable source or issue when the claim cannot be understood from the
  local implementation. Do not paste a long explanation that will drift.
- Never preserve an uncertain hardware claim as fact. Mark the uncertainty
  and the authority needed to resolve it, or omit the verdict.

## Tests

- State what behavior a regression variant uniquely exercises, not how the
  fixture is assembled.
- Do not leave notes about removed skips, wedges, temporary failures, or old
  implementations unless the historical fact still constrains current code.
- A comment does not prove coverage. Verify that the parameter reaches the
  operation under test and is not overwritten by later initialization.
- Explain an intentional mismatch between functional and perf coverage in the
  PR description; keep only the enduring constraint beside the test.

## TODO comments

Use a TODO only for actionable, scoped follow-up:

```text
TODO(#12345): Remove the fallback after Blackhole firmware exposes X.
```

Include the condition for removal and a tracking issue. Do not use a TODO as a
substitute for fixing a defect in the current change.

## Examples

Avoid narration:

```cpp
// Set the destination address.
set_dst_write_addr<...>(0);
```

Prefer the non-obvious invariant:

```cpp
// Pack release clears this Dest half before math reuses it.
_llk_pack_dest_section_done_<...>();
```

Avoid edit history:

```python
# The old wedge skip was removed because this works now.
```

Prefer no comment when the test name and assertion already explain the case.

Avoid duplicated detail:

```cpp
// Reserve one accumulator slot, leaving capacity - 1 chunks available.
// The final accumulator is reserved for the reduction result.
```

Prefer one line:

```cpp
// Reserve the final accumulator slot for the reduction result.
```

## Review checklist

- [ ] Explains why, a constraint, or an invariant that code does not show.
- [ ] One or two lines for local rationale.
- [ ] Does not duplicate nearby code or comments.
- [ ] Describes current behavior without stale history.
- [ ] Hardware and performance claims are grounded in an authoritative source.
- [ ] Test comments describe unique behavior that the test actually reaches.
- [ ] TODO has a tracking issue and removal condition.
