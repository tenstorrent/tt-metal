# Merging Findings and Output Format

Instructions for aggregating per-reviewer findings into the unified report.

## Input from reviewers

Each reviewer emits the format defined in `shared.md` — a `## Findings` bullet
list and a `## Learn Notes` list of absolute paths. Parse both sections for
each reviewer. The `file:<path>:<line>` and severity labels are the load-bearing
fields for merging.

## Check completeness before merging

Before merging, verify each dispatched reviewer returned a well-formed output:

- `## Findings` section present
- Either findings in the required bullet shape, or the literal "No issues found."
- `## Learn Notes` section present (or "None.")

If a reviewer's output is missing, truncated, or malformed, do **not** silently
drop it. Name the failure in the report header:

```
**Reviewers**: Architect, Programmer, Documentation, Fresh-eye (QA — failed, see note)
```

Record the failure reason in the note file so the human can retry.

## Duplicate Detection

Two findings are duplicates if **both**:
1. Same file and line within ±5 lines
2. Similar issue keywords (same problem, not just same location)

When merging duplicates:
- Keep the most detailed description and suggestion
- List all reviewers who flagged it: `flagged by: Architect, Programmer`
- Do not auto-promote severity based on reviewer count — multi-reviewer
  consensus is a signal to the human, not an algorithmic rule.

### Severity disagreement across reviewers

When reviewers flag the same issue at different severities (e.g., Architect
says MUST-FIX, Fresh-eye says CONSIDER), the disagreement is real information —
each reviewer's role lens assigns different impact. Handle it by:

- Taking the **highest** severity any reviewer assigned as the merged label
- Listing lower-severity flaggers with their severity in parentheses:
  `flagged by: Architect, Fresh-eye (CONSIDER)`

The reader sees both the bar and the dissent.

## Severity

Severity is defined **per reviewer** in each role's `## Severity Definitions`.
The aggregator preserves whatever severity each reviewer assigned and never
reinterprets it. There is no global severity taxonomy in this file.

## Output Format

**Persistent 1..N numbering across severities.** One contiguous sequence — do
not restart per group, do not skip numbers. The user must be able to say
"address 3 and 7" unambiguously.

**Sort order:** by severity (MUST-FIX → SHOULD-FIX → CONSIDER), then within a
severity by `(flagger count descending, file path ascending)`. Stable and
reproducible.

```
# Code Review

**Scope**: <scope>    **Files**: <count>    **Commit**: <sha>
**Reviewers**: <list, with failures noted>
**Note**: ~/.tt-agent/notes/findings-review-<ts>-<scope>.md

---

1. [MUST-FIX] <title>                    flagged by: Architect, Programmer
   File: path/to/file.cpp:123
   Issue: <what's wrong + concrete evidence — file:line, source path, invariant, or tt:learn note>
   Suggestion: <how to fix>

2. [MUST-FIX] <title>                    flagged by: QA
   File: path/to/test.py:45
   Issue: ...
   Suggestion: ...

3. [SHOULD-FIX] <title>                  flagged by: Programmer
   File: ...
   Issue: ...
   Suggestion: ...

4. [SHOULD-FIX] <title>                  flagged by: Fresh-eye, Documentation (CONSIDER)
   File: ...
   Issue: ...
   Suggestion: ...

5. [CONSIDER] <title>                    flagged by: Architect
   File: ...
   Issue: ...
   Suggestion: ...

---

Review complete. <N> findings from <K> reviewers.
```

If no issues:

```
# Code Review

**Scope**: <scope>    **Files**: <count>    **Commit**: <sha>
**Reviewers**: <list>
**Note**: ~/.tt-agent/notes/findings-review-<ts>-<scope>.md

No issues found.
```

### Reviewer → merged transformation

The reviewer wire format (bulleted, `## Findings`) and the merged format
(numbered, flat `File:/Issue:/Suggestion:`) differ. The aggregator performs:

1. Strip `- **` prefix and `**` bold from each finding's header line
2. Flatten nested sub-bullets to plain `File:`, `Issue:`, `Suggestion:` lines
3. Append `flagged by: <reviewers>` to the title line
4. Renumber contiguously 1..N per the sort order above

## Note File

Persist the same content (header + all findings in identical format) to:

```
~/.tt-agent/notes/findings-review-<YYYY-MM-DD-HHMMSS>-<scope-slug>.md
```

Seconds precision prevents collisions on back-to-back reviews.

`<scope-slug>`: short dash-delimited tag — `all-uncommitted`, `branch-vs-main`,
`staged`, `unstaged`, or `files-<first-file-stem>` for specific files.

The note additionally aggregates every path emitted in reviewers' `## Learn Notes`
blocks into a single evidence trail:

```
## tt:learn notes cited
- <absolute path> (produced by Programmer, cited in finding 3)
- <absolute path> (produced by Architect, cited in finding 7)
```

If a reviewer's output was malformed or missing, record that too:

```
## Reviewer failures
- QA: output missing (subagent timeout)
```

Never overwrite a prior review. One file per run, timestamp-disambiguated to
the second.
