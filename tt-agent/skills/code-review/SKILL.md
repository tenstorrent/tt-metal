---
name: code-review
description: "Strict, honest, TT-aware code review — 5 parallel reviewers produce a ranked, evidence-backed findings plan. Use before merging or as a workflow bookend."
metadata:
  layer: meta
---

# TT Code Review

## Purpose

Run the reviewers listed in the cast against a diff in parallel, merge their
findings, return a ranked action plan. Each reviewer is strict, honest,
challenging, and TT-aware. Pure report — no fixes, no workflow gating.

**Why `layer: meta`:** this skill is cross-cutting (any skill can call it as a
bookend) and produces a notes artifact. It is not pipeline-bound like `tt-run`
or `tt-profiler`. Same shape as `tt-learn`.

## When to Invoke

- `/tt:code-review`, "review my changes", "review this PR"
- Orchestrator dispatches here for "review code before merging"
- Any workflow skill calling here as a bookend before declaring done

## Pipeline

```
select scope → gather diff → capture metadata → dispatch reviewers → merge → note → return plan
```

1. **Scope**: If a caller passed a diff, use it. Otherwise ask the user which
   scope to review (see `Scope Selection` below).
2. **Gather diff**: Run the git command for the chosen scope (see mapping below).
   Empty diff → stop; emit the "No issues found" template with reason
   `"Diff was empty"` so the caller knows the skill ran but had nothing to review.
3. **Capture metadata**: `git rev-parse --short HEAD` for `<sha>`;
   `git diff --name-only <scope args> | wc -l` for `<file-count>`.
4. **Dispatch reviewers**: Use the `Task` tool. Emit one `Task` call per reviewer
   in a **single assistant message** for true parallel execution. Sequential
   dispatch defeats the purpose. Prompt template below.
5. **Merge**: Apply `merge.md`. Verify each reviewer returned a well-formed
   output; note failures in the header. Detect duplicates, apply severity
   tiebreak, tag flaggers.
6. **Note**: Write the full merged plan to
   `~/.tt-agent/notes/findings-review-<YYYY-MM-DD-HHMMSS>-<scope-slug>.md`.
   Aggregate `## Learn Notes` from every reviewer into the evidence trail.
7. **Return**: Stream the unified plan with persistent 1..N numbering across severities.

## Scope Selection

Ask the user to choose one (any harness's user-interaction primitive is fine —
Claude Code's `AskUserQuestion`, or a simple prompt):

| Scope | Git command |
|---|---|
| staged | `git diff --staged` |
| unstaged | `git diff` |
| branch vs main | `git diff main...HEAD` |
| all uncommitted (default) | `git diff HEAD` |
| specific files | `git diff HEAD -- <paths>` |

Scope slugs for the note filename: `staged`, `unstaged`, `branch-vs-main`,
`all-uncommitted`, `files-<first-file-stem>`.

## Reviewer Cast

| Reviewer | Name (→ file) | Focus |
|---|---|---|
| Architect | `architect` → `reviewers/architect.md` | structure |
| Programmer | `programmer` → `reviewers/programmer.md` | correctness |
| Documentation | `documentation` → `reviewers/documentation.md` | intent |
| Fresh-eye | `fresh-eye` → `reviewers/fresh-eye.md` | clarity |
| QA | `qa` → `reviewers/qa.md` | coverage |

The `Name` column is the exact on-disk filename stem; substitute it for
`<name>` in the dispatch template.

## Dispatch Template

For each reviewer, issue one `Task` call with this prompt. All `Task` calls
go in the same assistant message.

```
You are the <Role> reviewer for a TT code review.

Read these two files before proceeding:
1. <repo-root>/tt-agent/skills/code-review/shared.md   (shared contract — strictness,
   evidence, exploration mandate, output format)
2. <repo-root>/tt-agent/skills/code-review/reviewers/<name>.md   (your role file —
   mission, checklists, severity definitions)

Where <repo-root> is the output of `git rev-parse --show-toplevel` in the cwd.

Diff to review (full content):
<diff>

Apply the shared contract. Output only the Findings and Learn Notes blocks per
the shared output format. No preamble, no conclusion.
```

Substitute `<Role>`, `<name>`, `<repo-root>`, and `<diff>` at dispatch time.

## Caller Contract

tt:code-review is a **pure report producer**. Findings are advisory — it never
blocks the caller.

- **Human**: reads plan, fixes, iterates.
- **Workflow skill**: reads plan, acts on findings, re-runs its verification
  loop, may call back for a fresh review.

### Calling from a workflow skill

A workflow skill integrates code-review as a bookend like this:

1. Before declaring done, compute a diff of its own changes:
   `git diff <base>...HEAD` where `<base>` is the workflow's pre-change commit.
2. Invoke code-review with the diff as direct input (skip scope prompt).
3. Read the returned plan. For each MUST-FIX, decide: apply the fix and re-run
   the verification step, or escalate with context.
4. Cite the generated findings note path in the workflow's own experiment log
   so the review evidence is traceable.

## Progressive Load Table

Files the code-review skill itself loads as its pipeline advances:

| Sub-task | Load |
|---|---|
| Merging logic and output format | `merge.md` |
| Reviewer prompt template reference | see Dispatch Template above |

Files each reviewer subagent loads (not loaded by the orchestrator):

| Sub-task | Load |
|---|---|
| Shared reviewer contract | `shared.md` |
| Role-specific instructions | `reviewers/<name>.md` |
| Hardware invariants (when relevant) | `tt-agent/knowledge/hardware/*.md` |
| Canonical example pointers | `tt-agent/knowledge/references/*.md` |
| Per-repo recipes | `tt-agent/knowledge/recipes/<repo>/*.md` |
| Volatile pattern questions | invoke `tt:learn("<topic>")` |
