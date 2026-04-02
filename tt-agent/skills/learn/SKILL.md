---
name: learn
description: "Research live Tenstorrent codebases on demand — produces dated context notes from local code search and deepwiki, consumed by other skills and developers"
metadata:
  layer: meta
---

# TT Learn

## Purpose

Researches live Tenstorrent codebases and produces dated context notes. Other skills
invoke tt-learn when they need volatile knowledge (APIs, patterns, architecture, test
coverage) before proceeding. Developers invoke it directly to build understanding
before starting work. Works in any TT repo — detects context from the working directory.

tt-learn never guesses. It reads code, synthesizes findings, and writes them down.

## When to Invoke

- Another skill needs codebase context before it can proceed
- Developer wants to understand a subsystem before modifying it
- `tt-learn("ttnn op structure")`, `tt-learn("CCL teardown sequence")`
- "I need to understand how X works"
- Orchestrator dispatches here for the "need codebase context" row

## Pipeline

```
check existing → load references → research subagent → return note
```

1. **Check existing**: Look for `~/.tt-agent/notes/context-<topic-slug>.md`.
   If found and no refresh requested, return it immediately.

2. **Load references**: If `tt-agent/knowledge/references/` exists in the current repo,
   scan for files relevant to the topic. These provide starting pointers but are not
   required — the skill works without them.

3. **Dispatch research subagent**: Launch an Agent with `research-prompt.md` as
   instructions. Pass the topic, any matched reference content, and the refresh flag.
   The subagent does the actual Grep/Read/deepwiki work and writes the context note.

4. **Return note**: Return the context note content to the caller so they can use
   it immediately without reading the file.

## Note Format

All notes are written to `~/.tt-agent/notes/context-<topic-slug>.md`:

```markdown
# Context: <topic>

**Date**: YYYY-MM-DD  **Repo**: tenstorrent/tt-metal  **Commit**: abc1234

## Core Insight

<1-3 sentences: the single most important thing to know.>

## How It Works

- <Concise bullet points — only what's needed to act on the topic>

## Key Files

- `path/to/file` — one-line description
```

**Target: under 80 lines.** Notes become part of agent context — every line costs.

## Refresh

Notes are assumed fresh for a development session. To force re-research:
- User says "refresh" or "re-learn"
- Caller passes a refresh hint
- This skips step 1 and always researches from scratch

## Failure Mode

If neither local search nor deepwiki produces useful results, write a note
documenting what was attempted, what references were checked, and what's missing.
Escalate to the user — don't fabricate understanding.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| Research subagent instructions | `research-prompt.md` |
