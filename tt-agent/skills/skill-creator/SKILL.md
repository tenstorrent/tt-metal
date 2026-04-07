---
name: skill-creator
layer: meta
description: "Design and build skills for the tt-agent system — guides through rigorous design alignment before writing, then applies TT conventions. Use when creating, improving, or reviewing tt-agent skills"
---

# TT Skill Creator

## Purpose

Designs and builds high-quality skills for the tt-agent system. The most important
part of skill creation is **designing the skill well** — understanding what it should
do, exposing flaws in the spec, achieving full alignment with the developer before
writing anything.

Wraps `/skill-creator` for base mechanics (format, frontmatter, progressive load,
evals) and adds TT-specific guidance from `tt-guidelines.md`.

## When to Invoke

- Create a new tt-agent skill
- Improve or review an existing tt-agent skill
- Verify a skill follows tt-agent conventions

## Pipeline

```
design → align → write → validate
```

The design phase is where the real work happens.

## Phase 1: Design Alignment

**Hard gate: you must present a design summary and receive explicit developer
approval before writing a single line of skill content. No exceptions — even
if the request contains a full spec.**

### Step 1 — First output (always)

Before asking any questions, produce a design summary covering:

1. **Purpose** — what the skill does and explicitly does not do
2. **Layer** — orchestration / workflow / tool / meta, with justification
3. **Trigger** — what request patterns invoke it; boundary with adjacent skills
4. **Input → output** — what it receives, what it produces, where output lands
5. **Dependencies** — which skills it calls; which call it; device access needed?
6. **Open questions** — things you cannot resolve from context alone

End with: _"What did I get wrong? What did I miss?"_

Wait for the developer's response before proceeding.

### Step 2 — Interrogate gaps

For each open question and each correction from the developer:
- Ask one question at a time. Wait for the answer.
- Push back on vague answers. "Helps with X" is not a spec — ask for metric, target, iteration loop.
- After each answer, update your stated understanding so the developer can catch drift.

### Step 3 — Expose flaws before approval

Before asking for approval, actively check:
- Scope creep: if the skill does 5 things, should it be 2 skills?
- Overlap with existing skills — read the dispatch table
- Implicit knowledge assumptions — volatile info must go through tt-learn, not be assumed known
- Missing convergence criteria (workflow) or verification steps (tool)

### Step 4 — Gate

Once all questions are resolved, state: _"I'm ready to write. Shall I proceed?"_

**Do not write until the developer says yes.**

## Phase 2: Write

Only after receiving explicit approval to proceed:

1. **Invoke `/skill-creator`** for base mechanics.
2. **Load `tt-guidelines.md`** and apply TT-specific constraints.
3. Write SKILL.md + sub-files. Be concise — optimize token count in all skill content.

## Phase 3: Validate

Run the self-check from `tt-guidelines.md`. Run frontmatter tests.

## Progressive Load Table

| Sub-task | Load |
|---|---|
| TT-specific rules and constraints | `tt-guidelines.md` |
| Base skill format, frontmatter, evals | Invoke `/skill-creator` |
