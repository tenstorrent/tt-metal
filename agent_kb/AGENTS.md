# Agent KB Schema

This file defines how agents should use and maintain `/agent_kb`.

## Purpose

Use this KB to improve code generation for `tt-metal`, especially in areas where the raw code and docs contain subtle constraints that are easy to miss:

- kernel pipeline structure
- circular buffer protocols
- Dst register ownership and synchronization
- NoC ordering and barriers
- numerical accuracy pitfalls
- architecture-scoped behavior
- debugging workflows and observability

## Rules

1. Treat repo code and docs as the source of truth.
2. Treat KB pages as synthesized guidance, not authority.
3. Prefer updating an existing KB page over creating duplicates.
4. Every non-trivial claim should include a `Sources` section with repo paths.
5. Distinguish clearly between:
   - `Documented`: stated in repo docs or comments.
   - `Inferred`: concluded from examples or implementation patterns.
   - `Empirical`: observed from tests, debug logs, or experiments.
6. Scope guidance when needed:
   - hardware generation
   - kernel type
   - data format
   - memory layout
   - API family
7. When codegen is the task, always inspect at least one close example in the current checkout before editing code.

## Page Types

Use YAML frontmatter on KB pages with this minimum schema:

```yaml
---
title: Human readable title
type: concept | pitfall | recipe | debug | arch | source
status: seed | active | needs_review
confidence: high | medium | low
last_reviewed: YYYY-MM-DD
tags:
  - kernel
  - circular-buffer
source_files:
  - relative/path/to/source.md
---
```

Optional fields:

- `arch_scope`
- `kernel_scope`
- `api_scope`
- `related_pages`

## Content Style

- Write short pages with strong headings.
- Prefer checklists and invariants over long prose.
- Link to neighboring pages aggressively.
- Record uncertainty explicitly instead of smoothing it over.
- Keep examples small and source-linked.

## Ingest Workflow

When adding a new source:

1. Read the source.
2. Create or update a page under `sources/`.
3. Update relevant concept, pitfall, recipe, or debug pages.
4. Update `index.md`.
5. Append a short entry to `log.md`.

## Query Workflow

When answering a code-generation question:

1. Search `agent_kb/`.
2. Read the matching concept, pitfall, and recipe pages.
3. Search raw sources and examples to verify the current implementation pattern.
4. Produce the answer or patch with citations to both KB pages and raw sources when useful.

## Lint Workflow

Periodically check for:

- broken local links
- pages not listed in `index.md`
- orphan pages with no inbound links
- stale pages that refer to old APIs or missing files
- claims lacking a `Sources` section

Use `python3 tools/agent_kb/lint_kb.py`.
