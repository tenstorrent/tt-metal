# TT-Metal Agent Knowledge Base

This directory is a repo-local knowledge base for guiding LLM agents during `tt-metal` code generation.

It is not a replacement for the source tree. It sits between:

1. Raw sources in the repo such as `METALIUM_GUIDE.md`, `docs/`, `tech_reports/`, `tt_metal/programming_examples/`, and tests.
2. Agent tasks such as writing a kernel, debugging a hang, or choosing a reference implementation.

The goal is to avoid re-deriving the same operational knowledge on every prompt. The KB stores synthesized, source-linked guidance in a format that is easy for an agent to search, update, and cite.

## Layout

- `AGENTS.md`: schema and maintenance rules for the KB.
- `index.md`: content index for fast navigation.
- `log.md`: append-only record of ingests and major updates.
- `concepts/`: stable domain concepts.
- `pitfalls/`: known failure modes and non-obvious traps.
- `recipes/`: repeatable workflows for code generation and review.
- `debug_playbooks/`: debugging workflows.
- `arch/`: architecture and hardware-generation scoped notes.
- `sources/`: seeded summaries of important source documents.

## Intended Workflow

1. Search `agent_kb/` first.
2. Read the most relevant concept, pitfall, and recipe pages.
3. Validate against the current checkout before making changes.
4. When you learn something durable, update the KB and append to `log.md`.

## Tooling

- `python3 tools/agent_kb/search_kb.py <query>`
- `python3 tools/agent_kb/search_kb.py --sources <query>`
- `python3 tools/agent_kb/lint_kb.py`

The KB is deliberately markdown-first. If it grows beyond what `index.md` and `rg` can comfortably handle, add a stronger local search layer later.
