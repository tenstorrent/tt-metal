# Agent KB Log

## [2026-04-06] bootstrap | Initial tt-metal agent KB

- Created the initial KB structure.
- Added schema and maintenance rules in `agent_kb/AGENTS.md`.
- Seeded concept, pitfall, recipe, debug, and source-summary pages from core `tt-metal` documentation.
- Added lightweight search and lint scripts under `tools/agent_kb/`.

## [2026-04-06] ingest | Conv3D operation

- Added `agent_kb/sources/conv3d_op.md` summarizing the current `ttnn.experimental.conv3d` structure.
- Added `agent_kb/recipes/conv3d_codegen.md` for safe modification workflow.
- Added `agent_kb/pitfalls/conv3d_gotchas.md` capturing op-specific hazards and test-backed caveats.
- Updated `agent_kb/index.md` to surface the new Conv3D pages.
