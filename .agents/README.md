# tt-metal agent handbook

This directory contains agent-facing knowledge for working in this repo. The content is intended to be read by Claude Code, Cursor, Codex, and other coding agents.

## Notes — load these when working on the porting pipeline

| Topic | File |
|---|---|
| Mission, pipeline stages, durable output shape, and evidence standards | [notes/model-bringup-mission.md](notes/model-bringup-mission.md) |
| HF → TTNN porting pipeline structure, per-model layout, the Generator contract | [notes/porting-pipeline.md](notes/porting-pipeline.md) |
| tt_transformers Generator gotchas — five non-obvious requirements when wrapping it directly | [notes/tt-transformers-gotchas.md](notes/tt-transformers-gotchas.md) |
| Design conventions for shared porting infrastructure | [notes/design-conventions.md](notes/design-conventions.md) |

## Skills — invoke these on user request

| Skill | Purpose |
|---|---|
| [skills/functional-decoder/SKILL.md](skills/functional-decoder/SKILL.md) | HF decoder layer → correct TTNN `FunctionalDecoder`. |
| [skills/optimize-decoder/SKILL.md](skills/optimize-decoder/SKILL.md) | Correct TTNN decoder → faster `OptimizedDecoder` with evidence. |
| [skills/multichip-decoder/SKILL.md](skills/multichip-decoder/SKILL.md) | Single-chip decoder → multi-chip `MultichipDecoder`. |
| [skills/decoder-to-productized/SKILL.md](skills/decoder-to-productized/SKILL.md) | One stage of the porting pipeline: ported decoder → `tt/model.py` + `tt/generator.py` + `tt/generator_vllm.py` + vLLM registration. |

## When to read what

- **Working on agent-driven model bringup** → read [model-bringup-mission.md](notes/model-bringup-mission.md) first, then the skill for the current stage.
- **Working anywhere in `models/autoports/<model_name>/` or `models/common/readiness_check/`** → read [porting-pipeline.md](notes/porting-pipeline.md) first.
- **Writing a generator that builds on tt_transformers utilities** (e.g. `create_tt_model`, `prepare_generator_args`) → read [tt-transformers-gotchas.md](notes/tt-transformers-gotchas.md) before anything else; the gotchas are not documented in the tt_transformers code.
- **Adding new shared infra (contracts, runners, harnesses)** → [design-conventions.md](notes/design-conventions.md).
- **User asks to "port a model" / "productize a decoder"** → invoke `skills/decoder-to-productized`.

## Conventions for adding to this handbook

- Topic-per-file under `notes/`. Keep each file ≤200 lines; split if it grows.
- Lead with the rule or fact; follow with a short "Why" / "How to apply" if non-obvious.
- Cite file paths and code locations with `path:line` so agents can jump directly.
- Don't duplicate what's already in code comments — link to the code.
