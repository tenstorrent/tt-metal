# tt-metal agent handbook

This directory contains agent-facing knowledge for working in this repo. The content is intended to be read by Claude Code, Cursor, Codex, and other coding agents.

## Notes — load these when working on the porting pipeline

| Topic | File |
|---|---|
| HF → TTNN porting pipeline structure, per-model layout, the Generator contract | [notes/porting-pipeline.md](notes/porting-pipeline.md) |
| tt_transformers Generator gotchas — five non-obvious requirements when wrapping it directly | [notes/tt-transformers-gotchas.md](notes/tt-transformers-gotchas.md) |
| Dev-box essentials — Python env, mesh devices, weight paths, crash recovery, reading logs | [notes/dev-environment.md](notes/dev-environment.md) |
| Design conventions for shared porting infrastructure | [notes/design-conventions.md](notes/design-conventions.md) |

## Skills — invoke these on user request

| Skill | Purpose |
|---|---|
| [skills/decoder-to-productized/SKILL.md](skills/decoder-to-productized/SKILL.md) | One stage of the porting pipeline: ported decoder → `tt/model.py` + `tt/generator.py` + `tt/generator_vllm.py` + vLLM registration. |

## When to read what

- **Working anywhere in `models/autoports/<model_name>/` or `models/common/readiness_check/`** → read [porting-pipeline.md](notes/porting-pipeline.md) first.
- **Writing a generator that builds on tt_transformers utilities** (e.g. `create_tt_model`, `prepare_generator_args`) → read [tt-transformers-gotchas.md](notes/tt-transformers-gotchas.md) before anything else; the gotchas are not documented in the tt_transformers code.
- **Running anything on the device, debugging a TTNN crash, opening a mesh** → [dev-environment.md](notes/dev-environment.md).
- **Adding new shared infra (contracts, runners, harnesses)** → [design-conventions.md](notes/design-conventions.md).
- **User asks to "port a model" / "productize a decoder"** → invoke `skills/decoder-to-productized`.

## Conventions for adding to this handbook

- Topic-per-file under `notes/`. Keep each file ≤200 lines; split if it grows.
- Lead with the rule or fact; follow with a short "Why" / "How to apply" if non-obvious.
- Cite file paths and code locations with `path:line` so agents can jump directly.
- Don't duplicate what's already in code comments — link to the code.
