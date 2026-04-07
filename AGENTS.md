# AGENTS

## TTNN Op Creation Policy

- For any request to create or implement a **new TTNN operation**, invoke `$op-create-generic` first.
- Use the repository adapter at `.codex/op_pipeline.yaml`.
- Respect adapter mode strictly:
  - If `mode: pipeline_only`, do not manually freestyle op implementation before pipeline phases complete.
- If pipeline setup is missing or a phase command fails, stop and report the failure; do not silently fall back to manual implementation.
- Manual/freestyle implementation is allowed only if the user explicitly requests bypassing the pipeline.
