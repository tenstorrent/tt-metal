# AGENTS

## TTNN Op Creation Policy

- For any request to create or implement a **new TTNN operation**, invoke `$op-create-generic` first.
- Use the repository adapter at `.codex/op_pipeline.yaml`.
- Re-entry guard:
  - If task context explicitly contains both `PIPELINE_PHASE_SUBTASK: true` and
    `OP_CREATE_GENERIC_ENTRYPOINT_ALREADY_RUN: true`, the pipeline has already been
    started, so do **not** invoke `$op-create-generic` again.
  - In that case, execute only the requested phase task.
- Respect adapter mode strictly:
  - If `mode: pipeline_only`, do not manually freestyle op implementation before pipeline phases complete.
- If pipeline setup is missing or a phase command fails, stop and report the failure; do not silently fall back to manual implementation.
- Manual/freestyle implementation is allowed only if the user explicitly requests bypassing the pipeline.
