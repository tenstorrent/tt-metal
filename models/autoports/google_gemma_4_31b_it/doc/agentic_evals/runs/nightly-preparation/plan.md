# Nightly rung preparation

Prepare three independent source-only eval-registration patches based on TTI bb321bf0f066dd472f51bd0b3f8875d47a0b3cfb. Each enables only its own nightly suite and preserves frozen smoke membership, release limits and two-client concurrency. Nightly sizes are GPQA 10, SWE 5, Terminal 10. Each is nested over smoke. Do not apply to the active publication checkout, publish, dispatch, or advance a rung before valid smoke.

Use a detached host-only worktree to generate and verify each candidate independently. Confirm exact stock common/Harbor selection and disabled peer suites. No devices; device-hour box zero. These are prepared candidates; measured smoke projections and admission determine whether the coordinator may dispatch.
