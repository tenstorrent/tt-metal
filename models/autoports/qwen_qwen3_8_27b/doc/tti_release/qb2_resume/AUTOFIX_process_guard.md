# AutoFix: guarded startup and orphan cleanup

Symptom: the first QB2 baseline launch continued after a failed shell preflight, and stopping its readiness wrapper left engine1033315 alive during model loading. Driver owner entries also included PID0, whose identity was unknown.

Cause: the shell did not propagate the Python assertion into launch refusal. The dedicated process guard used only an environment marker for descendant cleanup; an engine can lose that marker. Its adopted child was therefore skipped.

Fix: `run_vllm_stage.sh` invokes the guard with `--require-idle-tt 4`. The guard refuses nonempty owner files, including PID0, before creating the runner. During cleanup only, it also accepts its own same-UID direct children, identified by PID and birth ticks and revalidated after opening a pidfd. It does not broaden normal marker-based signaling or send SIGKILL.

Validation: ten focused host tests passed, including a real exec-with-empty-environment descendant, preservation of another launch, PID-reuse refusal, TERM-ignoring-child reporting, and owner0/missing-device refusal. The independent narrow review found no required work. The subsequent guarded server reached health200 and served the exact252-output benchmark. Live shutdown validation also passed on 2026-09-15: guard, runner, API and engine exited, all four device-owner files were empty, and the post-run `tt-smi -s` exited 0. Server session49219 returned143 after the requested SIGTERM. No hardware reset was used. See `process_guard_shutdown.json`.

Recovery: signaled only the new launch's verified guard and its exact orphan engine. All four owner files became empty. PID0 entries disappeared too, but this correlation does not identify their opener. No reset, cache removal, or foreign-process signal occurred. Original server617d had already closed cleanly with health exit0.
