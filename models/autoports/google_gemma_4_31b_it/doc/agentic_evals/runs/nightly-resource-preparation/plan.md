# Nightly resource admission preparation

Goal: complete admission facts for frozen SWE nightly5 and Terminal nightly10, preserving task-defined resources. Device-hour box 0; speed saving 0; validity unblocks both suites.

Existing setup/task_resources.json already inventories all 15 tasks and definition hashes. Reuse it rather than duplicate that evidence. Confirm source definitions and pinned stock resource/default/preflight semantics, calculate per-suite totals and simultaneous C2 demand, and compare only against actual assigned-runner disk reports when present. Report unknown runner headroom explicitly. No task solutions, source edits, hooks, dispatch, or device work.
