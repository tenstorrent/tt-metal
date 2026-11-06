# Stock dispatch correction

Minimum repair: add -f impl-of-model=gemma4-31b-autoport-v3 to the Terminal workflow dispatch, retaining TTIa087be086b6e2cd6edf215dddb4c47c48463f13a, TTmetal08ee8790bcbcd9e85cb799593b2bbaf40b785134 and pluginca7a18246e4111629ed6cb4a7d1192002749fccf. No shared runtime or catalog-default modification. No optional cache or Harbor pin changes bundled.

Host confirmation passed against exact dispatched catalog: explicit selection yields correct model ID and adapter overrides. Existing live SWE independently proves stock explicit forwarding reaches this autoport. No local device run is needed to distinguish this input omission. Coordinator owns corrected dispatch and confirmation from its live emitted server args/registration; result pending.

No source edits, commit, push, dispatch, cancellation or device action performed by this investigator. Preserve failed run as invalid/no-score evidence and verify explicit impl on every subsequent dispatch.

Coordinator also verified GPQA nightly 42176391791 original dispatch in `bringup/artifacts/multigoal-runs/v3-20261105T172350Z-stage12/12-12-agentic-evals.resume-1.jsonl`: the command pins TTI `a0c8bde` and the plugin SHA but contains no `impl-of-model` argument. Disqualify that arm before any score, and replace with explicit implementation on the same immutable SHAs. No cancellation or device action by this investigator; hardware ownership is unknown. SWE explicit implementation and actual autoport remain confirmed by the coordinator.
