# Refreshed candidates confirmed

All three source-only patches independently apply to TTI2d738ec3910f7f13d1614e6196d9ff5f646b6fe1. Each isolated worktree `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-observable-<candidate>` remains uncommitted on `prep/observable-<candidate>`. No active publication edits, push, dispatch or device work.

GPQA198 resolves unique indices0..197 on stock common evals route, num_concurrent2, request14400s/output126976, pinned revision and max_length262080 (advertised262144 minus stock64 reserve). Original nightly10 cohort evidence is unchanged. SWE5 and Terminal10 resolve exactly their pre-frozen nightly lists from cohorts.json on stock agentic route, C2, agent21600s/request3600s/output32768, without task CPU/memory overrides. Each candidate disables both peers and preserves B8/maxseq8, HF/tokenizer/Harbor pins, native parsers, persistent cache paths, agent activation repair and model unique phase-log/one-second autoexport settings.

61 focused tests pass for each candidate, including stock selections, model config, references, Harbor revision pin, and generic activation tests. All three full hook runs pass ruff/ruff-format/pytest/copyright. git diff --check passes. Only model eval configuration and model-specific expectations change; no generic source/test harness edits.

- `gpqa198.patch`: SHA256 `59684ca42d92654f2cfd5fe4822ccaf3cc8646f4d3db7f8058546721f762f585`; `gpqa198-resolved.json`; hooks `bringup/artifacts/stage12-observable-gpqa198-hooks.log`.
- `swe5.patch`: SHA256 `7db565e5a94bb331b4e95228e04e577eb8347a1a87fb31131ec642f45eca5c53`; `swe5-resolved.json`; hooks `bringup/artifacts/stage12-observable-swe5-hooks.log`.
- `terminal10.patch`: SHA256 `3f5cb98f47e25a13a2bb99f9254b7cf87797c34d59d02cff65d753d5cc49417e`; `terminal10-resolved.json`; hooks `bringup/artifacts/stage12-observable-terminal10-hooks.log`.

Apply exactly one patch with `git apply <absolute candidate.patch>` from the common base. These replace the older stale preparations. Choose stock workflow `evals` for GPQA and `agentic` for SWE/Terminal; CI mode is ci-nightly. Dispatch pins remain TT439c22b0e991e5af626efafa7224edf28f84795b and vLLM plugin ca7a18246e4111629ed6cb4a7d1192002749fccf explicitly. Live rung eligibility, task-source lock verification, measured projection and admission remain coordinator decisions; preparation grants no dispatch. Host Docker absence caused fallback endpoint warning in SWE resolution, not a claim of live reachability. Actual/predicted device hours0; no speed claim.
