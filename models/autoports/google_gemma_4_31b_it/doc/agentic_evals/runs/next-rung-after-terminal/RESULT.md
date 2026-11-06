# Next-rung patches ready

Two independent source-only patches apply to TTI2f69c37e593d7a9b5c91af0f66ebc22e6632c338:

- gpqa198.patch: full fixed indices0..197, C2/B8, SWE0/Terminal0.
- swe5.patch: unchanged frozen SWE nightly5, C2/B8, GPQA0/Terminal0.

Each resulting tree is byte-identical across all1821 tracked files to its corresponding already-tested candidate in `current-observability-candidates`. Fresh stock resolution also matches prior resolved configuration exactly except the recorded base commit. Those candidate trees passed61 focused checks and full hooks. Per coordinator instruction, no unchanged tests/hooks were rerun. git diff --check passes for both patches.

All generic lifecycle/activation repairs, release time/output caps, cache paths, model adapter/HF/Harbor/plugin pins and periodic unique phase export are preserved. Comparison provenance, patch hashes and resolved configs are adjacent. No original cohort selection or prior evidence was changed.

Apply exactly one with `git apply <absolute patch path>` from the stated base. Branches `prep/after-terminal-gpqa198` and `prep/after-terminal-swe5` in isolated worktrees `/home/mvasiljevic/gemma4-31b-pipeline-v3/tti-after-terminal-<candidate>` are uncommitted. No publication checkout edits, commit, push, dispatch or devices. Preparation/device cost0. Live rung eligibility remains coordinator decision. Stock dispatch routes: GPQA evals, SWE agentic; pin TT439c22b0e991e5af626efafa7224edf28f84795b and vLLM plugin ca7a18246e4111629ed6cb4a7d1192002749fccf explicitly.
