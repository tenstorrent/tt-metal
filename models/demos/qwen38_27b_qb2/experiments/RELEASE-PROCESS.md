# Qwen release readiness against Models CI process

Reviewed the user-provided `PSE-Models CI Release Process-091026-162524 (1).pdf`
on Oct 9 2026. This is a readiness mapping, not an approved or published release.

The documented cycle is two weeks: announce/request inclusion and obtain model
engineering commitment during the preceding week; Week 1 triages nightlies and
cuts coordinated stable branches; Week 2 stabilizes release CI and publishes
approved models. Release engineering owns staging and automated promotion.

Prerequisites are TTIS integration with complete parameters in the **dev** model
catalogue, defined nightly/release jobs in `models-ci-config.json`, eval and
benchmark tests, at least one passing target-device On-dispatch in `tt-shield`,
and integration early enough to obtain a nightly before the cut-off. On-dispatch,
nightly and release jobs are triggered from `tt-shield`.

Only edit `workflows/model_specs/dev`; automation promotes approved entries to
prod and generates release documentation. Release fixes must reach both main
and the current stable branch. The release stack coordinates tt-metal, the
Tenstorrent vLLM fork and tt-inference-server; this implementation also depends
on an explicitly pinned vLLM plugin revision. Consult release engineering for
the active cycle and stable refs before preparing promotion PRs.

## Our candidate

- Model/experiments: `tenstorrent/tt-metal`, branch
  `anatarajan/qwen38-long-context-throughput-20261007`.
- Frozen BFP8 runtime: same repository, branch
  `anatarajan/qwen38-bfp8-control-runtime-20261009`, commit
  `20619e008a236aaf393937b222a60a5b03e49cdc`.
- TTIS packaging: `tenstorrent/tt-inference-server`, branch
  `anatarajan/qwen38-galaxy-release-20261009`, commit
  `bbb1ca07e7cd0f97af47745a1686545a16089b9c`.
- Plugin used by the evaluated runtime:
  `b7e4292e4193cba20abe9c7c68ce489201b2e36b`.
- Runtime Metal: `a08819ddbe23077f8037d3802303939064868ff6`.
- Checkpoint: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Built BFP8 image manifest:
  `sha256:79f7b4469a6ec2bcce5204399b37b2aeced8be7f260dd98f7007ad41f8813055`.

Existing P300X2 or Quetzal catalogue entries are not evidence of this eight-TP4
Galaxy candidate's release qualification. Source branches and a built image
alone do not satisfy the process. Pin release artifacts by commit/digest as
requested; do not deploy moving branch names or `latest` tags.

GPQA is accepted by the user as of Oct 9, 17:09 UTC. At the acceptance
check, 194/198 responses were complete with 174 correct and zero truncations.
The run continues to preserve the final 198-question result. This is explicit
user acceptance, not a retroactive claim that the original 177/198 gate passed;
immutable benchmark receipts and the running protocol remain unchanged.

Remaining gates: tool-calling/agentic acceptance, successful
container startup and inference, Galaxy catalogue/job integration, passing
target-device On-dispatch and nightly/release CI, stable-branch alignment and
release-team staging. The built BFP8 image passed runtime import and entrypoint
help checks, but its separate startup probe exited 2 because the probe omitted
the required `--tt-device` wrapper argument. This is a probe invocation failure,
not evidence of a broken image. Fix the probe, verify the packaged handoff, then
run container hardware/API checks; container inference remains unqualified. This review did not contact release engineering, change prod
catalogue entries, trigger shared CI, create release tags or publish a release.
