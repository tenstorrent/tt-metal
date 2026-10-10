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
  `58d68d4b872738fa9f3cccc3a68c9cb4e6d9dcf0`.
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

GPQA is accepted by the user as of Oct 9, 17:09 UTC. The final corrected run
finished at 17:30 UTC: **176/198 (88.89%)**, zero truncations, 50m49s.
See `galaxy-evidence/gpqa-first-final-v2/` for original receipts. This is explicit
user acceptance, not a retroactive claim that the original 177/198 gate passed;
immutable benchmark receipts and the running protocol remain unchanged.

**Oct 10 update:** the newer frozen compact-GDN source passed the strict gate
at 177/198 (89.39%) in 59m 53s, with five incorrect output-budget cutoffs and no
context cutoffs. Its completed-answer audit, eight-replica G0 and API checks
passed. See `galaxy-evidence/compact-qualified-v1/` for exact source hashes.
This source is newer than the frozen runtime/image listed above; those image
digests do not automatically contain or qualify the compact candidate. The
development branch also contains later default-off experiments. A release
must pin and test the exact selected source and rebuilt image.

Remaining gates: tool-calling/agentic acceptance, successful
container startup and inference, Galaxy catalogue/job integration, passing
target-device On-dispatch and nightly/release CI, stable-branch alignment and
release-team staging. The built BFP8 image passed runtime import and entrypoint
help checks. The separate startup probe initially exited 2 because it omitted
the required `--tt-device` wrapper argument. The corrected probe passed against
the actual immutable BFP8 image at 17:20 UTC, without opening hardware. Container
hardware/API checks are persistently queued; inference remains unqualified.
The first separate OpenBench run had eight client timeouts and requires a clean
rerun with explicit HTTP-client timeout and retry settings. This review did not
contact release engineering, change prod
catalogue entries, trigger shared CI, create release tags or publish a release.
