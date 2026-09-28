# QB2 workflow source compatibility audit

Status: reviewed, committed and normally pushed; **not dispatched or image-built**.
[Publication record](publication.md) contains exact refs and the vLLM access
blocker. No hardware was used by this compatibility work. The existing model/catalog
changes were preserved. No applicable AGENTS.md was present in the workflow
checkouts or inference paths touched by this change.

**Provenance correction, incorporated into the local build implementation:**
[runtime metadata](../../readiness_vllm/ttft_optimization/runtime_versions.json)
establishes installed engine `vllm==0.26.0+empty` from site-packages, not an
editable monorepo engine. Only the nested plugin is overlaid from `7f72…`.
The image tag's apparent 0.21 version was not engine-version evidence. The
initial monorepo-root installation plan was incorrect and has been removed.
Repository input routing now selects the nested plugin source independently
of the installed engine. This correction still requires independent review
and actual image validation before dispatch.

Read-only inspection of the current container's standalone installer confirms
the actual engine recipe is a **source-built empty-target distribution**, not a
proven TT-index wheel download: upstream v0.26.0 common requirements with
standalone dependency overrides, CPU torchvision0.26.0, then
`VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm vllm==0.26.0`.
Observed transformers5.12.1, torch2.11.0+cpu, tokenizers0.22.2, numpy1.26.4
are retained/verified, along with observed torchvision0.26.0+cpu and tblib3.2.2.
The image tag's `c9cfebc` resolves to standalone revision
`c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`. Its installer and override files
match the actual image byte-for-byte (SHA256):

- Installer: `9190fdd216ea96e95503b0caba88f40925c0c13819525fff67ef763c327731de`.
- Overrides: `5eff7fa73e2c5ec1b3eb0a776f279ab0bf4486aaee567a7fbdb00874f5f4c60c`.

Raw remote/image hash comparison:
[engine_installer_provenance.log](../../readiness_vllm/ttft_optimization/engine_installer_provenance.log).

## Checkouts and provenance

All three local branches are `mvasiljevic/gemma4-ttft-monorepo-compat`.

| Repository / local checkout | Original base SHA |
|---|---|
| tenstorrent/tt-agentic-bringup-qb2; `/home/mvasiljevic/gemma4-ttft-qb2-workflows` | `79921d592f17f3d780f99765de5dda9acd8b3666` |
| tenstorrent/tt-shield; `/home/mvasiljevic/gemma4-ttft-shield-workflows` | `ba2f03318608be52c5cb2085469598a89dcfa0fc` |
| tenstorrent/tt-inference-server; `/home/mvasiljevic/gemma4-ttft-inference-server` | `7896680ca8967884fbc9e6126ad7eb6a326c7785` |

Shield was cloned from `vvukoman/enable-on-dispatch-cross-repo-trigger`, the
original QB2 wrapper's reusable workflow ref. Inference already contained three
uncommitted catalog/launcher-test changes on `mvasiljevic/gemma4-ttft-bench`;
branch creation preserved them. Future commits and pushed SHAs must be recorded
separately; these base SHAs alone do not contain the modifications.

The tested **plugin source only** is the monorepo commit
`tenstorrent/vllm@7f72b1c6e905f5137fe3377f2e7b42738d3f271d`.
The old wrapper resolved its vllm input in `tenstorrent/vllm-tt-plugin`, so that
monorepo SHA could not be passed honestly. Standalone main inspected during the
audit was `35090660433d5606957ded97f7130b5cc75f94f7`: its installer selects vLLM
0.26.0 and its registry lacks this autoport architecture. Its engine version is
consistent with the observed installed engine; registry compatibility still
requires the tested nested plugin. The initial audit incorrectly inferred that
the monorepo engine was in use; runtime evidence supersedes that inference.
Installed EngineCore/UniProc source hashes also differ from the local monorepo.
Earlier local-source pipeline/GIL hypotheses therefore are not established for
the measured engine. No accepted production change relies on that assumed
equivalence; both host-yield experiments were rejected. Measured HTTP/observer
timings remain observations, not proof of those source-level causal hypotheses.

## Changed files and corrected contract

### Selected deployment scheduling profile

The independently selected `gemma4-autoport` catalog entry explicitly sets
`no-async-scheduling: true`, emitting `--no-async-scheduling` for the
latency-priority deployment. Canonical Gemma4 defaults remain unchanged.
The selected geometry's synchronous S128/O128 median TTFT was
95.241/95.357/95.028 ms across three cohorts. S128/O16 retained an initial
107.186 ms result followed by 97.385/95.592 ms; the initial result is not
discarded. This is a deployment tradeoff, not a universal speedup: C1 TPOT
increases approximately 6–9%, while C8/C32 E2EL increases approximately
0.85%/0.40% versus optimized async.

Async remains an explicit throughput-oriented option: the local serving tool
uses that path when `--no-async-scheduling` is omitted. A catalog override must
not be inferred from this statement: no additional inference-workflow CLI
override syntax is asserted here. The verified alternative is
`tools/ttft_server.py --output <owned-output-directory>`; the selected local
latency launch adds `--no-async-scheduling`. Full context (262144), 32 slots, selected precision, model
and tokenizer revisions, device sampling, and disabled prefix/chunked prefill
remain pinned as before. The focused launcher assertion now requires the
negative flag and rejects the positive flag. After the benchmark owner's
sampling-phase go-ahead, the relevant inference tests passed (114), shield
workflow contracts passed (6), and QB2 wrapper contract passed (1). New logs:
[inference](../../readiness_vllm/ttft_optimization/inference_latency_profile_tests.log),
[shield](../../readiness_vllm/ttft_optimization/shield_latency_profile_tests.log),
[QB2](../../readiness_vllm/ttft_optimization/qb2_latency_profile_tests.log).
Both inference build/install shell scripts pass `bash -n`; all three checkout
diffs pass `git diff --check`. These are CPU-only checks, not image-build or
remote serving validation.

QB2:

- `.github/workflows/manual-tt-shield-dispatch.yml`: optional `vllm-repository`
  choice, default `tenstorrent/vllm-tt-plugin`, alternative `tenstorrent/vllm`;
  forwards repository and ref to the owned shield branch.
- `tests/test_vllm_repository_contract.py`: wrapper forwarding/default contract.

Shield:

- `.github/workflows/on-dispatch.yml`: dispatch/call input and propagation.
- `.github/workflows/workflow_resolve-shas.yml`: allowlist validation and SHA
  resolution in the selected repository.
- `.github/workflows/workflow_build-inference-server.yml`: passes the **full**
  resolved SHA, forwards the repository for the non-default path, validates its
  allowlist, and reports repository/full SHA. Omitting the new CLI flag on the
  default path preserves compatibility with older standalone inference refs.
- `.github/scripts/test_vllm_repository_contract.py`: six CPU-only contracts.

Inference:

- `.gitignore`: narrow exception for the required compatibility manifest JSON.
- `scripts/build_single_docker.sh`: repository CLI/default/allowlist, resolution
  of non-full vLLM refs before builds, repository build argument, and distinct
  `monorepo-` image-tag component and separate installed-engine version argument.
- `vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile`: selected repository checkout,
  mandatory full SHA for monorepo, verification of checked-out HEAD against the
  requested commit, selected installer, complete source-tree runtime copy,
  plugin repository/revision labels, and a separate engine-version label.
- `scripts/install_vllm_bundled_plugin.sh` and
  `scripts/vllm_bundled_plugin_manifest.json`: hash-verified original engine
  recipe provenance, installed runtime version constraints/checks, and only
  the selected nested plugin installed editable with `--no-deps`.
- `tests/test_vllm_source_build.py`: eight contracts, including mocked Docker
  command forwarding and rejection of invalid repositories.
- `tests/test_run_vllm_api_server.py`: adapts the existing standalone checkout
  source-string assertion; retains the previously added autoport launch test.
- Existing `workflows/model_spec.py` and `workflows/model_specs/dev/llm.yaml`
  catalog edits remain unchanged by this compatibility task.

The monorepo is cloned into the existing `${HOME_DIR}/vllm-tt-plugin` source
root, but its root engine installer is **never sourced**. The owned helper
downloads/checks the pinned original installer and overrides; it preserves the
original common-requirements/CPU-torchvision recipe with additional constraints
to retain the measured runtime dependency versions, then runs:

```sh
VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm vllm==0.26.0
uv pip install --no-deps -e "$plugin_source_root/plugins/vllm-tt-plugin"
```

Only the nested plugin is editable. The engine lives in site-packages; the
helper asserts its location from a temporary working directory outside the
monorepo, avoiding accidental source shadowing. Copying the source root at the
same absolute path retains the nested plugin; copying the Python environment
retains the installed engine. The monorepo engine is not added to PYTHONPATH.
The manifest is retained at `/usr/local/share/tt-vllm-compat/manifest.json`.
Standalone still sources its own `docs/install-vllm-tt.sh` unchanged.

The selected monorepo build receives and verifies the full commit, so its image
revision label is not merely a short tag. Legacy direct standalone Dockerfile
callers can still supply their prior refs; the reviewed dispatch/build-script
path resolves those to full SHAs before build.

## Validation performed

Host Python: `/tmp/gemma4-ttft-catalog-tests-venv/bin/python`.

```sh
# inference checkout
python -m pytest -q tests/test_vllm_source_build.py tests/test_run_vllm_api_server.py tests/test_model_catalog_yaml.py tests/test_quetzal_image_build.py
# shield checkout
python -m pytest -q .github/scripts/test_vllm_repository_contract.py
# QB2 checkout
python -m pytest -q tests/test_vllm_repository_contract.py
```

Results: [inference 114 passed](../../readiness_vllm/ttft_optimization/inference_compat_host_tests.log),
[shield 6 passed](../../readiness_vllm/ttft_optimization/shield_compat_host_tests.log),
[QB2 1 passed](../../readiness_vllm/ttft_optimization/qb2_compat_host_tests.log).
`bash -n scripts/build_single_docker.sh` and whitespace checks pass. These tests
mock Docker; no daemon/image build, GitHub dispatch, or model execution occurs.

## Required eventual dispatch inputs and remaining risks

Use model `gemma-4-26B-A4B-it`, implementation `gemma4-autoport`, device
`p300x2`, workflow `benchmarks`, empty docker image, repository
`tenstorrent/vllm`, and **bundled-plugin source ref**
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d`. Metal and inference refs must be
reviewed, pushed **full SHAs containing the actual changes**, not the base
commits or moving branch tips. The inference ref is independently checked out
for model resolution/build/testing, so pinning it avoids branch movement drift.

The owned shield branch must exist remotely before GitHub can validate the
wrapper's reusable-workflow reference. This document does not authorize or
claim that push/dispatch has occurred. A built image still needs import/runtime
checks and native benchmark validation. Static/mocked tests do not establish
dependency resolution, registry credentials, runner availability, image pull
permissions, or hardware compatibility. Default standalone installation remains
available. Engine source-build reproducibility is not wheel-bit-identity: the
source version and measured runtime versions are pinned, but transitive build
tools and common requirements still need validation in the resulting image.
