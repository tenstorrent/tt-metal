# Multihost (exabox) model unit-test CI — progress log

**Branch:** `jameslee/models-tiered-ci-exabox-jobs`
**Reusable workflow:** `.github/workflows/models-unit-tests-impl.yaml`
**Dispatcher used for testing:** `.github/workflows/models-t1-unit-tests.yaml`
(`(Tier 1) Models Unit Tests`), dispatched with `model=all`, `sku="bh_spsg (BH Single Galaxy)"`.

## Goal

Add a job to `models-unit-tests-impl.yaml` that runs model unit tests across the exabox
multihost infrastructure, modeled on the `multihost-tests` job in
`demo-sp-multihost-impl.yaml`. Selection is **by SKU**: a test leg routes to the new job when its
SKU carries an `allocation` block in `.github/sku_config.yaml` (i.e. `runs_on:
exabox-multihost-with-nfs`). As a live example, a `flux-multihost-tests` entry runs the Flux.1-dev
DiT unit test on the single-galaxy `bh_spsg` SKU.

## ✅ Known-good commit: `3b23e6ebfc8`

Commit **`3b23e6ebfc8`** ("Use repo-relative mesh_graph_desc_path in Flux rank binding") is the
proven-working state. It was confirmed by two runs (the intervening two failures were transient
exabox infra, not the change):

- Run `29458588017` — pytest collected and ran the model on the galaxy mesh.
- Run `29521281774` — **full pipeline succeeded end-to-end** and the Flux pytest ran to completion:
  `2 failed, 2 skipped, 38 deselected`. The 2 "traced" cases skip by design in CI; the 2
  "not_traced" cases reached weight loading and failed only on **HuggingFace gating** —
  `GatedRepoError: 401 … black-forest-labs/FLUX.1-dev … access restricted`. That is a
  model-access/auth matter (no `HF_TOKEN`/cached weights in the multihost worker env), **not** a
  defect in the multihost CI wiring.

Infra-only failures seen after (unrelated to the change): `29459561439` (`curl (6)` DNS to the
GitHub OIDC endpoint) and `29518601121` (HelmRelease not `Ready` within 5m).

### Simplification: plain `mpirun` instead of `tt-run` — `fb666d4f5e1` (validated)

For a single rank, `tt-run` only translated the rank binding into four env vars and built the
`mpirun` command. Commit **`fb666d4f5e1`** replaces it with a direct `mpirun -np 1` that exports
`TT_MESH_ID` / `TT_MESH_HOST_RANK` / `TT_MESH_GRAPH_DESC_PATH` / `TT_VISIBLE_DEVICES` itself (host
selection, TCP interface, and env-forwarding all come from the job `env:`), and drops the
`rank_bindings_mapping.yaml` / `rank0_binding.yaml` (only `mesh_graph_descriptor.textproto`
remains). Run `29524718983` confirmed it reaches the **identical** result as the `tt-run` path
(`2 failed, 2 skipped, 38 deselected`, same HF-gating stop), with `python3` correctly resolving to
the worker venv via env-forwarding — no `bash -lc` needed.

## What is proven working

The **CI infrastructure** is fully validated on hardware. The `models-unit-tests-multihost` job
executes the complete exabox lifecycle end-to-end, repeatably:
allocation → get configs → create environment → download + `mpirun --pernode` install →
cluster reset/validation → ttnn-import & `/mnt/models` checks → **test run** → delete environment
→ delete allocation. SKU-based routing also works: the single-host container job is skipped and
the multihost leg is picked up by the new job.

The remaining work has been about the **Flux payload itself** running in the exabox environment;
each attempt peeled back one layer (permissions → k8s naming → interpreter/venv → shell quoting →
launcher model → mesh descriptor → path resolution).

## How it works

- `.github/scripts/utils/prepare_test_matrix.py` tags each matrix leg with `multihost: true` when
  its SKU has an `allocation` block.
- `models-unit-tests-impl.yaml` `load-test-matrix` splits the filtered matrix into `matrix`
  (single-host container job) and `matrix_multihost` (new exabox job).
- The `models-unit-tests-multihost` job provisions the cluster via the `ttop-*` composite actions
  and runs the leg's `cmd` via `tt-run` (`ttnn/ttnn/distributed/ttrun.py`).
- The Flux leg ships **static single-rank bindings** under
  `models/tt_dit/tests/models/flux1/scaleout_configs/single_galaxy/` and launches with
  `--rank-bindings-mapping` (avoiding runtime Phase-1 discovery, which writes to worker-local
  storage the launcher can't see).

## Commit → workflow-run mapping

Each dispatch was `models-t1-unit-tests.yaml` on the branch with
`model=all`, `sku="bh_spsg (BH Single Galaxy)"`. Runs are at
`https://github.com/tenstorrent/tt-metal/actions/runs/<run-id>`.

| # | Commit | Change | Run ID | Result / lesson |
|---|--------|--------|--------|-----------------|
| 1 | `97aaa080e55` | Add multihost job, SKU-based matrix split, Flux `bh_spsg` entry, budget, t1 dispatcher SKUs | `29439845427` | **startup_failure** — reusable workflow requested `id-token: write` the callers didn't grant |
| 2 | `19527ce8106` | Grant `id-token: write` to impl callers (t1/t2/t3/all-model-tests) | `29440103955` | Reached multihost job; **failed at Create allocation** — underscore in `bh_spsg` is an invalid k8s (RFC 1123) name |
| 3 | `b60568977116` | Derive k8s-safe allocation name (`_`→`-`) | `29441012958` | **Full lifecycle passed**; Flux test failed: `pytest: command not found` (bare `pytest` not on PATH) |
| 4 | `d41cee970f4` | Use `python3 -m pytest` | `29442505792` | Failed: `/usr/bin/python3: No module named pytest` (system interpreter, not the venv) |
| 5 | `2883afde56b` | Install pytest into venv + use venv interpreter in cmd | `29448536322` | **Canceled** (before result) |
| 6 | `e5129c516d0` | Add `pytest pytest-timeout` to the `mpirun --pernode uv pip install` step; revert Flux cmd | `29451092233` | **load-test-matrix failed** — apostrophe in a cmd comment (`job's`) broke `MATRIX='…'` shell quoting |
| 7 | `2eaa69f21e6` | Pass `MATRIX`/`TIER`/`MODEL` via `env:` (robust to quotes) | `29451912236` | Reached multihost job; Flux failed again: `No module named pytest` (`/usr/bin/python3`) |
| 8 | `7a5e02c7d91` | Run Flux pytest via `tt-run` (legacy `--rank-binding`) | `29453553869` | Failed: `tt-run` config validation — `/etc/ttop/mgd.textproto` not found (`bh_spsg` is a Count allocation, no MGD) |
| 9 | `e3931d397ed` | `tt-run` new mode: `--mesh-graph-descriptor` + `--hosts` (Phase-1 discovery) | `29455850020` | Phase-1 ran & wrote bindings, but launcher failed: **"Phase 1 output not found"** — bindings written to worker-local FS the separate launcher can't see |
| 10 | `703f85cfbbd` | Static single-rank bindings + `--rank-bindings-mapping` (no discovery); narrow Flux to `2x4` | `29458588017` | **pytest ran** — model imported, `38 deselected, 4 errors`; mesh open hit `TT_FATAL exists(mesh_graph_desc_path)` (bare path not resolvable on worker) |
| 11 | `3b23e6ebfc8` | Repo-relative `mesh_graph_desc_path` in the rank binding | `29459561439` | **In flight when session paused** — tests whether the `2x4` case now opens the mesh and runs the model |

## Files changed

- `.github/scripts/utils/prepare_test_matrix.py` — emit `multihost` marker from SKU `allocation`.
- `.github/workflows/models-unit-tests-impl.yaml` — `permissions`, matrix split, new
  `models-unit-tests-multihost` job, pernode pytest install, env-based `apply-filters`.
- `.github/workflows/models-t1-unit-tests.yaml` — exabox SKUs in `ALL_SKUS` + dispatch choices;
  `id-token: write`.
- `.github/workflows/models-t2-unit-tests.yaml`, `models-t3-unit-tests.yaml`,
  `all-model-tests.yaml` — `id-token: write`.
- `.github/time_budget.yaml` — `models → unit_tier1 → bh_spsg` budget.
- `tests/pipeline_reorg/models_unit_tests.yaml` — live `flux-multihost-tests` entry (`bh_spsg`).
- `models/tt_dit/tests/models/flux1/scaleout_configs/single_galaxy/` —
  `mesh_graph_descriptor.textproto`, `rank0_binding.yaml`, `rank_bindings_mapping.yaml`.

## Open item / next steps

- Confirm the result of run `29459561439` (`gh run view 29459561439`). If the `2x4` Flux case
  opens the mesh, the remaining outcome is a genuine model/PCC result rather than a wiring issue.
- **For merge:** the Flux entry is a live experiment on scarce `bh_spsg` hardware. If it is not
  passing, consider reverting the Flux entry in `models_unit_tests.yaml` to a commented template so
  the branch merges with the validated multihost infrastructure and no failing scheduled leg — the
  `models-unit-tests-multihost` job does not depend on the Flux entry.
