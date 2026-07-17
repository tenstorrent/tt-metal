# `.github/scripts/` — CI Script Reference

This directory contains scripts used by GitHub Actions workflows for Docker image builds, CI data collection, test infrastructure, and developer utilities.

## Docker Build & Image Management

| Script | Description |
|--------|-------------|
| `dockerfile-hash.sh` | Computes a content-addressed hash for a Dockerfile and all its `COPY` source files (including transitive Dockerfile dependencies). Used for Docker layer cache invalidation. |
| `compute-platform-data.sh` | Computes platform-specific (Ubuntu version) Docker image tags and checks whether they already exist in the registry. Outputs JSON with tags, existence flags, venv required flags, and metadata for `ci-build-light`, `ci-build`, `ci-test-light`, `ci-test`, `dev-light`, `dev`, basic images, manylinux, and Python venvs. Final image tags include canonical tool tag hash material and any consumed venv hash; `dev` also includes the preceding main image hashes so it can serve as the first registry canary. Venv manifests are inspected only when a missing final image requires them. Usage: `compute-platform-data.sh <version> <repo> [--force-rebuild] [--check-exists]` |
| `compute-tool-data.sh` | Computes tool image tags and checks registry existence for all tools. Outputs JSON with per-tool existence flags. Usage: `compute-tool-data.sh <repo> [--force-rebuild] [--check-exists]` |
| `compute-tool-tags.sh` | Single source of truth for content-addressed tool image tags (ccache, mold, doxygen, gdb, cmake, etc.). Extracts versions from `Dockerfile.tools`, hashes each tool's install script, and outputs canonical `ghcr.io/...` tags as JSON. Usage: `compute-tool-tags.sh [REPOSITORY]` |
| `get-target-tools.sh` | Lists tool names required by a given Docker Bake target or group. Parses `docker-bake.hcl` context keys. Usage: `get-target-tools.sh <bake-target-or-group>` (e.g., `ci-build`, `dev-light`, `basic-dev`, `tools`) |
| `validate-docker-bake-ci.py` | Validates CI-facing Docker Bake wiring (output settings, tag formats, target-specific venv contexts, Harbor prefixes) without building any images. Run as a pre-merge check. |
| `llk-build-docker-images.sh` | Builds Docker images for the LLK (Low-Level Kernel) test infrastructure. Patches LLK Dockerfiles to use the Metal repo's base image registry and builds both base and CI images. |
| `llk-get-docker-tag.sh` | Computes a content-addressed tag for LLK Docker images by hashing all relevant Dockerfiles, requirements, and install scripts. |

## Infrastructure & Installation

| Script | Description |
|--------|-------------|
| `install-slurm.sh` | Builds and installs Slurm headers + `libslurm` from source (v25.11.4). Provides development files for OpenMPI's `--with-slurm` configure option. No daemons or plugins are installed. |

## Cloud Utilities (`cloud_utils/`)

| Script | Description |
|--------|-------------|
| `cloud_utils/mount_weka.sh` | Mounts the WekaFS filesystem at `/mnt/MLPerf` and ensures hugepages are allocated. Handles both the systemd service and legacy `rc.local` paths. |

## Data Analysis & Reporting (`data_analysis/`)

Scripts that collect CI/CD metrics and benchmark data for upload to the analytics database.

| Script | Description |
|--------|-------------|
| `data_analysis/create_pipeline_json.py` | Generates a CICD pipeline JSON artifact from GitHub Actions workflow and job data. |
| `data_analysis/create_benchmark_with_environment_json.py` | Enriches partial benchmark data files with GitHub runner environment metadata. |
| `data_analysis/create_dummy_partial_benchmark_json.py` | Creates a synthetic benchmark pickle file for testing the data-collection pipeline. |
| `data_analysis/create_job_failure_cluster_json.py` | Converts job failure cluster data (from the `slack-output-analysis` action) into pydantic models and saves as JSON for database upload. |

## CI Health & Reporting

| Script | Description |
|--------|-------------|
| `ci_digest.py` | One named CI digest. For each watched workflow, reads the latest completed scheduled run's `ai_run_summary_<run_id>` JSON artifact (succeeded / failed / infra_failure, produced by the `ai_summary/run` action) and renders a consolidated report — per-workflow health bar, counts, and a collapsible failed-jobs table — to the step summary + an artifact; falls back to the run conclusion when no JSON is present. Stateless; scheduling/gating lives in the workflow (`github.event.schedule`). Driven by `.github/workflows/ci-digest.yaml`. Usage: `ci_digest.py --name <digest> --workflows <foo.yaml …>`. Tests: `ci_digest.py --self-test`. |

## Test & CI Utilities (`utils/`)

| Script | Description |
|--------|-------------|
| `utils/find-changed-files.sh` | Detects which files changed between `origin/main` and `HEAD`, then sets boolean flags for affected areas (CMake, tt-metalium, ttnn, models, docs, LLK, etc.). Used to conditionally gate CI jobs. |
| `utils/prepare_test_matrix.py` | Builds a filtered GitHub Actions test matrix from a YAML test definition file and enabled SKUs. Expands each `(test, sku)` into a matrix row with `runs_on` from `.github/sku_config.yaml`. Supports `ALL_SKUS_IN_TESTS`, optional `--sku-allowlist` change gating, and `--event merge_group` prio routing via `merge_queue_sku`. See [Prepare test matrix](#prepare-test-matrix) below. |
| `utils/count_pytests.py` | Counts total pytest cases in a directory, including `@pytest.mark.parametrize` expansions, by parsing Python ASTs. |
| `utils/verify_time_budget.py` | Validates that the sum of test timeouts per (team, SKU) pair stays within the time budget defined in the budget YAML. Optionally enforces `--max-per-test-timeout` (minutes) against each individual SKU timeout. |
| `utils/validate_perf_targets.py` | Validates model performance benchmark results against targets defined in `models/model_targets.yaml`. Exits non-zero if any metric regresses beyond tolerance. |
| `utils/validate_golden_csv_columns.sh` | Validates that golden bandwidth CSV files in the perf microbenchmark directory have the expected column headers. |
| `utils/hang_report.py` | Generates a JUnit XML test report for a hung test (dispatch timeout). Ensures the hung test appears in CI artifacts even if the process is killed before pytest finalizes. Supports a two-phase flow: initial report, then update with triage summary. |
| `utils/model-charts-sync.py` | Validates that featured models in `tt_metal/README.md` are consistent with the full model list in `tt_metal/models/README.md`. |
| `utils/extract_allocation_from_sku.py` | Extracts runner allocation configuration for a given SKU name from `.github/sku_config.yaml`. |
| `utils/generate_tray_mapping.py` | Generates tray-to-PCIe device mapping and TP2 (Tensor Parallel 2) Ethernet-connected device pairs for Galaxy/UBB multi-chip systems. |
| `utils/multi-user-create-files.py` | Generates `docker-compose.yml` and related files for multi-user container setups on Galaxy/UBB systems, assigning Tenstorrent devices to containers. |
| `utils/multi-user-configure-container.sh` | Entrypoint script for multi-user containers — installs Python dev requirements, installs ttnn wheel, and sleeps to keep the container alive. |

### Prepare test matrix

Gate pipelines (merge-gate / PR-gate) treat test-list SKU keys as **logical** coverage. Concrete merge-queue prio runners are resolved at matrix build time—not listed as twin keys in `tests/pipeline_reorg/*gate*` YAMLs.

```text
prepare_test_matrix.py <tests_yaml> <enabled_skus> <sku_config_yaml>
    [--event EVENT] [--sku-allowlist LIST]
```

| Input | Behavior |
|-------|----------|
| `enabled_skus` | Comma-separated SKUs, or `ALL_SKUS_IN_TESTS` to take every key under any test's `skus:` map. |
| `--event` | When `merge_group`, rewrite each logical SKU that defines `merge_queue_sku` in `.github/sku_config.yaml` to that concrete prio SKU (sets `logical_sku` on the row). Other events leave the logical key unchanged. |
| `--sku-allowlist` | Omit for no extra filter. Empty string → skip all (`matrix=[]`, exit 0). Otherwise CSV of logical SKUs intersected with coverage (used for LLK change gating). |

Gate workflows typically pass `enabled-skus: ALL_SKUS_IN_TESTS` so the test list owns device coverage. LLK jobs also pass `sku-allowlist` (`*` in the workflow means “omit flag / no filter”; empty skips; else CSV)—see `llk-unit-tests-impl.yaml` / `llk-smoke-impl.yaml`.

Do not list `*_prio` SKUs in gate test YAMLs; those entries live only under the merge-queue section of `sku_config.yaml` and are selected via `merge_queue_sku`.

Examples:

```bash
# All SKUs in the list, rewrite to prio runners in the merge queue
python3 .github/scripts/utils/prepare_test_matrix.py \
  tests/pipeline_reorg/runtime_validation_merge_gate_tests.yaml \
  ALL_SKUS_IN_TESTS .github/sku_config.yaml --event merge_group

# LLK wormhole-only change gating
python3 .github/scripts/utils/prepare_test_matrix.py \
  tests/pipeline_reorg/llk_merge_gate_tests.yaml \
  ALL_SKUS_IN_TESTS .github/sku_config.yaml \
  --event pull_request --sku-allowlist wh_n150_civ2
```
