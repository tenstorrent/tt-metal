# Slurm CI Infrastructure

This directory contains Slurm sbatch scripts that mirror the GitHub Actions CI/CD workflows defined in `.github/`. The migration moves CI execution from GitHub-hosted runners to on-premise Slurm-managed hardware, enabling direct access to Tenstorrent accelerators and shared Weka storage.

## Directory Structure

```
slurm/
├── submit.sh                    # Universal launcher - entry point for all workflows
├── README.md                    # This file
├── PLAN.md                      # Pointer to the full migration plan
├── config/
│   ├── site.sh                  # Site-specific storage paths (edit for new clusters)
│   ├── env.sh                   # Shared environment variables (registries, paths)
│   ├── partitions.conf          # Slurm partition definitions
│   ├── sku_map.sh               # SKU name -> partition/constraint mapping
│   ├── sku_config.yaml          # Hardware SKU configuration
│   ├── time_budget.yaml         # Job time budget definitions
│   ├── time_budgets.sh          # Time budget shell helpers
│   └── upstream-tests.json      # Upstream test matrix definition
├── lib/
│   ├── common.sh                # Core library: logging, git context, pipeline ID
│   ├── artifacts.sh             # Weka-backed artifact staging/fetching
│   ├── cleanup.sh               # Job epilogue: report staging, cleanup, notifications
│   ├── docker.sh                # Docker container execution helpers
│   ├── matrix.sh                # Job array / matrix strategy support
│   ├── notify.sh                # Slack notification helpers
│   ├── retry.sh                 # Retry-with-backoff command wrapper
│   ├── setup_job.sh             # Job prologue: workspace prep, artifact fetch
│   └── ttop.sh                  # TTOP allocation helpers for multi-node topologies
├── workflows/
│   ├── _helpers/
│   │   ├── resolve_docker_image.sh   # Docker image resolution logic
│   │   └── submit_dependent.sh       # Dependency-aware job submission
│   ├── build-artifact.sh             # Central build job
│   ├── build-docker-artifact.sh      # Docker image build (job array)
│   ├── package-and-release.sh        # Release artifact packaging
│   ├── publish-release-image.sh      # Push release image to GHCR
│   ├── release-build-test-publish.sh # Orchestrator: full release pipeline
│   ├── release-cleanup.sh            # Release artifact cleanup
│   ├── release-demo-tests.sh         # Release validation tests (array)
│   ├── release-verify-tag.sh         # Git tag verification
│   ├── docs-latest-public.sh         # Documentation build and publish
│   ├── tt-metal-l2-nightly.sh        # Orchestrator: nightly test pipeline
│   ├── apc-nightly-debug.sh          # Nightly APC debug tests
│   ├── vllm-nightly-tests.sh         # vLLM nightly tests
│   ├── ttnn-sweeps-slack-notify.sh   # Sweep results notification
│   ├── profiler-ci-selector.sh       # Orchestrator: profiler job selection
│   ├── profiler-tests.sh             # Profiler unit/integration tests
│   ├── run-profiler-regression.sh    # Tracy profiler regression
│   ├── unit-tests-infra.sh           # Infrastructure unit tests
│   ├── produce-data.sh               # CI data production for analytics
│   ├── test-wheels.sh                # Wheel install and smoke tests
│   ├── auto-retry-nightly.sh         # Failed nightly job resubmission
│   ├── auto-retry-post-commit.sh     # Failed post-commit job resubmission
│   ├── test-dispatch.sh              # Slurm dispatch validation
│   ├── test-installing-step.sh       # Install validation on hardware
│   ├── test-llk-metal-integration.sh # LLK integration tests
│   ├── temp-ccache-test.sh           # Ccache hit rate validation
│   ├── upstream-tests.sh             # Upstream integration tests (array)
│   ├── umd-unit-tests.sh             # UMD unit tests
│   ├── ttsim.sh                      # TT simulator tests
│   ├── test-calculate-version.sh     # Version calculation validation
│   ├── pipeline-select.sh            # Orchestrator: pipeline selection
│   ├── pipeline-select-galaxy.sh     # Orchestrator: Galaxy pipeline selection
│   └── pipeline-select-t3k.sh        # Orchestrator: T3K pipeline selection
└── crontab/
    └── schedules.crontab             # Scheduled job definitions for scrontab
```

## Site Configuration

All storage and mount paths are centralized in `config/site.sh` — the single file operators need to edit when deploying on a different cluster or storage back-end.

| Variable | Purpose | Default |
|----------|---------|---------|
| `CI_STORAGE_BASE` | Root of CI shared storage | `$(pwd)/.slurm-ci` |
| `MLPERF_BASE` | MLPerf data / models path | `/mnt/MLPerf` |
| `CONTAINER_WORKDIR` | Working directory inside containers | `/work` |
| `TT_DEVICE_PATH` | Host path to Tenstorrent device nodes | `/dev/tenstorrent` |
| `HUGEPAGES_PATH` | Host hugepages mount path | `/dev/hugepages-1G` |

All variables can be overridden via the environment before sourcing the config:

```bash
# Example: deploy on a cluster with dedicated CI storage
export CI_STORAGE_BASE="/mnt/nfs/ci"
export MLPERF_BASE="/mnt/nfs/mlperf"
./slurm/submit.sh all-post-commit-workflows
```

Derived paths (`LOG_BASE`, `ARTIFACT_BASE`, etc.) are computed automatically from `CI_STORAGE_BASE` — see `config/site.sh` for the full list.

## Quick Start

### Submit a Workflow

```bash
# Run the full post-commit pipeline
./slurm/submit.sh all-post-commit-workflows

# Run a specific test suite
./slurm/submit.sh fast-dispatch-frequent-tests --arch wormhole_b0

# Build only
./slurm/submit.sh build-artifact

# Dry run (prints what would be submitted)
./slurm/submit.sh tt-metal-l2-nightly --dry-run
```

### Common Options

| Option | Description |
|--------|-------------|
| `--pipeline-id ID` | Override auto-generated pipeline ID |
| `--docker-image IMG` | Docker image to use |
| `--arch ARCH` | Architecture (`wormhole_b0`, `blackhole`) |
| `--ref REF` | Git ref to build/test |
| `--dry-run` | Print submission command without executing |

## Checking Job Status

```bash
# List your running/pending jobs
squeue -u $(whoami)

# Detailed job info
squeue -j <JOBID> -l

# Job history (last 24 hours)
sacct --starttime=$(date -u -d '24 hours ago' '+%Y-%m-%dT%H:%M') \
      --format=JobID,JobName%40,Partition,State,ExitCode,Elapsed

# Specific pipeline jobs (grep by pipeline ID)
sacct --format=JobID,JobName%50,State,ExitCode,Elapsed | grep <PIPELINE_ID>
```

## Viewing Logs

Logs are written to shared storage at `${LOG_BASE}` (default: `$(pwd)/.slurm-ci/logs/`, configured in `config/site.sh`):

```
${LOG_BASE}/<job-name>/<job-id>/<array-task-id>.log   # stdout
${LOG_BASE}/<job-name>/<job-id>/<array-task-id>.err   # stderr
```

```bash
# Tail a running job's output
tail -f ${LOG_BASE}/build-artifact/<JOBID>/0.log

# View a specific array task
cat ${LOG_BASE}/upstream-tests/<JOBID>/2.log
```

## Managing Artifacts

Artifacts are stored on shared storage at `${ARTIFACT_BASE}` (derived from `CI_STORAGE_BASE` in `config/site.sh`), organized by pipeline ID:

```
${ARTIFACT_BASE}/<pipeline-id>/
├── build/                  # Build tarballs and wheels
│   ├── ttm_any.tar.zst
│   └── tt_metal_wheels/
├── docker/                 # Docker image tags
│   └── image_tags.env
├── reports/                # Test reports per job
│   ├── <job-name>/
│   └── ...
├── data/                   # Analytics data
└── pipeline.env            # Pipeline metadata
```

```bash
# List artifacts for a pipeline
ls -la ${ARTIFACT_BASE}/<PIPELINE_ID>/

# Check build artifact
ls -lh ${ARTIFACT_BASE}/<PIPELINE_ID>/build/ttm_any.tar.zst

# View test reports
ls ${ARTIFACT_BASE}/<PIPELINE_ID>/reports/
```

## Partition Configuration

| Partition | Hardware | Use Case |
|-----------|----------|----------|
| `build` | CPU-only | Builds, infra tests, notifications |
| `wh-n150` | Wormhole N150 | Single-card WH tests |
| `wh-n300` | Wormhole N300 | Dual-card WH tests |
| `wh-t3k` | Wormhole T3000 | Multi-card WH (T3K/LLMBox) |
| `wh-galaxy` | Wormhole Galaxy | Full Galaxy topology |
| `bh-p100` | Blackhole P100 | Single-card BH tests |
| `bh-p150` | Blackhole P150 | Single-card BH tests |
| `bh-p300` | Blackhole P300 | Dual-card BH tests |
| `bh-llmbox` | Blackhole LLMBox | Multi-card BH |
| `bh-loudbox` | Blackhole Loudbox | Multi-card BH |
| `bh-deskbox` | Blackhole Deskbox | BH development |
| `exabox` | Multi-host | Cross-node fabric tests |
| `perf` | Performance nodes | Performance regression tests |

## SKU to Partition Mapping

The `config/sku_map.sh` file maps GitHub Actions SKU names to Slurm partitions:

```bash
# In your script:
source slurm/config/sku_map.sh
eval "$(get_slurm_args wh_n150)"
sbatch ${SLURM_PARTITION} ${SLURM_CONSTRAINT} job.sh

# Quick lookup:
get_partition wh_llmbox_perf    # -> wh-t3k
get_arch_name bh_p100           # -> blackhole
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PIPELINE_ID` | Unique pipeline identifier | Auto-generated (timestamp-sha) |
| `DOCKER_IMAGE` | Docker image override | From `config/env.sh` |
| `ARCH_NAME` | Target architecture | `wormhole_b0` |
| `GIT_SHA` / `GIT_REF` | Git context | Auto-detected |
| `SLACK_WEBHOOK_URL` | Slack notification webhook | (none) |
| `CI_STORAGE_BASE` | Root of CI shared storage (from `site.sh`) | `$(pwd)/.slurm-ci` |
| `LOG_BASE` | Log output directory (derived) | `${CI_STORAGE_BASE}/logs` |
| `ARTIFACT_BASE` | Artifact storage root (derived) | `${CI_STORAGE_BASE}/artifacts` |
| `ARTIFACT_DIR` | Pipeline artifact directory | `${CI_STORAGE_BASE}/artifacts/<PIPELINE_ID>` |
| `CONTAINER_WORKDIR` | Working directory inside containers (from `site.sh`) | `/opt/tt-metal` |
| `TT_DEVICE_PATH` | Host Tenstorrent device path (from `site.sh`) | `/dev/tenstorrent` |
| `HUGEPAGES_PATH` | Host hugepages mount (from `site.sh`) | `/dev/hugepages-1G` |
| `BUILD_ARTIFACT` | Fetch build tarball in setup | `0` |
| `INSTALL_WHEEL` | Install Python wheel in setup | `0` |
| `ENABLE_WATCHER` | Enable TT Metal Watcher | `0` |
| `ENABLE_KERNEL_CCACHE` | Enable kernel ccache | `0` |
| `CCACHE_REMOTE_STORAGE` | Redis URL for ccache | (none) |
| `FORCE_ALL` | Force all pipeline selectors to run everything | `0` |

## Adding New Workflows

1. Create a new `.sh` file in `slurm/workflows/`.

2. **For test jobs** (direct sbatch), include SBATCH directives and source the libraries:

   ```bash
   #!/usr/bin/env bash
   #SBATCH --job-name=my-new-test
   #SBATCH --partition=wh-n150
   #SBATCH --time=02:00:00

   set -euo pipefail
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   source "${SCRIPT_DIR}/../lib/common.sh"
   source_lib artifacts
   source_lib docker
   source_lib setup_job
   source_lib cleanup
   source_config env

   require_env PIPELINE_ID
   BUILD_ARTIFACT=1 setup_job
   IMAGE="$(resolve_image "${DOCKER_IMAGE:-}")"
   docker_login
   docker_pull_with_retry "${IMAGE}"
   trap 'cleanup_job --exit-code $?' EXIT

   docker_run "${IMAGE}" "cd ${TT_METAL_HOME} && pytest tests/my_tests/"
   ```

3. **For orchestrators** (submit sub-jobs), omit SBATCH directives:

   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   source "${SCRIPT_DIR}/../lib/common.sh"
   source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"

   require_env PIPELINE_ID
   BUILD_JOB=$(submit_after "" "${SCRIPT_DIR}/build-artifact.sh" --partition=build)
   TEST_JOB=$(submit_after "${BUILD_JOB}" "${SCRIPT_DIR}/my-new-test.sh" --partition=wh-n150)
   ```

4. Make the script executable: `chmod +x slurm/workflows/my-new-test.sh`

5. Test with dry run: `./slurm/submit.sh my-new-test --dry-run`

## Scheduled Jobs (scrontab)

Nightly and periodic jobs are managed via Slurm's `scrontab`:

```bash
# Install the schedule
scrontab slurm/crontab/schedules.crontab

# View current schedule
scrontab -l

# Remove schedule
scrontab -r
```

See `slurm/crontab/schedules.crontab` for the full schedule definition.

## Troubleshooting

### Job is stuck in PENDING

```bash
# Check why the job is pending
squeue -j <JOBID> -o "%i %j %T %r"

# Common reasons:
#   Priority        - waiting for higher-priority jobs
#   Resources       - partition is full
#   Dependency      - waiting for a prerequisite job
#   QOSMaxJobsPerUser - too many jobs queued
```

### Job failed immediately

```bash
# Check exit code and state
sacct -j <JOBID> --format=JobID,State,ExitCode,Reason

# View the error log
cat ${LOG_BASE}/<job-name>/<JOBID>/0.err

# Common failures:
#   Exit 1     - script error (check logs)
#   Exit 137   - OOM killed (increase --mem)
#   TIMEOUT    - exceeded --time limit
#   NODE_FAIL  - hardware issue, will auto-retry
```

### Docker pull failures

```bash
# Verify registry access
docker login ghcr.io

# Check if Harbor mirror is available
timeout 3 bash -c 'echo >/dev/tcp/harbor.ci.tenstorrent.net/443' && echo "Harbor OK"

# Force a specific image
./slurm/submit.sh <workflow> --docker-image ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
```

### Artifacts not found

```bash
# Verify the pipeline ID
echo $PIPELINE_ID
ls ${ARTIFACT_BASE}/$PIPELINE_ID/

# Check if storage is mounted (path depends on config/site.sh)
df -h ${CI_STORAGE_BASE:-.slurm-ci}
mount | grep -E 'nfs|weka'

# Check artifact staging from build job logs
grep "Staging" ${LOG_BASE}/build-artifact/<BUILD_JOBID>/0.log
```

### Cancel a pipeline

```bash
# Cancel a specific job
scancel <JOBID>

# Cancel all jobs for a user
scancel -u $(whoami)

# Cancel all jobs with a specific name prefix
scancel --name="release-*"
```
