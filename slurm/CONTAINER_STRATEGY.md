# Container Strategy for Slurm CI: Singularity/Apptainer vs OCI Container Hooks

Analysis of two alternative container strategies for the Slurm CI infrastructure:

- **Approach A** -- Singularity/Apptainer with SIF images on NFS
- **Approach B** -- Slurm OCI container integration via Pyxis+Enroot (or native `oci.conf`)

Covers architecture differences, code changes required, level of effort, and trade-offs against the current plain-Docker baseline.

---

## Current Baseline

The existing Slurm CI infrastructure uses **plain Docker** on compute nodes. Every test/build job follows this pattern:

1. `docker_login` to GHCR or Harbor
2. `docker_pull_with_retry "$IMAGE"` to fetch the image over the network
3. `docker_run "$IMAGE" "$COMMANDS"` to execute inside the container

Core files implementing this:

- `lib/docker.sh` -- `docker_login`, `docker_pull_with_retry`, `docker_run`, `resolve_image`
- `workflows/_helpers/resolve_docker_image.sh` -- `resolve_workflow_docker_image`
- `config/env.sh` -- registry URLs, default image tags, `build_docker_run_opts`
- `config/site.sh` -- host paths for devices, hugepages, storage

Every workflow script (100+) calls `docker_run` with the resolved image. The Docker daemon must be running with appropriate permissions on every compute node.

### Current Container Execution Flow

```
  Workflow Script (on compute node)
  ┌──────────────────────────────────────────────────────────┐
  │  1. source lib/docker.sh                                 │
  │  2. resolve_workflow_docker_image "dev"                   │
  │  3. docker_login (GHCR or Harbor)                        │
  │  4. docker_pull_with_retry "$DOCKER_IMAGE"               │
  │  5. docker_run "$DOCKER_IMAGE" "$COMMANDS"               │
  │     └─> docker run --rm --network host                   │
  │            --device /dev/tenstorrent                      │
  │            -v workspace:/work                             │
  │            -v hugepages -v /etc/passwd ...                │
  │            -e TT_METAL_HOME -e ARCH_NAME ...             │
  │            $IMAGE bash -c "$COMMANDS"                     │
  └──────────────────────────────────────────────────────────┘
```

### Key Dependencies on Docker

- `dockerd` daemon running on every compute node
- Docker group membership for the Slurm job user
- Per-job network pull from GHCR (with Harbor mirror fallback)
- Docker's local layer cache for deduplication across pulls
- `docker image prune` and `docker container prune` for cleanup

---

## Approach A: Singularity/Apptainer -- SIF Images on NFS

### How It Works

```
                                     NFS / Weka
  GHCR / Harbor                   +-----------------+
  +------------+    apptainer     | /weka/ci/sif/   |
  | OCI Image  | ---  pull  ---> | image-abc.sif   |
  +------------+   (one-time)    +-----------------+
                                        |
                                   bind mount
                                        |
                                 +------v-------+
                                 | Compute Node |
                                 | apptainer    |
                                 |   exec       |
                                 |  image.sif   |
                                 |  --bind ...  |
                                 +--------------+
```

Docker/OCI images are converted to **SIF** (Singularity Image Format) files -- single, immutable, read-only squashfs archives. These are stored on shared Weka/NFS storage. Compute nodes run `apptainer exec /path/to/image.sif <command>` instead of `docker run`.

### Architecture Changes

**New file: `lib/singularity.sh`** (replaces `lib/docker.sh`)

- `singularity_pull_or_cache IMAGE` -- convert Docker image to SIF on NFS if not cached, keyed by content-hash tag
- `singularity_run SIF_PATH COMMANDS [extra_args...]` -- execute in container with device bind-mounts, env forwarding
- No login function needed (auth via `APPTAINER_DOCKER_USERNAME` / `APPTAINER_DOCKER_PASSWORD` env vars)

**New file: `workflows/build-sif-cache.sh`** (or added to `build-docker-artifact.sh`)

- Job that runs after Docker image builds to convert new tags to SIF on NFS
- Could be a scheduled cron job or triggered in the pipeline

**Modified files:**

| File | Change |
|------|--------|
| `config/site.sh` | Add `SIF_CACHE_DIR="${CI_STORAGE_BASE}/sif"` |
| `config/env.sh` | Replace Docker registry defaults with SIF path defaults; remove `build_docker_run_opts` |
| `workflows/_helpers/resolve_docker_image.sh` | Resolve to SIF path instead of registry URL; rename to `resolve_container_image.sh` |
| `lib/docker.sh` | Delete or gut; replaced by `lib/singularity.sh` |
| All 100+ workflow scripts | `docker_run "$IMAGE" "..."` becomes `singularity_run "$SIF" "..."` |

**Key syntax translation:**

| Docker | Singularity/Apptainer |
|--------|-----------------------|
| `docker run --rm` | `apptainer exec` (no daemon, ephemeral by nature) |
| `-v /host:/container` | `--bind /host:/container` or `-B /host:/container` |
| `--device /dev/tenstorrent` | `--bind /dev/tenstorrent` (plain bind mount; no `--device` concept) |
| `-e VAR=val` | `--env VAR=val` or `APPTAINERENV_VAR=val` |
| `-u uid:gid` | Not needed (runs as calling user by default) |
| `-w /work` | `--pwd /work` |
| `--network host` | Host networking is the default |
| `docker login` | `APPTAINER_DOCKER_USERNAME` + `APPTAINER_DOCKER_PASSWORD` exports |

**Example: what `singularity_run` would look like:**

```bash
singularity_run() {
    local sif_path="$1"; shift
    local commands="$1"; shift
    local -a args=(
        --bind "${WORKSPACE:-.}:${CONTAINER_WORKDIR:-/work}"
        --bind "/etc/passwd:/etc/passwd:ro"
        --bind "/etc/shadow:/etc/shadow:ro"
        --pwd "${CONTAINER_WORKDIR:-/work}"
        --writable-tmpfs
    )
    if [[ -e "${TT_DEVICE_PATH:-/dev/tenstorrent}" ]]; then
        args+=(--bind "${TT_DEVICE_PATH}")
    fi
    if [[ -d "${HUGEPAGES_PATH:-/dev/hugepages-1G}" ]]; then
        args+=(--bind "${HUGEPAGES_PATH}")
    fi
    # env forwarding...
    apptainer exec "${args[@]}" "$sif_path" bash -c "set -euo pipefail; $commands"
}
```

**Example: what `singularity_pull_or_cache` would look like:**

```bash
singularity_pull_or_cache() {
    local docker_uri="$1"
    local tag="${docker_uri##*:}"
    local repo="${docker_uri%%:*}"
    local safe_name
    safe_name="$(echo "$repo" | tr '/:' '_')"
    local sif_path="${SIF_CACHE_DIR}/${safe_name}_${tag}.sif"

    if [[ -f "$sif_path" ]]; then
        log_info "SIF cache hit: $sif_path"
        echo "$sif_path"
        return 0
    fi

    log_info "Converting Docker image to SIF: $docker_uri -> $sif_path"
    local tmp_sif="${sif_path}.tmp.$$"
    if apptainer pull "$tmp_sif" "docker://${docker_uri}"; then
        mv "$tmp_sif" "$sif_path"
        log_info "SIF cached: $sif_path"
        echo "$sif_path"
    else
        rm -f "$tmp_sif"
        log_error "Failed to convert image: $docker_uri"
        return 1
    fi
}
```

**Example: what a workflow script becomes (t3000-unit-tests.sh):**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=t3000-unit-tests
#SBATCH --partition=wh-t3k
#SBATCH --time=02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/singularity.sh"       # was: lib/docker.sh
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"
source "${SCRIPT_DIR}/workflows/_helpers/resolve_container_image.sh"

parse_common_args "$@"
resolve_workflow_container_image dev             # sets SIF_PATH instead of DOCKER_IMAGE

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

export SINGULARITY_EXTRA_ENV="GTEST_OUTPUT=xml:${TT_METAL_HOME}/generated/test_reports/
HF_HUB_OFFLINE=1
HF_HOME=${MLPERF_BASE}/huggingface
LD_LIBRARY_PATH=${TT_METAL_HOME}/build/lib"
export SINGULARITY_EXTRA_BINDS="${MLPERF_BASE}:${MLPERF_BASE}:ro"

singularity_run "$SIF_PATH" "
    mkdir -p \${TT_METAL_HOME}/generated/test_reports
    source tests/scripts/t3000/run_t3000_unit_tests.sh
    echo '${TEST_CMD}'
    ${TEST_CMD}
"

log_info "T3000 unit test '${TEST_NAME}' complete"
```

### Pros

- **No Docker daemon on compute nodes.** This is the single biggest win. Docker requires root-level daemon access; Singularity/Apptainer runs entirely unprivileged. This eliminates a large class of security and permissions issues on Slurm clusters.
- **Cached on shared storage.** SIF files live on Weka/NFS. Once converted, every node has instant access -- no per-job network pulls. This dramatically reduces job startup latency, especially for large images (the CI images are multi-GB).
- **Battle-tested in HPC/Slurm.** Singularity was designed for exactly this use case. It is the de facto standard for running containers on Slurm clusters.
- **Immutable images.** SIF files are read-only squashfs. No risk of layer corruption, no dangling images to clean up, no `docker image prune` needed.
- **Simple cleanup.** One file per image. `rm /weka/ci/sif/old-image.sif` -- done. No image layer graph to manage.
- **Preserves user identity.** Runs as the calling user's UID/GID by default -- no `-u $(id -u):$(id -g)` gymnastics.
- **Existing Docker images work.** `apptainer pull docker://ghcr.io/...` converts any Docker/OCI image to SIF.
- **MPI-aware.** Singularity/Apptainer has first-class support for host MPI injection (`--bind` the host MPI into the container), critical for multi-node Galaxy/Exabox jobs.

### Cons

- **SIF conversion step.** Each new Docker image tag requires a one-time conversion (`apptainer pull`). This takes 2-5 minutes per image and must be managed as a pipeline stage or cron job.
- **NFS storage cost.** SIF files are compressed but still large (2-8 GB each). With 6 image variants and regular tag rotation, expect 20-50 GB of SIF cache at steady state.
- **Writable filesystem limitations.** SIF is read-only. Jobs that write inside the container filesystem need `--writable-tmpfs` (tmpfs overlay) or `--overlay`. This uses RAM-backed tmpfs. Tests that write many GBs of temp data inside the container may need to be restructured to write to bind-mounted host paths.
- **Behavioral differences from Docker.** Home directory handling, `/tmp` handling, and environment inheritance all differ from Docker defaults. Expect a debugging/adaptation period.
- **Apptainer must be installed on all compute nodes.** Requires cluster admin cooperation for initial deployment (though it is a standard HPC package).
- **`docker build` still needed.** Image *building* still uses Docker/Buildx. Only the *runtime* changes to Singularity. The `build-docker-artifact.sh` workflow remains Docker-based or needs a separate builder node.
- **Less ecosystem tooling.** Docker Compose, Buildx multi-platform, layer caching -- none of this exists in the Singularity world. Not relevant for CI runtime, but relevant for image builds.

### Level of Effort

| Work Item | Estimate |
|-----------|----------|
| Write `lib/singularity.sh` (pull, run, env forwarding) | 1-2 days |
| Write SIF cache management (conversion job, GC cron) | 1 day |
| Rewrite `resolve_docker_image.sh` to resolve SIF paths | 0.5 days |
| Update `config/env.sh` and `config/site.sh` | 0.5 days |
| Mechanical update of 100+ workflow scripts (`docker_run` -> `singularity_run`) | 2-3 days (scriptable) |
| Debug and validate writable-tmpfs / overlay edge cases | 1-2 days |
| Validate multi-node MPI jobs (Galaxy, Exabox) | 1-2 days |
| Cluster admin: install Apptainer on all nodes | External dependency |
| **Total** | **7-11 days** |

---

## Approach B: OCI Container Hooks via Slurm Integration

There are two sub-variants here. Both move the container boundary from "inside the job script" to "around the job script" -- Slurm itself manages the container lifecycle.

### Sub-variant B1: Pyxis + Enroot (NVIDIA)

```
  sbatch --container-image=ghcr.io/...
         --container-mounts=/dev/tenstorrent,...
         job.sh
            |
            v
  +--- Slurm slurmstepd ---+
  |  Pyxis SPANK plugin     |
  |    |                     |
  |    v                     |
  |  Enroot runtime          |
  |  +--pull & squashfs--+   |
  |  |  job.sh runs HERE |   |
  |  +-------------------+   |
  +--------------------------+
```

[Pyxis](https://github.com/NVIDIA/pyxis) is an NVIDIA-developed SPANK plugin for Slurm. [Enroot](https://github.com/NVIDIA/enroot) is a lightweight, unprivileged container runtime that converts OCI images to squashfs and runs them without a daemon.

With Pyxis, container configuration moves to `#SBATCH` directives or `srun` flags:

```bash
#SBATCH --container-image=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
#SBATCH --container-mounts=/dev/tenstorrent:/dev/tenstorrent,/dev/hugepages-1G:/dev/hugepages-1G
#SBATCH --container-workdir=/work
```

The job script body runs *inside* the container automatically. No `docker_run` wrapper.

### Sub-variant B2: Slurm Native OCI (`oci.conf`)

Slurm 21.08+ supports `oci.conf` which configures an OCI-compatible runtime (runc, crun). This is lower-level than Pyxis -- you prepare OCI bundles and Slurm's step daemon uses the runtime to execute job steps.

This approach is **less mature** and requires more manual bundle management. Most production deployments use Pyxis+Enroot instead.

### Architecture Changes (Pyxis+Enroot -- the practical choice)

**Modified files:**

| File | Change |
|------|--------|
| `config/env.sh` | Add Enroot config vars (`ENROOT_CACHE_PATH`, `ENROOT_DATA_PATH`), remove `build_docker_run_opts` |
| `config/site.sh` | Add `ENROOT_CACHE_PATH="${CI_STORAGE_BASE}/enroot/cache"` |
| `lib/docker.sh` | Delete entirely. Container management is Slurm's responsibility. |
| `submit.sh` | Add `--container-image`, `--container-mounts` to sbatch arguments |
| `workflows/_helpers/resolve_docker_image.sh` | Still needed for image resolution, but output feeds `--container-image` flag instead of `docker_run` |
| All 100+ workflow scripts | Remove `docker_login`, `docker_pull_with_retry`, `docker_run` calls. Script body IS the container payload. Add `#SBATCH --container-image` directive. |
| Slurm cluster config | Install Pyxis plugin + Enroot on all nodes; configure `plugstack.conf` |

**The fundamental shift:** The current scripts have a two-phase pattern:

```
  Phase 1 (host): setup_job, docker_login, docker_pull
  Phase 2 (container): docker_run "$IMAGE" "$COMMANDS"
```

With Pyxis, the entire script runs in the container. There is no host phase inside the script -- setup must happen either:

- In Slurm prologs/epilogs (cluster-level config)
- Inside the container itself (requires shared storage mounts to be available)
- At submission time in `submit.sh` (before `sbatch`)

**Example: what a workflow script becomes (t3000-unit-tests.sh):**

```bash
#!/usr/bin/env bash
#SBATCH --job-name=t3000-unit-tests
#SBATCH --partition=wh-t3k
#SBATCH --time=02:00:00
#SBATCH --container-image=${DOCKER_IMAGE}
#SBATCH --container-mounts=/dev/tenstorrent:/dev/tenstorrent,/dev/hugepages-1G:/dev/hugepages-1G,/weka/ci:/weka/ci:ro,/mnt/MLPerf:/mnt/MLPerf:ro
#SBATCH --container-workdir=/work

set -euo pipefail

# We are already inside the container.
# No docker_run, no docker_pull, no docker_login.
# setup_job and cleanup_job must work from inside the container
# (shared storage is available via --container-mounts).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"
source "${SCRIPT_DIR}/lib/artifacts.sh"
source "${SCRIPT_DIR}/lib/setup_job.sh"
source "${SCRIPT_DIR}/lib/cleanup.sh"
source "${SCRIPT_DIR}/lib/matrix.sh"

parse_common_args "$@"

export BUILD_ARTIFACT=1
export INSTALL_WHEEL=1
setup_job
trap 'cleanup_job $?' EXIT

TASK_ID="$(get_array_task_id)"
TEST_CMD="$(get_task_field "$MATRIX_FILE" "$TASK_ID" cmd)"
TEST_NAME="$(get_task_field "$MATRIX_FILE" "$TASK_ID" name)"

log_info "Running array task ${TASK_ID}: ${TEST_NAME}"

export GTEST_OUTPUT="xml:${TT_METAL_HOME}/generated/test_reports/"
export HF_HUB_OFFLINE=1
export HF_HOME="${MLPERF_BASE}/huggingface"
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build/lib"

mkdir -p "${TT_METAL_HOME}/generated/test_reports"
source tests/scripts/t3000/run_t3000_unit_tests.sh
echo "${TEST_CMD}"
${TEST_CMD}

log_info "T3000 unit test '${TEST_NAME}' complete"
```

**Example: what `submit.sh` changes look like:**

```bash
# Before (current):
sbatch --parsable \
    "--job-name=ci-${WORKFLOW_NAME}-${PIPELINE_ID}" \
    "--output=${LOG_DIR}/%x-%j.out" \
    "--error=${LOG_DIR}/%x-%j.err" \
    "--export=ALL,PIPELINE_ID=${PIPELINE_ID},ARTIFACT_DIR=${ARTIFACT_DIR}" \
    ${EXTRA_SBATCH_ARGS} \
    "${WORKFLOW_SCRIPT}"

# After (Pyxis):
CONTAINER_IMAGE="$(resolve_image)"
CONTAINER_MOUNTS="/dev/tenstorrent:/dev/tenstorrent"
CONTAINER_MOUNTS+=",${HUGEPAGES_PATH}:${HUGEPAGES_PATH}"
CONTAINER_MOUNTS+=",${CI_STORAGE_BASE}:${CI_STORAGE_BASE}"
CONTAINER_MOUNTS+=",${MLPERF_BASE}:${MLPERF_BASE}:ro"

sbatch --parsable \
    "--job-name=ci-${WORKFLOW_NAME}-${PIPELINE_ID}" \
    "--output=${LOG_DIR}/%x-%j.out" \
    "--error=${LOG_DIR}/%x-%j.err" \
    "--export=ALL,PIPELINE_ID=${PIPELINE_ID},ARTIFACT_DIR=${ARTIFACT_DIR}" \
    "--container-image=${CONTAINER_IMAGE}" \
    "--container-mounts=${CONTAINER_MOUNTS}" \
    "--container-workdir=${CONTAINER_WORKDIR:-/work}" \
    ${EXTRA_SBATCH_ARGS} \
    "${WORKFLOW_SCRIPT}"
```

### Pros

- **No Docker daemon on compute nodes.** Same as Singularity -- Enroot is rootless and daemonless.
- **Simpler workflow scripts.** No `docker_run` wrapper, no `docker_login`, no `docker_pull_with_retry`. Scripts just contain the test commands. The container is transparent infrastructure.
- **Native Slurm integration.** Container lifecycle is managed by Slurm's step daemon. Job accounting, resource limits, signal handling all work correctly without shelling into a nested `docker run`.
- **Automatic image caching.** Enroot converts OCI images to squashfs and caches them. With `ENROOT_CACHE_PATH` on shared storage, images are pulled once across the cluster.
- **OCI-native.** Works directly with GHCR/Harbor OCI images -- no conversion step needed (Enroot does it transparently).
- **Credential handling.** Enroot uses `$ENROOT_CONFIG_PATH/.credentials` or standard Docker credential helpers.
- **GPU/device GRES integration.** Device passthrough can be tied to Slurm's GRES system rather than per-script `--device` flags.
- **Used in production at scale.** Pyxis+Enroot is deployed at NVIDIA, major national labs, and large AI clusters.

### Cons

- **Requires Slurm admin cooperation.** Pyxis is a SPANK plugin -- it must be installed on the Slurm controller and all compute nodes. Enroot must be installed on all compute nodes. This is a cluster infrastructure change, not just a CI code change.
- **`oci.conf` native approach is immature.** If going the native Slurm OCI route (without Pyxis), the tooling is rough. Pyxis is strongly recommended.
- **Deeper script restructuring.** Every workflow script changes fundamentally. Code that runs before `docker_run` (workspace prep, artifact fetch) must now either run outside the container (as a Slurm prolog) or be restructured to run inside. The current two-phase pattern (host setup + container execution) needs rethinking.
- **`setup_job` and `cleanup_job` complexity.** These currently run on the host before/after `docker_run`. With Pyxis, the entire job step runs in the container. Host-side setup (artifact fetch from Weka, workspace creation) must move to Slurm prologs/epilogs or be done inside the container (which needs the mounts).
- **`#SBATCH --container-image` is static.** SBATCH directives are parsed at submission time. Dynamic image resolution (Harbor mirror fallback, staged tags from build job) must happen before `sbatch` is called, not inside the script. This pushes complexity into `submit.sh` and orchestrator scripts.
- **Multi-node container jobs.** For Galaxy/Exabox multi-node MPI jobs, Pyxis handles `srun` steps but the interaction between Slurm's MPI integration and containerized steps needs careful validation.
- **Enroot squashfs caching.** Enroot's cache can grow large and needs garbage collection. Cache invalidation on shared NFS can cause thundering-herd issues when many nodes pull the same new image simultaneously.
- **Less flexible than Docker for build jobs.** `build-docker-artifact.sh` (which builds Docker images) cannot itself run inside a Pyxis container easily. Build jobs may need to remain Docker-based.
- **Debugging is harder.** You can't easily `docker exec` into a running Pyxis container. Debugging requires Slurm-level tools.

### Level of Effort

| Work Item | Estimate |
|-----------|----------|
| Cluster admin: install Pyxis + Enroot on controller + all nodes | External dependency (1-3 days admin work) |
| Configure `plugstack.conf`, Enroot credentials, shared cache | 1 day |
| Redesign `submit.sh` to pass `--container-image` and `--container-mounts` | 1-2 days |
| Refactor `resolve_docker_image.sh` for submission-time resolution | 1 day |
| Restructure `setup_job.sh` / `cleanup_job` for container-native execution | 2-3 days |
| Delete `lib/docker.sh` | 0.5 days |
| Rewrite 100+ workflow scripts to remove `docker_run` and add SBATCH container directives | 3-5 days (deeper restructuring than Approach A) |
| Handle edge cases: build jobs, `docker build` jobs, CPU-only jobs | 1-2 days |
| Validate multi-node MPI jobs | 1-2 days |
| Validate Enroot squashfs caching on Weka/NFS | 1 day |
| **Total** | **12-18 days** (including external admin dependency) |

---

## Side-by-Side Comparison

| Dimension | Current (Docker) | A: Singularity/Apptainer | B: Pyxis + Enroot |
|-----------|-----------------|--------------------------|-------------------|
| **Daemon required** | Yes (dockerd) | No | No |
| **Root/privileged** | Docker group | Unprivileged | Unprivileged |
| **Image format** | OCI layers (Docker cache) | SIF (squashfs, single file) | squashfs (Enroot cache) |
| **Image pull** | Per-job `docker pull` | One-time convert to SIF on NFS | Enroot auto-caches on first pull |
| **Image storage** | Docker layer store on each node | Single SIF on shared NFS | Enroot squashfs on shared storage |
| **Job script pattern** | `docker_run "$IMAGE" "$CMD"` | `singularity_run "$SIF" "$CMD"` | Script body IS the command (no wrapper) |
| **Device passthrough** | `--device /dev/tenstorrent` | `--bind /dev/tenstorrent` | `--container-mounts=...` or GRES |
| **Env forwarding** | `-e VAR=val` | `--env VAR=val` | Inherited automatically |
| **MPI support** | Host MPI calls `docker run` | Host MPI bind-mounted into container | Slurm `srun` manages MPI+container |
| **Cluster admin effort** | Docker installed + group perms | Install Apptainer (standard HPC pkg) | Install Pyxis + Enroot (less common) |
| **CI code effort** | Baseline | Moderate (swap runtime, mechanical script updates) | High (restructure script architecture) |
| **Maturity in HPC** | Low (not standard for Slurm) | Very high (de facto HPC standard) | High (common in AI/GPU clusters) |
| **Writable filesystem** | Full (overlay fs) | Limited (`--writable-tmpfs` or `--overlay`) | Limited (Enroot has `--rw` but RAM-backed) |
| **Debugging** | `docker exec` into running | `singularity shell` | Harder (Slurm-level only) |
| **Cleanup** | `docker image prune`, dangling layers | `rm *.sif` | Enroot cache GC |

---

## Recommendation Factors

**Choose Singularity/Apptainer (Approach A) if:**

- You want the lowest-risk, most proven path for Slurm container execution
- Cluster admins are familiar with Apptainer (common in HPC shops)
- You want to minimize architectural changes -- the `container_run` wrapper pattern stays the same
- You value simple, predictable image management (SIF files on NFS)
- Multi-node MPI jobs are a critical path

**Choose Pyxis+Enroot (Approach B) if:**

- You want the cleanest long-term architecture (scripts become container-native)
- The cluster already has or is willing to deploy Pyxis (common in NVIDIA/AI-focused clusters)
- You want Slurm-native container lifecycle management (accounting, signals, GRES)
- You're willing to invest more upfront for a simpler steady-state

**Hybrid option:**

- Use Singularity/Apptainer for the runtime (Approach A) with the current script architecture
- Migrate to Pyxis/Enroot later when cluster infrastructure matures
- The SIF cache on NFS is useful regardless -- Enroot can also be configured to use pre-cached squashfs images

---

## Files Affected Summary

Both approaches touch the same core set of files:

- `lib/docker.sh` -- replaced or deleted
- `config/env.sh` -- registry/path config changes
- `config/site.sh` -- new cache directory variable
- `workflows/_helpers/resolve_docker_image.sh` -- resolution target changes
- `submit.sh` -- Approach B adds container flags; Approach A minimal changes
- 100+ workflow scripts -- Approach A: swap function call; Approach B: remove wrapper + add SBATCH directives
- `workflows/build-docker-artifact.sh` -- unchanged in both (still builds Docker images)
