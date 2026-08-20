#!/usr/bin/env bash
#
# Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4)
#
# Standalone port of the models_e2e_tests.yaml entry. The multihost CI job
# (.github/workflows/models-e2e-tests-multihost-impl.yaml) supplies the env
# below before eval'ing the YAML cmd; this script applies the same defaults
# so a login shell on the runner (or an equivalent cluster) can invoke it
# directly. Override any variable in the environment before launching.
#
# Required:
#   TT_METAL_HOME           tt-metal source root (CI: /home/user/tt-metal)
#
# For reference, the CI env is (job-level, models-e2e-tests-multihost-impl.yaml):
#   PATH                    /opt/venv/bin:...  (venv first)
#   VIRTUAL_ENV             /opt/venv
#   HOME                    /home/user
#   PIP_CACHE_DIR           /home/user/.cache/pip
#   LD_LIBRARY_PATH         $TT_METAL_HOME/build/lib  (wheel lib may be prepended)
#   TT_METAL_RUNTIME_ROOT   $TT_METAL_HOME
#   TT_METAL_CACHE          /home/user/.cache
#   PYTHONPATH              $TT_METAL_HOME
#   PIPELINE_DIR            /ci/models-e2e-<run_id>-<job-index>   (NFS scratch)
#   TTRUN_CWD               $PIPELINE_DIR/ttrun-cwd               (empty automapper cwd)
#   HF_TOKEN                HuggingFace token (forwarded to mpirun ranks)
#
# Launch env (YAML cmd, not job-level):
#   TT_METAL_ALLOCATOR_MODE_HYBRID=1
#   TT_METAL_SLOW_DISPATCH_MODE=1
#
# Usage:
#   ./run_deepseek_blitz_sc16.sh              # full 16-Galaxy run
#   ./run_deepseek_blitz_sc16.sh --dry-run    # print the command only
#   ./run_deepseek_blitz_sc16.sh --quiet      # launch.log only (CI-like, no tee)
#   PIPELINE_DIR=/data/$USER/deepseek-v3-b1-sc16/my-run HOSTS=h1,h2,... ./run_deepseek_blitz_sc16.sh
#
# name:        Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4)
# model:       deepseek-v3-b1
# sku:         bh_sc16 (timeout: 60 min)
# owner_id:    U08E1JCMAJF
# team:        scaleout

set -euo pipefail

DRY_RUN=0
QUIET=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --quiet)   QUIET=1 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $arg" >&2
      echo "Usage: TT_METAL_HOME=<path> $0 [--dry-run] [--quiet]" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# CI environment (same names/roles as models-e2e-tests-multihost-impl.yaml)
# ---------------------------------------------------------------------------

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  echo "ERROR: TT_METAL_HOME is required" >&2
  exit 1
fi
: "${TT_METAL_RUNTIME_ROOT:=${TT_METAL_HOME}}"
if [[ -z "${PYTHONPATH:-}" ]]; then
  PYTHONPATH="${TT_METAL_HOME}"
elif [[ ":${PYTHONPATH}:" != *":${TT_METAL_HOME}:"* ]]; then
  PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
fi

# CI container user + caches. Do not override an already-set HOME.
: "${HOME:=/home/user}"
: "${TT_METAL_CACHE:=${HOME}/.cache}"
: "${PIP_CACHE_DIR:=${HOME}/.cache/pip}"

# Native libs from the cmake tarball. CI may later prepend the installed
# wheel's ttnn/build/lib when prefer-wheel-libs is true; honour a pre-set path.
: "${LD_LIBRARY_PATH:=${TT_METAL_HOME}/build/lib}"

# Shared NFS scratch for automapper (empty ttrun-cwd + ranklogs).
: "${PIPELINE_DIR:=/data/${USER}/deepseek-v3-b1-sc16/$(date +%Y%m%d-%H%M%S)}"
: "${TTRUN_CWD:=${PIPELINE_DIR}/ttrun-cwd}"

# Venv used by the CI container. Prepend its bin only when present so a
# non-container shell with python3 already on PATH still works.
: "${VIRTUAL_ENV:=/opt/venv}"
if [[ -d "${VIRTUAL_ENV}/bin" ]]; then
  case ":${PATH}:" in
    *":${VIRTUAL_ENV}/bin:"*) ;;
    *) PATH="${VIRTUAL_ENV}/bin:${PATH}" ;;
  esac
fi

# HF_TOKEN is optional here; gated downloads need it on every rank (CI sets it
# from secrets.HUGGINGFACE_TOKEN). Leave unset if the weights are already local.

export PATH VIRTUAL_ENV HOME PIP_CACHE_DIR
export LD_LIBRARY_PATH
export TT_METAL_HOME TT_METAL_RUNTIME_ROOT TT_METAL_CACHE PYTHONPATH
export PIPELINE_DIR TTRUN_CWD
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

# YAML cmd env (not job-level).
export TT_METAL_ALLOCATOR_MODE_HYBRID="${TT_METAL_ALLOCATOR_MODE_HYBRID:-1}"
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"

# ---------------------------------------------------------------------------
# Launch inputs (YAML cmd)
# ---------------------------------------------------------------------------

# Hostfile is set only in CI.
TCP_INTERFACE="${TCP_INTERFACE:-ens5f0np0}"
MGD="${MGD:-${TT_METAL_HOME}/models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto}"
MODEL_PATH="${MODEL_PATH:-/mnt/models/deepseek-ai/DeepSeek-R1-0528-dequantized/DeepSeek-R1-dequantized}"
CACHE_PATH="${CACHE_PATH:-/mnt/models/deepseek-ai/cache-2026-03-22}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
PROMPT="${PROMPT:-Solve this step by step: Design a cache for sharded LLM weights across multiple hosts with local NVMe and shared NFS. Minimize startup latency, avoid duplicate reads, support cache invalidation, and explain tradeoffs between per-host caches, content-addressable storage, and pre-sharded artifacts.}"

if [[ -n "${HOSTS:-}" ]]; then
  RESOLVED_HOSTS="${HOSTS}"
else
  echo "ERROR: HOSTS is required" >&2
  exit 1
fi

missing=0
for path in "${MGD}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${path}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

echo "TT_METAL_HOME: ${TT_METAL_HOME}"
echo "PYTHONPATH:    ${PYTHONPATH}"
echo "PIPELINE_DIR:  ${PIPELINE_DIR}"
echo "TTRUN_CWD:     ${TTRUN_CWD}"
echo "MGD:           ${MGD}"
echo "Hosts:         ${RESOLVED_HOSTS}"
echo "TCP iface:     ${TCP_INTERFACE}"

set -- python3 "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
  --skip-executable-check \
  --tcp-interface "${TCP_INTERFACE}" \
  --mesh-graph-descriptor "${MGD}" \
  --hosts "${RESOLVED_HOSTS}" \
  --mpi-args "--bind-to none --tag-output --wdir ${TT_METAL_HOME} --output-filename ${PIPELINE_DIR}/ranklogs" \
  -- \
  python3 -m models.demos.deepseek_v3_b1.demo.cli \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --no-stop-at-eos \
    --weights real \
    --cache-path "${CACHE_PATH}" \
    --prompt "${PROMPT}" \
    --model-path "${MODEL_PATH}"

printf 'Command:      '
printf '%q ' "$@"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

# The YAML cmd cds into TTRUN_CWD because automapper phase 1 writes generated/ttrun/
# into the working directory. ttrun.py and the MGD are absolute, so the source tree
# is not copied onto NFS.
mkdir -p "${PIPELINE_DIR}/ranklogs" "${TTRUN_CWD}"
cd "${TTRUN_CWD}"

if [[ "${QUIET}" -eq 1 ]]; then
  # CI: stdout/stderr go only to launch.log; the job step has its own capture.
  "$@" > "${PIPELINE_DIR}/launch.log" 2>&1
else
  # Standalone: keep CI's launch.log and also show the stream.
  "$@" 2>&1 | tee "${PIPELINE_DIR}/launch.log"
fi
