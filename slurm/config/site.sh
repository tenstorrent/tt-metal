#!/usr/bin/env bash
# site.sh -- Site-specific storage paths.
# Edit this file when deploying on a new cluster / storage vendor.
# All other scripts derive their paths from these variables.

# Root of CI shared storage (Weka, Pure Storage, NFS, etc.)
# Defaults to .slurm-ci/ under the launch directory so it works on any NFS
# mount without assumptions about vendor-specific paths.  Override with:
#   export CI_STORAGE_BASE=/weka/ci  (or any shared-storage path)
export CI_STORAGE_BASE="${CI_STORAGE_BASE:-$(pwd)/.slurm-ci}"

# MLPerf / model data mount point
export MLPERF_BASE="${MLPERF_BASE:-/mnt/MLPerf}"

# Container working directory (where the repo is mounted inside Docker)
export CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/work}"

# Device paths (only override if non-standard)
export TT_DEVICE_PATH="${TT_DEVICE_PATH:-/dev/tenstorrent}"
export HUGEPAGES_PATH="${HUGEPAGES_PATH:-/dev/hugepages-1G}"
