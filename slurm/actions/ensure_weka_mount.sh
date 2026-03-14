#!/usr/bin/env bash
# ensure_weka_mount.sh - Ensure Weka filesystem is mounted and hugepages available
# Usage: ensure_weka_mount.sh
#
# Port of .github/actions/ensure-active-weka-mount/action.yml
# and .github/scripts/cloud_utils/mount_weka.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_config env

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help]

Ensure the Weka mount is active and hugepages are available.

Steps:
  1. Restart the Weka mount systemd unit
  2. Verify the mount is accessible
  3. Restart or initialise hugepages (1G pages)
  4. Wait for hugepages to become non-zero

Options:
  -h, --help    Show this help message

Environment:
  MLPERF_BASE           Base mount path from config/env.sh (default: /mnt/MLPerf)
  WEKA_MOUNT_POINT      Mount path to verify (default: \${MLPERF_BASE})
  WEKA_SYSTEMD_UNIT     Systemd mount unit name (default: mnt-MLPerf.mount)
  HUGEPAGES_TIMEOUT     Seconds to wait for hugepages (default: 60)
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage 0 ;;
        *)         log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEKA_MOUNT_POINT="${WEKA_MOUNT_POINT:-${MLPERF_BASE:-/mnt/MLPerf}}"
WEKA_SYSTEMD_UNIT="${WEKA_SYSTEMD_UNIT:-mnt-MLPerf.mount}"
HUGEPAGES_TIMEOUT="${HUGEPAGES_TIMEOUT:-60}"
HUGEPAGES_SYSFS="/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages"

# ---------------------------------------------------------------------------
# Step 1: Restart Weka mount
# ---------------------------------------------------------------------------

log_info "Restarting Weka mount unit (${WEKA_SYSTEMD_UNIT})..."
sudo systemctl restart "${WEKA_SYSTEMD_UNIT}"

log_info "Verifying Weka mount at ${WEKA_MOUNT_POINT}..."
if ls -al "${WEKA_MOUNT_POINT}/bit_error_tests" &>/dev/null; then
    log_info "Weka mount is accessible"
else
    log_error "Weka mount verification failed — ${WEKA_MOUNT_POINT}/bit_error_tests not accessible"
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Restart hugepages
# ---------------------------------------------------------------------------

check_status=0
sudo systemctl status tenstorrent-hugepages.service &>/dev/null || check_status=$?

if [[ "$check_status" -eq 4 ]]; then
    log_warn "Hugepages systemd service not found, falling back to /etc/rc.local"
    sudo /etc/rc.local
else
    log_info "Hugepages service found (status=$check_status), restarting..."
    sudo systemctl restart tenstorrent-hugepages.service
fi

# ---------------------------------------------------------------------------
# Step 3: Wait for hugepages to be allocated
# ---------------------------------------------------------------------------

log_info "Waiting up to ${HUGEPAGES_TIMEOUT}s for hugepages..."
start_ts=$(date +%s)

while [[ "$(cat "$HUGEPAGES_SYSFS" 2>/dev/null || echo 0)" -eq 0 ]]; do
    if (( $(date +%s) - start_ts > HUGEPAGES_TIMEOUT )); then
        log_error "nr_hugepages still 0 after ${HUGEPAGES_TIMEOUT}s"
        exit 1
    fi
    sleep 1
done

log_info "Hugepages available: $(cat "$HUGEPAGES_SYSFS")"
log_info "Weka mount and hugepages ready"
