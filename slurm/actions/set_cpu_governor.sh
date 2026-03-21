#!/usr/bin/env bash
# set_cpu_governor.sh - Set CPU frequency governor for all cores
# Usage: set_cpu_governor.sh [--governor MODE]
#
# Port of .github/actions/set-cpu-governor/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

GOVERNOR="performance"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Set the CPU frequency scaling governor for all cores via sysfs.

Options:
  --governor MODE   Governor to set (default: performance)
                    Choices: performance, ondemand, conservative,
                             powersave, userspace, schedutil
  -h, --help        Show this help message
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --governor) GOVERNOR="$2"; shift 2 ;;
        -h|--help)  usage 0 ;;
        *)          log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Check for scaling_governor sysfs entries (may be absent on VMs)
# ---------------------------------------------------------------------------

shopt -s nullglob
scaling_governors=(/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
shopt -u nullglob

if [[ ${#scaling_governors[@]} -eq 0 ]]; then
    log_warn "No scaling_governor found — this may be a virtual machine"
    log_warn "Skipping CPU governor configuration"
    exit 0
fi

# ---------------------------------------------------------------------------
# Set governor
# ---------------------------------------------------------------------------

log_info "Setting CPU governor to '${GOVERNOR}' for all cores"

if echo "$GOVERNOR" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null; then
    log_info "Successfully set ${GOVERNOR} governor for all CPU cores"
else
    log_error "Failed to set ${GOVERNOR} governor"
    exit 1
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

log_info "Verifying governor settings:"
for gov_file in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -f "$gov_file" ]] || continue
    cpu_num="${gov_file#/sys/devices/system/cpu/cpu}"
    cpu_num="${cpu_num%%/*}"
    current="$(cat "$gov_file" 2>/dev/null || echo unknown)"
    log_info "  CPU ${cpu_num}: ${current}"
done
