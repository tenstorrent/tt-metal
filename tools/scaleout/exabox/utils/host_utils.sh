#!/bin/bash

# Host-list validation utilities shared by the exabox cluster scripts.

# Aborts if a comma-separated --hosts list contains the same host more than once.
# A duplicated host silently corrupts MPI rank assignment (two ranks land on the
# same node), which later surfaces as a cryptic failure deep inside the test or
# the topology mapper instead of a clear input error. Catch it up front.
#
# Usage: check_duplicate_hosts "$HOSTS" || exit 1
check_duplicate_hosts() {
    local host_list="$1"
    local -a hosts
    IFS=',' read -ra hosts <<< "$host_list"

    local -A seen=()
    local -A dupes=()
    local h
    for h in "${hosts[@]}"; do
        # Tolerate empty entries from stray/trailing commas.
        [[ -z "$h" ]] && continue
        if [[ -n "${seen[$h]:-}" ]]; then
            dupes["$h"]=1
        fi
        seen["$h"]=1
    done

    if [[ "${#dupes[@]}" -gt 0 ]]; then
        echo "Error: --hosts contains duplicate host(s): ${!dupes[*]}" >&2
        echo "" >&2
        echo "Each host must appear exactly once in --hosts. Passing the same host" >&2
        echo "twice corrupts MPI rank assignment and causes cryptic test failures." >&2
        echo "Provided list: $host_list" >&2
        return 1
    fi
    return 0
}

# Minimum tool/driver/firmware versions required on every host before running recovery or
# validation. Bump these as the supported baseline moves; both recover.sh and run_validation.sh
# check against the same values.
TT_SMI_MIN_VERSION="5.2.0"   # `tt-smi --version`
KMD_MIN_VERSION="2.9.0"      # `cat /sys/module/tenstorrent/version`
FW_MIN_VERSION="19.11"       # `cat /sys/class/tenstorrent/tenstorrent!<n>/tt_fw_bundle_ver`

# Assert the minimum tt-smi / KMD / firmware versions on every host via mpirun. This is a host-level
# check (independent of any Docker image), so it always runs via plain mpirun. Each rank prints
# per-check OK/ERROR lines and exits non-zero if anything is below the baseline, so mpirun (and this
# function) returns non-zero if any host fails.
#
# Usage: check_cluster_versions "$HOSTS" "$MPI_IF" [mpi_extra_args...]
# Returns 0 if all hosts meet the minimums, non-zero otherwise.
check_cluster_versions() {
    local hosts="$1"
    local mpi_if="$2"
    shift 2
    local mpi_extra_args=("$@")

    local version_check_cmd
    # Single-quoted heredoc: nothing expands here. Minimum versions are passed as positional args
    # ($1/$2/$3) so the script body stays literal and runs identically on every rank.
    read -r -d '' version_check_cmd <<'VERSION_CHECK_CMD' || true
set -o pipefail
h=$(hostname)
tt_smi_min="$1"
kmd_min="$2"
fw_min="$3"
fail=0

# version_ge A B -> true (0) if A >= B, using version-aware ordering.
version_ge() {
    [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)" == "$2" ]]
}

# 1) tt-smi version
tt_smi_ver=$(tt-smi --version 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)+' | head -n1)
if [[ -z "$tt_smi_ver" ]]; then
    printf '[%s] ERROR: could not determine tt-smi version (is tt-smi installed?)\n' "$h"
    fail=1
elif ! version_ge "$tt_smi_ver" "$tt_smi_min"; then
    printf '[%s] ERROR: tt-smi version %s < required %s\n' "$h" "$tt_smi_ver" "$tt_smi_min"
    fail=1
else
    printf '[%s] OK: tt-smi version %s (>= %s)\n' "$h" "$tt_smi_ver" "$tt_smi_min"
fi

# 2) KMD (kernel driver) version
kmd_ver=$(cat /sys/module/tenstorrent/version 2>/dev/null)
if [[ -z "$kmd_ver" ]]; then
    printf '[%s] ERROR: could not read /sys/module/tenstorrent/version (is the tenstorrent KMD loaded?)\n' "$h"
    fail=1
elif ! version_ge "$kmd_ver" "$kmd_min"; then
    printf '[%s] ERROR: KMD version %s < required %s\n' "$h" "$kmd_ver" "$kmd_min"
    fail=1
else
    printf '[%s] OK: KMD version %s (>= %s)\n' "$h" "$kmd_ver" "$kmd_min"
fi

# 3) Firmware bundle version, checked on every tenstorrent device present on the host
fw_found=0
for f in /sys/class/tenstorrent/tenstorrent!*/tt_fw_bundle_ver; do
    [[ -e "$f" ]] || continue
    fw_found=1
    fw_ver=$(cat "$f" 2>/dev/null)
    if [[ -z "$fw_ver" ]]; then
        printf '[%s] ERROR: could not read %s\n' "$h" "$f"
        fail=1
    elif ! version_ge "$fw_ver" "$fw_min"; then
        printf '[%s] ERROR: firmware version %s on %s < required %s\n' "$h" "$fw_ver" "$f" "$fw_min"
        fail=1
    fi
done
if [[ "$fw_found" -eq 0 ]]; then
    printf '[%s] ERROR: no tenstorrent devices found under /sys/class/tenstorrent\n' "$h"
    fail=1
elif [[ "$fail" -eq 0 ]]; then
    printf '[%s] OK: firmware version >= %s on all devices\n' "$h" "$fw_min"
fi

exit "$fail"
VERSION_CHECK_CMD

    mpirun --host "$hosts" \
        --mca btl_tcp_if_include "$mpi_if" \
        "${mpi_extra_args[@]}" \
        bash -c "$version_check_cmd" _ "$TT_SMI_MIN_VERSION" "$KMD_MIN_VERSION" "$FW_MIN_VERSION"
}

# Runs check_cluster_versions and prints standard status messages, deciding whether the caller
# should proceed. Distinguishes two failure modes from the check output:
#   - a version below the required minimum  -> hard failure (return 1: caller should abort)
#   - versions could not be read at all     -> can't verify, but don't block (yellow warning, return 0)
# This keeps recovery/validation running when e.g. tt_fw_bundle_ver is transiently unreadable
# (device busy) instead of failing the whole procedure. Returns 0 to proceed, 1 to abort.
#
# Usage: run_version_check_gate "$HOSTS" "$MPI_IF" [mpi_extra_args...]
run_version_check_gate() {
    local hosts="$1"
    local mpi_if="$2"
    shift 2
    local mpi_extra_args=("$@")

    local yellow=$'\033[1;33m'
    local reset=$'\033[0m'

    echo "Checking tt-smi/KMD/firmware versions on all hosts..."

    local out rc
    out=$(check_cluster_versions "$hosts" "$mpi_if" "${mpi_extra_args[@]}" 2>&1)
    rc=$?
    printf '%s\n' "$out"

    # A genuine version-below-minimum is always a hard failure. Use a bash pattern match rather than
    # a `... | grep -q` pipeline: under `set -o pipefail`, grep can close the pipe on an early match,
    # the upstream printf gets SIGPIPE, and the pipeline returns non-zero, masking the match.
    if [[ "$out" == *"< required"* ]]; then
        echo ""
        echo "Error: version check failed on one or more hosts (see above)."
        echo "       Required: tt-smi >= $TT_SMI_MIN_VERSION, KMD >= $KMD_MIN_VERSION, firmware >= $FW_MIN_VERSION."
        echo "       Re-run with --skip-version-check to bypass."
        return 1
    fi

    # Otherwise a non-zero result means some version could not be read (e.g. tt_fw_bundle_ver busy).
    # We can't verify the versions, but don't block the run — warn (yellow) and continue.
    if [[ $rc -ne 0 ]]; then
        printf '%s\n' ""
        printf '%sWarning: could not read tool/driver/firmware versions on one or more hosts (see above).%s\n' "$yellow" "$reset"
        printf '%s         Skipping the version check and continuing to the tests.%s\n' "$yellow" "$reset"
        printf '%s         Pass --skip-version-check to skip this check explicitly.%s\n' "$yellow" "$reset"
        printf '%s\n' ""
        return 0
    fi

    echo "Version check passed on all hosts."
    echo ""
    return 0
}
