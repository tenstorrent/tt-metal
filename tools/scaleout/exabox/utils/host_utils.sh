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
