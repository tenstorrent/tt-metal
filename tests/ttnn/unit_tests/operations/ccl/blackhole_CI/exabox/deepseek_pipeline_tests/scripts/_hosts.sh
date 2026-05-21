#!/bin/bash
# Shared host-list parser. Sourced by reset_chips.sh, recover_hung_run.sh,
# bootstrap_pipeline_dir.sh, run_pipeline_smoke.sh.
#
# HOSTS must be set by the caller — there is no default. Cluster
# membership is operator-specific; baking a host list into the repo
# would silently target the wrong machines on any other cluster.
#
# Set per-shell:
#   export HOSTS="hostA hostB hostC hostD"     # space-separated
#   export HOSTS="hostA,hostB,hostC,hostD"     # comma-separated also OK
#
: "${HOSTS:?HOSTS is not set; export HOSTS=\"<space- or comma-separated 4-host list>\" first}"

# Normalize: commas → spaces, collapse multiple spaces, strip leading/trailing
HOSTS="$(echo "$HOSTS" | tr ',' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//')"

# Empty after normalization (e.g. HOSTS=" ") still counts as unset.
if [ -z "$HOSTS" ]; then
    echo "ERROR: HOSTS is empty after normalization. Set it to a non-empty host list." >&2
    return 1 2>/dev/null || exit 1
fi
