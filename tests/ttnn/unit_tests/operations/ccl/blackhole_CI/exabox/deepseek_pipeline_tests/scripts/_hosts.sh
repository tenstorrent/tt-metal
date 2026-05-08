#!/bin/bash
# Shared host list for the single-pod scripts. Sourced by reset_chips.sh,
# recover_hung_run.sh, bootstrap_pipeline_dir.sh, _run_common.sh.
#
# Override per-shell via HOSTS="..." (space- OR comma-separated)
# if the 4-host bindings change. Both forms are accepted; we normalize to
# space-separated below so `for h in $HOSTS` iterates correctly.
HOSTS="${HOSTS:-bh-glx-110-c07u02 bh-glx-110-c07u08 bh-glx-110-c08u02 bh-glx-110-c08u08}"

# Normalize: commas → spaces, collapse multiple spaces, strip leading/trailing
HOSTS="$(echo "$HOSTS" | tr ',' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//')"
