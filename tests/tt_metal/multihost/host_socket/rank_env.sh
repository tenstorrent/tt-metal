#!/usr/bin/env bash
# Per-rank env shim: mpirun's -x gives every rank the same value, but TT_MESH_ID
# must differ. Any VAR_PER_RANK="a:b" is split on ':' and the rank's element
# exported as VAR.
#
# Used instead of mpirun's MPMD (colon) form, which does not reliably honour one
# --host per segment.
set -u

RANK="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PMI_RANK:-0}}}"
export TT_MESH_ID="$RANK"
export TT_MESH_HOST_RANK=0

pick() {  # var_name, colon_list
    local name="$1" list="$2"
    IFS=':' read -r -a parts <<< "$list"
    local idx=$RANK
    (( idx < ${#parts[@]} )) || idx=$(( ${#parts[@]} - 1 ))
    export "$name=${parts[$idx]}"
}

[[ -n "${TT_VISIBLE_DEVICES_PER_RANK:-}" ]] && {
    pick TT_VISIBLE_DEVICES "$TT_VISIBLE_DEVICES_PER_RANK"
    export TT_METAL_VISIBLE_DEVICES="$TT_VISIBLE_DEVICES"
}

echo "rank $RANK on $(hostname): mesh=$TT_MESH_ID chip=${TT_VISIBLE_DEVICES:-unset}" >&2
exec "$@"
