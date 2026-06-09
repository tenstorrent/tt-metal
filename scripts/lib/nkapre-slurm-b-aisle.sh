#!/usr/bin/env bash
# Galaxy Slurm helpers for nkapre parity jobs.
# Allowed: B-aisle (bh-glx-b*) and 110-row C nodes (bh-glx-110-c*).
# Not allowed: legacy C-aisle (bh-glx-c* without 110 prefix).

NKAPRE_SLURM_PARTITION="${NKAPRE_SLURM_PARTITION:-bh_sc5_B2B9_D12}"
NKAPRE_110_PARTITION="${NKAPRE_110_PARTITION:-bh_sp_4x32_110_C1_C10}"

is_legacy_c_aisle_node() {
    local node="$1"
    [[ "$node" == bh-glx-c* && "$node" != bh-glx-110-c* ]]
}

is_allowed_nkapre_node() {
    local node="$1"
    [[ "$node" == bh-glx-b* || "$node" == bh-glx-110-c* ]]
}

validate_nkapre_node() {
    local node="$1"
    if is_legacy_c_aisle_node "$node"; then
        echo "ERROR: legacy C-aisle nodes (bh-glx-c*, not 110-row) cannot run nkapre parity (got: ${node})." >&2
        return 1
    fi
    if [[ -n "$node" ]] && ! is_allowed_nkapre_node "$node"; then
        echo "ERROR: node must be bh-glx-b* or bh-glx-110-c* (got: ${node})." >&2
        return 1
    fi
    return 0
}

partition_for_node() {
    local node="$1"
    if [[ "$node" == bh-glx-110-c* ]]; then
        echo "${NKAPRE_110_PARTITION}"
    else
        echo "${NKAPRE_SLURM_PARTITION}"
    fi
}

pick_idle_nkapre_node() {
    local exclude_csv="${1:-}"
    local -a exclude_arr=()
    if [[ -n "$exclude_csv" ]]; then
        IFS=',' read -r -a exclude_arr <<<"$exclude_csv"
    fi

    _pick_from_partition() {
        local part="$1"
        local pattern="$2"
        sinfo -N -p "$part" -t idle -h -o '%N' | tr ' ' '\n' | while read -r n; do
            [[ -n "$n" ]] || continue
            [[ "$n" == $pattern ]] || continue
            validate_nkapre_node "$n" || continue
            local skip=0 ex
            for ex in "${exclude_arr[@]}"; do
                [[ "$n" == "$ex" ]] && skip=1 && break
            done
            [[ "$skip" -eq 0 ]] && echo "$n" && break
        done
    }

    local picked
    picked="$(_pick_from_partition "${NKAPRE_SLURM_PARTITION}" "bh-glx-b*")"
    if [[ -n "$picked" ]]; then
        echo "$picked"
        return 0
    fi
    _pick_from_partition "${NKAPRE_110_PARTITION}" "bh-glx-110-c*"
}

# Back-compat aliases
reject_c_aisle_node() { validate_nkapre_node "$@"; }
pick_idle_b_aisle_node() { pick_idle_nkapre_node "$@"; }
