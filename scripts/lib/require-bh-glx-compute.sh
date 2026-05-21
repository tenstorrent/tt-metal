#!/usr/bin/env bash
# Abort if not running on a bh-glx Galaxy compute node.
require_bh_glx_compute() {
    local host
    host="$(hostname -s 2>/dev/null || hostname)"
    if [[ "$host" != bh-glx-* ]]; then
        echo "ERROR: this script must run on a bh-glx compute node (got: ${host})." >&2
        echo "From the login node, submit only:" >&2
        echo "  ./scripts/submit-nkapre-parity-slurm.sh" >&2
        echo "Or interactively:" >&2
        echo "  salloc --partition bh_sc5_B2B9_D12 --nodelist bh-glx-b06u08" >&2
        echo "  srun --pty /bin/bash" >&2
        echo "  ./scripts/slurm-nkapre-parity-job.sh" >&2
        exit 1
    fi
}
