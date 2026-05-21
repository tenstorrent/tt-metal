#!/usr/bin/env bash
# Abort unless running on a bh-glx Galaxy compute node (never the login node).
require_bh_glx_compute() {
    local host
    host="$(hostname -s 2>/dev/null || hostname)"
    if [[ "$host" == bh-glx-* ]]; then
        return 0
    fi

    echo "ERROR: this must run on a bh-glx Galaxy compute node (got: ${host})." >&2
    echo "Do not run parity, multiprocess, or LLK sweeps on the login node." >&2
    echo "From the login node, submit Slurm jobs only:" >&2
    echo "  ./scripts/submit-nkapre-parity-ttsim-slurm.sh   # full 57-suite parity" >&2
    echo "  ./scripts/submit-nkapre-llk-ttsim-slurm.sh      # LLK weekly + nightly WH" >&2
    echo "  sbatch scripts/slurm-nkapre-mp-ttsim-job.sh       # multiprocess only" >&2
    echo "  sbatch scripts/slurm-llk-smoke-ttsim-job.sh       # quick LLK smoke" >&2
    echo "Or allocate Galaxy interactively:" >&2
    echo "  salloc --partition bh_sc5_B2B9_D12 --nodelist bh-glx-b06u08 --cpus-per-task=1" >&2
    echo "  srun --pty /bin/bash" >&2
    exit 1
}
