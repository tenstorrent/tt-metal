#!/usr/bin/env bash
#SBATCH --job-name=hsock
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# Runs run_host_socket_tests.sh once per mode named on the command line, all
# inside one two-node allocation, so a full qualification is one job:
#
#   cd tests/tt_metal/multihost/host_socket
#   sbatch -p <galaxy-partition> run_modes_exabox.sbatch smoke latency burst perf
#   TT_HOST_SOCKET_NUM_CORES=4 sbatch -p <galaxy-partition> run_modes_exabox.sbatch perf
#
# TT_METAL_HOME is taken from the environment, else found by walking up from the
# submit directory. Build on a compute node (or in the 22.04 CI container): the
# login hosts carry a newer glibc than the galaxy nodes.
set -uo pipefail
if [[ -z "${TT_METAL_HOME:-}" ]]; then
    candidate="$(cd "${SLURM_SUBMIT_DIR:-.}" && pwd)"
    while [[ "$candidate" != "/" ]]; do
        if [[ -d "$candidate/tt_metal" && -d "$candidate/tests/tt_metal/multihost/host_socket" ]]; then
            TT_METAL_HOME="$candidate"
            break
        fi
        candidate="$(dirname "$candidate")"
    done
fi
[[ -n "${TT_METAL_HOME:-}" ]] || { echo "error: set TT_METAL_HOME or submit from inside the repo" >&2; exit 2; }
export TT_METAL_HOME
cd "$TT_METAL_HOME" || exit 2
echo "== nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd' ')"
echo "== commit: $(git rev-parse --short HEAD 2>/dev/null)"
rc=0
for mode in "$@"; do
    echo; echo "################ MODE $mode ################"; date
    tests/tt_metal/multihost/host_socket/run_host_socket_tests.sh "$mode" || rc=$?
done
echo; echo "== all modes done rc=$rc"; date
exit $rc
