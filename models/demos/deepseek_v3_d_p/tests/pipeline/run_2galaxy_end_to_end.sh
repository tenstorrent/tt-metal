#!/usr/bin/env bash
# End-to-end 2-galaxy cross-mesh socket smoke + perf.
#
# Does what you'd otherwise type by hand:
#   1. Validate prerequisites (HAL bump, SSH, validation tool built)
#   2. Run tools/scaleout/exabox/recover.sh N times back-to-back to train
#      the inter-galaxy ETH PHYs (a single recover.sh call sometimes isn't
#      enough — empirically 2 with --num-iterations 10 each is reliable).
#   3. Launch runme_2galaxy_cross_mesh_smoke.sh on the trained fabric.
#
# Default hosts/interface match the d04 reservation; override via flags.

set -eo pipefail

HOST_A="bh-glx-d04u02"
HOST_B="bh-glx-d04u08"
TT_TCP_INTERFACE="ens5f0np0"
NUM_RECOVERY_PASSES=2
RECOVERY_ITERS=10
SKIP_RECOVERY=false

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

End-to-end 2-galaxy cross-mesh socket smoke + perf, including the
inter-galaxy ETH PHY training step.

Options:
    --host-a HOST           First host (rank 0, mesh_id=0). Default: ${HOST_A}
    --host-b HOST           Second host (rank 1, mesh_id=1). Default: ${HOST_B}
    --tcp-interface IFACE   NIC for inter-host traffic. Default: ${TT_TCP_INTERFACE}
    --recovery-passes N     Times to run recover.sh back-to-back. Default: ${NUM_RECOVERY_PASSES}
    --recovery-iters N      --num-iterations passed to each recover.sh. Default: ${RECOVERY_ITERS}
    --skip-recovery         Skip recovery (use only if you know links are trained)
    -h, --help              Show this help

Example:
    $0 --host-a bh-glx-d04u02 --host-b bh-glx-d04u08
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-a)           HOST_A="$2"; shift 2 ;;
        --host-b)           HOST_B="$2"; shift 2 ;;
        --tcp-interface)    TT_TCP_INTERFACE="$2"; shift 2 ;;
        --recovery-passes)  NUM_RECOVERY_PASSES="$2"; shift 2 ;;
        --recovery-iters)   RECOVERY_ITERS="$2"; shift 2 ;;
        --skip-recovery)    SKIP_RECOVERY=true; shift ;;
        -h|--help)          show_help; exit 0 ;;
        *)                  echo "ERROR: unknown arg: $1" >&2; show_help >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TT_METAL_ROOT="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"
TT_BLAZE_ROOT="$(cd "${TT_METAL_ROOT}/.." && pwd)"

RUNME="${SCRIPT_DIR}/runme_2galaxy_cross_mesh_smoke.sh"
RECOVER="${TT_METAL_ROOT}/tools/scaleout/exabox/recover.sh"
VALIDATION_BIN="${TT_METAL_ROOT}/build/tools/scaleout/run_cluster_validation"
HAL_HEADER="${TT_METAL_ROOT}/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h"

echo "=========================================="
echo "2-galaxy cross-mesh socket smoke (end-to-end)"
echo "  HOST_A             = ${HOST_A}"
echo "  HOST_B             = ${HOST_B}"
echo "  TT_TCP_INTERFACE   = ${TT_TCP_INTERFACE}"
echo "  recovery passes    = ${NUM_RECOVERY_PASSES} x --num-iterations=${RECOVERY_ITERS}"
echo "  skip recovery      = ${SKIP_RECOVERY}"
echo "  tt-metal root      = ${TT_METAL_ROOT}"
echo "=========================================="
echo ""

# --- Prereqs ---------------------------------------------------------------

echo "[prereq] checking HAL bump (MEM_ERISC_KERNEL_CONFIG_SIZE)..."
if ! grep -q "MEM_ERISC_KERNEL_CONFIG_SIZE (32 \* 1024)" "${HAL_HEADER}"; then
    echo "ERROR: ${HAL_HEADER} does not have the 32 KB bump." >&2
    echo "       Check out jjovicic/socket-experiment and rebuild tt-metal." >&2
    exit 1
fi
echo "[prereq] OK"

echo "[prereq] checking launcher + recover.sh + validation tool..."
[[ -f "${RUNME}" ]]          || { echo "ERROR: missing ${RUNME}" >&2; exit 1; }
[[ -f "${RECOVER}" ]]        || { echo "ERROR: missing ${RECOVER}" >&2; exit 1; }
if [[ "${SKIP_RECOVERY}" != true && ! -x "${VALIDATION_BIN}" ]]; then
    echo "ERROR: ${VALIDATION_BIN} not built." >&2
    echo "       Build it (scaleout tools must be enabled), or rerun with --skip-recovery" >&2
    echo "       if you've trained the ETH PHYs another way." >&2
    exit 1
fi
echo "[prereq] OK"

echo "[prereq] checking SSH to both hosts in BatchMode (mpirun's launcher path)..."
# mpirun spawns remote MPI procs by ssh-ing from THIS host to each --host entry,
# so what we need to verify is that this host can reach both A and B without
# interactive prompts. We don't need to test "A reaching B" — mpirun never does
# that (it launches from here to both, not via a hop).
ssh -o BatchMode=yes -o ConnectTimeout=5 "${HOST_A}" hostname >/dev/null 2>&1 \
    || { echo "ERROR: cannot SSH to ${HOST_A} from here without password" >&2; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=5 "${HOST_B}" hostname >/dev/null 2>&1 \
    || { echo "ERROR: cannot SSH to ${HOST_B} from here without password" >&2; exit 1; }
echo "[prereq] OK"

# --- Recovery -------------------------------------------------------------

if [[ "${SKIP_RECOVERY}" == true ]]; then
    echo ""
    echo "[recovery] SKIPPED (--skip-recovery)"
else
    echo ""
    echo "[recovery] Running recover.sh ${NUM_RECOVERY_PASSES} time(s) (each --num-iterations ${RECOVERY_ITERS})"
    cd "${TT_METAL_ROOT}"
    for pass in $(seq 1 "${NUM_RECOVERY_PASSES}"); do
        echo ""
        echo "[recovery] pass ${pass}/${NUM_RECOVERY_PASSES}"
        echo "----------------------------------------------------------"
        if ! "${RECOVER}" \
                --hosts "${HOST_A},${HOST_B}" \
                --num-iterations "${RECOVERY_ITERS}"; then
            echo "[recovery] pass ${pass} reported failures — continuing to next pass" >&2
        fi
    done
    echo ""
    echo "[recovery] all passes done"
fi

# --- Test -----------------------------------------------------------------

echo ""
echo "[test] launching runme_2galaxy_cross_mesh_smoke.sh"
echo "       NOTE: do not run tt-smi -glx_reset between this and recovery —"
echo "       that would undo the ETH PHY training."
echo "----------------------------------------------------------"

cd "${TT_BLAZE_ROOT}"
source ./env.sh

HOST_A="${HOST_A}" \
HOST_B="${HOST_B}" \
TT_TCP_INTERFACE="${TT_TCP_INTERFACE}" \
    bash "${RUNME}"
