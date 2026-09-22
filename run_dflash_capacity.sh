#!/usr/bin/env bash
# SCRATCH -- issue #56487 experiment. Untracked on purpose; do not commit, do not add to a PR.
#
# Max concurrent users on a Galaxy quad, with and without the DFlash drafter cache.
#
#   ./run_dflash_capacity.sh dflash        # drafter on  (expect the LAST rank to bind)
#   ./run_dflash_capacity.sh plain         # verifier only (expect rank 0 to bind)
#
# Both arms run PREFILL_USE_TRACE=0 so the delta is the drafter cache alone -- the dflash path is
# not trace-captured, so a trace-on comparison would also move 256 MB/chip of trace region.
set -euo pipefail

ARM="${1:-dflash}"
MODULE="${MODULE:-dflash_capacity_probe}"   # or dflash_forward_pressure
BASE_USERS="${BASE_USERS:-1}"
HOSTS="${PREFILL_HOSTS:-bh-glx-120-b06u02,bh-glx-120-b06u08,bh-glx-120-b07u02,bh-glx-120-b07u08}"
CONFIG="${CONFIG:-sc4}"

export TT_METAL_HOME="${TT_METAL_HOME:-/data/nmilicevic/tt-metal}"
export PYTHONPATH="${TT_METAL_HOME}"
MANIFEST_DIR="${TT_METAL_HOME}/models/demos/deepseek_v3_d_p/tt/runners/manifests"
MGD="${TT_METAL_HOME}/models/demos/common/prefill/runners/topology_configuration/ci/${CONFIG}_mgd.textproto"

case "${ARM}" in
  dflash) MANIFEST="${MANIFEST_DIR}/kimi27_dflash.json" ;;
  plain)  MANIFEST="${MANIFEST_DIR}/kimi27.json" ;;
  *) echo "usage: $0 [dflash|plain]" >&2; exit 2 ;;
esac
[ -f "${MGD}" ] || { echo "missing ${MGD}" >&2; exit 2; }

OUT="${OUT:-/data/nmilicevic/dflash_capacity}/${MODULE#dflash_}_${ARM}_${CONFIG}_$(date +%m%d-%H%M%S)"
RANKLOGS="${OUT}/ranklogs"
mkdir -p "${RANKLOGS}" "${OUT}/ttrun-cwd"

# prterun reparents away from ttrun, so a killed launcher otherwise leaves the ranks holding the mesh.
cleanup() {
  if pgrep -f "prterun.*${OUT}" >/dev/null 2>&1; then
    pkill -TERM -f "prterun.*${OUT}" 2>/dev/null || true
    for _ in $(seq 1 30); do pgrep -f "prterun.*${OUT}" >/dev/null 2>&1 || break; sleep 1; done
    pkill -KILL -f "prterun.*${OUT}" 2>/dev/null || true
  fi
  echo "==================== CAPACITY ===================="
  grep -ahE "CAPACITY rank=|COMBINED rank=|SERVED rank=" "${RANKLOGS}"/* 2>/dev/null | sed -E "s/.*(CAPACITY|COMBINED|SERVED)/\1/" | sort -u \
    || echo "no CAPACITY lines -- check ${RANKLOGS}"
  echo "logs: ${OUT}"
}
trap cleanup EXIT INT TERM

cd "${OUT}/ttrun-cwd"
python3 "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
  --skip-executable-check \
  --tcp-interface "${PREFILL_TCP_IFACE:-ens5f0np0}" \
  --mesh-graph-descriptor "${MGD}" \
  --hosts "${HOSTS}" \
  --mpi-args "--bind-to none --tag-output --wdir ${TT_METAL_HOME} --output-filename ${RANKLOGS}/probe -x PATH -x LD_LIBRARY_PATH" \
  bash -lc "cd '${TT_METAL_HOME}'; \
    export PYTHONPATH='${TT_METAL_HOME}'; \
    export PYTHONUNBUFFERED=1; \
    export PREFILL_MANIFEST='${MANIFEST}'; \
    export PREFILL_CHUNK_SIZE=5120; \
    export PREFILL_MAX_SEQ_LEN=256000; \
    export PREFILL_NUM_USERS=${BASE_USERS}; \
    export PREFILL_USE_TRACE=0; \
    export PREFILL_CAPACITY_PROBE_COMPILE=${PREFILL_CAPACITY_PROBE_COMPILE:-1}; \
    export PREFILL_PRESSURE_POSITIONS=${PREFILL_PRESSURE_POSITIONS:-6}; \
    export PREFILL_PRESSURE_MODE=${PREFILL_PRESSURE_MODE:-explore}; \
    export PREFILL_PRESSURE_FROM=${PREFILL_PRESSURE_FROM:-0}; \
    export PREFILL_PRESSURE_TO=${PREFILL_PRESSURE_TO:-96}; \
    export LOGURU_LEVEL=INFO; \
    exec python3 -m ${MODULE}"
