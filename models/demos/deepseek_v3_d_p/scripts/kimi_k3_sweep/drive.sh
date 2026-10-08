#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# drive.sh <rundir> <runner users> <phase>...   phase = tag:users:isl[:prefix]
# Run on HOSTS' first host (rank 0 owns the H2D service the producer attaches to).
# One bringup, then the phases in order. Each producer must finish and the last rank drain it before
# the next starts: two producers on the H2D stream at once corrupt the chunk metadata.
set -uo pipefail
D=${1:?rundir}; RU=${2:?runner users}; shift 2
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOME_DIR=$(cd "$S/../../../../.." && pwd)
cd "$HOME_DIR"
source python_env/bin/activate
# Pipeline order must follow the physical galaxy ring; any rotation or reversal of it works.
read -r -a RING <<< "${HOSTS:-bh-glx-120-b06u08 bh-glx-120-b06u02 bh-glx-120-b07u02 bh-glx-120-b07u08}"
IFACE=${TCP_IFACE:-ens5f0np0}
SCRATCH=${SCRATCH:-/var/tmp/$USER-k3}  # host-local: a shared JIT cache dies on NFS ESTALE
HOSTLIST=$(printf '%s:1,' "${RING[@]}"); HOSTLIST=${HOSTLIST%,}
SSH="ssh -o BatchMode=yes"
mkdir -p "$D"; rm -rf "$D/timing" "$D/phases.txt"; mkdir -p "$D/timing"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
stop_runner() {
  for h in "${RING[@]}"; do timeout 60 $SSH $h "pkill -9 -f '[p]refill_runner'; pkill -9 -f '[p]refill_producer'; pkill -9 -f '[t]trun.py'; pkill -9 -f '[p]rterun'; pkill -9 -f '[p]rted'; exit 0" >/dev/null 2>&1; done
}
log "start users=$RU hosts=$HOSTLIST phases=$*"
stop_runner; sleep 4
for h in "${RING[@]}"; do timeout 30 $SSH $h "rm -f /dev/shm/tt_socket_manifest_* /dev/shm/tt_prefill_layer_completion_ring_* /dev/shm/*prefill* /dev/shm/tt_h2d_*; mkdir -p $SCRATCH/jit $SCRATCH/cc" >/dev/null 2>&1; done
# All four in parallel, or the fabric fails with "Fabric Router Sync: Timeout".
for h in "${RING[@]}"; do (timeout 550 $SSH $h 'tt-smi -glx_reset' >/dev/null 2>&1) & done
wait
for h in "${RING[@]}"; do
  n=$(timeout 60 $SSH $h 'ls /dev/tenstorrent 2>/dev/null | wc -l' 2>/dev/null)
  [ "${n:-0}" -ge 32 ] || { log "$h devices=$n, aborting"; exit 1; }
done
sed -e "s#@USERS@#$RU#" -e "s#@DIR@#$D#" -e "s#@HOME@#$HOME_DIR#g" -e "s#@SCRATCH@#$SCRATCH#" \
    -e "s#@ACK3@#${ACK3:-1}#" "$S/runner.yaml.in" > "$D/runner.yaml"
RLOG=$D/runner.log
export TT_METAL_HOME=$HOME_DIR PP_TT_METAL_CACHE=$SCRATCH/jit TMPDIR=$SCRATCH/cc
export PREFILL_MANIFEST=models/demos/deepseek_v3_d_p/tt/runners/manifests/kimi_k3.json PREFILL_MODEL=kimi_k3
setsid nohup ./models/demos/common/prefill/runners/run_pipeline_prefill.sh "$D/runner.yaml" "$HOSTLIST" "$IFACE" > "$RLOG" 2>&1 < /dev/null &
FATAL="Out of Memory|bad_alloc|TT_FATAL|TT_THROW|Bus error|Traceback|not mapped|Router Sync|usable links"
ready=0
for i in $(seq 1 720); do
  sleep 10
  [ "$(grep -ac 'request (unbounded) loop start' "$RLOG")" -ge 4 ] && { ready=1; break; }
  grep -qaE "$FATAL" "$RLOG" && break
  pgrep -f "[t]trun.py" >/dev/null || break
done
grep -a "DRAM " "$RLOG" | sed 's/.*\[pp rank/[pp rank/' | sort
[ $ready = 1 ] || { log "runner not ready"; grep -aoE "($FATAL)[^|]{0,200}" "$RLOG" | sort -u | head -8; stop_runner; exit 1; }
log "runner ready"
rows() { cat "$D/timing/rank3.csv" 2>/dev/null | wc -l; }
for ph in "$@"; do
  IFS=: read -r TAG PU ISL PREFIX <<< "$ph"; PREFIX=${PREFIX:-0}
  want=$(( $(rows) + (ISL + 5119) / 5120 * PU ))
  echo "$TAG $(for r in 0 1 2 3; do cat "$D/timing/rank$r.csv" 2>/dev/null | wc -l; done | paste -sd,) $PU $ISL $PREFIX" >> "$D/phases.txt"
  log "phase $TAG users=$PU isl=$ISL prefix=$PREFIX"
  bash "$S/producer.sh" "$D" "$TAG" "$PU" "$ISL" "$PREFIX" &
  pid=$!; ok=0
  for i in $(seq 1 2400); do
    sleep 5
    [ "$(rows)" -ge $want ] && ! kill -0 $pid 2>/dev/null && { ok=1; break; }
    grep -qaE "$FATAL" "$RLOG" && break
    if ! kill -0 $pid 2>/dev/null && ! grep -qa "DONE wall" "$D/producer_$TAG.log"; then break; fi
  done
  [ $ok = 1 ] || { log "phase $TAG failed"; tail -5 "$D/producer_$TAG.log"; break; }
  log "phase $TAG done"
  sleep 10
done
echo "end $(for r in 0 1 2 3; do cat "$D/timing/rank$r.csv" 2>/dev/null | wc -l; done | paste -sd,)" >> "$D/phases.txt"
stop_runner
python3 "$S/report.py" "$D"
