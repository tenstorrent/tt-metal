#!/usr/bin/env bash
# Multi-process, no-MPI routing bring-up test for the fabric-manager ServiceCoordinator.
#
# Spins up one controller process and N agent processes (all instances of run_fabric_manager),
# wired together over TCP. Each agent loads its own mock cluster descriptor and its mesh binding,
# then builds a full ControlPlane (physical discovery + topology mapping + routing-table
# configuration) via --role routing-bringup, so every cross-host exchange (discovery, topology
# mapping, intermesh connectivity, router-port-directions) is routed through the controller instead
# of MPI. The test asserts that:
#   * every agent exits 0 and prints ROUTING_OK,
#   * every agent converged on the identical global fabric-mapping fingerprint (map_hash),
#   * the controller shuts down cleanly once all agents finish.
#
# Usage:
#   run_service_coordinator_routing_test.sh [--binary PATH] [--port PORT]
#
# Defaults to the 2-host dual_t3k_ci cluster with the dual_t3k mesh graph descriptor (two 2x4 meshes,
# one host each) -- the same mock the MPI MultiHost.TestDual2x4ControlPlaneInit test uses.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

BINARY="${REPO_ROOT}/build/tools/scaleout/run_fabric_manager"
PORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --help|-h)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Two-host dual-T3K: each agent owns one 2x4 mesh (mesh 0 / mesh 1), one host rank each.
MGD="${REPO_ROOT}/tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto"
DESCS=(
  "${REPO_ROOT}/tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc/dual_t3k_ci_cluster_desc_f10cs03_rank_0.yaml"
  "${REPO_ROOT}/tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc/dual_t3k_ci_cluster_desc_f10cs04_rank_1.yaml"
)
MESH_IDS=(0 1)
MESH_HOST_RANKS=(0 0)

N=${#DESCS[@]}
if [[ ! -x "$BINARY" ]]; then
  echo "error: run_fabric_manager binary not found/executable at: $BINARY" >&2
  echo "       build it with: ninja -C build run_fabric_manager" >&2
  exit 1
fi
if [[ ! -f "$MGD" ]]; then
  echo "error: mesh graph descriptor not found: $MGD" >&2
  exit 1
fi
for d in "${DESCS[@]}"; do
  if [[ ! -f "$d" ]]; then
    echo "error: mock cluster descriptor not found: $d" >&2
    exit 1
  fi
done

# Pick a free ephemeral port if the caller did not pin one (controller binds it; agents dial it).
if [[ "$PORT" == "0" ]]; then
  PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null \
        || echo $(( (RANDOM % 20000) + 20000 )))
fi

WORKDIR="$(mktemp -d)"
CONTROLLER_LOG="${WORKDIR}/controller.log"
cleanup() {
  [[ -n "${CONTROLLER_PID:-}" ]] && kill "${CONTROLLER_PID}" 2>/dev/null
  for pid in "${AGENT_PIDS[@]:-}"; do kill "$pid" 2>/dev/null; done
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "=== ServiceCoordinator multi-process routing bring-up test ==="
echo "binary : $BINARY"
echo "agents : $N"
echo "port   : $PORT"
echo "mgd    : $MGD"
echo "workdir: $WORKDIR"

cd "$REPO_ROOT"

# 1. Controller.
"$BINARY" --role controller --world-size "$N" --port "$PORT" >"$CONTROLLER_LOG" 2>&1 &
CONTROLLER_PID=$!

# Wait for the controller to announce it is listening (bounded).
for _ in $(seq 1 50); do
  grep -q "listening on port" "$CONTROLLER_LOG" && break
  if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    echo "error: controller exited early" >&2; cat "$CONTROLLER_LOG" >&2; exit 1
  fi
  sleep 0.1
done

# 2. Agents.
AGENT_PIDS=()
AGENT_LOGS=()
for ((i = 0; i < N; i++)); do
  log="${WORKDIR}/agent_${i}.log"
  AGENT_LOGS+=("$log")
  "$BINARY" --role routing-bringup \
    --controller "127.0.0.1:${PORT}" \
    --world-index "$i" --world-size "$N" \
    --mock-cluster-desc "${DESCS[$i]}" \
    --mesh-graph-desc "$MGD" \
    --mesh-id "${MESH_IDS[$i]}" --mesh-host-rank "${MESH_HOST_RANKS[$i]}" >"$log" 2>&1 &
  AGENT_PIDS+=($!)
done

# 3. Join agents, collect exit codes.
rc=0
for ((i = 0; i < N; i++)); do
  if ! wait "${AGENT_PIDS[$i]}"; then
    echo "error: agent $i exited non-zero" >&2
    rc=1
  fi
done

# 4. Join controller (it self-terminates once all agents complete).
if ! wait "$CONTROLLER_PID"; then
  echo "error: controller exited non-zero" >&2
  rc=1
fi

# 5. Validate outputs: every agent ROUTING_OK + identical map_hash fingerprint.
echo "--- agent output ---"
FIRST_FP=""
for ((i = 0; i < N; i++)); do
  line=$(grep "routing-bringup] agent" "${AGENT_LOGS[$i]}" | grep -E "ROUTING_OK|ROUTING_FAIL" | tail -1)
  echo "agent $i: ${line:-<no result line>}"
  if [[ "$line" != *"ROUTING_OK"* ]]; then
    echo "error: agent $i did not report ROUTING_OK" >&2
    rc=1
    continue
  fi
  fp="${line#*meshes=}"       # canonical fingerprint tail: "meshes=.. mapped_nodes=.. map_hash=0x.."
  fp="meshes=${fp}"
  if [[ -z "$FIRST_FP" ]]; then
    FIRST_FP="$fp"
  elif [[ "$fp" != "$FIRST_FP" ]]; then
    echo "error: agent $i fingerprint diverged" >&2
    echo "       expected: $FIRST_FP" >&2
    echo "       got:      $fp" >&2
    rc=1
  fi
done

if [[ $rc -eq 0 ]]; then
  echo "=== PASS: $N agents converged on identical global fabric mapping ($FIRST_FP) with no MPI ==="
else
  echo "=== FAIL (see errors above) ===" >&2
  echo "--- controller log ---" >&2
  cat "$CONTROLLER_LOG" >&2
  for ((i = 0; i < N; i++)); do
    echo "--- agent $i log (tail) ---" >&2
    tail -n 30 "${AGENT_LOGS[$i]}" >&2
  done
fi
exit $rc
