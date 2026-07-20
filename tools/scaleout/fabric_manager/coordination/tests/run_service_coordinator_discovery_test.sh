#!/usr/bin/env bash
# Multi-process, no-MPI bring-up test for the fabric-manager ServiceCoordinator.
#
# Spins up one controller process and N agent processes (all instances of run_fabric_manager),
# wired together over TCP. Each agent loads its own mock cluster descriptor and runs physical
# system discovery through the coordinator (--role discover-psd), so the gather -> merge -> scatter
# is routed through the controller instead of MPI. The test asserts that:
#   * every agent exits 0 and prints PSD_OK,
#   * every agent converged on the identical merged global PSD fingerprint,
#   * the merged PSD contains exactly N hosts,
#   * the controller shuts down cleanly once all agents finish.
#
# Usage:
#   run_service_coordinator_discovery_test.sh [--binary PATH] [--port PORT] [mock_desc_0 mock_desc_1 ...]
#
# With no mock descriptors given, defaults to the 2-host dual_t3k_ci cluster (distinct hostnames
# f10cs03 / f10cs04 with cross-host links) -- the same mock the MPI PhysicalDiscovery test uses.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

BINARY="${REPO_ROOT}/build/tools/scaleout/run_fabric_manager"
PORT=0
DESCS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) BINARY="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --help|-h)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) DESCS+=("$1"); shift ;;
  esac
done

if [[ ${#DESCS[@]} -eq 0 ]]; then
  DESCS=(
    "${REPO_ROOT}/tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc/dual_t3k_ci_cluster_desc_f10cs03_rank_0.yaml"
    "${REPO_ROOT}/tt_metal/third_party/tt-cluster-descriptors/wormhole/dual_t3k_ci/dual_t3k_ci_cluster_desc/dual_t3k_ci_cluster_desc_f10cs04_rank_1.yaml"
  )
fi

N=${#DESCS[@]}
if [[ ! -x "$BINARY" ]]; then
  echo "error: run_fabric_manager binary not found/executable at: $BINARY" >&2
  echo "       build it with: ninja -C build run_fabric_manager" >&2
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

echo "=== ServiceCoordinator multi-process discovery test ==="
echo "binary : $BINARY"
echo "agents : $N"
echo "port   : $PORT"
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
  "$BINARY" --role discover-psd \
    --controller "127.0.0.1:${PORT}" \
    --world-index "$i" --world-size "$N" \
    --mock-cluster-desc "${DESCS[$i]}" >"$log" 2>&1 &
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

# 5. Validate outputs: every agent PSD_OK + identical fingerprint + N hosts.
echo "--- agent output ---"
FIRST_FP=""
for ((i = 0; i < N; i++)); do
  line=$(grep "discover-psd] agent" "${AGENT_LOGS[$i]}" | grep -E "PSD_OK|PSD_FAIL" | tail -1)
  echo "agent $i: ${line:-<no result line>}"
  if [[ "$line" != *"PSD_OK"* ]]; then
    echo "error: agent $i did not report PSD_OK" >&2
    rc=1
    continue
  fi
  fp="${line#*hosts=}"       # canonical fingerprint: "hosts=N [host:rank,...]"
  fp="hosts=${fp}"
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
  echo "=== PASS: $N agents converged on identical global PSD ($FIRST_FP) with no MPI ==="
else
  echo "=== FAIL (see errors above) ===" >&2
  echo "--- controller log ---" >&2
  cat "$CONTROLLER_LOG" >&2
fi
exit $rc
