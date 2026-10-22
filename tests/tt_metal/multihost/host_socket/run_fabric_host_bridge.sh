#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# E2H2H2E on one 4-chip Blackhole box: two ranks, one 1x2 mesh each, the cables between the meshes
# host-bridged. Rank r gets chips from HOST_SOCKET_VISIBLE_DEVICES ("a,b:c,d", default "0,1:2,3") and
# mesh id r (rank_env.sh).
#
#   tests/tt_metal/multihost/host_socket/run_fabric_host_bridge.sh [extra gtest args]
#
# Env: HOST_SOCKET_MPIRUN (default the ULFM one the binary links), HOST_SOCKET_VISIBLE_DEVICES,
#      HOST_SOCKET_TIMEOUT (seconds, default 600), TT_METAL_FABRIC_HOST_BRIDGE (default intermesh),
#      TT_HOST_SOCKET_BRIDGE_WAIT_S (stall watchdog, default 60).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$ROOT}"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$ROOT}"
MPIRUN="${HOST_SOCKET_MPIRUN:-/opt/openmpi-v5.0.7-ulfm/bin/mpirun}"
[[ -x "$MPIRUN" ]] || MPIRUN=$(command -v mpirun) || { echo "error: no mpirun" >&2; exit 2; }
BIN="${HOST_SOCKET_BIN:-$ROOT/build/test/tt_metal/multi_host_socket_transport_tests}"
[[ -x "$BIN" ]] || { echo "error: $BIN not built (target multi_host_socket_transport_tests)" >&2; exit 2; }

export TT_MESH_GRAPH_DESC_PATH="${HOST_SOCKET_MGD:-$HERE/config/two_bh_1x2_intermesh_mgd.textproto}"
export TT_VISIBLE_DEVICES_PER_RANK="${HOST_SOCKET_VISIBLE_DEVICES:-0,1:2,3}"
export TT_METAL_FABRIC_HOST_BRIDGE="${TT_METAL_FABRIC_HOST_BRIDGE:-intermesh}"
# Bridged routers run single-erisc: the second erisc's PCIe writes are never acked on Blackhole.
export TT_METAL_DISABLE_MULTI_AERISC=1
RUN_TIMEOUT="${HOST_SOCKET_TIMEOUT:-600}"

echo "== mpirun=$MPIRUN bin=$BIN chips=$TT_VISIBLE_DEVICES_PER_RANK bridge=$TT_METAL_FABRIC_HOST_BRIDGE"
timeout --kill-after=30s "$RUN_TIMEOUT" \
    "$MPIRUN" -n 2 --host localhost:2 --oversubscribe --bind-to none --allow-run-as-root --tag-output \
    -x TT_METAL_HOME -x TT_METAL_RUNTIME_ROOT -x TT_MESH_GRAPH_DESC_PATH -x TT_VISIBLE_DEVICES_PER_RANK \
    -x TT_METAL_FABRIC_HOST_BRIDGE -x TT_METAL_DISABLE_MULTI_AERISC \
    ${TT_METAL_WATCHER:+-x TT_METAL_WATCHER} ${TT_METAL_LOGGER_LEVEL:+-x TT_METAL_LOGGER_LEVEL} \
    "$HERE/rank_env.sh" "$BIN" --gtest_filter='FabricHostBridgeTest.*' "$@"
