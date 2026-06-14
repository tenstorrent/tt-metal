#!/usr/bin/env bash
# Run a command inside the Qwen3.6-27B P300x2 test container with TT devices attached.
#
#   ./run.sh <NUM_DEVICES> <command...>
#
# NUM_DEVICES: how many TT chips to expose (1..4). Defaults to 4 (P300x2 = 4 chips).
# Examples:
#   ./run.sh 1 python3 -m models.demos.qwen36_27b.demo.demo --dummy-weights --max-layers 4 --prompt Hi
#   ./run.sh 4 python3 -m models.demos.qwen36_27b.evaluation.run_perf_bench
set -euo pipefail

WT=/home/ttuser/ttwork/tt-metal-qwen36
WEIGHTS=/home/ttuser/ttwork/qwen36-weights
IMAGE=qwen36-test:latest

N="${1:-4}"; shift || true
[ "$#" -eq 0 ] && set -- bash

DEV_ARGS=()
for i in $(seq 0 $((N-1))); do
  DEV_ARGS+=(--device "/dev/tenstorrent/$i")
done

# P300 boards are a CUSTOM fabric cluster: ttnn requires a mesh-graph-descriptor
# path (TT_MESH_GRAPH_DESC_PATH) or device open fails with
# "Custom fabric mesh graph descriptor path must be specified". Pick the descriptor
# that matches the exposed chip count: 1->p150 (1x1), 2->p300 (1x2), 4->p300_x2 (2x2).
MGD_DIR="$WT/tt_metal/fabric/mesh_graph_descriptors"
case "$N" in
  1) MGD="$MGD_DIR/p150_mesh_graph_descriptor.textproto" ;;
  2) MGD="$MGD_DIR/p300_mesh_graph_descriptor.textproto" ;;
  4) MGD="$MGD_DIR/p300_x2_mesh_graph_descriptor.textproto" ;;
  *) echo "unsupported NUM_DEVICES=$N (use 1, 2, or 4)" >&2; exit 1 ;;
esac

# All 4 chips report ETH_LIVE_STATUS=0x0 (no live eth links). On a single chip,
# firmware init waits on the dangling on-board eth core to its (unexposed) partner;
# SKIP_ETH_CORES_WITH_RETRAIN drops it so open_device doesn't hang. For single-chip
# the caller should also `ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)`.
# NOTE: a crashed container that held /dev/tenstorrent becomes a Dead zombie that
# locks the device and makes every later open_device hang. If you see hangs, run:
#   docker rm -f $(docker ps -aq --filter ancestor=qwen36-test:latest); tt-smi -r
exec docker run --rm -it \
  "${DEV_ARGS[@]}" \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  --cap-add ALL \
  -v "$WT:$WT" \
  -v "$WEIGHTS:$WEIGHTS" \
  -v /home/ttuser/.cache/huggingface:/home/ttuser/.cache/huggingface \
  -e "TT_MESH_GRAPH_DESC_PATH=$MGD" \
  -e "TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1" \
  -e "TT_LOGGER_LEVEL=ERROR" \
  -w "$WT" \
  "$IMAGE" "$@"
