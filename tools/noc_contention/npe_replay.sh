#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Replay a tt-metal NoC trace in tt-npe and report the per-link demand over time.
#
#   npe_replay.sh <logs_dir> <out_dir> [device] [-- extra noc_link_demand.py args]
#
# <logs_dir> is generated/profiler/.logs after a run with TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1: it holds
# noc_trace_dev<d>_ID<program>.json (one per device and program), soc_descriptor.yaml and topology.json. Every
# trace in it is replayed; the timeline (npe_timeline_<name>.json) and the link report (links_<name>.txt, .csv) go
# to <out_dir>. device defaults to P150 (tt_npe.py -d: wormhole_b0, N150, N300, T3K, blackhole, P100, P150).
#
# Needs a tt-npe checkout built with ./build-npe.sh; point TT_NPE_ROOT at it (default: ~/tt-npe). The pybind module
# is taken from its build/ dir, the device models from install/share.
set -euo pipefail
LOGS=${1:?logs dir}
OUT=${2:?out dir}
DEVICE=${3:-P150}
shift $(( $# >= 3 ? 3 : $# ))
if [[ "${1:-}" == "--" ]]; then shift; fi
NPE=${TT_NPE_ROOT:-$HOME/tt-npe}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$NPE/build/tt_npe:$NPE/install/bin${PYTHONPATH:+:$PYTHONPATH}"
export TT_NPE_DEVICE_MODEL_CONFIG_DIR="$NPE/install/share/tt_npe/device/models/"
mkdir -p "$OUT"
shopt -s nullglob
for trace in "$LOGS"/noc_trace_dev*_ID*.json; do
    name=$(basename "$trace" .json)
    echo "== replaying $name on $DEVICE"
    python3 "$NPE/install/bin/tt_npe.py" -d "$DEVICE" -t -w "$trace" -e \
        --timeline-json-filepath "$OUT/npe_timeline_$name.json" \
        --soc-descriptor-file "$LOGS/soc_descriptor.yaml" --topology-json "$LOGS/topology.json" \
        | grep -E "finished|util|demand|impact" || true
    python3 -I "$HERE/noc_link_demand.py" "$OUT/npe_timeline_$name.json" --csv "$OUT/links_$name.csv" "$@" \
        | tee "$OUT/links_$name.txt" | head -60
done
