#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Depth runner: one node id, chosen sites and delays, repeated enough times to
# turn a flaky race into a failure rate. Each site also gets a delay-0 control
# that still detours through the cave, so "the jump did it" is ruled out.
#
#   TTNOP_SITES=unpack:3 TTNOP_DELAYS=8,16 TTNOP_REPEATS=50 \
#       ./focus.sh 'test_x.py::test_y[params]'
#
# Same poke loop as ci.sh; only the planning and the reporting differ.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: focus.sh <pytest-node-id>" >&2
    exit 4
fi
NODE_ID="$1"

export TTNOP_REPEATS="${TTNOP_REPEATS:-50}"
export TTNOP_REPORT_DIR="${TTNOP_REPORT_DIR:-$HERE/reports/focus}"

build_scanner
cd "$PYTHON_TESTS"

echo ">> delays=${TTNOP_DELAYS:-1-100} threads=${TTNOP_THREADS:-unpack,math}" \
     "sites=${TTNOP_SITE_MODE:-sync} filler=${TTNOP_FILLER:-auto} repeats=${TTNOP_REPEATS}"
echo ">> case=${NODE_ID}"

# Build this one variant if the shared tree does not already hold it.
echo ">> [1/2] compiling"
flock "$BUILD_LOCK" python3 -m pytest --compile-producer -q "$NODE_ID"

exec 9>"$DEVICE_LOCK"
flock 9
echo ">> [2/2] sweeping"
started=$SECONDS
status=0
python3 -m pytest --compile-consumer -p ttnop_plugin -p no:randomly -q "$NODE_ID" || status=$?
echo ">> timing: sweep=$((SECONDS - started))s total=${SECONDS}s"
exit "$status"
