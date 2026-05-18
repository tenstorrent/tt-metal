#!/usr/bin/env bash
# Usage: ./probe_1d.sh <W_tiles_per_shard>
#
# Single-device companion to ./probe.sh. Runs the
# SubtractFp32ColBBcastSingleDeviceTest suite in one gtest invocation:
#   1. ColBBroadcast_Fp32Output_Hangs  (single-device mirror, uses $W)
#
# No MGD env var needed — the test opens the default single-chip device. Useful
# for bisecting whether the multi-device sharded path is actually a precondition
# for the hang, or whether the same per-chip local shape triggers it on one chip.
#
# Pair with ./probe.sh <same W> for an apples-to-apples comparison at the same
# per-chip width. ALWAYS `tt-smi -r` between probes — a hang leaves the cluster
# wedged.
set -u
W="${1:?usage: $0 <W_tiles_per_shard>}"

BIN=../build_Release/tt-train/tests/ttml_tests
FILTER='SubtractFp32ColBBcastSingleDeviceTest.*'

echo "== probe-1d W_tiles=$W =="
REPRO_W_TILES_PER_SHARD="$W" \
timeout 60 "$BIN" --gtest_filter="$FILTER"
rc=$?
echo "== W=$W exit=$rc  (0=PASS, 124=HANG) =="
exit $rc
