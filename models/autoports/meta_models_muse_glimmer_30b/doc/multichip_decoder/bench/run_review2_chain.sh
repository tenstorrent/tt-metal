#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# The device jobs review round 2 asked for, in order, one at a time
# ($tt-device-usage):
#   1. the three probes that predate the 8192 B fabric packet change, re-run at
#      the shipped packet size (the decode-shape and prefill-shape topology
#      probes, and the rebuilt fractured-residual probe);
#   2. one whole-layer A/B that measures every reducer candidate in a single
#      invocation, so the README's reducer table is like-with-like from one log,
#      the prefill `rs_ag` row is measured at the shipped 4 workers, and the
#      duplicate candidate serves as an explicit same-config repeat control;
#   3. the fabric packet size itself, whole-layer, one process per value.
#
# Each step writes its own log under logs/; those are the committed artifacts.  Do
# not commit a console transcript of this script -- it drifts from them.
#
#   bash bench/run_review2_chain.sh
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."          # repo root
D=$PWD/models/autoports/meta_models_muse_glimmer_30b/doc/multichip_decoder

echo "=== 1/6 fractured-residual decode probe, at the shipped 8192 B packet"
python "$D/bench/fractured_decode_probe.py" > "$D/logs/fractured_decode_probe.log" 2>&1
echo "exit=$?"; grep -E '^FRAC' "$D/logs/fractured_decode_probe.log"

echo "=== 2/6 tuned topology probe, decode shape, at the shipped 8192 B packet"
python "$D/bench/topology_probe.py" --rows 32 --traced > "$D/logs/topology_probe_decode32_tuned.log" 2>&1
echo "exit=$?"; grep -E '^TOPO' "$D/logs/topology_probe_decode32_tuned.log"

echo "=== 3/6 tuned topology probe, prefill shape, at the shipped 8192 B packet"
python "$D/bench/topology_probe.py" --rows 8192 > "$D/logs/topology_probe_prefill8192.log" 2>&1
echo "exit=$?"; grep -E '^TOPO' "$D/logs/topology_probe_prefill8192.log"

echo "=== 4/6 every reducer candidate, whole layer, one invocation"
python "$D/bench/layer_ab.py" --mesh 1x4 \
  --candidates tp4,ccl_all_reduce,ccl_rs_ag,ccl_rs_ag_prefill,ccl_rs_ag_prefill_w1 \
  --prefill-seq 8192 --decode-context 2048 > "$D/logs/layer_ab_reducer_final.log" 2>&1
echo "exit=$?"; grep -E '^AB|^FAILED' "$D/logs/layer_ab_reducer_final.log"

echo "=== 5/6 the shipped 8192 B fabric packet, whole layer"
python "$D/bench/layer_ab.py" --mesh 1x4 --candidates tp4 --packet-bytes 8192 \
  --prefill-seq 8192 --decode-context 2048 > "$D/logs/layer_ab_packet8192.log" 2>&1
echo "exit=$?"; grep -E '^AB|^FAILED' "$D/logs/layer_ab_packet8192.log"

echo "=== 6/6 the rejected 4352 B fabric packet, whole layer, same harness"
python "$D/bench/layer_ab.py" --mesh 1x4 --candidates tp4 --packet-bytes 4352 \
  --prefill-seq 8192 --decode-context 2048 > "$D/logs/layer_ab_packet4352.log" 2>&1
echo "exit=$?"; grep -E '^AB|^FAILED' "$D/logs/layer_ab_packet4352.log"

echo "=== review2 chain done ==="
