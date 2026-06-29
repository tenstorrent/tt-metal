#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sweep repro_socket_transfer.py over (TENSOR_SHAPE, NUM_TENSORS) x (transfer MODE) to map which
# configs PASS / FAIL_VERIFY / HANG. Self-contained: for each combo it sets the env vars, launches
# mesh_socket_debug/runner.sh under a timeout, captures a per-run log, classifies the outcome,
# kills orphaned ranks, and resets the devices after any non-PASS so the next run starts clean.
# Prints a summary table at the end.
#
# This script does NOT switch git branches -- check out + build the branch you want
# (ichovpan/socket-baseline or ichovpan/socket-patched) yourself first. The current branch is
# auto-detected; logs land in sweep_logs/<branch>/ and the branch is tagged on every line, so it
# is always obvious from the folder name which branch (baseline vs patched) the results are from.
#
# All runs use the local8 config => 4 devices (N=4). The host tensor is sharded across the 4
# devices, so each shard = TENSOR_SHAPE, and NUM_TENSORS = tensors streamed before the (single)
# synchronize. The three MODES are:
#   bigmesh_1sock  one [1,4] mesh, ONE socket with 4 connections
#   bigmesh_Nsock  one [1,4] mesh, 4 sockets (1 connection each)
#   submesh_Nsock  four [1,1] submeshes, 4 sockets (1 per submesh)
#
# Usage:
#   export TT_METAL_HOME=/path/to/tt-metal     # if not already set
#   bash mesh_socket_debug/bisect_sweep.sh     # from tt-train/sources/examples/grpo
#
# Tunables (env): TIMEOUT_S (per run, default 240), RESET_TIMEOUT_S (default 420),
#                 RESET_EACH=1 (reset before EVERY run for max isolation; slower).
#
# Rough time: 10 (shape,count) x 3 modes = 30 runs. PASS/FAIL_VERIFY runs are seconds; each HANG
# burns the full timeout + a ~4 min tt-smi reset, so a hang-heavy sweep can take a couple hours.

set -uo pipefail   # NOT -e: a failing / hanging run must not abort the sweep.

: "${TT_METAL_HOME:?set TT_METAL_HOME first}"

DEBUG_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/mesh_socket_debug"
RUNNER="${DEBUG_DIR}/runner.sh"
BRANCH="$(git -C "${DEBUG_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
BRANCH_LABEL="${BRANCH//\//_}"          # ichovpan/socket-baseline -> ichovpan_socket-baseline
LOGDIR="${DEBUG_DIR}/sweep_logs/${BRANCH_LABEL}"
TIMEOUT_S="${TIMEOUT_S:-240}"
RESET_TIMEOUT_S="${RESET_TIMEOUT_S:-420}"
RESET_EACH="${RESET_EACH:-0}"

export REPRO_BRANCH="${BRANCH}"         # so runner.sh / repro tag the right branch
mkdir -p "${LOGDIR}"
SUMMARY="${LOGDIR}/summary.txt"
: > "${SUMMARY}"

# (shape, NUM_TENSORS) grid. 64 MiB shard = [1,1,8192,4096]; 512 MiB = [1,1,16384,16384].
# NUM_TENSORS is capped per shape by the worst-case resident DRAM (bigmesh_Nsock holds
# N*NUM_TENSORS full-mesh templates = N*NUM_TENSORS*shard_bytes per chip; chip DRAM = 12 GiB):
#   64 MiB:  N*16*64MiB  = 4 GiB  -> NUM_TENSORS up to 16 fits.
#   512 MiB: N*4*512MiB  = 8 GiB  -> NUM_TENSORS=8 would need 16 GiB > 12 GiB, so cap at 4.
RUNS=(
  "1,1,8192,4096:1"
  "1,1,8192,4096:2"
  "1,1,8192,4096:3"
  "1,1,8192,4096:4"
  "1,1,8192,4096:8"
  "1,1,8192,4096:16"
  "1,1,16384,16384:1"
  "1,1,16384,16384:2"
  "1,1,16384,16384:3"
  "1,1,16384,16384:4"
)

# Each entry: "label:USE_SUBMESH:SINGLE_SOCKET".
MODES=(
  "bigmesh_1sock:0:1"
  "bigmesh_Nsock:0:0"
  "submesh_Nsock:1:0"
)

cleanup_procs() {
  pkill -9 -f repro_socket_transfer 2>/dev/null || true
  pkill -9 -f 'ttrun\.py'           2>/dev/null || true
  pkill -9 -f prterun               2>/dev/null || true
  pkill -9 -f orterun               2>/dev/null || true
  sleep 3
}

reset_devices() {
  echo "[sweep] tt-smi -r (device reset, ~4 min)..."
  timeout --signal=KILL "${RESET_TIMEOUT_S}" tt-smi -r > "${LOGDIR}/last_reset.log" 2>&1 \
    || echo "[sweep] WARN: tt-smi -r returned non-zero (see ${LOGDIR}/last_reset.log)"
}

classify() {  # $1=logfile  $2=rc  -> echoes a verdict
  local log="$1" rc="$2"
  if   grep -q  "tensors correct: True"  "${log}"; then echo "PASS"
  elif grep -q  "tensors correct: False" "${log}"; then echo "FAIL_VERIFY"
  elif grep -qi "heartbeat"              "${log}"; then echo "DEVICE_ERR"
  elif grep -qiE "ERISC_APP_KERNEL_CODE|Failed to generate binaries" "${log}"; then echo "BUILD_ERR"
  elif grep -qiE "Out of Memory|OOM|Allocator|out of memory" "${log}"; then echo "OOM"
  elif grep -qi "Waiting for lock"       "${log}"; then echo "LOCKED_stale_procs"
  elif [ "${rc}" -eq 137 ]; then echo "HANG_timeout"
  else echo "ERROR_rc${rc}"
  fi
}

echo "[sweep] branch=${BRANCH}  per-run logs in ${LOGDIR}"
cleanup_procs
reset_devices            # begin from a known-clean slate

declare -a RESULTS
for entry in "${RUNS[@]}"; do
  shape="${entry%:*}"
  nt="${entry#*:}"
  for mode_entry in "${MODES[@]}"; do
    mode="${mode_entry%%:*}"
    rest="${mode_entry#*:}"
    use_submesh="${rest%%:*}"
    single_socket="${rest#*:}"
    log="${LOGDIR}/shape_${shape//,/_}_n${nt}_${mode}.log"

    if [ "${RESET_EACH}" = "1" ]; then cleanup_procs; reset_devices; fi

    echo "[sweep] RUN  mode=${mode}  TENSOR_SHAPE=${shape}  NUM_TENSORS=${nt}  (timeout ${TIMEOUT_S}s)"
    start=${SECONDS}
    REPRO_TENSOR_SHAPE="${shape}" REPRO_NUM_TENSORS="${nt}" \
      REPRO_USE_SUBMESH="${use_submesh}" REPRO_SINGLE_SOCKET="${single_socket}" \
      timeout --signal=KILL "${TIMEOUT_S}" bash "${RUNNER}" > "${log}" 2>&1
    rc=$?
    dur=$(( SECONDS - start ))

    verdict="$(classify "${log}" "${rc}")"
    line="$(printf '%-14s %-20s N=%-3s -> %-18s (rc=%-3s %4ss)' "${mode}" "${shape}" "${nt}" "${verdict}" "${rc}" "${dur}")"
    echo "[sweep]   ${verdict}   (${dur}s, rc=${rc}, log: ${log})"
    echo "${line}" >> "${SUMMARY}"
    RESULTS+=("${line}")

    cleanup_procs
    if [ "${verdict}" != "PASS" ]; then reset_devices; fi   # recover wedged devices
  done
done

echo
echo "==================== SWEEP SUMMARY (branch=${BRANCH}) ===================="
printf '%s\n' "${RESULTS[@]}"
echo "========================================================================="
echo "Per-run logs + summary.txt in: ${LOGDIR}"
