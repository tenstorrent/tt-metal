#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# Bisect one stage's user capacity at the 1M window (1,064,960 tokens per slot), as #62125 did.
#
#   run_capacity_bisect.sh <name> <first_layer> <num_layers> <start_users> <traced|eager>
#
# Each step is one run_stage_probe.sh run with PREFILL_NUM_USERS=N: bring-up (weights, every user slot of the
# KV caches, compile, trace capture) plus a short request (STEP_WARMUP + STEP_CHUNKS chunks). It steps by one user
# from start_users until it has a pass at N and a fail at N + 1, then confirms N with the full 1,044,480-token
# request (stepping down if that fails). CAP_PHASE=bisect stops after the short steps; CAP_PHASE=confirm runs
# only the full-request confirmation of the bisection recorded in the result file. A failing step's "Out of Memory" line is copied into the result.
# Before every device job it waits while a device-priority marker sorting before $CAP_PRIORITY exists.
#
# Env: STAGE_PROBE_LOG_DIR (required), CAP_RUNNER_ENV / CAP_PRODUCER_ENV / CAP_RUNNER_MODULE (passed through as
# STAGE_PROBE_RUNNER_ENV / _PRODUCER_ENV / _RUNNER_MODULE), CAP_STEP_CHUNKS (default 10).
set -euo pipefail

NAME="${1:?name}"
FIRST="${2:?first layer}"
NUM="${3:?layer count}"
N="${4:?start users}"
MODE="${5:?traced|eager}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${STAGE_PROBE_LOG_DIR:?set STAGE_PROBE_LOG_DIR}"
STEP_CHUNKS="${CAP_STEP_CHUNKS:-10}"
FULL_TOKENS=1044480
RESULT="${LOG_DIR}/capacity-${NAME}-${MODE}.result"
MODE_ENV=""
[ "${MODE}" = eager ] && MODE_ENV="export PREFILL_USE_TRACE=0;"
[ "${MODE}" = traced ] || [ "${MODE}" = eager ] || { echo "mode must be traced or eager" >&2; exit 2; }

say() { echo "[$(date -u +%FT%TZ)] cap ${NAME}/${MODE}: $*" | tee -a "${RESULT}"; }
trap '[ -z "${TT_DEVICE_PRIORITY_DIR:-}" ] || rm -f "${TT_DEVICE_PRIORITY_DIR}/${CAP_PRIORITY:-x}-${CAP_MARKER_NAME:-glm54-cap}"' EXIT

wait_priority() {
  # Wait while a marker whose priority (the name up to the first '-') sorts before ours exists.
  local dir="${TT_DEVICE_PRIORITY_DIR:-}" own="${CAP_PRIORITY:?set CAP_PRIORITY, e.g. P0c}" ahead f waited=0
  [ -n "${dir}" ] || return 0
  while true; do
    ahead=""
    for f in $(ls "${dir}" 2>/dev/null); do
      [[ "${f%%-*}" < "${own}" ]] && ahead="${ahead} ${f}"
    done
    [ -n "${ahead}" ] || return 0
    [ $((waited % 60)) -eq 0 ] && say "waiting for higher-priority device work:${ahead}"
    sleep 15
    waited=$((waited + 15))
  done
}

# probe <users> <request_tokens> <warmup> -> 0 pass / 1 fail; sets LAST_DIR, LAST_OOM.
probe() {
  local users="$1" tokens="$2" warmup="$3" out rc mark="${TT_DEVICE_PRIORITY_DIR:-}/${CAP_PRIORITY}-${CAP_MARKER_NAME:-glm54-cap}"
  wait_priority
  # Hold the priority marker only while this job runs, so the gaps between steps do not block others.
  [ -z "${TT_DEVICE_PRIORITY_DIR:-}" ] || touch "${mark}"
  set +e
  out="$(STAGE_PROBE_WARMUP_CHUNKS="${warmup}" \
    STAGE_PROBE_RUNNER_ENV="${MODE_ENV} export PREFILL_NUM_USERS=${users}; ${CAP_RUNNER_ENV:-}" \
    STAGE_PROBE_PRODUCER_ENV="${CAP_PRODUCER_ENV:-}" \
    STAGE_PROBE_RUNNER_MODULE="${CAP_RUNNER_MODULE:-models.demos.common.prefill.runners.prefill_runner}" \
    "${HERE}/run_stage_probe.sh" "cap-${NAME}-${MODE}-u${users}" "${FIRST}" "${NUM}" "${tokens}" 2>&1)"
  rc=$?
  set -e
  [ -z "${TT_DEVICE_PRIORITY_DIR:-}" ] || rm -f "${mark}"
  LAST_DIR="$(sed -n 's/.*STAGE_PROBE_RESULT.* dir=\(.*\)$/\1/p' <<< "${out}" | tail -1)"
  [ -n "${LAST_DIR}" ] || { say "STOP: u${users} printed no STAGE_PROBE_RESULT (it never held the device lock)"; exit 1; }
  LAST_OOM="$(grep -m1 -o 'Out of Memory[^"]*' "${LAST_DIR}/runner.log" 2>/dev/null | cut -c1-400 || true)"
  return "${rc}"
}

PHASE="${CAP_PHASE:-all}"  # bisect: short steps only; confirm: the full request at the recorded bisection; all: both
PASS=""
FAIL=""
if [ "${PHASE}" = confirm ]; then
  read -r PASS FAIL < <(sed -n 's/.*bisected: pass \([0-9]*\) \/ fail \([0-9]*\).*/\1 \2/p' "${RESULT}" | tail -1)
  [ -n "${PASS}" ] || { echo "no bisection recorded in ${RESULT}" >&2; exit 2; }
  FAIL_OOM="$(sed -n "s/.*u${FAIL} FAIL (short) .* oom='\(.*\)'$/\1/p" "${RESULT}" | tail -1)"
  FAIL_DIR="$(sed -n "s/.*u${FAIL} FAIL (short) \([^ ]*\) .*/\1/p" "${RESULT}" | tail -1)"
else
  say "start: layers [${FIRST}, $((FIRST + NUM))) from ${N} users, step requests ${STEP_CHUNKS} chunks"
fi
while [ "${PHASE}" != confirm ] && { [ -z "${PASS}" ] || [ -z "${FAIL}" ]; }; do
  if probe "${N}" $((STEP_CHUNKS * 5120)) 2; then
    say "u${N} PASS (short) ${LAST_DIR}"
    PASS="${N}"
    [ -n "${FAIL}" ] || N=$((N + 1))
  else
    say "u${N} FAIL (short) ${LAST_DIR} oom='${LAST_OOM:-<no Out of Memory line; see runner.log>}'"
    [ -n "${LAST_OOM}" ] || { say "STOP: failure without an out-of-memory line"; exit 1; }
    FAIL="${N}"
    FAIL_OOM="${LAST_OOM}"
    FAIL_DIR="${LAST_DIR}"
    [ -n "${PASS}" ] || N=$((N - 1))
  fi
  [ "${N}" -gt 0 ] || { say "STOP: no passing count"; exit 1; }
done
[ "${PHASE}" = confirm ] || say "bisected: pass ${PASS} / fail ${FAIL}"
[ "${PHASE}" != bisect ] || exit 0

while ! probe "${PASS}" "${FULL_TOKENS}" 10; do
  say "u${PASS} FAIL on the full 1M request ${LAST_DIR} oom='${LAST_OOM:-<none>}'"
  [ -n "${LAST_OOM}" ] || { say "STOP: full-request failure without an out-of-memory line"; exit 1; }
  FAIL="${PASS}"
  FAIL_OOM="${LAST_OOM}"
  FAIL_DIR="${LAST_DIR}"
  PASS=$((PASS - 1))
done
say "u${PASS} PASS on the full 1M request ${LAST_DIR}"
say "RESULT pass=${PASS} fail=${FAIL} fail_dir=${FAIL_DIR} oom='${FAIL_OOM}'"
