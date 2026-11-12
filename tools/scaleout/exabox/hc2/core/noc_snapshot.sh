#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# noc_snapshot.sh: read-only snapshot of a hung galaxy while the hung process is still alive (plan R1h / 2.5).
# ub0_repro.py --hold-on-hang runs it when the CCL watchdog fires; it can also run by hand or from another harness
# (an H3 pytest on its op-timeout throw). Order: lspci -vvv of each root port and endpoint, dmesg tail, then tt-triage
# with NOC0, then NOC1 (a NOC0 deadlock leaves NOC1 readable). Each step runs under its own timeout; a step that
# fails or times out is recorded and the next one still runs. Nothing here resets, writes to or re-powers a chip.
#   noc_snapshot.sh OUT_DIR [ROOT_PORT=ENDPOINT ...]        e.g. noc_snapshot.sh /tmp/snap 0000:00:01.2=0000:02:00.0
# Writes OUT_DIR/rp_<bdf>.txt, ep_<bdf>.txt, dmesg_tail.txt, triage_help.txt, triage_noc<N>.txt (+ .csv where the
# triage takes --llm-output-path), and steps.tsv: step, exit code (124 = timed out), seconds, output file.
# env:
#   SNAP_LSPCI            lspci command (default "timeout 10 lspci"; ub0_repro.py passes its --lspci, e.g. with sudo -n)
#   SNAP_TRIAGE           triage command (default: python3 $TT_METAL_HOME/tools/triage/triage.py, else tt-triage on PATH)
#   SNAP_TRIAGE_TIMEOUT_S per triage pass (default 120)
#   SNAP_TRIAGE_RUN       comma list of triage scripts (default below); each becomes one --run=<script>
#   SNAP_TRIAGE_ARGS      extra triage args (default "--disable-progress --skip-version-check")
#   SNAP_NOCS             NOC passes, in order (default "0 1")
#   SNAP_DMESG_LINES      dmesg lines kept (default 200)
# Exit 0 when the snapshot ran (whatever each step gave), 1 when OUT_DIR cannot be written.
set -u
out=${1:?usage: noc_snapshot.sh OUT_DIR [ROOT_PORT=ENDPOINT ...]}; shift
mkdir -p "$out" && [ -w "$out" ] || { echo "noc_snapshot: cannot write $out" >&2; exit 1; }
read -r -a LSPCI <<< "${SNAP_LSPCI:-timeout 10 lspci}"
TO=${SNAP_TRIAGE_TIMEOUT_S:-120}
RUNS=${SNAP_TRIAGE_RUN:-check_arc,check_noc_status,check_eth_status,dump_fabric_stream_regs,dump_callstacks,dump_running_operations,device_telemetry}
read -r -a XARGS <<< "${SNAP_TRIAGE_ARGS---disable-progress --skip-version-check}"
steps=$out/steps.tsv
printf 'step\trc\tseconds\tfile\n' > "$steps"
log() { echo "SNAPSHOT|$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)|$*"; }
step() {  # $1 name, $2 output file, rest: the command (stdout+stderr to the file)
  local name=$1 f=$2 t0 rc; shift 2
  t0=$(date +%s.%N)
  "$@" > "$f" 2>&1; rc=$?
  printf '%s\t%s\t%.1f\t%s\n' "$name" "$rc" "$(awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN { print b - a }')" "$(basename "$f")" >> "$steps"
  log "$name rc $rc$([ "$rc" = 124 ] && echo ' (timed out)') -> $(basename "$f")"
  return 0
}
log "start: $out, ${#@} root port/endpoint pair(s)"

# 1. the link: root port first (host side, always readable), then the endpoint (the chip's own config space)
for pair in "$@"; do
  rp=${pair%%=*}; ep=${pair#*=}
  [ -n "$rp" ] && [ "$rp" != "$pair" ] && step "lspci_rp_$rp" "$out/rp_${rp//:/_}.txt" "${LSPCI[@]}" -vvv -s "$rp"
  [ -n "$ep" ] && step "lspci_ep_$ep" "$out/ep_${ep//:/_}.txt" "${LSPCI[@]}" -vvv -s "$ep"
done

# 2. the kernel's view: AER, link down, KMD messages
dmesg_tail() { { timeout 20 dmesg -T 2>/dev/null || timeout 20 sudo -n dmesg -T; } | tail -n "${SNAP_DMESG_LINES:-200}"; }
step dmesg "$out/dmesg_tail.txt" dmesg_tail

# 3. tt-triage, NOC0 then NOC1
if [ -n "${SNAP_TRIAGE:-}" ]; then read -r -a TRI <<< "$SNAP_TRIAGE"
elif [ -f "${TT_METAL_HOME:-/nonexistent}/tools/triage/triage.py" ]; then
  # run triage under a venv python that has ttexalens, not bare system python3.
  # tools_venv_python (lib/tools_probe.sh) when sourced; else inline fallback: the run's venv python, ~/venv, ~/venv3
  VP=""
  type tools_venv_python >/dev/null 2>&1 && VP=$(tools_venv_python "${PYTHON:-python3}") || VP=""
  # the candidates: tools_venv_candidates when sourced, else the same list inline, quoted ($HOME with a space stays one path)
  VC=()
  if declare -F tools_venv_candidates > /dev/null; then mapfile -t VC < <(tools_venv_candidates)
  elif [ -n "${TRAY_VENV_SEARCH:-}" ]; then
    # shellcheck disable=SC2206  # TRAY_VENV_SEARCH is word-split and glob-expanded on purpose
    VC=($TRAY_VENV_SEARCH)
  else VC=("$HOME"/venv/bin/python "$HOME"/venv3/bin/python); fi
  [ -z "$VP" ] &&
  for c in "${PYTHON:-python3}" "${VC[@]}"; do
    [ -x "$c" ] && [ "$c" != python3 ] && "$c" -I -c 'import ttexalens' >/dev/null 2>&1 && { VP=$c; break; }
  done 2>/dev/null
  [ -n "$VP" ] && TRI=("$VP" "$TT_METAL_HOME/tools/triage/triage.py") || TRI=(python3 "$TT_METAL_HOME/tools/triage/triage.py")
elif command -v tt-triage > /dev/null; then TRI=(tt-triage)
else TRI=(); fi
if [ ${#TRI[@]} = 0 ]; then
  printf 'triage\tnot-found\t0\t-\n' >> "$steps"
  log "triage: not found (set SNAP_TRIAGE or TT_METAL_HOME); lspci and dmesg only"
  exit 0
fi
# --help reads no device: it records which flags this triage has (--noc-id, --llm-output-path)
step triage_help "$out/triage_help.txt" timeout 30 "${TRI[@]}" --help
help=$(cat "$out/triage_help.txt")
RA=(); IFS=, read -r -a rs <<< "$RUNS"; for r in "${rs[@]}"; do [ -n "$r" ] && RA+=("--run=$r"); done
nocs=${SNAP_NOCS:-0 1}
if [[ $help != *--noc-id* ]]; then
  log "triage --help has no --noc-id: one pass on its default NOC"; nocs=default
fi
for n in $nocs; do
  a=("${XARGS[@]}")
  [ "$n" != default ] && a+=("--noc-id=$n")
  [[ $help == *--llm-output-path* ]] && a+=("--llm-output-path=$out/triage_noc$n.csv")
  step "triage_noc$n" "$out/triage_noc$n.txt" timeout -k 10 "$TO" "${TRI[@]}" "${a[@]}" "${RA[@]}"
done
log "end: $out"
