#!/usr/bin/env bash
# Kill leftover vLLM/python/MPI processes and clear UMD locks across the B12 quad.
# Lighter than b12-quad-reset.sh: does NOT do an ipmitool HW reset.
# Use this as the "clean state" step before re-launching vLLM server.

set -euo pipefail

readonly RESET_HOSTS=(
  "UF-MN-B1-GWH01"
  "UF-MN-B1-GWH02"
  "UF-MN-B2-GWH01"
  "UF-MN-B2-GWH02"
)

# Patterns are stored as a sentinel-separated string and written into a temp
# file on each host. We avoid embedding the patterns inline in the
# `bash -c "..."` ssh argument because pgrep -f would otherwise match the
# bash process running this script (its cmdline contains the patterns).
read -r -d '' KILL_PATTERNS_RAW <<'PATTERNS' || true
python.*vllm[./]entrypoints[./]openai[./]api_server
python.*models/demos/deepseek_v3
python.*tt-inference-server
ttnn/ttnn/distributed/ttrun.py
mpirun.*api_server
mpirun.*deepseek_v3
mpirun.*deepseek/demo
prterun.*api_server
prterun.*deepseek_v3
prterun.*deepseek/demo
prterun.*tt_core_launcher
prted
python.*tt-run.*tt_core_launcher
python.*vllm\.v1\.engine\.tt_core_launcher
vllm.entrypoints.openai.api_server
VLLM::DPCoordinator
VLLM::EngineCore_DP
PATTERNS

readonly KILL_PATTERNS_RAW

# Self-contained payload: reads patterns from /tmp/.b12quad-kill-patterns,
# excludes itself + ancestors, TERM->KILL stragglers, then clears UMD locks.
read -r -d '' REMOTE_PAYLOAD <<'PAYLOAD' || true
#!/usr/bin/env bash
set -u

PATTERN_FILE="${1:-/tmp/.b12quad-kill-patterns}"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)"

if [[ ! -f "${PATTERN_FILE}" ]]; then
  echo "[ERROR] ${HOST_SHORT}: pattern file ${PATTERN_FILE} not found" >&2
  exit 1
fi

self_pid=$$
self_bashpid="${BASHPID:-$self_pid}"

declare -A skip_pids=()
skip_pids[$self_pid]=1
skip_pids[$self_bashpid]=1

ancestor=$self_pid
for _ in 1 2 3 4 5 6 7 8 9 10; do
  ppid="$(ps -o ppid= -p "${ancestor}" 2>/dev/null | tr -d ' ' || true)"
  [[ -z "${ppid}" || "${ppid}" == "0" || "${ppid}" == "1" ]] && break
  skip_pids["${ppid}"]=1
  ancestor="${ppid}"
done

declare -A pids_to_kill=()
while IFS= read -r pat; do
  [[ -z "${pat}" ]] && continue
  while IFS= read -r pid; do
    [[ -z "${pid}" ]] && continue
    [[ -n "${skip_pids[${pid}]:-}" ]] && continue
    cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ -z "${cmdline}" ]] && continue
    if [[ "${cmdline}" == *"b12quad-kill-python"* ]] \
       || [[ "${cmdline}" == *"/tmp/.b12quad-kill-"* ]]; then
      continue
    fi
    pids_to_kill["${pid}"]="${cmdline}"
  done < <(pgrep -f "${pat}" 2>/dev/null || true)
done < "${PATTERN_FILE}"

if [[ ${#pids_to_kill[@]} -eq 0 ]]; then
  echo "[INFO] ${HOST_SHORT}: no leftover python/vllm/mpi processes."
else
  echo "[INFO] ${HOST_SHORT}: TERMing ${#pids_to_kill[@]} pid(s):"
  for pid in "${!pids_to_kill[@]}"; do
    echo "  ${pid}: ${pids_to_kill[$pid]}"
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 4
  remaining=()
  for pid in "${!pids_to_kill[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      remaining+=("${pid}")
    fi
  done
  if [[ ${#remaining[@]} -gt 0 ]]; then
    echo "[WARN] ${HOST_SHORT}: KILLing ${#remaining[@]} survivor(s): ${remaining[*]}"
    kill -KILL "${remaining[@]}" 2>/dev/null || true
  fi
fi

rm -f /dev/shm/TT_UMD_LOCK* 2>/dev/null || true
echo "[INFO] ${HOST_SHORT}: UMD locks cleared."
PAYLOAD

readonly REMOTE_PAYLOAD

run_remote() {
  local host="$1"
  local script_path="$2"
  local pattern_path="$3"

  ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" "bash ${script_path} ${pattern_path}" 2>&1
}

push_files() {
  local host="$1"
  local pattern_path="$2"
  local script_path="$3"

  printf '%s\n' "${KILL_PATTERNS_RAW}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" "cat > ${pattern_path}"
  printf '%s' "${REMOTE_PAYLOAD}" \
    | ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" "cat > ${script_path} && chmod +x ${script_path}"
}

is_local_host() {
  local host="$1"
  local local_short="$2"
  local host_short="${host%%.*}"
  [[ "${host_short}" == "${local_short}" ]]
}

main() {
  local local_host
  local local_short
  local pattern_path="/tmp/.b12quad-kill-patterns"
  local script_path="/tmp/.b12quad-kill-payload.sh"
  local failed=0

  local_host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "${HOSTNAME:-}")"
  local_short="${local_host%%.*}"

  echo "[INFO] Killing leftover python/vllm/mpi processes and clearing UMD locks on: ${RESET_HOSTS[*]}"

  printf '%s\n' "${KILL_PATTERNS_RAW}" > "${pattern_path}"
  printf '%s' "${REMOTE_PAYLOAD}" > "${script_path}"
  chmod +x "${script_path}"

  local host
  for host in "${RESET_HOSTS[@]}"; do
    if is_local_host "${host}" "${local_short}"; then
      echo "[INFO] Cleaning local host ${host}..."
      if ! bash "${script_path}" "${pattern_path}"; then
        echo "[ERROR] Local cleanup failed on ${host}."
        failed=1
      fi
      continue
    fi

    echo "[INFO] Pushing payload to ${host}..."
    if ! push_files "${host}" "${pattern_path}" "${script_path}"; then
      echo "[ERROR] Failed to push payload to ${host}."
      failed=1
      continue
    fi
    echo "[INFO] Cleaning remote host ${host} via ssh..."
    if ! run_remote "${host}" "${script_path}" "${pattern_path}"; then
      echo "[ERROR] Remote cleanup failed on ${host}."
      failed=1
    fi
  done

  if (( failed )); then
    echo "[ERROR] Clean state completed with errors." >&2
    return 1
  fi
  echo "[INFO] Clean state complete (no HW reset)."
}

main "$@"
