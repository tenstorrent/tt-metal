#!/usr/bin/env bash
set -euo pipefail

readonly RESET_HOSTS=(
  "UF-MN-B1-GWH01"
  "UF-MN-B1-GWH02"
  "UF-MN-B2-GWH01"
  "UF-MN-B2-GWH02"
)
readonly RESET_SETTLE_SECONDS=100

run_remote_ssh() {
  local host="$1"
  local remote_cmd="$2"
  local action="$3"
  local ssh_output=""
  local fallback_output=""

  if ssh_output="$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${host}" "${remote_cmd}" 2>&1)"; then
    [[ -n "${ssh_output}" ]] && printf '%s\n' "${ssh_output}"
    return 0
  fi

  if [[ "${ssh_output}" == *"Could not resolve hostname"* || "${ssh_output}" == *"Temporary failure in name resolution"* ]]; then
    # Retry with an explicit HostName override so local ssh HostName mappings
    # (for example, stale ".maas" entries) cannot rewrite the target host.
    if fallback_output="$(ssh -o BatchMode=yes -o ConnectTimeout=10 -o HostName="${host}" "${host}" "${remote_cmd}" 2>&1)"; then
      [[ -n "${fallback_output}" ]] && printf '%s\n' "${fallback_output}"
      echo "[WARN] SSH HostName mapping for ${host} failed; using direct hostname override."
      return 0
    fi
    ssh_output+=$'\n'"${fallback_output}"
  fi

  [[ -n "${ssh_output}" ]] && printf '%s\n' "${ssh_output}" >&2
  if [[ "${ssh_output}" == *"Could not resolve hostname"* || "${ssh_output}" == *"Temporary failure in name resolution"* ]]; then
    echo "[ERROR] Failed to ${action} on remote host ${host}: hostname is not resolvable from this machine."
  else
    echo "[ERROR] Failed to ${action} on remote host ${host}."
  fi
  return 1
}

cleanup_umd_locks() {
  local host
  local host_short
  local local_host
  local local_short
  local cleanup_failed=0
  local cleanup_cmd='rm -f /dev/shm/TT_UMD_LOCK* 2>/dev/null || true'

  local_host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "${HOSTNAME:-}")"
  local_short="${local_host%%.*}"

  echo "[INFO] Clearing UMD shared-memory locks on hosts: ${RESET_HOSTS[*]}"
  for host in "${RESET_HOSTS[@]}"; do
    host_short="${host%%.*}"

    if [[ "${host}" == "${local_host}" || "${host_short}" == "${local_short}" || "${host_short}" == "${HOSTNAME%%.*}" ]]; then
      if ! bash -lc "${cleanup_cmd}"; then
        echo "[ERROR] Failed to clear UMD locks on local host ${host}."
        cleanup_failed=1
      fi
      continue
    fi

    if ! run_remote_ssh "${host}" "bash -lc '${cleanup_cmd}'" "clear UMD locks"; then
      cleanup_failed=1
    fi
  done

  if [[ "${cleanup_failed}" -ne 0 ]]; then
    return 1
  fi
}

reset_hw() {
  local host
  local host_short
  local local_host
  local local_short
  local reset_failed=0
  local reset_cmd="sudo -n ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF"

  local_host="$(hostname -f 2>/dev/null || hostname 2>/dev/null || echo "${HOSTNAME:-}")"
  local_short="${local_host%%.*}"

  echo "[INFO] Resetting hardware on hosts: ${RESET_HOSTS[*]}"
  for host in "${RESET_HOSTS[@]}"; do
    host_short="${host%%.*}"

    if [[ "${host}" == "${local_host}" || "${host_short}" == "${local_short}" || "${host_short}" == "${HOSTNAME%%.*}" ]]; then
      echo "[INFO] Resetting local host ${host} via ipmitool..."
      if ! sudo -n ipmitool raw 0x30 0x8B 0xF 0xFF 0x0 0xF; then
        echo "[ERROR] Failed to reset local host ${host}."
        reset_failed=1
      fi
      continue
    fi

    echo "[INFO] Resetting remote host ${host} via ssh..."
    if ! run_remote_ssh "${host}" "${reset_cmd}" "reset hardware"; then
      reset_failed=1
    fi
  done

  if [[ "${reset_failed}" -ne 0 ]]; then
    return 1
  fi

  sleep "${RESET_SETTLE_SECONDS}"
}

kill_demo_processes() {
  local -a patterns=(
    "/data/yalrawwash/scripts/run/quadb12-demo.sh"
    "/data/yalrawwash/scripts/run/quadb56-demo.sh"
    "/data/yalrawwash/scripts/ds-run"
    "pytest -svvv .*models/demos/deepseek_v3/demo/test_demo.py"
    "python .*models/demos/deepseek_v3/demo/demo.py"
    "python .*tt-metal/generated/debug_prefill_row_boundary.py"
    "ttnn/ttnn/distributed/ttrun.py"
    "mpirun .*test_demo.py"
    "mpirun .*demo.py"
    "mpirun .*debug_prefill_row_boundary.py"
    "prterun .*test_demo.py"
    "prterun .*demo.py"
    "prterun .*debug_prefill_row_boundary.py"
    "build_metal.sh"
    "cmake -B build_Release"
    "ninja .*build_Release"
  )
  local pattern
  local pid
  local -A pids_to_kill=()
  local current_pid="$$"
  local current_bashpid="${BASHPID:-$$}"
  local parent_pid="${PPID:-0}"

  for pattern in "${patterns[@]}"; do
    while IFS= read -r pid; do
      [[ -z "${pid}" ]] && continue
      [[ "${pid}" == "${current_pid}" || "${pid}" == "${current_bashpid}" || "${pid}" == "${parent_pid}" ]] && continue
      pids_to_kill["${pid}"]=1
    done < <(pgrep -f "${pattern}" || true)
  done

  if [[ ${#pids_to_kill[@]} -eq 0 ]]; then
    echo "[INFO] No active build/demo processes to kill."
    return 0
  fi

  echo "[INFO] Terminating ${#pids_to_kill[@]} active build/demo process(es)..."
  for pid in "${!pids_to_kill[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done

  sleep 3

  local -a remaining_pids=()
  for pid in "${!pids_to_kill[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      remaining_pids+=("${pid}")
    fi
  done

  if [[ ${#remaining_pids[@]} -gt 0 ]]; then
    echo "[WARN] Forcing kill for ${#remaining_pids[@]} process(es): ${remaining_pids[*]}"
    kill -KILL "${remaining_pids[@]}" 2>/dev/null || true
  fi
}

main() {
  local had_errors=0

  if ! kill_demo_processes; then
    had_errors=1
  fi
  if ! cleanup_umd_locks; then
    had_errors=1
  fi
  if ! reset_hw; then
    had_errors=1
  fi
  if ! cleanup_umd_locks; then
    had_errors=1
  fi

  if [[ "${had_errors}" -ne 0 ]]; then
    echo "[ERROR] B12 quad reset completed with errors."
    return 1
  fi

  echo "[INFO] B12 quad reset complete."
}

main "$@"
