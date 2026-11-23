#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# device_safety.sh - board reset safety shared by the device runners.
#
# Sourced by scripts/run_safe_pytest.sh, scripts/tt-probe.sh and tt-blaze's scripts/run_safe_pytest.sh, after they
# hold the device lock (/tmp/tt-device.lock). It replaces their bare `tt-smi -r` with a guarded reset:
#
#   1. Idle check before a reset: the runner's own leftover children are killed and waited for, then the reset waits
#      until no process (in this PID namespace) holds /dev/tenstorrent/*. If one still does, nothing is reset.
#   2. Health gate after a reset: every chip that was there before must be enumerated again, report trained GDDR, and
#      show an advancing firmware heartbeat in `tt-smi -s`, then a short settle. Only then is the device handed on.
#   3. Circuit breaker: every reset after a fault (hang, crash, killed run, stuck Ethernet cores) is logged. With
#      TT_DEVICE_BREAKER_RESETS fault resets inside TT_DEVICE_BREAKER_WINDOW_S, new device runs are refused.
#   4. A failed reset or health gate marks the device unhealthy; every later run is refused until a person checks the
#      board and runs `scripts/device_safety.sh clear`.
#
# Run directly: `scripts/device_safety.sh status` shows the state; `scripts/device_safety.sh clear` clears the
# unhealthy flag and the breaker's reset history (it does not reset the board; the next runner resets a dirty board).
#
# State (shared by all runners on the machine, next to the lock and the dirty flag):
#   /tmp/tt-device.resets     one line per reset: <epoch> <fault|routine> <reason> pid=<runner pid>
#   /tmp/tt-device.unhealthy  present = refuse all device runs; holds the reason

DS_RESET_LOG="${TT_DEVICE_RESET_LOG:-/tmp/tt-device.resets}"
DS_UNHEALTHY_FLAG="${TT_DEVICE_UNHEALTHY_FLAG:-/tmp/tt-device.unhealthy}"
DS_BREAKER_RESETS="${TT_DEVICE_BREAKER_RESETS:-3}"
DS_BREAKER_WINDOW_S="${TT_DEVICE_BREAKER_WINDOW_S:-1800}"
DS_HEALTH_TIMEOUT_S="${TT_DEVICE_HEALTH_TIMEOUT_S:-180}"
DS_SETTLE_S="${TT_DEVICE_SETTLE_S:-5}"
DS_IDLE_TIMEOUT_S="${TT_DEVICE_IDLE_TIMEOUT_S:-60}"
DS_TAG="${DS_TAG:-DEVICE_SAFETY}"

ds_log() { echo "${DS_TAG}: $*"; }

ds_chip_count() {
    local n=0 f
    for f in /dev/tenstorrent/[0-9]*; do [[ -e "$f" ]] && n=$((n + 1)); done
    echo "$n"
}

# Galaxy needs `tt-smi -glx_reset`; other systems `tt-smi -r`. Each chip's PCI subsystem ID (cached config space,
# readable while the chip is hung) is the board UPI: 0x35 WH UBB, 0x47 BH UBB, 0x202 BH UBB CF.
# Override with SAFE_PYTEST_SYSTEM=galaxy|non-galaxy.
ds_detect_system() {
    if [[ -n "${SAFE_PYTEST_SYSTEM:-}" ]]; then
        echo "$SAFE_PYTEST_SYSTEM"
        return
    fi
    local f subsys n=0
    for f in /sys/class/tenstorrent/*/device/subsystem_device; do
        [[ -r "$f" ]] || continue
        n=$((n + 1))
        subsys=$(<"$f")
        case "$((subsys))" in
            $((0x35)) | $((0x47)) | $((0x202)))
                echo galaxy
                return
                ;;
        esac
    done
    if [[ "$n" -ge 32 ]]; then echo galaxy; else echo non-galaxy; fi
}

# "pid cmdline" of every process in this PID namespace with /dev/tenstorrent/* open.
ds_device_holders() {
    local pid_dir fd target pid
    for pid_dir in /proc/[0-9]*; do
        pid=${pid_dir##*/}
        for fd in "$pid_dir"/fd/*; do
            target=$(readlink "$fd" 2>/dev/null) || continue
            if [[ "$target" == /dev/tenstorrent/* ]]; then
                echo "$pid $(tr '\0' ' ' < "$pid_dir/cmdline" 2>/dev/null | cut -c1-160)"
                break
            fi
        done
    done
}

# Kill (SIGKILL) and reap every descendant of this runner, so a dying test process cannot still hold the board when
# the reset starts.
ds_kill_own_tree() {
    local victims=() frontier=($$) next p c
    while [[ ${#frontier[@]} -gt 0 ]]; do
        next=()
        for p in "${frontier[@]}"; do
            for c in $(pgrep -P "$p" 2>/dev/null); do
                victims+=("$c")
                next+=("$c")
            done
        done
        frontier=("${next[@]}")
    done
    [[ ${#victims[@]} -eq 0 ]] && return 0
    kill -KILL "${victims[@]}" 2>/dev/null || true
    local deadline=$(($(date +%s) + 10)) alive
    while [[ $(date +%s) -lt $deadline ]]; do
        alive=0
        for p in "${victims[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 0.2
    done
    wait 2>/dev/null || true
}

ds_wait_idle() {
    ds_kill_own_tree
    local deadline=$(($(date +%s) + DS_IDLE_TIMEOUT_S)) holders
    while true; do
        holders=$(ds_device_holders)
        [[ -z "$holders" ]] && return 0
        if [[ $(date +%s) -ge $deadline ]]; then
            ds_log "device still open after ${DS_IDLE_TIMEOUT_S}s by:"
            echo "$holders" | sed "s/^/${DS_TAG}:   /"
            DS_LAST_REASON="device in use by another process: $(echo "$holders" | head -3 | tr '\n' ';')"
            return 1
        fi
        sleep 1
    done
}

# One health reading: chip count, GDDR training, heartbeat advancing between two snapshots.
ds_health_check() {
    local expected=$1 a b
    a=$(mktemp) && b=$(mktemp) || return 1
    if ! timeout 60 tt-smi -s --snapshot_no_tty > "$a" 2>/dev/null; then
        DS_LAST_REASON="tt-smi snapshot failed"
        rm -f "$a" "$b"
        return 1
    fi
    sleep 2
    if ! timeout 60 tt-smi -s --snapshot_no_tty > "$b" 2>/dev/null; then
        DS_LAST_REASON="tt-smi snapshot failed"
        rm -f "$a" "$b"
        return 1
    fi
    DS_LAST_REASON=$(python3 - "$a" "$b" "$expected" <<'EOF'
import json, sys
try:
    a, b = (json.load(open(p))["device_info"] for p in sys.argv[1:3])
except Exception as e:  # noqa: BLE001 - any unreadable snapshot is unhealthy
    print(f"unreadable tt-smi snapshot: {e}")
    sys.exit(0)
expected = int(sys.argv[3])
problems = []
if len(b) != expected:
    problems.append(f"{len(b)} chips enumerated, expected {expected}")
for i, (da, db) in enumerate(zip(a, b)):
    if not db.get("board_info", {}).get("dram_status", False):
        problems.append(f"chip {i}: DRAM status false")
    for ch in db.get("gddr_telemetry", {}).get("channels", []):
        if ch.get("enabled") and ch.get("training") != "pass":
            problems.append(f"chip {i}: GDDR channel {ch.get('channel')} training {ch.get('training')}")
    ha, hb = (int(d.get("telemetry", {}).get("heartbeat", 0) or 0) for d in (da, db))
    if hb <= ha:
        problems.append(f"chip {i}: firmware heartbeat not advancing ({ha} -> {hb})")
print("; ".join(problems))
EOF
)
    rm -f "$a" "$b"
    [[ -z "$DS_LAST_REASON" ]]
}

ds_health_gate() {
    local expected=$1 deadline=$(($(date +%s) + DS_HEALTH_TIMEOUT_S))
    while true; do
        if ds_health_check "$expected"; then
            sleep "$DS_SETTLE_S"
            ds_log "health gate passed: ${expected} chips enumerated, GDDR trained, firmware heartbeat advancing"
            return 0
        fi
        if [[ $(date +%s) -ge $deadline ]]; then
            ds_log "health gate FAILED after ${DS_HEALTH_TIMEOUT_S}s: ${DS_LAST_REASON}"
            return 1
        fi
        sleep 3
    done
}

ds_mark_unhealthy() {
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') pid=$$ $*" > "$DS_UNHEALTHY_FLAG"
    ds_log "device marked UNHEALTHY: $*"
}

ds_recent_fault_resets() {
    [[ -f "$DS_RESET_LOG" ]] || { echo 0; return; }
    awk -v since=$(($(date +%s) - DS_BREAKER_WINDOW_S)) '$1 >= since && $2 == "fault"' "$DS_RESET_LOG" | wc -l
}

# Call right after taking the device lock. Returns 0 when a run may use the device; otherwise prints why (with the
# DEVICE_UNHEALTHY / DEVICE_BREAKER_OPEN token callers and agents grep for) and returns 1.
ds_preflight() {
    if [[ -f "$DS_UNHEALTHY_FLAG" ]]; then
        ds_log "DEVICE_UNHEALTHY: refusing to run. $(cat "$DS_UNHEALTHY_FLAG")"
        ds_log "A person must check the board, then run: ${BASH_SOURCE[0]} clear"
        return 1
    fi
    local n
    n=$(ds_recent_fault_resets)
    if [[ "$n" -ge "$DS_BREAKER_RESETS" ]]; then
        ds_log "DEVICE_BREAKER_OPEN: ${n} resets after faults in the last $((DS_BREAKER_WINDOW_S / 60)) min, refusing to run."
        ds_log "Debug without the device (compile-only, static sync checks), or ask a person to run: ${BASH_SOURCE[0]} clear"
        awk -v since=$(($(date +%s) - DS_BREAKER_WINDOW_S)) '$1 >= since && $2 == "fault"' "$DS_RESET_LOG" |
            while read -r ts kind rest; do ds_log "  $(date -u -d "@$ts" '+%H:%M:%S') $rest"; done
        return 1
    fi
    return 0
}

# Guarded reset. kind = fault (after a hang, crash, killed run or stuck cores; counted by the breaker) or routine.
# Returns 0 only when the board was reset and passed the health gate.
ds_reset() {
    local kind=$1 reason=$2 expected cmd
    expected=$(ds_chip_count)
    if ! ds_wait_idle; then
        ds_mark_unhealthy "reset skipped (${reason}): ${DS_LAST_REASON}"
        return 1
    fi
    echo "$(date +%s) ${kind} ${reason// /_} pid=$$" >> "$DS_RESET_LOG"
    if [[ "$(ds_detect_system)" == galaxy ]]; then cmd=(tt-smi -glx_reset); else cmd=(tt-smi -r); fi
    ds_log "resetting (${kind}: ${reason}): ${cmd[*]}"
    if ! "${cmd[@]}"; then
        ds_mark_unhealthy "${cmd[*]} failed (${reason})"
        return 1
    fi
    if ! ds_health_gate "$expected"; then
        ds_mark_unhealthy "health gate failed after reset (${reason}): ${DS_LAST_REASON}"
        return 1
    fi
    return 0
}

# Appended to TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE: after triage, end the test process (the hook runs through
# std::system, so $PPID is that process). Metal runs the hook once per process, so without this a pytest session goes
# on running its remaining tests on the hung board. TT_SAFE_PYTEST_CONTINUE_AFTER_HANG=1 keeps the old behaviour.
ds_stop_after_hang_suffix() {
    if [[ "${TT_SAFE_PYTEST_CONTINUE_AFTER_HANG:-0}" == 1 ]]; then
        echo ""
    else
        echo '; kill -TERM $PPID 2>/dev/null'
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-status}" in
        status)
            echo "chips enumerated: $(ds_chip_count)  system: $(ds_detect_system)"
            if [[ -f "$DS_UNHEALTHY_FLAG" ]]; then echo "UNHEALTHY: $(cat "$DS_UNHEALTHY_FLAG")"; else echo "unhealthy flag: none"; fi
            echo "fault resets in the last $((DS_BREAKER_WINDOW_S / 60)) min: $(ds_recent_fault_resets) (breaker opens at ${DS_BREAKER_RESETS})"
            [[ -f "$DS_RESET_LOG" ]] && tail -10 "$DS_RESET_LOG" | while read -r ts kind rest; do
                echo "  $(date -u -d "@$ts" '+%Y-%m-%d %H:%M:%S') ${kind} ${rest}"
            done
            echo "processes holding the device:"
            ds_device_holders | sed 's/^/  /'
            ;;
        clear)
            rm -f "$DS_UNHEALTHY_FLAG"
            [[ -f "$DS_RESET_LOG" ]] && mv "$DS_RESET_LOG" "${DS_RESET_LOG}.$(date +%s).old"
            echo "cleared the unhealthy flag and the breaker history (the board itself was not reset)"
            ;;
        *)
            echo "usage: $0 [status|clear]" >&2
            exit 2
            ;;
    esac
fi
