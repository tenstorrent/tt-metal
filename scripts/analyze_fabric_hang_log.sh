#!/usr/bin/env bash
# analyze_fabric_hang_log.sh — Parse a CI log file and produce a structured summary
# for AI agent troubleshooting of fabric hang/crash bugs in tt-metal.
#
# Usage: analyze_fabric_hang_log.sh <logfile>
#        cat logfile | analyze_fabric_hang_log.sh -
#
# Dependencies: bash, grep, sed, sort, python3

set -eo pipefail

LOGFILE="${1:--}"
if [[ "$LOGFILE" == "-" ]]; then
    TMPF=$(mktemp /tmp/fabric_log_XXXXXX)
    cat > "$TMPF"
    LOGFILE="$TMPF"
    trap "rm -f $TMPF" EXIT
elif [[ ! -f "$LOGFILE" ]]; then
    echo "ERROR: File not found: $LOGFILE" >&2
    exit 1
fi

# Strip ANSI escape codes for clean processing
CLEAN=$(mktemp /tmp/fabric_clean_XXXXXX)
trap "rm -f $CLEAN ${TMPF:-}" EXIT
sed 's/\x1b\[[0-9;]*m//g; s/\[0m//g; s/\[90m//g; s/\[35m//g; s/\[37m//g; s/\[36;1m//g' "$LOGFILE" > "$CLEAN"

echo "========================================================================"
echo "  FABRIC HANG LOG ANALYSIS"
echo "========================================================================"
echo ""

# ─── RUNNER INFO ───
echo "=== RUNNER INFO ==="
grep -m1 "Runner name:" "$CLEAN" | sed "s/^.*Runner name: /Runner: /" || true
grep -m1 "Machine name:" "$CLEAN" | sed "s/^.*Machine name: /Machine: /" || true
grep -m1 "Complete job name:" "$CLEAN" | sed "s/^.*Complete job name: /Job: /" || true
echo ""

# ─── TIMELINE ───
echo "=== TIMELINE (fabric-relevant, deduplicated, relative seconds) ==="
grep -E '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+' "$CLEAN" | \
grep -E '(info|warning|error)' | \
grep -iE '(Phase|edm_status|quiesce|fabric|TERMINATE|wait_for|configure_fabric|write_launch|ENTRY|Pass [0-9]|health|AllGather|READY_FOR_TRAFFIC|summary|pre-init|stale|corrupt|skipping|Timeout|read failed|cancel|launch_msg|newly_dead|initialized)' | \
grep -viE '(hugepage|bind_area|motherboard|topology_mapper|num_routing_planes|errno|hwloc|cpuset)' | \
python3 -c "
import sys, re

first = None
seen_patterns = {}
output_lines = []

for line in sys.stdin:
    line = line.rstrip()
    m = re.search(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+)', line)
    if not m:
        continue
    h, mi, s, ms = int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7)[:3])
    ts_ms = h*3600000 + mi*60000 + s*1000 + ms
    if first is None:
        first = ts_ms
    rel = (ts_ms - first) / 1000.0

    msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always|UMD|Device)\s*\|\s*', '', line)

    # Dedup key: normalize device/chan numbers
    dedup = re.sub(r'Device \d+', 'Dev N', msg)
    dedup = re.sub(r'chan[= ]\d+', 'chan=N', dedup)
    dedup = re.sub(r'channel \d+', 'channel N', dedup)
    dedup = re.sub(r'eth_chan \d+', 'eth_chan N', dedup)
    dedup = re.sub(r'chip_id=\d+', 'chip_id=N', dedup)
    dedup = re.sub(r'logical \([^)]+\)', 'logical (N,N)', dedup)
    dedup = re.sub(r'device \d+', 'device N', dedup)

    # Always keep: summary lines, phase transitions, errors/warnings, entry/exit markers
    # But NOT per-channel noise like 'skipping soft reset', 'Fabric initialized on Device N'
    is_important = bool(re.search(
        r'(summary|entering Phase|Phase [0-9].*complete|Phase [0-9].*skip'
        r'|Phase [0-9].*failed|Phase [0-9].*read failed'
        r'|error|Timeout|timeout|deadbeef|threw|broken'
        r'|ENTRY|Pass \d|quiesce_internal'
        r'|AllGather|cancel)',
        msg, re.I))
    # Per-channel noise: demote to dedup even if it has 'skip'/'init' keywords
    is_per_chan_noise = bool(re.search(
        r'(skipping soft reset|Fabric initialized on Device|base-UMD relay firmware|configure_fabric_cores:)',
        msg))
    if is_per_chan_noise:
        is_important = False

    if is_important:
        pass  # always emit
    elif dedup in seen_patterns:
        seen_patterns[dedup] += 1
        continue  # skip duplicate
    else:
        seen_patterns[dedup] = 1

    if len(msg) > 150:
        msg = msg[:147] + '...'
    output_lines.append(f'+{rel:7.1f}s  {msg}')

collapsed = sum(v - 1 for v in seen_patterns.values() if v > 1)
for line in output_lines[:100]:
    print(line)
if collapsed > 0:
    print(f'          [{collapsed} repetitive per-channel lines collapsed]')
"
echo ""

# ─── PHASES ───
echo "=== PHASES ==="
grep -iE 'Phase [0-9]' "$CLEAN" | \
grep -iE '(info|warning|error).*(Metal|Test)' | \
python3 -c "
import sys, re
seen = set()
for line in sys.stdin:
    line = line.rstrip()
    msg = re.sub(r'^.*\|\s*(Metal|Test)\s*\|\s*', '', line)
    msg = msg.replace('quiesce_and_restart_fabric_workers: ', '')
    msg = msg.replace('wait_for_fabric_workers_ready: ', '')
    # Dedup
    key = re.sub(r'Device \d+', 'Dev N', msg)
    key = re.sub(r'eth_chan \d+', 'eth_chan N', key)
    key = re.sub(r'chan \d+', 'chan N', key)
    key = re.sub(r'logical \([^)]+\)', 'logical (N,N)', key)
    is_important = bool(re.search(r'(complete|entering|skip|failed|timeout|deadbeef|summary)', msg, re.I))
    if not is_important and key in seen:
        continue
    seen.add(key)
    if len(msg) > 140:
        msg = msg[:137] + '...'
    print(msg)
" | head -50
echo ""

# ─── EDM STATUS VALUES ───
echo "=== EDM STATUS ==="
echo "edm_status values:"
grep -oE 'edm_status=0x[0-9a-fA-F]+' "$CLEAN" | sort | uniq -c | sort -rn | head -15
echo ""
echo "status= values (from health checks):"
grep -oE 'status=0x[0-9a-fA-F]+' "$CLEAN" | sort | uniq -c | sort -rn | head -10
echo ""

# ─── RELAY PATH BROKEN ESCALATION ───
echo "=== RELAY PATH BROKEN ==="
python3 -c "
import sys, re
with open('$CLEAN') as f:
    lines = f.readlines()
broken_events = []
for line in lines:
    if re.search(r'relay.*path.*broken|fabric_relay_path_broken_', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        dev = re.search(r'Device (\d+)', msg)
        devid = dev.group(1) if dev else '?'
        phase = re.search(r'Phase [0-9.]+', msg)
        phase_str = phase.group(0) if phase else '(no phase)'
        broken_events.append((devid, phase_str, msg[:120]))
if not broken_events:
    print('  (none detected — relay path not set broken in this log)')
else:
    seen = set()
    for devid, phase, msg in broken_events:
        key = re.sub(r'Device \d+', 'Dev N', msg)
        key = re.sub(r'chan[= ]\d+', 'chan=N', key)
        if key not in seen:
            seen.add(key)
            print(f'  Device {devid} / {phase}: {msg}')
    devs = sorted(set(d for d,_,_ in broken_events))
    print(f'  => Devices with relay_path_broken: {devs} ({len(broken_events)} total events)')
"
echo ""

# ─── FORCE RESETS / PHASE 2 TIMEOUTS ───
echo "=== FORCE RESETS (assert_risc_reset_at_core events) ==="
grep -E 'assert_risc_reset_at_core|force.reset|risc_reset' "$CLEAN" 2>/dev/null | \
grep -iE '(info|warning|error)' | \
python3 -c "
import sys, re, signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
seen = set()
for line in sys.stdin:
    msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
    key = re.sub(r'Device \d+', 'Dev N', msg)
    key = re.sub(r'chan[= ]\d+', 'chan=N', key)
    if key not in seen:
        seen.add(key)
        print('  ' + msg[:140])
" || true
MUX_RESETS=$(grep -c 'assert_risc_reset_at_core\|force.reset.*ETH\|ETH.*force.reset' "$CLEAN" 2>/dev/null || echo 0)
echo "  (total matching lines: ${MUX_RESETS:-0})"
echo ""

# ─── NEWLY DEAD CHANNELS ───
echo "=== NEWLY DEAD CHANNELS ==="
grep -E 'newly.dead|newly_dead|dead.*channel|channel.*dead' "$CLEAN" 2>/dev/null | \
grep -iE '(info|warning|error)' | \
python3 -c "
import sys, re, signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
seen = set()
for line in sys.stdin:
    msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
    key = re.sub(r'Device \d+', 'Dev N', msg)
    key = re.sub(r'chan[= ]\d+', 'chan=N', key)
    if key not in seen:
        seen.add(key)
        print('  ' + msg[:140])
" || true
DEAD_CH=$(grep -cE 'newly.dead|newly_dead' "$CLEAN" 2>/dev/null || echo 0)
[[ "${DEAD_CH:-0}" -eq 0 ]] && echo "  (none detected)"
echo ""

# ─── PRE-INIT STATE ───
echo "=== PRE-INIT STATE (stale/corrupt channel scan) ==="
grep -E 'pre.?init|stale|0x49706550|canary|kBaseUmd|base.UMD.*relay|terminate_stale' "$CLEAN" 2>/dev/null | \
grep -iE '(info|warning|error)' | \
python3 -c "
import sys, re, signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
seen = set()
for line in sys.stdin:
    msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
    key = re.sub(r'Device \d+', 'Dev N', msg)
    key = re.sub(r'chan[= ]\d+', 'chan=N', key)
    if key not in seen:
        seen.add(key)
        print('  ' + msg[:140])
" || true
SENTINEL=$(grep -c '0x49706550' "$CLEAN" 2>/dev/null || echo 0)
echo "  (0x49706550 sentinel occurrences: ${SENTINEL:-0})"
GLOBAL_DL=$(grep -cE 'Global deadline expired|global.*deadline.*channel' "$CLEAN" 2>/dev/null || echo 0)
echo "  (Global deadline teardown events: ${GLOBAL_DL:-0})"
echo ""

# ─── ENTRY SNAPSHOT vs PHASE 5 STATUS COMPARISON ───
echo "=== ENTRY SNAPSHOT vs PHASE 5 STATUS (per device) ==="
python3 -c "
import sys, re
with open('$CLEAN') as f:
    content = f.read()

# Collect ENTRY snapshot lines: 'ENTRY ... edm_status=0x...'
entry = {}
for m in re.finditer(r'ENTRY.*?Device (\d+).*?chan[= ](\d+).*?edm_status=(0x[0-9a-fA-F]+)', content):
    key = (m.group(1), m.group(2))
    entry[key] = m.group(3)
for m in re.finditer(r'Device (\d+).*?ENTRY.*?chan[= ](\d+).*?edm_status=(0x[0-9a-fA-F]+)', content):
    key = (m.group(1), m.group(2))
    entry[key] = m.group(3)

# Collect Phase 5 / Phase 5b final status lines
phase5 = {}
for m in re.finditer(r'Phase 5[b]?.*?Device (\d+).*?chan[= ](\d+).*?(?:status|edm_status)=(0x[0-9a-fA-F]+)', content):
    key = (m.group(1), m.group(2))
    phase5[key] = m.group(3)

all_keys = sorted(set(list(entry.keys()) + list(phase5.keys())), key=lambda x: (int(x[0]), int(x[1])))
if not all_keys:
    print('  (no per-channel status data found in log)')
else:
    print(f'  {\"Device\":>6} {\"Chan\":>4} {\"ENTRY\":>12} {\"Phase5\":>12} {\"Changed?\":>10}')
    for dev, chan in all_keys:
        e = entry.get((dev, chan), '?')
        p = phase5.get((dev, chan), '?')
        changed = '  YES' if e != p and e != '?' and p != '?' else ''
        print(f'  {dev:>6} {chan:>4} {e:>12} {p:>12} {changed:>10}')
"
echo ""

# ─── ERRORS/WARNINGS ───
echo "=== ERRORS/WARNINGS (filtered, deduplicated) ==="
grep -iE '(warning|error|TT_THROW|TT_FATAL|Fatal|Abort)' "$CLEAN" | \
grep -iE '(Metal|Test|Fabric)' | \
grep -viE '(hugepage|bind_area|motherboard|Node\.js|DeprecationWarning|digest-mismatch|Buffer\(\)|topology_mapper|num_routing_planes|hwloc|cpuset|errno|issue_record_event|reserve begin|reserve ok|sub_device)' | \
python3 -c "
import sys, re, signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
seen = set()
for line in sys.stdin:
    line = line.rstrip()
    msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line)
    if len(msg) > 160:
        msg = msg[:157] + '...'
    key = re.sub(r'Device \d+', 'Device N', msg)
    key = re.sub(r'chan[= ]\d+', 'chan=N', key)
    key = re.sub(r'eth_chan \d+', 'eth_chan N', key)
    if key not in seen:
        seen.add(key)
        print(msg)
" | head -25
echo ""

# ─── HANG INDICATORS ───
echo "=== HANG INDICATORS (gaps > 30s between consecutive GHA timestamps) ==="
python3 -c "
import sys, re
prev_sec = None
prev_line = ''
with open('$CLEAN') as f:
    for line in f:
        line = line.rstrip()
        m = re.match(r'^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', line)
        if m:
            sec = int(m.group(4))*3600 + int(m.group(5))*60 + int(m.group(6))
            if prev_sec is not None:
                gap = sec - prev_sec
                if gap < 0: gap += 86400
                if gap > 30:
                    print(f'  GAP: {gap}s')
                    print(f'    BEFORE: {prev_line[:140]}')
                    print(f'    AFTER:  {line[:140]}')
                    print()
            prev_sec = sec
            prev_line = line
"
echo ""

# ─── LAST 50 LINES ───
echo "=== LAST 50 LINES (before cancel/cleanup) ==="
CANCEL_LINE=$(grep -n 'operation was canceled' "$CLEAN" | head -1 | cut -d: -f1)
if [[ -n "$CANCEL_LINE" && "$CANCEL_LINE" -gt 0 ]]; then
    START=$((CANCEL_LINE - 50))
    [[ $START -lt 1 ]] && START=1
    sed -n "${START},${CANCEL_LINE}p" "$CLEAN" | \
        grep -E '(info|warning|error|Phase|edm_status|quiesce|fabric|TERMINATE|Timeout|cancel|READY|read failed)' | \
        python3 -c "
import sys, re
for line in sys.stdin:
    line = line.rstrip()
    line = re.sub(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+Z ', '', line)
    if len(line) > 160:
        line = line[:157] + '...'
    print(line)
" | tail -50
else
    tail -50 "$CLEAN"
fi
echo ""

# ─── SUMMARY ───
echo "=== SUMMARY ==="
RUNNER=$(grep -m1 "Runner name:" "$CLEAN" | sed "s/^.*Runner name: //" | tr -d "'" | xargs)
JOB=$(grep -m1 "Complete job name:" "$CLEAN" | sed "s/^.*Complete job name: //" | xargs)
CANCEL_TS=$(grep -m1 'operation was canceled' "$CLEAN" | grep -oE '^[0-9-]*T[0-9:]*' || echo "unknown")

LAST_METAL_TS=$(grep -E '(info|warning).*Metal' "$CLEAN" | tail -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1 || echo "unknown")
LAST_METAL_MSG=$(grep -E '(info|warning).*Metal' "$CLEAN" | tail -1 | sed 's/^.*| *Metal *| *//' | head -c 160)

P5B_FAILS=$(grep -c 'Phase 5b.*read failed' "$CLEAN" 2>/dev/null || true)
P5B_FAILS=${P5B_FAILS:-0}
P5B_SKIP=$(grep -c 'Phase 5b.*skipping' "$CLEAN" 2>/dev/null || true)
P5B_SKIP=${P5B_SKIP:-0}
PROBLEM_DEVS=$(grep -iE '(read failed|Timeout|0xDEAD5B5B|0xDEADECE7|deadbeef)' "$CLEAN" | grep -oE 'Device [0-9]+' | sort -u | tr '\n' ', ' | sed 's/,\s*$//')

HANG_DUR="unknown"
if [[ "$CANCEL_TS" != "unknown" && "$LAST_METAL_TS" != "unknown" ]]; then
    C_SEC=$(date -d "${CANCEL_TS/T/ }" +%s 2>/dev/null || echo 0)
    L_SEC=$(date -d "$LAST_METAL_TS" +%s 2>/dev/null || echo 0)
    if [[ $C_SEC -gt 0 && $L_SEC -gt 0 ]]; then
        HANG_DUR="$((C_SEC - L_SEC))s"
    fi
fi

# Detect which failure patterns are present, then emit a targeted diagnosis.
HAS_RELAY_BROKEN=$(grep -c 'relay.*path.*broken\|fabric_relay_path_broken_' "$CLEAN" 2>/dev/null || echo 0)
HAS_P4_TIMEOUT=$(grep -cE 'Phase 4.*TIMEOUT|Phase 4.*timeout|MUX.*timeout|Timeout.*MUX' "$CLEAN" 2>/dev/null || echo 0)
HAS_EXCEPTION=$(grep -cE 'TT_THROW|TT_FATAL|Fatal|Abort' "$CLEAN" 2>/dev/null || echo 0)
HAS_FORCE_RESET=$(grep -c 'assert_risc_reset_at_core\|force.reset' "$CLEAN" 2>/dev/null || echo 0)

if [[ "${HAS_RELAY_BROKEN:-0}" -gt 0 ]]; then
    DIAGNOSIS="UMD relay path breakdown (fabric_relay_path_broken_ set). After Phase 3 loaded
fabric firmware on the MMIO device's relay ERISCs, subsequent non-MMIO reads via those
relay channels either timed out (5s each) or hung indefinitely. Check RELAY PATH BROKEN
and ENTRY SNAPSHOT sections above for affected devices/channels."
elif [[ "${HAS_P4_TIMEOUT:-0}" -gt 0 ]]; then
    DIAGNOSIS="Phase 4 Tensix MUX timeout: a MUX core did not reach READY_FOR_TRAFFIC within
the allotted window after firmware relaunch. The MUX was force-reset and the job aborted.
Check Phase 4 log lines and FORCE RESETS section for the affected channel."
elif [[ "${HAS_EXCEPTION:-0}" -gt 0 && "${HAS_RELAY_BROKEN:-0}" -eq 0 ]]; then
    DIAGNOSIS="Exception (TT_THROW/TT_FATAL) during quiesce without relay-path-broken flag set.
Check ERRORS/WARNINGS section for the thrown message and the surrounding log context."
elif [[ "${HAS_FORCE_RESET:-0}" -gt 0 ]]; then
    DIAGNOSIS="Force resets (assert_risc_reset_at_core) were applied during Phase 2. Check
whether the reset succeeded and whether Phase 3 proceeded on a still-running MUX core.
Check FORCE RESETS and Phase 2 timing in the PHASES section."
else
    DIAGNOSIS="No known failure pattern detected. Check ERRORS/WARNINGS and NEWLY DEAD CHANNELS
sections above for clues. Consider examining the raw PHASES timeline for unexpected gaps."
fi

cat <<SUMMARY_EOF
Runner: $RUNNER
Job: $JOB
Canceled at: $CANCEL_TS
Last Metal log: $LAST_METAL_TS
Hang duration (last log -> cancel): $HANG_DUR
Phase 5b read failures: $P5B_FAILS
Phase 5b skipped (relay broken): $P5B_SKIP
Problem devices: ${PROBLEM_DEVS:-none detected}

Last logged action: $LAST_METAL_MSG

Diagnosis: $DIAGNOSIS
SUMMARY_EOF
echo ""
echo "========================================================================"
