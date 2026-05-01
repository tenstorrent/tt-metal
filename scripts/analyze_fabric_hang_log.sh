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
grep -iE '(Phase|edm_status|quiesce|fabric|TERMINATE|wait_for|configure_fabric|write_launch|ENTRY|Pass[- ][0-9]|Pass-0|health|AllGather|READY_FOR_TRAFFIC|summary|pre-init|pre-launch|stale|corrupt|skipping|Timeout|read failed|cancel|launch_msg|newly.dead|newly_dead|initialized|deferred|degraded|FIX AB extension|FIX AC|FIX AE|FIX AJ|FIX AK|FIX AL|FIX AM|FIX AN|FIX AQ|FIX AT|FIX AU|FIX AV|FIX AW|FIX AX|FIX AY|FIX AZ|FIX BA|FIX M2|FIX NS|FIX NT|FIX NU|FIX NX|FIX NY|FIX PL|FIX TE|FIX TF|FIX X|teardown:.*relay|post_teardown:.*FIX|canary|force.reset|NOT ready after|UMD ready after|marking dead|relay confirmed dead|relay-dead|relay-broken non-MMIO|deferred.*ERISC|restored relay|STARTED early.exit|skipping Phase 5b|Pass-0 timeout.*handshake|master chan.*FIX AS|edm_status_address.*sentinel|ROM postcode|channels_not_ready_for_traffic|STARTED.*adding.*relay_broken|fabric_teardown_timed_out.*set|wait_for_non_mmio_flush.*threw|mark_relay_broken.*close_device|Marking relay broken|topology discovery|redundant.*topology|Physical chip id not found|EthCoord.*missing|chip_locations.*incomplete|Captured EthCoord.*MMIO|EthCoord.*FIX NT|EthCoord.*FIX NU|relay already known broken|relay_broken_chips|non-base firmware running|training status will never be written|ETH_TRAIN_STATUS_ADDR|l1_barrier timed out.*dead ERISC|dram_barrier timed out.*non-MMIO|WriteInitMagic.*read_core timed out|T3K topology check FAILED|chips visible|No forwarding direction|chip excluded by FIX TB)' | \
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
grep -iE 'Phase [0-9]|Pass-0|SUMMARY|teardown: FIX AC|FIX AB extension|FIX AE|FIX AJ|FIX AK|FIX AL|FIX AM|FIX AN|FIX AQ|FIX AT|FIX AU|FIX AV|FIX AW|FIX AX|FIX AY|FIX AZ|FIX BA|FIX NS|FIX NT|FIX NU|FIX NX|FIX NY|FIX TE|FIX TF|FIX X|post_teardown:.*FIX AB|pre-launch|deferred|degraded|STARTED early.exit|skipping Phase 5b|Pass-0 timeout.*handshake|master chan.*FIX AS|edm_status_address.*sentinel|ROM postcode|channels_not_ready_for_traffic|STARTED.*adding.*relay_broken|fabric_teardown_timed_out.*set|wait_for_non_mmio_flush.*threw|Marking relay broken|Physical chip id not found|Captured EthCoord.*MMIO|relay already known broken|non-base firmware running|ETH_TRAIN_STATUS_ADDR|No forwarding direction|chip excluded by FIX TB' "$CLEAN" | \
grep -iE '(info|warning|error).*(Metal|Test|Always)' | \
python3 -c "
import sys, re, signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
seen = set()
for line in sys.stdin:
    line = line.rstrip()
    msg = re.sub(r'^.*\|\s*(Metal|Test|Always)\s*\|\s*', '', line)
    msg = msg.replace('quiesce_and_restart_fabric_workers: ', '')
    msg = msg.replace('wait_for_fabric_workers_ready: ', '')
    msg = msg.replace('launch_eth_cores_for_quiesce: ', 'launch_eth: ')
    # Dedup
    key = re.sub(r'Device \d+', 'Dev N', msg)
    key = re.sub(r'eth_chan \d+', 'eth_chan N', key)
    key = re.sub(r'chan \d+', 'chan N', key)
    key = re.sub(r'logical \([^)]+\)', 'logical (N,N)', key)
    key = re.sub(r'0x[0-9a-fA-F]+', '0xNN', key)
    is_important = bool(re.search(r'(complete|entering|skip|failed|timeout|deadbeef|SUMMARY|Pass-0.*complete|deferred|degraded|FIX AC|FIX AB extension|teardown_timed_out)', msg, re.I))
    if not is_important and key in seen:
        continue
    seen.add(key)
    if len(msg) > 140:
        msg = msg[:137] + '...'
    print(msg)
" | head -50 || true
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
# Also capture FIX AS Pass-0 "marking dead" messages (the log text uses "marking dead"
# rather than "newly_dead", but the semantic meaning is identical)
grep -E 'newly.dead|newly_dead|dead.*channel|channel.*dead|configure_fabric_cores.*newly.dead|marking dead.*skipping launch|Pass-0.*marking dead' "$CLEAN" 2>/dev/null | \
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
DEAD_CH=$(grep -cE 'newly.dead|newly_dead|configure_fabric_cores.*newly.dead' "$CLEAN" 2>/dev/null || echo 0)
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
# Canary detections
CANARY_COUNT=$(grep -c '0xA0A0A0A0\|A0A0A0A0' "$CLEAN" 2>/dev/null || echo 0)
if [ "$CANARY_COUNT" -gt 0 ]; then
    echo "  => CANARY 0xA0A0A0A0 detected $CANARY_COUNT times (fabric firmware crashed before INITIALIZATION_STARTED)"
    grep '0xA0A0A0A0\|A0A0A0A0' "$CLEAN" | head -5 | while IFS= read -r line; do
        echo "    $line" | cut -c1-140
    done
else
    echo "  0xA0A0A0A0 canary: (none detected)"
fi
echo ""

# ─── PHASE 5B SENTINELS ───
echo "=== PHASE 5B SENTINELS ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
deadline_skip = []
read_exc = []
for line in lines:
    dev = re.search(r'Device (\d+)', line)
    chan = re.search(r'chan[= ](\d+)', line)
    devid = dev.group(1) if dev else '?'
    chanid = chan.group(1) if chan else '?'
    if '0xDEAD5B5B' in line or 'DEAD5B5B' in line.upper():
        deadline_skip.append((devid, chanid))
    if '0xDEADECE7' in line or 'DEADECE7' in line.upper():
        read_exc.append((devid, chanid))
if not deadline_skip and not read_exc:
    print('  (none detected)')
else:
    for devid, chanid in deadline_skip:
        print(f'  Device {devid} chan {chanid}: deadline-skipped (0xDEAD5B5B)')
    for devid, chanid in read_exc:
        print(f'  Device {devid} chan {chanid}: read-exception (0xDEADECE7)')
    print(f'  => deadline-skipped: {len(deadline_skip)}, read-exception: {len(read_exc)}')
"
echo ""

# ─── QUIESCE INVOCATION COUNT ───
echo "=== QUIESCE INVOCATION COUNT ==="
python3 -c "
import re, sys
from collections import defaultdict
with open('$CLEAN') as f:
    lines = f.readlines()
counts = defaultdict(int)
for line in lines:
    if 'ENTRY snapshot' in line:
        dev = re.search(r'Device (\d+)', line)
        if dev:
            counts[dev.group(1)] += 1
if not counts:
    print('  (no ENTRY snapshot lines found)')
else:
    for dev, cnt in sorted(counts.items(), key=lambda x: int(x[0])):
        flag = ' *** CASCADING QUIESCE ***' if cnt > 1 else ''
        print(f'  Device {dev}: {cnt} quiesce invocation(s){flag}')
    if any(c > 1 for c in counts.values()):
        print('  => Double-quiesce cascade detected. Second quiesce is triggered by GTest TearDown after test failure.')
"
echo ""

# ─── RESCUE STUCK DISPATCH ───
echo "=== RESCUE STUCK DISPATCH ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
events = []
for line in lines:
    if re.search(r'rescue_stuck_dispatch|rescue.*dispatch|stuck.*dispatch', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        events.append(msg[:140])
if not events:
    print('  (none detected)')
else:
    seen = set()
    for msg in events:
        key = re.sub(r'\d+', 'N', msg)
        if key not in seen:
            seen.add(key)
            print(f'  {msg}')
    print(f'  => {len(events)} rescue_stuck_dispatch event(s). Each stream write = 5s UMD timeout; many streams = minutes of hang.')
"
echo ""

# ─── FIX AD DETECTION ───
echo "=== FIX AD (hard-reset stuck dispatch cores) ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
events = []
for line in lines:
    if re.search(r'rescue_stuck_dispatch_cores.*hard.*reset|hard BRISC reset|performing hard BRISC reset', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        events.append(msg[:140])
if not events:
    print('  (none detected)')
else:
    seen = set()
    for msg in events:
        key = re.sub(r'\d+', 'N', msg)
        if key not in seen:
            seen.add(key)
            print(f'  {msg}')
    print(f'  => FIX AD triggered: {len(events)} hard-BRISC-reset event(s) on stuck dispatch cores')
"
echo ""

# ─── FIX W DETECTION ───
echo "=== FIX W (all-dead clean return) ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
events = []
for line in lines:
    if re.search(r'FIX W|Phase 5b.*all.*truly.*unhealthy.*stuck at 0x0|all.*dead.*clean return', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        events.append(msg[:140])
if not events:
    print('  (none detected)')
else:
    seen = set()
    for msg in events:
        key = re.sub(r'\d+', 'N', msg)
        if key not in seen:
            seen.add(key)
            print(f'  {msg}')
    print(f'  => FIX W triggered: entire ERISC layer dead, clean return path taken ({len(events)} event(s))')
"
echo ""

# ─── FIX AA DETECTION ───
echo "=== FIX AA (AllGather skip — relay path broken) ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
events = []
for line in lines:
    if re.search(r'FIX AA|relay path broken.*skipping AllGather|skipping AllGather', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        events.append(msg[:140])
if not events:
    print('  (none detected)')
else:
    seen = set()
    for msg in events:
        key = re.sub(r'\d+', 'N', msg)
        if key not in seen:
            seen.add(key)
            print(f'  {msg}')
    print(f'  => FIX AA triggered: AllGather skipped due to broken relay path ({len(events)} event(s))')
"
echo ""

# ─── PHASE 4 MUX STATUS PROGRESSION ───
echo "=== PHASE 4 MUX STATUS PROGRESSION ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
events = []
raw_statuses = []  # (chan_key, hex_status) tuples in order
for line in lines:
    if re.search(r'Phase 4.*still waiting.*status=0x|Phase 4.*MUX', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        events.append(msg[:140])
        # Extract (chan, status) for stuck-detection
        chan_m   = re.search(r'chan[= ](\d+)', msg, re.I)
        status_m = re.search(r'status=(0x[0-9a-fA-F]+)', msg, re.I)
        chan_key = chan_m.group(1) if chan_m else '?'
        hex_val  = status_m.group(1).lower() if status_m else None
        if hex_val:
            raw_statuses.append((chan_key, hex_val))
if not events:
    print('  (none detected)')
else:
    seen = set()
    for msg in events:
        key = re.sub(r'status=0x[0-9a-fA-F]+', 'status=0xN', msg)
        key = re.sub(r'\d+ms', 'Nms', key)
        key = re.sub(r'Device \d+', 'Dev N', key)
        key = re.sub(r'chan[= ]\d+', 'chan=N', key)
        if key not in seen:
            seen.add(key)
            print(f'  {msg}')
    print(f'  => {len(events)} Phase 4 MUX line(s). Check whether status=0x values progress or repeat (stuck).')
    # Stuck detection: look for >=3 consecutive identical (chan, status) pairs
    if raw_statuses:
        # Group consecutive runs by (chan, status)
        run_chan, run_val, run_len = raw_statuses[0][0], raw_statuses[0][1], 1
        stuck_reported = set()
        for chan_key, hex_val in raw_statuses[1:]:
            if chan_key == run_chan and hex_val == run_val:
                run_len += 1
            else:
                if run_len >= 3:
                    key = (run_chan, run_val)
                    if key not in stuck_reported:
                        stuck_reported.add(key)
                        print(f'  [STUCK] MUX status unchanged across {run_len} polls ({run_val}) on chan {run_chan} — MUX is stuck, not slow')
                run_chan, run_val, run_len = chan_key, hex_val, 1
        # Check final run
        if run_len >= 3:
            key = (run_chan, run_val)
            if key not in stuck_reported:
                print(f'  [STUCK] MUX status unchanged across {run_len} polls ({run_val}) on chan {run_chan} — MUX is stuck, not slow')
"
echo ""

# ─── PHASE 2.5 ERISC TERMINATION LATENCY ───
echo "=== PHASE 2.5 ERISC TERMINATION LATENCY ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
timing_events = []
timeout_events = []
for line in lines:
    if re.search(r'Phase 2\.?5.*ERISC.*TERMINATED.*ms|Phase 2\.?5.*terminated.*[0-9]+ms', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        timing_events.append(msg[:140])
    elif re.search(r'Phase 2\.?5.*timeout', line, re.I):
        msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
        timeout_events.append(msg[:140])
if not timing_events and not timeout_events:
    print('  (none detected)')
else:
    if timing_events:
        print(f'  Termination timing ({len(timing_events)} line(s)):')
        seen = set()
        for msg in timing_events:
            key = re.sub(r'[0-9]+ms', 'Nms', msg)
            key = re.sub(r'Device \d+', 'Dev N', key)
            key = re.sub(r'chan[= ]\d+', 'chan=N', key)
            if key not in seen:
                seen.add(key)
                print(f'    {msg}')
    if timeout_events:
        print(f'  Timeouts ({len(timeout_events)} line(s)):')
        seen = set()
        for msg in timeout_events:
            key = re.sub(r'Device \d+', 'Dev N', msg)
            key = re.sub(r'chan[= ]\d+', 'chan=N', key)
            if key not in seen:
                seen.add(key)
                print(f'    {msg}')
"
echo ""

# ─── PHASE 2.5 → PHASE 3 DANGER CHECK ───
echo "=== PHASE 2.5 → PHASE 3 DANGER CHECK ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    lines = f.readlines()
# Find every Phase 2.5 timeout line; then check if a Phase 3 configure line appears
# within the next 5 lines.  If so, ERISC L1 may have been overwritten while live.
danger_found = False
for i, line in enumerate(lines):
    if re.search(r'Phase 2\.?5.*(timeout|did not terminate)', line, re.I):
        for j in range(i + 1, min(i + 6, len(lines))):
            if re.search(r'Phase 3.*(configure|write_launch|entering Phase 3)', lines[j], re.I):
                danger_found = True
                timeout_msg = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', line).rstrip()
                phase3_msg  = re.sub(r'^.*\|\s*(Metal|Test|Fabric|Always)\s*\|\s*', '', lines[j]).rstrip()
                print(f'  [DANGER] Phase 2.5 timed out but Phase 3 proceeded — ERISC L1 may have been overwritten while ERISC was live')
                print(f'    timeout: {timeout_msg[:120]}')
                print(f'    phase3:  {phase3_msg[:120]}')
                break
if not danger_found:
    print('  (no Phase 2.5 timeout → Phase 3 overlap detected)')
"
echo ""

# ─── DEVICE MMIO CLASSIFICATION ───
echo "=== DEVICE MMIO CLASSIFICATION ==="
python3 -c "
import re, sys
with open('$CLEAN') as f:
    content = f.read()
mmio_devs = set()
non_mmio_devs = set()
# grep for mmio=true / mmio=false lines and extract device IDs
for line in content.splitlines():
    if re.search(r'mmio=true', line, re.I):
        for m in re.finditer(r'[Dd]evice[_ ](\d+)|chip_id=(\d+)', line):
            devid = m.group(1) or m.group(2)
            mmio_devs.add(devid)
    if re.search(r'mmio=false', line, re.I):
        for m in re.finditer(r'[Dd]evice[_ ](\d+)|chip_id=(\d+)', line):
            devid = m.group(1) or m.group(2)
            non_mmio_devs.add(devid)
if not mmio_devs and not non_mmio_devs:
    print('  (no mmio= labels found in log)')
else:
    mmio_sorted = sorted(mmio_devs, key=lambda x: int(x))
    non_mmio_sorted = sorted(non_mmio_devs, key=lambda x: int(x))
    print(f'  MMIO devices ({len(mmio_sorted)}): {mmio_sorted}')
    print(f'  non-MMIO devices ({len(non_mmio_sorted)}): {non_mmio_sorted}')
    if non_mmio_sorted:
        print(f'  => non-MMIO failures indicate relay path issues (firmware loaded via MMIO relay)')
    if mmio_sorted:
        print(f'  => MMIO failures indicate direct hardware or firmware init problems')
"
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

# ─── DISPATCH CASCADE (FIX PA/PB/PC) ───
echo "=== DISPATCH CASCADE (500ms fw_launch_addr stale — FIX PA/PB/PC/PD) ==="
python3 -c "
import re, sys, collections
with open('$CLEAN') as f:
    lines = f.readlines()

timeout_events = []
force_reset_events = []
fix_aq_events = []
fix_pa_events = []
per_device_cores = collections.defaultdict(set)

for line in lines:
    if re.search(r'Timeout \\(500 ms\\) waiting for physical cores', line):
        msg = re.sub(r'^.*\\|\\s*(Metal|Test|Fabric|Always|critical)\\s*\\|\\s*', '', line).rstrip()
        timeout_events.append(msg[:140])
        dev_m = re.search(r'Device (\\d+):', msg)
        cores_m = re.search(r'finish: ([0-9, -]+)', msg)
        if dev_m and cores_m:
            dev = dev_m.group(1)
            cores = cores_m.group(1).strip()
            per_device_cores[dev].add(cores)
    elif re.search(r'Force-resetting stale ETH cores on device \\d+ to prevent', line):
        msg = re.sub(r'^.*\\|\\s*(Metal|Test|Fabric|Always)\\s*\\|\\s*', '', line).rstrip()
        force_reset_events.append(msg[:140])
    elif re.search(r'FIX AQ.*Failed to init remote device', line):
        fix_aq_events.append(line.rstrip())
    elif re.search(r'FIX PA|erisc_app_still_running.*force.reset|fw_launch_addr.*cleared', line, re.I):
        msg = re.sub(r'^.*\\|\\s*(Metal|Test|Fabric|Always)\\s*\\|\\s*', '', line).rstrip()
        fix_pa_events.append(msg[:140])

total_cascades = len(timeout_events)
total_force_resets = len(force_reset_events)
fix_aq_count = len(fix_aq_events)

if total_cascades == 0:
    print('  (no 500ms dispatch cascade detected — FIX PA/PB/PC appear effective)')
else:
    print(f'  [CASCADE DETECTED] {total_cascades} x 500ms timeout(s), {total_force_resets} force-reset(s)')
    print(f'  FIX AQ overhead: {fix_aq_count} x 5s UMD timeout(s) = ~{fix_aq_count * 5}s total')
    print(f'  Estimated cascade overhead: {total_cascades * 0.5 + fix_aq_count * 5:.0f}s ({(total_cascades * 0.5 + fix_aq_count * 5)/60:.1f} min)')
    print()
    print('  Affected devices and core coordinates:')
    for dev in sorted(per_device_cores.keys(), key=int):
        for core_set in per_device_cores[dev]:
            print(f'    Device {dev}: {core_set}')
    print()
    if fix_pa_events:
        print(f'  FIX PA mitigation active ({len(fix_pa_events)} event(s)):')
        seen = set()
        for msg in fix_pa_events:
            key = re.sub(r'\\d+', 'N', msg)
            if key not in seen:
                seen.add(key)
                print(f'    {msg}')
    else:
        print('  FIX PA mitigation: NOT detected in log')
    print()
    if total_cascades > 1:
        print('  [ROOT CAUSE] fw_launch_addr not cleared after fabric teardown force-reset.')
        print('  FIX PD (device.cpp quiesce Pass-0 deassert) should eliminate this — FIX PC was in the wrong path.')
        print('  If FIX PC is present and cascade persists, check: write_core_immediate')
        print('  failed silently for affected channel coordinates (catch(...) swallows it).')
"
echo ""

# ─── DISPATCH CQ DEADLOCK ───
echo "=== DISPATCH CQ DEADLOCK (fetch_queue_reserve_back timeout — GAP-58) ==="
grep -E 'fetch_queue_reserve_back|cq_id=.*in_flight=128|Timeout detected' "$CLEAN" | \
python3 -c "
import sys, re, signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)
lines = [l.rstrip() for l in sys.stdin]
cq_timeouts = [l for l in lines if 'fetch_queue_reserve_back timeout' in l]
if not cq_timeouts:
    print('  (no fetch_queue_reserve_back timeout detected — dispatch CQ appears healthy)')
else:
    print(f'  [CQ DEADLOCK DETECTED] {len(cq_timeouts)} fetch_queue_reserve_back timeout(s)')
    for l in cq_timeouts[:5]:
        m = re.search(r'cq_id=(\d+) in_flight=(\d+) ptrs=(0x[0-9a-f]+) fences=(0x[0-9a-f]+)', l)
        if m:
            cq_id, inflight, ptrs, fences = m.group(1), m.group(2), m.group(3), m.group(4)
            print(f'  cq_id={cq_id} in_flight={inflight} write_ptr={ptrs} completion_ptr={fences}')
        else:
            print(f'  {l[-120:]}')
    print()
    print('  Root cause: Dispatch CQ filled to 128 in-flight commands; device stopped consuming.')
    print('  Likely fabric/dispatch deadlock: dispatch waiting for fabric, fabric waiting for dispatch.')
    print('  This is GAP-58 — the actual race condition being hunted (cascade was masking this).')
    print('  Next: Investigate system_memory_manager.cpp:fetch_queue_reserve_back + AllGather interaction.')
" || true
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
" | head -25 || true
echo ""

# ─── OPERATION TIMEOUT EVENTS ───
echo "=== OPERATION TIMEOUT EVENTS (TT_METAL_OPERATION_TIMEOUT_SECONDS) ==="
python3 -c "
import sys, re
events = []
with open('$CLEAN') as f:
    for line in f:
        line = line.rstrip()
        if re.search(r'TT_METAL_OPERATION_TIMEOUT|operation.*timeout.*seconds|OperationTimeout|Operation.*timed out', line, re.I):
            events.append(line)
if events:
    for e in events[:20]:
        print('  ' + e[:200])
    if len(events) > 20:
        print(f'  ... and {len(events)-20} more')
    print(f'  => {len(events)} operation-level timeout event(s) detected')
else:
    print('  (none detected)')
"
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
CANCEL_LINE=$(grep -n 'operation was canceled' "$CLEAN" | head -1 | cut -d: -f1 || true)
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
RUNNER=$(grep -m1 "Runner name:" "$CLEAN" | sed "s/^.*Runner name: //" | tr -d "'" | xargs || true)
JOB=$(grep -m1 "Complete job name:" "$CLEAN" | sed "s/^.*Complete job name: //" | xargs || true)
CANCEL_TS=$(grep -m1 'operation was canceled' "$CLEAN" | grep -oE '^[0-9-]*T[0-9:]*' || echo "unknown")

LAST_METAL_TS=$(grep -E '(info|warning).*Metal' "$CLEAN" | tail -1 | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1 || echo "unknown")
LAST_METAL_MSG=$(grep -E '(info|warning).*Metal' "$CLEAN" | tail -1 | sed 's/^.*| *Metal *| *//' | head -c 160)

P5B_FAILS=$(grep -c 'Phase 5b.*read failed' "$CLEAN" 2>/dev/null || true)
P5B_FAILS=${P5B_FAILS:-0}
P5B_SKIP=$(grep -c 'Phase 5b.*skipping' "$CLEAN" 2>/dev/null || true)
P5B_SKIP=${P5B_SKIP:-0}
PROBLEM_DEVS=$(grep -iE '(read failed|Timeout|0xDEAD5B5B|0xDEADECE7|deadbeef)' "$CLEAN" | grep -oE 'Device [0-9]+' | sort -u | tr '\n' ', ' | sed 's/,\s*$//' || true)

HANG_DUR="unknown"
if [[ "$CANCEL_TS" != "unknown" && "$LAST_METAL_TS" != "unknown" ]]; then
    C_SEC=$(date -d "${CANCEL_TS/T/ }" +%s 2>/dev/null || echo 0)
    L_SEC=$(date -d "$LAST_METAL_TS" +%s 2>/dev/null || echo 0)
    if [[ $C_SEC -gt 0 && $L_SEC -gt 0 ]]; then
        HANG_DUR="$((C_SEC - L_SEC))s"
    fi
fi

# Detect which failure patterns are present, then emit a targeted diagnosis.
HAS_DISPATCH_CASCADE=$(grep -c 'Timeout (500 ms) waiting for physical cores' "$CLEAN" 2>/dev/null || echo 0)
HAS_RELAY_BROKEN=$(grep -c 'relay.*path.*broken\|fabric_relay_path_broken_' "$CLEAN" 2>/dev/null || echo 0)
HAS_P4_TIMEOUT=$(grep -cE 'Phase 4.*TIMEOUT|Phase 4.*timeout|MUX.*timeout|Timeout.*MUX|Timeout.*MUX READY_FOR_TRAFFIC|MUX READY_FOR_TRAFFIC.*[Tt]imeout|Timeout waiting for fabric MUX' "$CLEAN" 2>/dev/null || echo 0)
HAS_EXCEPTION=$(grep -cE 'TT_THROW|TT_FATAL|Fatal|Abort' "$CLEAN" 2>/dev/null || echo 0)
HAS_FORCE_RESET=$(grep -c 'assert_risc_reset_at_core\|force.reset' "$CLEAN" 2>/dev/null || echo 0)
FIX_Z=$(grep -c 'is_fabric_relay_path_broken\|relay.*broken.*completion_queue\|CQ.*relay.*broken' "$CLEAN" 2>/dev/null || echo 0)
FIX_AB=$(grep -cE 'hard-reset.*MMIO|RiscFirmwareInitializer.*teardown|MMIO ETH.*reset|fabric_teardown_timed_out_.*set.*device|FIX AB extension' "$CLEAN" 2>/dev/null || echo 0)
# FIX AB extension: post_teardown flag that quiesce timed out; triggers Step 5 FIX AC reset
FIX_AB_EXT=$(grep -cE 'FIX AB extension|post_teardown.*FIX AB|fabric_teardown_timed_out.*set.*hard-reset|teardown: FIX AC \(timeout-only\)' "$CLEAN" 2>/dev/null || echo 0)
FIX_AD=$(grep -cE 'rescue_stuck_dispatch_cores.*hard.*reset|hard BRISC reset|performing hard BRISC reset' "$CLEAN" 2>/dev/null || echo 0)
FIX_W=$(grep -cE 'FIX W|Phase 5b.*all.*truly.*unhealthy.*stuck at 0x0|all.*dead.*clean return' "$CLEAN" 2>/dev/null || echo 0)
FIX_AA=$(grep -ciE 'FIX AA|relay path broken.*skipping AllGather|skipping AllGather' "$CLEAN" 2>/dev/null || echo 0)
FIX_V=$(grep -cE 'FIX V|Setting fabric_relay_path_broken_=true to skip relay ops in subsequent quiesce|Phase 5.*timeout.*0x0.*non-MMIO|status still 0x0 on non-MMIO device' "$CLEAN" 2>/dev/null || echo 0)
RELAY_RESTORED=$(grep -c 'relay-broken flag reset by configure_fabric' "$CLEAN" 2>/dev/null || echo 0)
# FIX-1: MMIO device Phase 5 timeout now also sets fabric_relay_path_broken_ (removed !is_mmio_capable() guard)
FIX_1_MMIO=$(grep -cE 'Setting fabric_relay_path_broken_=true|Phase 5.*timeout.*0x0.*MMIO|fabric_relay_path_broken_.*MMIO' "$CLEAN" 2>/dev/null || echo 0)
# FIX AS: Pass-0 canary poll events (per-channel polling before write_launch_msg)
FIX_AS_PASS0=$(grep -cE 'Pass-0 \(FIX AS\)|Pass-0 \(FIX AR\+AS\) complete' "$CLEAN" 2>/dev/null || echo 0)
# FIX_AS_TIMEOUT: actual log messages are "NOT ready after ... marking dead" and
# "did NOT reach UMD ready after ... marking dead, skipping launch"
FIX_AS_TIMEOUT=$(grep -cE 'Pass-0.*NOT ready after|Pass-0.*did NOT reach.*UMD ready|Pass-0.*marking dead|canary not seen' "$CLEAN" 2>/dev/null || echo 0)
# FIX AC: teardown MMIO ETH reset events
FIX_AC_FIRES=$(grep -cE 'FIX AC' "$CLEAN" 2>/dev/null || echo 0)
# FIX AU: relay-broken non-MMIO channels bypassed poll loop (FIX AU #42429)
FIX_AU_FIRES=$(grep -cE 'FIX AU|bypassed the poll loop.*relay-broken' "$CLEAN" 2>/dev/null || echo 0)
# FIX AX: relay-confirmed-dead non-MMIO channels skipped assert_risc_reset
FIX_AX_FIRES=$(grep -cE 'FIX AX|relay confirmed dead.*skipping assert_risc_reset' "$CLEAN" 2>/dev/null || echo 0)
# FIX AJ: relay path confirmed dead during assert_risc_reset (marks relay_dead_devices)
FIX_AJ_FIRES=$(grep -cE 'FIX AJ|relay path confirmed dead during force-reset' "$CLEAN" 2>/dev/null || echo 0)
# FIX AK (FabricFirmware): transitive relay guard — skipping l1_barrier for ALL non-MMIO
FIX_AK_FIRES=$(grep -cE 'relay-dead device.*confirmed.*skipping l1_barrier.*FIX AK|FIX AK.*skipping l1_barrier' "$CLEAN" 2>/dev/null || echo 0)
# FIX AQ: secondary edm_status_address sentinel poll after FIX AR heartbeat poll
FIX_AQ_FIRES=$(grep -cE 'FIX AQ' "$CLEAN" 2>/dev/null || echo 0)
FIX_AQ_TIMEOUT=$(grep -cE 'FIX AQ.*ROM postcode.*after.*ms' "$CLEAN" 2>/dev/null || echo 0)
# FIX AT: Phase 5 handshake poll skipped when MMIO master chan was FIX AS Pass-0 timeout'd
FIX_AT_FIRES=$(grep -cE 'FIX AT|Pass-0 timeout.*skipping.*handshake|master chan.*FIX AS.*Pass-0 timeout' "$CLEAN" 2>/dev/null || echo 0)
# FIX AY: deferred non-MMIO ETH ERISC reset via restored MMIO relay
# FIX AV: skip remaining ETH cores on same device when first assert_risc_reset times out
FIX_AY_FIRES=$(grep -cE 'FIX AY' "$CLEAN" 2>/dev/null || echo 0)
FIX_AY_SUCCEEDED=$(grep -cE 'FIX AY.*all.*reset to base firmware|FIX AY.*succeeded' "$CLEAN" 2>/dev/null || echo 0)
FIX_AY_FAILED=$(grep -cE 'FIX AY.*failed|FIX AY.*non-std exception|FIX AY/AV.*failed' "$CLEAN" 2>/dev/null || echo 0)
FIX_AV_FIRES=$(grep -cE 'FIX AV #42429|FIX AY/AV.*Skipping all remaining' "$CLEAN" 2>/dev/null || echo 0)
# FIX AL: STARTED early-exit — Phase 5 master chan stuck at EDMStatus::STARTED after kStartedTimeoutMs
FIX_AL_FIRES=$(grep -cE 'FIX AL|STARTED early-exit after.*ms.*master chan' "$CLEAN" 2>/dev/null || echo 0)
# FIX AM: Phase 5b skip when master chan still at STARTED after FIX AL break
FIX_AM_FIRES=$(grep -cE 'FIX AM|skipping Phase 5b.*FIX AM|channels_not_ready_for_traffic.*FIX AM|still at STARTED.*skipping Phase 5b' "$CLEAN" 2>/dev/null || echo 0)
# FIX AW: ~Cluster destructor runs driver_->close_device() in detached thread to avoid wait_for_non_mmio_flush hang
FIX_AW_FIRES=$(grep -cE 'FIX AW|relay-broken non-MMIO.*running driver.*close_device.*background thread|close_device.*did not complete.*5s.*FIX AW' "$CLEAN" 2>/dev/null || echo 0)
# FIX BA: STARTED-state non-MMIO devices added to relay_broken_non_mmio (FIX AM fired, relay_broken=false)
FIX_BA_FIRES=$(grep -cE 'FIX BA|channels_not_ready_for_traffic.*relay not marked broken.*Adding to relay_broken_non_mmio|teardown: FIX BA' "$CLEAN" 2>/dev/null || echo 0)
# FIX AE: catch wait_for_non_mmio_flush() timeout in write_core/write_reg/noc_multicast_write + mark_relay_broken() for ~Cluster()
# Pattern: "FIX AE: wait_for_non_mmio_flush(chip N) threw: ... Marking relay broken."
FIX_AE_FLUSH_TIMEOUT=$(grep -cE 'FIX AE.*wait_for_non_mmio_flush.*threw|FIX AE.*Marking relay broken' "$CLEAN" 2>/dev/null || echo 0)
# FIX AE also marks all remote chips relay-broken in ~Cluster() before close_device() — no specific log (silent path).
# FIX NS: eliminate redundant topology discovery in initialize_base_objects (no runtime log — structural fix)
# Regression evidence: 14+ min SIGALRM before any test, "unit_tests_ttnn" hangs without FIX AQ warning
# FIX NT: preserve EthCoord for FIX-AQ-skipped remote chips in chip_locations
# Log: "FIX AQ: Failed to init remote device ... Skipping" (FIX AQ fires; NT adds EthCoord silently)
# Regression evidence: TT_FATAL "Physical chip id not found for eth coord" @ tt_cluster.cpp:575
FIX_NT_CRASH=$(grep -cE 'Physical chip id not found for eth coord|TT_FATAL.*chip id not found.*eth' "$CLEAN" 2>/dev/null || echo 0)
# FIX NU: capture MMIO EthCoord (via PCIe NODE_INFO) before FIX W heartbeat guard
# Log: "FIX NU: Captured EthCoord for MMIO device ASIC ID" (debug-level only, may not appear in CI)
# Regression evidence: same TT_FATAL as FIX NT but MMIO chip missing; FIX W skipped ALL ETH channels
FIX_NU_COORD=$(grep -cE 'FIX NU: Captured EthCoord|Captured EthCoord.*MMIO.*before relay' "$CLEAN" 2>/dev/null || echo 0)
# FIX NX+NY: write_core() relay guard for non-MMIO chips — FIX NX catches first timeout exception,
# FIX NY caches the failure in relay_broken_chips_ so subsequent calls for same chip skip UMD.
# FIX NX log: "FIX NX: write_core(chip N) threw: ... Marking relay broken (FIX NX+NY)."
# FIX NY log: "FIX NY: write_core(chip N) skipped — relay already known broken." (debug-level)
# Regression evidence (FIX NX missing): write_core exception propagates to MetalContext::initialize.
# Regression evidence (FIX NY missing): N_channels × 5s serial stall per dead chip — GHA timeout.
#   CI ref: run 25086219070 (job 73503180670): 6 channels × 5s = 30s → GHA 5-min action timeout.
FIX_NX_THROWS=$(grep -cE 'FIX NX: write_core.*threw|Marking relay broken \(FIX NX\+NY\)' "$CLEAN" 2>/dev/null || echo 0)
FIX_NY_SKIPS=$(grep -cE 'FIX NY: write_core.*skipped.*relay already known broken' "$CLEAN" 2>/dev/null || echo 0)
# FIX TE (#42429): control_plane.cpp callers of try_get_asic_id_from_fabric_node_id skip chips
# excluded by FIX TB (no entry in topology mapper — degraded topology from corrupt ERISC L1).
# Log: "FIX TE (#42429): Skipping order_ethernet_channels for FabricNodeId {}"
#      "FIX TE (#42429): FabricNodeId (M{}, D{}) not found in topology mapper"
# Regression evidence: without FIX TE, TT_FATAL in configure_routing_tables or order_ethernet_channels
#   for FIX TB-excluded chips → SIGABRT; with FIX TE, warning logged and chip skipped.
FIX_TE_SKIPS=$(grep -cE 'FIX TE.*Skipping order_ethernet_channels|FIX TE.*not found in topology mapper' "$CLEAN" 2>/dev/null || echo 0)
# FIX TF (#42429): assemble_2d_fabric_packet_header_args TT_FATAL guard — replaces opaque
# bad_optional_access thrown when inter-mesh relay is broken (get_forwarding_direction returns nullopt).
# Log (TT_FATAL, appears before SIGABRT): "FIX TF: No forwarding direction from physical chip N"
# Regression evidence: without FIX TF, GTest sees "bad optional access" with no chip context;
#   with FIX TF, TT_FATAL names src/dst chip IDs and fabric node IDs.
FIX_TF_FIRES=$(grep -cE 'FIX TF: No forwarding direction' "$CLEAN" 2>/dev/null || echo 0)
# FIX M2 (#42429): Secondary check in compile_and_configure_fabric() — channel showed 0x49706550 (base-UMD relay)
# but peer non-MMIO device is confirmed dead-relay → remove from base_umd_channels so configure_fabric_cores()
# performs a hard soft-reset (no relay reads in flight, safe to reset).
FIX_M2_FIRES=$(grep -cE 'FIX M2.*dead-relay|compile_and_configure_fabric: FIX M2' "$CLEAN" 2>/dev/null || echo 0)
# FIX PL (#42429): opt-in timeout guards on l1_barrier / dram_barrier / read_core for non-MMIO chips.
# Fires when the ERISC relay path is dead and the barrier/read would otherwise hang indefinitely.
FIX_PL_FIRES=$(grep -cE 'clear_l1_state: l1_barrier timed out.*dead ERISC relay|clear_dram_state: dram_barrier timed out|terminate_active_ethernet_cores_on_all_chips: l1_barrier timed out|WriteInitMagic: read_core timed out' "$CLEAN" 2>/dev/null || echo 0)

if [[ "${HAS_DISPATCH_CASCADE:-0}" -gt 0 ]]; then
    DIAGNOSIS="500ms dispatch cascade (FIX PA/PB/PC pattern): ${HAS_DISPATCH_CASCADE} Timeout(500ms)
events on ETH dispatch cores. Root cause: fw_launch_addr not cleared after fabric teardown
force-reset. FIX PD (device.cpp quiesce Pass-0 deassert) should prevent this.
If FIX PC is committed, cascade may persist for channels where write_core_immediate
failed silently (catch(...) swallows it). Check DISPATCH CASCADE section above."
elif [[ "${HAS_RELAY_BROKEN:-0}" -gt 0 ]]; then
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
if [ "${HAS_DISPATCH_CASCADE:-0}" -gt 0 ]; then
    echo "  => [DISPATCH CASCADE] ${HAS_DISPATCH_CASCADE} x 500ms fw_launch_addr stale timeout(s) — FIX PD needed in device.cpp quiesce_and_restart_fabric_workers Pass-0"
fi
if [ "${FIX_Z:-0}" -gt 0 ]; then
    echo "  => FIX Z triggered: relay path broken check in read_completion_queue_event — test was skipped or fast-failed"
fi
if [ "${FIX_AB:-0}" -gt 0 ]; then
    echo "  => FIX AB triggered: hard-reset of MMIO ETH channels at process teardown"
fi
if [ "${FIX_AB_EXT:-0}" -gt 0 ]; then
    echo "  => [FIX AB extension] post_teardown: quiesce timed out (fabric_teardown_timed_out_ set); Step 5 FIX AC fired to hard-reset MMIO ETH (${FIX_AB_EXT} event(s)). Relay was intact (no relay_broken_non_mmio). This is the 'timeout-only' path — channels didn't reach TERMINATED in time but relay survived."
fi
if [ "${FIX_AD:-0}" -gt 0 ]; then
    echo "  => FIX AD triggered: hard BRISC reset on stuck dispatch cores (${FIX_AD} event(s))"
fi
if [ "${FIX_W:-0}" -gt 0 ]; then
    echo "  => FIX W triggered: all-dead clean return path — entire ERISC layer was dead (${FIX_W} event(s))"
fi
if [ "${FIX_AA:-0}" -gt 0 ]; then
    echo "  => FIX AA triggered: AllGather skipped due to broken relay path (${FIX_AA} event(s))"
fi
if [ "${FIX_V:-0}" -gt 0 ]; then
    echo "  => [FIX V] triggered: non-MMIO device Phase 5 timeout with status=0x0 — fabric_relay_path_broken_ set (${FIX_V} event(s))"
fi
if [ "${RELAY_RESTORED:-0}" -gt 0 ]; then
    echo "  => [RELAY RESTORED] relay-broken flag cleared by configure_fabric — UMD relay path restored (${RELAY_RESTORED} event(s))"
fi
if [ "${FIX_1_MMIO:-0}" -gt 0 ]; then
    echo "  => [FIX-1] MMIO device Phase 5 timeout set fabric_relay_path_broken_ (${FIX_1_MMIO} event(s)) — FIX AC teardown path should have fired"
fi
if [ "${FIX_AS_PASS0:-0}" -gt 0 ]; then
    echo "  => [FIX AS] Pass-0 canary poll fired (${FIX_AS_PASS0} event(s)) — force-reset channels polling for UMD canary before write_launch_msg"
fi
if [ "${FIX_AS_TIMEOUT:-0}" -gt 0 ]; then
    echo "  => [FIX AS TIMEOUT] Pass-0 canary poll timed out on ${FIX_AS_TIMEOUT} channel(s) — channels marked newly-dead; degraded mesh"
fi
if [ "${FIX_AC_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AC] teardown ETH reset path fired (${FIX_AC_FIRES} event(s)) — MMIO ETH cores PCIe-reset at process teardown"
fi
if [ "${FIX_AU_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AU] relay-broken non-MMIO channels bypassed poll loop (${FIX_AU_FIRES} event(s)) — pre-populated relay_dead_devices skipped heartbeat"
fi
if [ "${FIX_AX_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AX] relay-confirmed-dead non-MMIO ETH: assert_risc_reset_at_core skipped (${FIX_AX_FIRES} event(s)) — ERISCs left in FABRIC fw; FIX AY should clean up next"
fi
if [ "${FIX_AJ_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AJ] relay path confirmed dead during force-reset pass (${FIX_AJ_FIRES} event(s)) — device added to relay_dead_devices set"
fi
if [ "${FIX_AK_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AK] transitive relay guard: l1_barrier skipped for ALL non-MMIO devices (${FIX_AK_FIRES} event(s)) — relay-dead device(s) present"
fi
if [ "${FIX_AQ_FIRES:-0}" -gt 0 ]; then
    if [ "${FIX_AQ_TIMEOUT:-0}" -eq 0 ]; then
        echo "  => [FIX AQ] edm_status_address sentinel poll complete (${FIX_AQ_FIRES} event(s)) — UMD relay wrote 0x49706550 before next session started"
    else
        echo "  => [FIX AQ] edm_status_address sentinel poll TIMED OUT (${FIX_AQ_TIMEOUT} channel(s)) — 0x49705180 ROM postcode persisted; next session may still see corrupt L1"
    fi
fi
if [ "${FIX_AY_FIRES:-0}" -gt 0 ]; then
    if [ "${FIX_AY_FAILED:-0}" -eq 0 ]; then
        echo "  => [FIX AY] deferred non-MMIO ERISC reset succeeded (${FIX_AY_FIRES} event(s), ${FIX_AY_SUCCEEDED} ERISCs reset) — next session should find base fw"
    else
        if [ "${FIX_AV_FIRES:-0}" -gt 0 ]; then
            echo "  => [FIX AY/AV] deferred non-MMIO ERISC reset: relay dead on first core, remaining cores skipped (FIX AV fired ${FIX_AV_FIRES}×) — ${FIX_AY_SUCCEEDED} ok / ${FIX_AY_FAILED} failed/skipped"
        else
            echo "  => [FIX AY] deferred non-MMIO ERISC reset PARTIAL (${FIX_AY_FIRES} event(s), ${FIX_AY_SUCCEEDED} ok / ${FIX_AY_FAILED} failed) — some ERISCs may retain FABRIC fw"
        fi
    fi
fi
# FIX AV: sysmem_manager_->reset() skipped for relay-broken non-MMIO devices
FIX_AV_SKIP=$(grep -cE 'running in degraded mode|configure_fabric.*degraded' "$CLEAN" 2>/dev/null || echo 0)
if [ "${FIX_AV_SKIP:-0}" -gt 0 ]; then
    echo "  => [FIX AV / DEGRADED] configure_fabric degraded mode fired (${FIX_AV_SKIP} event(s)) — pre-dead channels skipped; relay-broken sysmem reset guard may have applied"
fi
if [ "${FIX_AT_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AT] Phase 5 handshake poll skipped (${FIX_AT_FIRES} event(s)) — master chan was FIX AS Pass-0 timeout'd (WH BRISC boot >500ms, status=0x0 after deassert). No firmware was loaded so Phase 5 poll would waste 10s; FIX AT early-exits + sets fabric_relay_path_broken_=true to skip Phase 5b. Saves 10s per affected MMIO device."
fi
if [ "${FIX_AL_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AL] STARTED early-exit fired (${FIX_AL_FIRES} event(s)) — master chan stuck at EDMStatus::STARTED for >3s (out-of-mesh peer unreachable, ETH handshake never completes). Phase 5 poll exits early; FIX AM should fire next to skip Phase 5b."
fi
if [ "${FIX_AM_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AM] Phase 5b skipped after FIX AL STARTED early-exit (${FIX_AM_FIRES} event(s)) — master chan still at STARTED after poll exits; subordinates stuck at REMOTE_HANDSHAKE_COMPLETE; Phase 5b is pointless. Sets fabric_channels_not_ready_for_traffic_=true. FIX BA should fire at teardown to clean up."
fi
if [ "${FIX_AW_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AW] ~Cluster destructor: driver_->close_device() running in background thread with 5s timeout (${FIX_AW_FIRES} event(s)) — relay-broken non-MMIO chips registered by FIX BA/teardown; wait_for_non_mmio_flush would hang indefinitely on stale UMD relay CMD queue entries. NOTE: FIX AW was superseded by FIX AE (commit 561f7abd505) which marks chips broken before close_device() — no thread detach needed."
fi
if [ "${FIX_AE_FLUSH_TIMEOUT:-0}" -gt 0 ]; then
    echo "  => [FIX AE] wait_for_non_mmio_flush() timed out mid-session in write_core/write_reg/noc_multicast_write (${FIX_AE_FLUSH_TIMEOUT} event(s)) — relay died while a write was in progress. FIX AE caught the exception and called mark_relay_broken() so all subsequent flushes for that chip return instantly (0ms instead of 5s per call). Also: ~Cluster() marks all remote chips broken before close_device() to prevent UMD destructor/constructor race (supersedes FIX AW background-thread approach)."
fi
if [ "${FIX_NT_CRASH:-0}" -gt 0 ]; then
    echo "  => [FIX NT REGRESSION] 'Physical chip id not found for eth coord' TT_FATAL detected (${FIX_NT_CRASH} occurrence(s))."
    echo "     Root cause: FIX AQ skipped a remote chip during topology discovery but its EthCoord was NOT"
    echo "     preserved in chip_locations (FIX NT missing/reverted). Code that called get_physical_chip_id_"
    echo "     from_eth_coord() for the skipped chip's coord found no entry → TT_FATAL → SIGABRT."
    echo "     Fix: UMD topology_discovery.cpp FIX NT — after FIX AQ 'continue', emplace EthCoord in eth_coords."
    echo "     CI refs: run 25077304186 (FIX NT), run 25079761804 (FIX NU — same crash, different root cause)."
fi
if [ "${FIX_NU_COORD:-0}" -gt 0 ]; then
    echo "  => [FIX NU] MMIO EthCoord captured before relay-safety guards (${FIX_NU_COORD} event(s)) — FIX W heartbeat guard"
    echo "     fired for some MMIO ETH channels but get_local_eth_coord() PCIe read ran first → MMIO chip in chip_locations."
fi
if [ "${FIX_BA_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX BA] teardown: STARTED-state non-MMIO device(s) added to relay_broken_non_mmio (${FIX_BA_FIRES} event(s)) — FIX AM set channels_not_ready_for_traffic=true but NOT fabric_relay_path_broken_. Without FIX BA: FIX AC + FIX AY skip these devices; STARTED ERISCs remain; next session stalls 5s/device in topology discovery. FIX BA forces FIX AC + FIX AY to clean them up."
fi
if [ "${FIX_NX_THROWS:-0}" -gt 0 ]; then
    echo "  => [FIX NX] write_core() relay timeout caught for non-MMIO chip(s) (${FIX_NX_THROWS} throw(s)) — write_to_device() timed out (5s) on dead relay; FIX NX caught the exception so it did not propagate to MetalContext::initialize(). FIX NY should have cached the broken chip to skip subsequent channels immediately."
fi
if [ "${FIX_NY_SKIPS:-0}" -gt 0 ]; then
    echo "  => [FIX NY] write_core() relay-broken cache hit (${FIX_NY_SKIPS} skip(s)) — relay_broken_chips_ set by FIX NX after first timeout; all subsequent write_core() calls for that chip returned immediately (0ms). Note: FIX NY log is debug-level and may not appear in CI logs."
elif [ "${FIX_NX_THROWS:-0}" -gt 0 ]; then
    echo "  => [FIX NY] NOTE: FIX NX fired (${FIX_NX_THROWS} throw(s)) but FIX NY skip log not found in this log."
    echo "     FIX NY log is debug-level (tt::LogDevice) — typically suppressed in CI."
    echo "     If testee took >35s for MetalContext::initialize, FIX NY may be missing/reverted."
    echo "     CI ref: run 25086219070 (job 73503180670): 6 FIX NX throws at 5s intervals = 30s stall."
    echo "     FIX NY fix: relay_broken_chips_ unordered_set in Cluster; check before write_to_device()."
fi
if [ "${FIX_M2_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX M2] compile_and_configure_fabric: secondary base-UMD channel check fired (${FIX_M2_FIRES} channel(s))."
    echo "     Channel(s) showed 0x49706550 sentinel (base-UMD relay active) BUT peer non-MMIO device was"
    echo "     already in dead_relay_devices_ (confirmed unreachable by PHASE 1 probe). FIX M2 removes these"
    echo "     channels from base_umd_channels_map so configure_fabric_cores() performs a hard soft-reset"
    echo "     (assert+deassert ERISC0) instead of skipping — safe because no relay reads are in flight."
    echo "     FIX I handles the resulting dead-peer sync handshake skip in PHASE 2."
fi
if [ "${FIX_PL_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX PL] Non-MMIO relay barrier/read timed out (${FIX_PL_FIRES} occurrence(s))."
    echo "     Affected call sites: clear_l1_state l1_barrier, clear_dram_state dram_barrier,"
    echo "     terminate_active_ethernet_cores_on_all_chips l1_barrier, WriteInitMagic read_core."
    echo "     All four routes through the ERISC relay on non-MMIO chips — when the relay is dead,"
    echo "     the call would hang indefinitely without the FIX PL timeout guard."
    echo "     Each timed-out call logs a warning and continues (best-effort, no throw)."
fi
if [ "${FIX_TE_SKIPS:-0}" -gt 0 ]; then
    echo "  => [FIX TE] control_plane: FIX TB-excluded chip(s) skipped in routing table config (${FIX_TE_SKIPS} skip(s))."
    echo "     FIX TB excluded chip(s) from topology mapper (unknown ASIC ID — degraded topology)."
    echo "     FIX TE guards configure_routing_tables_for_fabric_ethernet_channels() and"
    echo "     order_ethernet_channels() so they skip chips with no topology mapper entry"
    echo "     instead of TT_FATAL on the mandatory lookup."
fi
if [ "${FIX_TF_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TF] assemble_2d_fabric_packet_header_args: no inter-mesh route found (${FIX_TF_FIRES} occurrence(s))."
    echo "     get_forwarding_direction() returned nullopt for a cross-mesh chip pair."
    echo "     Before FIX TF: .value() threw bad_optional_access → GTest caught opaque error in SetUp()."
    echo "     FIX TF: TT_FATAL with src/dst physical chip IDs and fabric node IDs for diagnosis."
    echo "     Indicates inter-mesh relay is broken (chips in degraded mode, no routing table entry)."
elif grep -qE 'bad optional access' "$CLEAN" 2>/dev/null; then
    echo "  => [FIX TF MISSING] 'bad optional access' found in log — FIX TF may be absent or reverted."
    echo "     Source: assemble_2d_fabric_packet_header_args in relay_mux.hpp calling .value() on"
    echo "     get_forwarding_direction() result without has_value() check."
    echo "     When inter-mesh relay is broken, this throws bad_optional_access caught by GTest in SetUp()."
    echo "     Fix: add TT_FATAL(forwarding_direction.has_value(), \"FIX TF: ...\") before .value() call."
fi
echo ""
echo "========================================================================"
