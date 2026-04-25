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
grep -iE '(Phase|edm_status|quiesce|fabric|TERMINATE|wait_for|configure_fabric|write_launch|ENTRY|Pass [0-9]|health|AllGather|READY_FOR_TRAFFIC|summary|pre-init|stale|corrupt|skipping|Timeout|read failed|cancel|launch_msg|newly.dead|newly_dead|initialized)' | \
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
grep -E 'newly.dead|newly_dead|dead.*channel|channel.*dead|configure_fabric_cores.*newly.dead' "$CLEAN" 2>/dev/null | \
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
HAS_P4_TIMEOUT=$(grep -cE 'Phase 4.*TIMEOUT|Phase 4.*timeout|MUX.*timeout|Timeout.*MUX|Timeout.*MUX READY_FOR_TRAFFIC|MUX READY_FOR_TRAFFIC.*[Tt]imeout|Timeout waiting for fabric MUX' "$CLEAN" 2>/dev/null || echo 0)
HAS_EXCEPTION=$(grep -cE 'TT_THROW|TT_FATAL|Fatal|Abort' "$CLEAN" 2>/dev/null || echo 0)
HAS_FORCE_RESET=$(grep -c 'assert_risc_reset_at_core\|force.reset' "$CLEAN" 2>/dev/null || echo 0)
FIX_Z=$(grep -c 'is_fabric_relay_path_broken\|relay.*broken.*completion_queue\|CQ.*relay.*broken' "$CLEAN" 2>/dev/null || echo 0)
FIX_AB=$(grep -cE 'hard-reset.*MMIO|RiscFirmwareInitializer.*teardown|MMIO ETH.*reset|fabric_teardown_timed_out_.*set.*device|FIX AB extension' "$CLEAN" 2>/dev/null || echo 0)
FIX_AD=$(grep -cE 'rescue_stuck_dispatch_cores.*hard.*reset|hard BRISC reset|performing hard BRISC reset' "$CLEAN" 2>/dev/null || echo 0)
FIX_W=$(grep -cE 'FIX W|Phase 5b.*all.*truly.*unhealthy.*stuck at 0x0|all.*dead.*clean return' "$CLEAN" 2>/dev/null || echo 0)
FIX_AA=$(grep -ciE 'FIX AA|relay path broken.*skipping AllGather|skipping AllGather' "$CLEAN" 2>/dev/null || echo 0)
FIX_V=$(grep -cE 'FIX V|Setting fabric_relay_path_broken_=true to skip relay ops in subsequent quiesce|Phase 5.*timeout.*0x0.*non-MMIO|status still 0x0 on non-MMIO device' "$CLEAN" 2>/dev/null || echo 0)
RELAY_RESTORED=$(grep -c 'relay-broken flag reset by configure_fabric' "$CLEAN" 2>/dev/null || echo 0)

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
if [ "${FIX_Z:-0}" -gt 0 ]; then
    echo "  => FIX Z triggered: relay path broken check in read_completion_queue_event — test was skipped or fast-failed"
fi
if [ "${FIX_AB:-0}" -gt 0 ]; then
    echo "  => FIX AB triggered: hard-reset of MMIO ETH channels at process teardown"
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
echo ""
echo "========================================================================"
