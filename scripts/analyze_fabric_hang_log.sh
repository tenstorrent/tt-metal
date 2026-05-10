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

# Strip ANSI/VT100 escape codes and GitHub Actions group markers for clean processing.
# Handles: color codes (\x1b[...m), cursor movement, character set switches (\x1b(B),
# carriage returns, and ##[group]/##[endgroup] GitHub Actions annotations.
CLEAN=$(mktemp /tmp/fabric_clean_XXXXXX)
trap "rm -f $CLEAN ${TMPF:-}" EXIT
sed $'s/\x1b\\[[0-9;]*[A-Za-z]//g; s/\x1b(B//g; s/\r//g; s/##\\[.*\\]//g' "$LOGFILE" > "$CLEAN"

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
grep -iE '(Phase|edm_status|quiesce|fabric|TERMINATE|wait_for|configure_fabric|write_launch|ENTRY|Pass[- ][0-9]|Pass-0|health|AllGather|READY_FOR_TRAFFIC|summary|pre-init|pre-launch|stale|corrupt|skipping|Timeout|read failed|cancel|launch_msg|newly.dead|newly_dead|initialized|deferred|degraded|fixture_teardown|fixture_setup|run_mailbox|failed to initialize FW|FIX AB extension|FIX AC|FIX AE|FIX AJ|FIX AK|FIX AL|FIX AM|FIX AN|FIX AQ|FIX AR2|FIX AT|FIX AU|FIX AV|FIX AW|FIX AX|FIX AY|FIX AZ|FIX BA|FIX BB|FIX BC|FIX CD|FIX DT-1|FIX GS|FIX M2|FIX NS|FIX NT|FIX NU|FIX NW|FIX NX|FIX NY|FIX PF|FIX PG|FIX PL|FIX PY|FIX QU|FIX QV|FIX QW|FIX RM|FIX RR|FIX RS|FIX RX|FIX RZ|FIX SA|FIX SB|FIX SB2-R|FIX SC|FIX TE|FIX TF|FIX TG|FIX TH|FIX TJ|FIX TV|FIX TW|FIX XY-2|FIX EXT|FIX X|teardown:.*relay|post_teardown:.*FIX|canary|force.reset|NOT ready after|UMD ready after|marking dead|relay confirmed dead|relay-dead|relay-broken non-MMIO|relay_broken cleared|relay path restored|deferred.*ERISC|restored relay|STARTED early.exit|skipping Phase 5b|Pass-0 timeout.*handshake|master chan.*FIX AS|edm_status_address.*sentinel|ROM postcode|channels_not_ready_for_traffic|STARTED.*adding.*relay_broken|fabric_teardown_timed_out.*set|wait_for_non_mmio_flush.*threw|mark_relay_broken.*close_device|Marking relay broken|topology discovery|redundant.*topology|Physical chip id not found|EthCoord.*missing|chip_locations.*incomplete|Captured EthCoord.*MMIO|EthCoord.*FIX NT|EthCoord.*FIX NU|relay already known broken|relay_broken_chips|non-base firmware running|training status will never be written|ETH_TRAIN_STATUS_ADDR|l1_barrier timed out.*dead ERISC|dram_barrier timed out.*non-MMIO|WriteInitMagic.*read_core timed out|T3K topology check FAILED|chips visible|No forwarding direction|chip excluded by FIX TB|has no host rank in topology mapper|no available dispatch links|invalid for WORMHOLE_B0|FIX TK|FIX TL|FIX TM|FIX TN|not in fabric cluster|skipping create_unit_meshes|worker tensix info map|cluster too degraded|NOT a valid EDMStatus|zeroed edm_status_address|0xdeaddead|open_devices_internal failed|FabricSwitchManager.*setup failed|warm-up complete|warm-up failed|Fabric health check failed|still-initializing|extending fabric_router_sync_timeout|skipping L1 clear|all_gather.*barrier|ENTERING.*barrier|EXITED.*barrier|topology damaged|topology recovered|topology still degraded|fabric_telemetry_dump|fabric_baseline_compare|rr_recovered|PCIe-direct soft reset|FIX BH|FIX BO|FIX BP)' | \
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
grep -iE 'Phase [0-9]|Pass-0|SUMMARY|teardown: FIX AC|FIX AB extension|FIX AE|FIX AJ|FIX AK|FIX AL|FIX AM|FIX AN|FIX AQ|FIX AR2|FIX AT|FIX AU|FIX AV|FIX AW|FIX AX|FIX AY|FIX AZ|FIX BA|FIX BB|FIX BC|FIX CD|FIX DT-1|FIX GS|FIX NS|FIX NT|FIX NU|FIX NW|FIX NX|FIX NY|FIX PF|FIX PG|FIX PY|FIX QU|FIX QV|FIX QW|FIX RM|FIX RR|FIX RS|FIX RX|FIX RZ|FIX SA|FIX SB|FIX SB2-R|FIX SC|FIX TE|FIX TF|FIX TG|FIX TH|FIX TJ|FIX TV|FIX TW|FIX XY-2|FIX EXT|FIX X|post_teardown:.*FIX AB|pre-launch|deferred|degraded|fixture_teardown|fixture_setup|run_mailbox|failed to initialize FW|STARTED early.exit|skipping Phase 5b|Pass-0 timeout.*handshake|master chan.*FIX AS|edm_status_address.*sentinel|ROM postcode|channels_not_ready_for_traffic|STARTED.*adding.*relay_broken|fabric_teardown_timed_out.*set|wait_for_non_mmio_flush.*threw|Marking relay broken|relay_broken cleared|relay path restored|Physical chip id not found|Captured EthCoord.*MMIO|relay already known broken|non-base firmware running|ETH_TRAIN_STATUS_ADDR|No forwarding direction|chip excluded by FIX TB|has no host rank in topology mapper|no available dispatch links|invalid for WORMHOLE_B0|FIX TK|FIX TL|FIX TM|FIX TN|not in fabric cluster|skipping create_unit_meshes|worker tensix info map|cluster too degraded|NOT a valid EDMStatus|zeroed edm_status_address|0xdeaddead|open_devices_internal failed|FabricSwitchManager.*setup failed|warm-up complete|warm-up failed|Fabric health check failed|still-initializing|extending fabric_router_sync_timeout|skipping L1 clear|all_gather.*barrier|ENTERING.*barrier|EXITED.*barrier|topology damaged|topology recovered|topology still degraded|rr_recovered|PCIe-direct soft reset|FIX BH|FIX BO|FIX BP' "$CLEAN" | \
grep -iE '(info|warning|error).*(Metal|Test|Always|Fabric)' | \
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
    is_important = bool(re.search(r'(complete|entering|skip|failed|timeout|deadbeef|SUMMARY|Pass-0.*complete|deferred|degraded|FIX AC|FIX AB extension|FIX BC|FIX CD|FIX GS|FIX RM|FIX RZ|FIX SA|FIX RX|teardown_timed_out|fixture_teardown|fixture_setup|run_mailbox|failed to initialize FW|open_devices_internal|FabricSwitchManager|warm-up)', msg, re.I))
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
MUX_RESETS=$(grep -c 'assert_risc_reset_at_core\|force.reset.*ETH\|ETH.*force.reset' "$CLEAN" 2>/dev/null; :)
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
DEAD_CH=$(grep -cE 'newly.dead|newly_dead|configure_fabric_cores.*newly.dead' "$CLEAN" 2>/dev/null; :)
[[ "${DEAD_CH:-0}" -eq 0 ]] && echo "  (none detected)"
echo ""

# ─── PRE-INIT STATE ───
echo "=== PRE-INIT STATE (stale/corrupt channel scan) ==="
grep -E 'pre.?init|stale|0x49706550|canary|kBaseUmd|base.UMD.*relay|terminate_stale|NOT a valid EDMStatus|zeroed edm_status_address|0xdeaddead' "$CLEAN" 2>/dev/null | \
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
SENTINEL=$(grep -c '0x49706550' "$CLEAN" 2>/dev/null; :)
echo "  (0x49706550 sentinel occurrences: ${SENTINEL:-0})"
GLOBAL_DL=$(grep -cE 'Global deadline expired|global.*deadline.*channel' "$CLEAN" 2>/dev/null; :)
echo "  (Global deadline teardown events: ${GLOBAL_DL:-0})"
# Canary detections
CANARY_COUNT=$(grep -c '0xA0A0A0A0\|A0A0A0A0' "$CLEAN" 2>/dev/null; :)
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
    elif re.search(r'FIX PA|erisc_app_still_running.*force.reset', line, re.I):
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
HAS_DISPATCH_CASCADE=$(grep -c 'Timeout (500 ms) waiting for physical cores' "$CLEAN" 2>/dev/null; :)
HAS_RELAY_BROKEN=$(grep -c 'relay.*path.*broken\|fabric_relay_path_broken_' "$CLEAN" 2>/dev/null; :)
HAS_P4_TIMEOUT=$(grep -cE 'Phase 4.*TIMEOUT|Phase 4.*timeout|MUX.*timeout|Timeout.*MUX|Timeout.*MUX READY_FOR_TRAFFIC|MUX READY_FOR_TRAFFIC.*[Tt]imeout|Timeout waiting for fabric MUX' "$CLEAN" 2>/dev/null; :)
HAS_EXCEPTION=$(grep -cE 'TT_THROW|TT_FATAL|Fatal|Abort' "$CLEAN" 2>/dev/null; :)
HAS_FORCE_RESET=$(grep -c 'assert_risc_reset_at_core\|force.reset' "$CLEAN" 2>/dev/null; :)
FIX_Z=$(grep -c 'is_fabric_relay_path_broken\|relay.*broken.*completion_queue\|CQ.*relay.*broken' "$CLEAN" 2>/dev/null; :)
FIX_AB=$(grep -cE 'hard-reset.*MMIO|RiscFirmwareInitializer.*teardown|MMIO ETH.*reset|fabric_teardown_timed_out_.*set.*device|FIX AB extension' "$CLEAN" 2>/dev/null; :)
# FIX AB extension: post_teardown flag that quiesce timed out; triggers Step 5 FIX AC reset
FIX_AB_EXT=$(grep -cE 'FIX AB extension|post_teardown.*FIX AB|fabric_teardown_timed_out.*set.*hard-reset|teardown: FIX AC \(timeout-only\)' "$CLEAN" 2>/dev/null; :)
FIX_AD=$(grep -cE 'rescue_stuck_dispatch_cores.*hard.*reset|hard BRISC reset|performing hard BRISC reset' "$CLEAN" 2>/dev/null; :)
FIX_W=$(grep -cE 'FIX W|Phase 5b.*all.*truly.*unhealthy.*stuck at 0x0|all.*dead.*clean return' "$CLEAN" 2>/dev/null; :)
FIX_AA=$(grep -ciE 'FIX AA|relay path broken.*skipping AllGather|skipping AllGather' "$CLEAN" 2>/dev/null; :)
FIX_V=$(grep -cE 'FIX V|Setting fabric_relay_path_broken_=true to skip relay ops in subsequent quiesce|Phase 5.*timeout.*0x0.*non-MMIO|status still 0x0 on non-MMIO device' "$CLEAN" 2>/dev/null; :)
RELAY_RESTORED=$(grep -c 'relay-broken flag reset by configure_fabric' "$CLEAN" 2>/dev/null; :)
# FIX-1: MMIO device Phase 5 timeout now also sets fabric_relay_path_broken_ (removed !is_mmio_capable() guard)
FIX_1_MMIO=$(grep -cE 'Setting fabric_relay_path_broken_=true|Phase 5.*timeout.*0x0.*MMIO|fabric_relay_path_broken_.*MMIO' "$CLEAN" 2>/dev/null; :)
# FIX AS: Pass-0 canary poll events (per-channel polling before write_launch_msg)
FIX_AS_PASS0=$(grep -cE 'Pass-0 \(FIX AS\)|Pass-0 \(FIX AR\+AS\) complete' "$CLEAN" 2>/dev/null; :)
# FIX_AS_TIMEOUT: actual log messages are "NOT ready after ... marking dead" and
# "did NOT reach UMD ready after ... marking dead, skipping launch"
FIX_AS_TIMEOUT=$(grep -cE 'Pass-0.*NOT ready after|Pass-0.*did NOT reach.*UMD ready|Pass-0.*marking dead|canary not seen' "$CLEAN" 2>/dev/null; :)
# FIX AC: teardown MMIO ETH reset events
FIX_AC_FIRES=$(grep -cE 'FIX AC' "$CLEAN" 2>/dev/null; :)
# FIX AU / FIX AU-2: relay-broken non-MMIO TERMINATE/l1_barrier attempt (AU-2 tries; AU skipped)
FIX_AU_FIRES=$(grep -cE 'FIX AU(-2)?[^-]|FIX AU-2|bypassed the poll loop.*relay-broken|relay path broken.*TERMINATE|TERMINATE.*relay path broken' "$CLEAN" 2>/dev/null; :)
# FIX AX / FIX AX-2: relay-confirmed-dead non-MMIO assert_risc_reset (AX-2 attempts; AX skipped)
FIX_AX_FIRES=$(grep -cE 'FIX AX(-2)?[^-]|FIX AX-2|relay confirmed dead.*assert_risc_reset|assert_risc_reset.*relay confirmed dead' "$CLEAN" 2>/dev/null; :)
# FIX XY-2: relay_broken cleared after successful ERISC force-reset (relay path restored)
FIX_XY2_RELAY_RESTORED=$(grep -cE 'FIX XY-2.*relay_broken cleared|relay_broken cleared.*FIX XY-2|FIX XY-2: Clearing relay_broken' "$CLEAN" 2>/dev/null; :)
# FIX SB2-R: relay_broken NOT set because probe was clean (clean boot, not a real failure)
FIX_SB2R_CLEAN_BOOT=$(grep -cE 'relay_broken NOT set.*FIX SB2-R|FIX SB2-R.*relay_broken NOT set|probe was clean.*relay_broken NOT set' "$CLEAN" 2>/dev/null; :)
# FIX AJ: relay path confirmed dead during assert_risc_reset (marks relay_dead_devices)
FIX_AJ_FIRES=$(grep -cE 'FIX AJ|relay path confirmed dead during force-reset' "$CLEAN" 2>/dev/null; :)
# FIX AK (FabricFirmware): transitive relay guard — skipping l1_barrier for ALL non-MMIO
FIX_AK_FIRES=$(grep -cE 'relay-dead device.*confirmed.*skipping l1_barrier.*FIX AK|FIX AK.*skipping l1_barrier' "$CLEAN" 2>/dev/null; :)
# FIX AQ: secondary edm_status_address sentinel poll after FIX AR heartbeat poll
FIX_AQ_FIRES=$(grep -cE 'FIX AQ' "$CLEAN" 2>/dev/null; :)
FIX_AQ_TIMEOUT=$(grep -cE 'FIX AQ.*edm_status_address still.*after' "$CLEAN" 2>/dev/null; :)
FIX_AQ_SUCCESS=$(grep -cE 'FIX AQ.*sentinel poll complete' "$CLEAN" 2>/dev/null; :)
# FIX RP (#42429): ROM postcode 0x49705180 transition polling on non-MMIO channels
# Success: "FIX RP Device N chan=N ROM postcode ... transitioned to base-UMD sentinel"
# Timeout: "FIX RP Device N chan=N ROM postcode ... did NOT transition ... within Nms"
# Unexpected: "FIX RP Device N chan=N ROM postcode ... transitioned to unexpected value"
FIX_RP_SUCCESS=$(grep -cE 'FIX RP.*transitioned to base-UMD sentinel' "$CLEAN" 2>/dev/null; :)
FIX_RP_TIMEOUT=$(grep -cE 'FIX RP.*did NOT transition.*within' "$CLEAN" 2>/dev/null; :)
FIX_RP_UNEXPECTED=$(grep -cE 'FIX RP.*transitioned to unexpected value' "$CLEAN" 2>/dev/null; :)
# FIX RR (#42429): PCIe-direct soft reset for MMIO pre-confirmed-dead channels in configure_fabric_cores.
# Success: "configure_fabric_cores: device N channel N FIX RR — PCIe-direct soft reset succeeded"
# Failure: "configure_fabric_cores: device N channel N FIX RR — PCIe-direct soft reset FAILED"
# FIX RS (#42429): configure_fabric() log on recovered channel propagation:
#   "configure_fabric: Device N FIX RS — N channel(s) recovered by FIX RR propagated back"
# Also captured in configure_fabric exit summary as "rr_recovered=N".
FIX_RR_SUCCESS=$(grep -cE 'FIX RR.*PCIe-direct soft reset succeeded' "$CLEAN" 2>/dev/null; :)
FIX_RR_FAIL=$(grep -cE 'FIX RR.*PCIe-direct soft reset FAILED' "$CLEAN" 2>/dev/null; :)
FIX_RS_RECOVERED=$(grep -oE 'FIX RS.*([0-9]+) channel\(s\) recovered by FIX RR' "$CLEAN" 2>/dev/null | grep -oE '[0-9]+' | awk '{s+=$1} END{print s+0}'; :)
# FIX RZ2: stale_base_umd flag cleared after ring-sync + health check pass
FIX_RZ2_CLEAR_COUNT=$(grep -cE 'FIX RZ2.*clearing fabric_stale_base_umd|RZ2.*base-UMD channels confirmed healthy' "$CLEAN" 2>/dev/null; :)
# FIX M: base-UMD channel transitions via launch_msg (sets stale_base_umd=true)
FIX_M_TRANSITION_COUNT=$(grep -cE 'FIX M\b.*Setting fabric_stale_base_umd|base-UMD channel.*transitioned via launch_msg|Setting fabric_stale_base_umd_channels_=true' "$CLEAN" 2>/dev/null; :)
# FIX QW SKIP: test skip due to stale_base_umd or channels_not_ready in fixture SetUp
FIX_QW_SKIP_COUNT=$(grep -cE 'stale_base_umd=true|channels_not_ready=true|FIX QW.*skip|stale_base_umd_channels=true.*skipping' "$CLEAN" 2>/dev/null; :)
# FIX AR2: post-deassert delay before heartbeat poll (avoids stale 0xABCD false positive)
FIX_AR2_FIRES=$(grep -cE 'FIX AR2.*post-deassert delay' "$CLEAN" 2>/dev/null; :)
# FIX AT: Phase 5 handshake poll skipped when MMIO master chan was FIX AS Pass-0 timeout'd
FIX_AT_FIRES=$(grep -cE 'FIX AT|Pass-0 timeout.*skipping.*handshake|master chan.*FIX AS.*Pass-0 timeout' "$CLEAN" 2>/dev/null; :)
# FIX AY: deferred non-MMIO ETH ERISC reset via restored MMIO relay
# FIX AV: skip remaining ETH cores on same device when first assert_risc_reset times out
FIX_AY_FIRES=$(grep -cE 'FIX AY' "$CLEAN" 2>/dev/null; :)
FIX_AY_SUCCEEDED=$(grep -cE 'FIX AY.*all.*reset to base firmware|FIX AY.*succeeded' "$CLEAN" 2>/dev/null; :)
FIX_AY_FAILED=$(grep -cE 'FIX AY.*failed|FIX AY.*non-std exception|FIX AY/AV.*failed' "$CLEAN" 2>/dev/null; :)
FIX_AV_FIRES=$(grep -cE 'FIX AV #42429|FIX AY/AV.*Skipping all remaining' "$CLEAN" 2>/dev/null; :)
# FIX AL: read-failure early-exit (relay exception, firmware unusable — not a STARTED timeout)
FIX_AL_FIRES=$(grep -cE 'FIX AL|firmware.*unusable.*early.exit|relay.*read.*FAILED.*early.exit' "$CLEAN" 2>/dev/null; :)
# FIX AM: Phase 5b skip when master chan still at STARTED after FIX AL break
FIX_AM_FIRES=$(grep -cE 'FIX AM|skipping Phase 5b.*FIX AM|channels_not_ready_for_traffic.*FIX AM|still at STARTED.*skipping Phase 5b' "$CLEAN" 2>/dev/null; :)
# FIX AO (#42429): wait_for_fabric_router_sync() STARTED early-exit — master chan stuck at
# EDMStatus::STARTED (0xa0b0c0d0) after 1s.  Peer is not responding (out-of-mesh or non-MMIO
# base-UMD).  Skips ring sync cleanly instead of waiting full timeout (10-120s).
# Log: "wait_for_fabric_router_sync: Device N master chan=N stuck at STARTED (0xa0b0c0d0) after Nms
#       — peer is not responding (likely out-of-mesh). Skipping ring sync cleanly. (FIX AO #42429)"
FIX_AO_FIRES=$(grep -cE 'stuck at STARTED.*peer is not responding.*FIX AO|Skipping ring sync cleanly.*FIX AO #42429' "$CLEAN" 2>/dev/null; :)
# FIX BD/BF (#42429): deterministic MMIO-peer-preference master channel selection.
# Log (log_info): "FabricBuilder::create_routers: Device N selected master_router_chan=N
#                  (peer_chip=N peer_is_mmio=true/false total_router_candidates=N) (FIX BD/BF #42429)"
# Counts how many devices selected an MMIO peer vs non-MMIO peer for master channel.
FIX_BF_MMIO_MASTER=$(grep -cE 'selected master_router_chan=.*peer_is_mmio=true.*FIX BD/BF' "$CLEAN" 2>/dev/null; :)
FIX_BF_NONMMIO_MASTER=$(grep -cE 'selected master_router_chan=.*peer_is_mmio=false.*FIX BD/BF' "$CLEAN" 2>/dev/null; :)
# FIX BE (#42429): per-cycle stale state clearing — structural fix, no direct log line.
# Prevents external_umd_channels_map_ / has_base_umd_channels_ / timeout_on_base_umd_devices_ /
# ring_sync_already_timed_out_ from accumulating across teardown+reinit cycles.
# Regression evidence: base-UMD channel count grows 2→4→6+ per cycle without this fix.
FIX_BE_STRUCTURAL=1  # always active in this branch
# FIX BG (#42429): host-pre-launch (0xdeadb07e) sentinel early-exit in wait_for_fabric_router_sync.
# Fires when ERISC was soft-reset (FIX RR) but never started executing — stuck at host-written
# pre-launch canary.  Device marked dead-master-chan; sync skipped.
# Log: "Device N master chan=N stuck at host-pre-launch (0xdeadb07e) after Nms —
#       ERISC not executing after soft-reset (FIX RR). Marking device as dead-master-chan. (FIX BG #42429)"
FIX_BG_FIRES=$(grep -cE 'stuck at host-pre-launch.*0xdeadb07e.*FIX BG|Marking device as dead-master-chan.*FIX BG' "$CLEAN" 2>/dev/null; :)
# FIX BH (#42429): poll for ERISC ROM-to-UMD boot after FIX RR deassert.
# Without FIX BH: ERISC could still be in ROM-init when Phase 3 writes L1 → 0xdeadb07e race (FIX BG).
# Success: "configure_fabric_cores: device N channel N FIX BH — ERISC booted from ROM to 0x..."
# Timeout: "configure_fabric_cores: device N channel N FIX BH — ERISC did not exit ROM phase within..."
FIX_BH_SUCCESS=$(grep -cE 'FIX BH.*ERISC booted from ROM' "$CLEAN" 2>/dev/null; :)
FIX_BH_TIMEOUT=$(grep -cE 'FIX BH.*ERISC did not exit ROM phase' "$CLEAN" 2>/dev/null; :)
# FIX BO (#42429): Phase 5 kSyncTimeoutMs extended 10s→120s when stale base-UMD channels present.
# Without FIX BO: cluster with N base-UMD channels × 10s each → "Fabric health check failed" TT_THROW.
# Log: "wait_for_fabric_workers_ready: Device N FIX BO — stale base-UMD channels detected, extending Phase 5..."
FIX_BO_FIRES=$(grep -cE 'FIX BO.*stale base-UMD channels' "$CLEAN" 2>/dev/null; :)
# FIX BP (#42429): fabric_context null guard — teardown ordering race recovered.
# Without FIX BP: ControlPlane::get_fabric_context() after teardown → TT_FATAL → infinite hang.
# Log 1 (throw site): "FIX BP: fabric_context null — ControlPlane::get_fabric_context() called after teardown"
# Log 2 (catch site): "teardown: FIX AQ — fabric_context already torn down (FIX BP)"
FIX_BP_FIRES=$(grep -cE 'FIX BP.*fabric_context null|fabric_context already torn down.*FIX BP' "$CLEAN" 2>/dev/null; :)
# FIX ST (#42429): use device's effective fabric_pre_dead_channels_ (post-FIX-RR) for
# mmio_dead_master_chan check instead of raw probe_dead_channels_map.
# FIX ST dead:      "... NOT recovered by FIX RR ... Sync will be skipped. (#42429 FIX AN / FIX ST)"
# FIX ST recovered: "... Firmware loaded; sync will proceed normally. (#42429 FIX ST)"
FIX_ST_DEAD=$(grep -cE 'NOT recovered by FIX RR.*Sync will be skipped|FIX AN / FIX ST' "$CLEAN" 2>/dev/null; :)
FIX_ST_RECOVERED=$(grep -cE 'Firmware loaded; sync will proceed normally.*FIX ST' "$CLEAN" 2>/dev/null; :)
# FIX AW: ~Cluster destructor runs driver_->close_device() in detached thread to avoid wait_for_non_mmio_flush hang
FIX_AW_FIRES=$(grep -cE 'FIX AW|relay-broken non-MMIO.*running driver.*close_device.*background thread|close_device.*did not complete.*5s.*FIX AW' "$CLEAN" 2>/dev/null; :)
# FIX BA: STARTED-state non-MMIO devices added to relay_broken_non_mmio (FIX AM fired, relay_broken=false)
FIX_BA_FIRES=$(grep -cE 'FIX BA|channels_not_ready_for_traffic.*relay not marked broken.*Adding to relay_broken_non_mmio|teardown: FIX BA' "$CLEAN" 2>/dev/null; :)
# FIX AE: catch wait_for_non_mmio_flush() timeout in write_core/write_reg/noc_multicast_write + mark_relay_broken() for ~Cluster()
# Pattern: "FIX AE: wait_for_non_mmio_flush(chip N) threw: ... Marking relay broken."
FIX_AE_FLUSH_TIMEOUT=$(grep -cE 'FIX AE.*wait_for_non_mmio_flush.*threw|FIX AE.*Marking relay broken' "$CLEAN" 2>/dev/null; :)
# FIX AE also marks all remote chips relay-broken in ~Cluster() before close_device() — no specific log (silent path).
# FIX NS: eliminate redundant topology discovery in initialize_base_objects (no runtime log — structural fix)
# Regression evidence: 14+ min SIGALRM before any test, "unit_tests_ttnn" hangs without FIX AQ warning
# FIX NT: preserve EthCoord for FIX-AQ-skipped remote chips in chip_locations
# Log: "FIX AQ: Failed to init remote device ... Skipping" (FIX AQ fires; NT adds EthCoord silently)
# Regression evidence: TT_FATAL "Physical chip id not found for eth coord" @ tt_cluster.cpp:575
FIX_NT_CRASH=$(grep -cE 'Physical chip id not found for eth coord|TT_FATAL.*chip id not found.*eth' "$CLEAN" 2>/dev/null; :)
# FIX NU: capture MMIO EthCoord (via PCIe NODE_INFO) before FIX W heartbeat guard
# Log: "FIX NU: Captured EthCoord for MMIO device ASIC ID" (debug-level only, may not appear in CI)
# Regression evidence: same TT_FATAL as FIX NT but MMIO chip missing; FIX W skipped ALL ETH channels
FIX_NU_COORD=$(grep -cE 'FIX NU: Captured EthCoord|Captured EthCoord.*MMIO.*before relay' "$CLEAN" 2>/dev/null; :)
# FIX NX+NY: write_core() relay guard for non-MMIO chips — FIX NX catches first timeout exception,
# FIX NY caches the failure in relay_broken_chips_ so subsequent calls for same chip skip UMD.
# FIX NX log: "FIX NX: write_core(chip N) threw: ... Marking relay broken (FIX NX+NY)."
# FIX NY log: "FIX NY: write_core(chip N) skipped — relay already known broken." (debug-level)
# Regression evidence (FIX NX missing): write_core exception propagates to MetalContext::initialize.
# Regression evidence (FIX NY missing): N_channels × 5s serial stall per dead chip — GHA timeout.
#   CI ref: run 25086219070 (job 73503180670): 6 channels × 5s = 30s → GHA 5-min action timeout.
FIX_NX_THROWS=$(grep -cE 'FIX NX: write_core.*threw|Marking relay broken \(FIX NX\+NY\)' "$CLEAN" 2>/dev/null; :)
FIX_NY_SKIPS=$(grep -cE 'FIX NY: write_core.*skipped.*relay already known broken' "$CLEAN" 2>/dev/null; :)
# FIX TE (#42429): control_plane.cpp callers of try_get_asic_id_from_fabric_node_id skip chips
# excluded by FIX TB (no entry in topology mapper — degraded topology from corrupt ERISC L1).
# Log: "FIX TE (#42429): Skipping order_ethernet_channels for FabricNodeId {}"
#      "FIX TE (#42429): FabricNodeId (M{}, D{}) not found in topology mapper"
# Regression evidence: without FIX TE, TT_FATAL in configure_routing_tables or order_ethernet_channels
#   for FIX TB-excluded chips → SIGABRT; with FIX TE, warning logged and chip skipped.
FIX_TE_SKIPS=$(grep -cE 'FIX TE.*Skipping order_ethernet_channels|FIX TE.*not found in topology mapper' "$CLEAN" 2>/dev/null; :)
# FIX TF (#42429): assemble_2d_fabric_packet_header_args TT_FATAL guard — replaces opaque
# bad_optional_access thrown when inter-mesh relay is broken (get_forwarding_direction returns nullopt).
# Log (TT_FATAL, appears before SIGABRT): "FIX TF: No forwarding direction from physical chip N"
# Regression evidence: without FIX TF, GTest sees "bad optional access" with no chip context;
#   with FIX TF, TT_FATAL names src/dst chip IDs and fabric node IDs.
FIX_TF_FIRES=$(grep -cE 'FIX TF: No forwarding direction' "$CLEAN" 2>/dev/null; :)
# FIX TF control_plane (#42429): FabricNodeId not in router_port_directions map — excluded chip
# skipped in routing-plane-count and routing-table conversion (control_plane.cpp).
# Without this guard: std::out_of_range BEFORE MPI all_gather → peer rank hangs indefinitely.
# Log: "FIX TF (#42429): FabricNodeId (M0, D3) not in router_port_directions map"
FIX_TF_CP_FIRES=$(grep -cE 'FIX TF \(#42429\).*not in router_port_directions map' "$CLEAN" 2>/dev/null; :)
# FIX SC (GAP-76): risc_firmware_initializer.cpp pre-scan + llrt.cpp inline rescue for Tensix
# cores with stale go_msg (0x55) after hard BRISC reset by rescue_stuck_dispatch_cores().
# FIX SC asserts BRISC reset BEFORE writing RUN_MSG_DONE (fixes the FIX SB race condition where
# stale firmware overwrote the DONE write).  Also handles cores that transition to 0x55 between
# the pre-scan and the wait_until_cores_done poll loop.
# FIX SC-ADDR (GAP-76): risc_firmware_initializer.cpp — FIX SC pre-scan was always using the
# TENSIX go_msg address even for ETH cores (ACTIVE_ETH / IDLE_ETH) in not_done_cores.  ETH cores
# keep go_msg at a different L1 offset.  Reading wrong address returned garbage (0x02) → FIX SC
# fired falsely → wrote RUN_MSG_DONE to wrong ETH L1 address → ETH dispatch FW corruption →
# deterministic 1000ms teardown timeout for all 8 devices every open/close cycle.
# Fixed by calling llrt::get_core_type() per core and using hal_.get_dev_addr(core_type, GO_MSG).
# Core-type annotation added to log messages so ETH vs Tensix stale fires are distinguishable.
# Log (pre-scan, stale):  "FIX SC (GAP-76): Device N core X (TENSIX) has stale go_msg=0xNN"
#                         "FIX SC (GAP-76): Device N core X (ACTIVE_ETH) has stale go_msg=0xNN"
# Log (debug, SC-ADDR):   "FIX SC-ADDR (GAP-76): Device N core X (ACTIVE_ETH) go_msg_addr=0xNNN signal=0xNN — valid"
# Log (inline):           "FIX SC (GAP-76): Tensix core X run_mailbox=0xNN — asserting reset + writing DONE"
# Log (failure):          "FIX SC (GAP-76): assert_risc_reset failed for Device N core X"
#                         "FIX SC (GAP-76): inline rescue failed for core X"
FIX_SC_PRESCAN=$(grep -cE 'FIX SC \(GAP-76\).*stale go_msg' "$CLEAN" 2>/dev/null; :)
FIX_SC_ETH_FIRE=$(grep -cE 'FIX SC \(GAP-76\).*(ACTIVE_ETH|IDLE_ETH).*stale go_msg' "$CLEAN" 2>/dev/null; :)
FIX_SC_TENSIX_FIRE=$(grep -cE 'FIX SC \(GAP-76\).*TENSIX.*stale go_msg' "$CLEAN" 2>/dev/null; :)
FIX_SC_INLINE=$(grep -cE 'FIX SC \(GAP-76\).*Tensix core.*asserting reset' "$CLEAN" 2>/dev/null; :)
FIX_SC_FAIL=$(grep -cE 'FIX SC \(GAP-76\).*(assert_risc_reset failed|inline rescue failed)' "$CLEAN" 2>/dev/null; :)
# FIX SC-ADDR debug logs (only present when METAL_LOG_LEVEL=DEBUG or TT_LOGGER_LEVEL=DEBUG)
FIX_SC_ADDR_ETH_VALID=$(grep -cE 'FIX SC-ADDR \(GAP-76\).*(ACTIVE_ETH|IDLE_ETH).*valid \(no FIX SC\)' "$CLEAN" 2>/dev/null; :)
FIX_SC_ADDR_ETH_STALE=$(grep -cE 'FIX SC-ADDR \(GAP-76\).*(ACTIVE_ETH|IDLE_ETH).*STALE' "$CLEAN" 2>/dev/null; :)
# FIX TG (#42429): control_plane.cpp configure_routing_tables_for_fabric_ethernet_channels —
# TT_ASSERT (no-op in Release) guarded host_rank_for_chip.value() for connected chips excluded
# by FIX TB.  Fixed by guarding with has_value() check and continuing (skip), matching FIX TE.
# Log: "FIX TG (#42429): Mesh {} Chip {} has no host rank in topology mapper"
# Regression evidence: without FIX TG, std::bad_optional_access thrown in ControlPlane ctor
#   during T3K SetUp() on degraded cluster → GTest catches it as opaque "bad optional access".
FIX_TG_FIRES=$(grep -cE 'FIX TG.*has no host rank in topology mapper' "$CLEAN" 2>/dev/null; :)
# FIX TH (#42429): relay_mux.cpp GenerateStaticConfigs() — preflight check for empty forwarding link indices.
# After FIX NY/NY+ verified both src and dst chips are in the fabric cluster, the MMIO-to-MMIO dispatch relay
# path can still lose ALL ETH channels if progressive teardown corruption killed every link in the routing
# direction.  Calling get_dispatch_link_index() would TT_FATAL.  FIX TH calls get_forwarding_link_indices()
# first; if empty, marks device channels_not_ready_for_traffic so FIX SA can GTEST_SKIP gracefully.
# Log: "FIX TH (#42429): RelayMux::GenerateStaticConfigs — no available dispatch links from FabricNodeId"
# Regression evidence: without FIX TH, TT_FATAL "No links available from (M0,D2) to (M0,D3)" in SetUp().
FIX_TH_FIRES=$(grep -cE 'FIX TH.*no available dispatch links' "$CLEAN" 2>/dev/null; :)
# FIX TH control_plane.cpp (#42429): convert_fabric_routing_table_to_chip_routing_table skips
# excluded chips in the get_chip_ids() loop.  Without this guard: TT_FATAL "Fabric node id
# (M0, D0) not found in mapping" → both ranks crash, rank 1 hangs 15 min in MPI barrier.
# Log: "FIX TH (#42429): convert_fabric_routing_table — skipping excluded chip FabricNodeId"
FIX_TH_CP_FIRES=$(grep -cE 'FIX TH.*convert_fabric_routing_table.*skipping excluded chip' "$CLEAN" 2>/dev/null; :)
# FIX TJ (#42429): topology_mapper.cpp prefilter for WH_B0-invalid mesh shapes (e.g. 3x1).
# Log: "FIX TJ (#42429): TopologyMapper: skipping shape {}x{} — invalid for WORMHOLE_B0"
FIX_TJ_FIRES=$(grep -cE 'FIX TJ.*invalid for WORMHOLE_B0' "$CLEAN" 2>/dev/null; :)
# FIX TK (#42429): BaseFabricFixture::DoSetUpTestSuite — filter chip IDs against
# is_physical_chip_in_fabric_cluster() after topology discovery.  Prevents TT_FATAL
# in create_unit_meshes() when topology mapper degrades to 1x1 (all ETH links dead).
# Log: "FIX TK (#42429): Physical chip {} not in fabric cluster — excluding from unit meshes."
# Log (all excluded): "FIX TK (#42429): No chips in fabric cluster — skipping SetUpTestSuite."
# Regression evidence: TT_FATAL "Physical chip id {} not found in control plane chip mapping"
#   in SetUpTestSuite() after TopologyMapper downgraded to 1x1.
FIX_TK_FIRES=$(grep -cE 'FIX TK.*not in fabric cluster|FIX TK.*No chips in fabric cluster' "$CLEAN" 2>/dev/null; :)
# FIX TL (#42429): DoSetUpTestSuite — bail BEFORE create_unit_meshes() when FIX TK sets
# cluster_degraded_skip_=true (partial chip set), to avoid UDM/Tensix builder TT_FATAL
# which requires the full expected topology (>= 2 chips) and can't handle a 1x1 cluster.
# Log: "FIX TL (#42429): Fabric cluster has only {}/{} chips — skipping create_unit_meshes"
# Regression evidence: TT_FATAL "Device {} not found in worker tensix info map" at
#   fabric_tensix_builder.cpp:874 in Fabric2DUDMModeFixture::SetUpTestSuite().
FIX_TL_FIRES=$(grep -cE 'FIX TL.*skipping create_unit_meshes' "$CLEAN" 2>/dev/null; :)
# FIX TM (#42429): expand_one_or_all_to_all_unicast in tt_fabric_test_config.cpp — return
# early when all_pairs is empty (0 routing planes on severely degraded cluster).
# Log: "FIX TM (#42429): No device pairs found for one_to_all_unicast — cluster too degraded"
# Regression evidence: crash accessing all_pairs[0].first on empty vector.
FIX_TM_FIRES=$(grep -cE 'FIX TM.*cluster too degraded' "$CLEAN" 2>/dev/null; :)
# FIX TH2 (#42429): fabric_firmware_initializer.cpp get_fabric_router_sync_timeout_ms() —
# extends per-device sync timeout from 10s to 30s when base-UMD channels are present.
# Base-UMD ERISCs transition via launch_msg (not soft reset), requiring more time for
# relay quiesce + new firmware init + ring handshake.  Without FIX TH2, sequential polling
# of 12 base-UMD channels at 10s each times out → "Fabric health check failed" TT_THROW
# with "N still-initializing" in the breakdown.
# Log: "FIX TH2 (#42429): base-UMD channels detected — extending fabric_router_sync_timeout"
# Regression evidence: "Fabric health check failed ... 12 still-initializing" in run 25278856721.
FIX_TH2_FIRES=$(grep -cE 'FIX TH2.*extending fabric_router_sync_timeout' "$CLEAN" 2>/dev/null; :)
# FIX TH3 (#42429): extends ring-sync timeout from 30s (3x) to 120s (12x) for base-UMD channels.
FIX_TH3_FIRES=$(grep -cE 'FIX TH3.*extending fabric_router_sync_timeout' "$CLEAN" 2>/dev/null; :)
# FIX TG L1-clear (#42429): fabric_init.cpp configure_fabric_cores() — preserve edm_status_address
# (0x49706550 sentinel) for base-UMD relay channels so the next session's
# terminate_stale_erisc_routers() correctly identifies base-UMD state and fires FIX M.
# FIX TG2 (#42429): partial L1 clear — zeros sync-critical addresses (edm_local_sync_address,
# edm_local_tensix_sync_address, termination_signal_address) while preserving edm_status_address.
# Original FIX TG skipped ALL clears, leaving stale handshake state that caused REMOTE_HANDSHAKE_COMPLETE
# stalls across smi-reset cycles (runs 25293661493 + 25294660215).
# Log (FIX TG2 preserve): "configure_fabric_cores: device N channel N base-UMD relay — preserving edm_status_address (0x49706550 sentinel) [FIX TG #42429]"
# Log (FIX TG2 clear):    "configure_fabric_cores: device N channel N base-UMD relay — clearing sync address 0xNNNNNNNN to prevent stale handshake state [FIX TG2 #42429]"
FIX_TG_L1_FIRES=$(grep -cE 'preserving edm_status_address.*0x49706550.*FIX TG' "$CLEAN" 2>/dev/null; :)
FIX_TG2_SYNC_CLEARS=$(grep -cE 'clearing sync address.*stale handshake.*FIX TG2' "$CLEAN" 2>/dev/null; :)
# FIX EXT (#42429): external ETH channel classification — channel at 0x49706550 with no in-cluster
# peer (e.g. physically wired to out-of-mesh host).  Excluded from soft-reset AND firmware launch.
# Fires in wait_for_fabric_router_sync() and verify_all_fabric_channels_healthy() when such
# channels are encountered.  Without this fix, the channel would block ring-sync or health-check.
# Log: "wait_for_fabric_router_sync: Device N master chan=N is an external ETH channel — skipping cleanly. (FIX EXT #42429)"
FIX_EXT_FIRES=$(grep -cE 'external ETH channel.*skipping cleanly.*FIX EXT|FIX EXT.*external ETH channel' "$CLEAN" 2>/dev/null; :)
# FIX TI ring-sync (#42429): fabric_firmware_initializer.cpp verify_all_fabric_channels_healthy() —
# skip health check for devices whose ring barrier timed out while base-UMD channels were present.
# Sets fabric_channels_not_ready_for_traffic_ + fabric_ring_sync_timed_out_ so test fixtures SKIP
# and FIX BA does NOT add the device to relay_broken_non_mmio (FIX TK guard).
# Log: "verify_all_fabric_channels_healthy: Device N ring barrier timed out ... (#42429 FIX TI + FIX TK)"
FIX_TI_RING_FIRES=$(grep -cE 'ring barrier timed out during base-UMD.*FIX TI' "$CLEAN" 2>/dev/null; :)
# FIX TJ ring-sync fast-skip (#42429): fabric_firmware_initializer.cpp wait_for_fabric_router_sync() —
# once one device times out on the base-UMD ring barrier, all remaining devices are immediately
# marked as timed-out instead of waiting the full 30s each (ring barrier requires every member).
# Log: "wait_for_fabric_router_sync: Device N skipped — ring sync already timed out ... (FIX TJ #42429)."
FIX_TJ_RING_FIRES=$(grep -cE 'ring sync already timed out.*FIX TJ' "$CLEAN" 2>/dev/null; :)
# FIX TI first-timeout (#42429): fabric_firmware_initializer.cpp wait_for_fabric_router_sync() —
# the first device that actually waits the full 30s and times out (sets ring_sync_already_timed_out_).
# Log: "timeout_on_base_umd_devices_ ... ring_sync_already_timed_out_ set to fast-skip ... (FIX TI + FIX TJ #42429)."
FIX_TI_FIRST_TIMEOUT=$(grep -cE 'ring_sync_already_timed_out_ set to fast-skip.*FIX TI.*FIX TJ' "$CLEAN" 2>/dev/null; :)
# FIX TK guard log (#42429): risc_firmware_initializer.cpp teardown step 1 — FIX BA was skipped
# because device has fabric_ring_sync_timed_out (FIX TI path, not FIX AM STARTED-state).
# Log: "teardown: FIX TK — non-MMIO device N has fabric_channels_not_ready ... Skipping FIX BA"
FIX_TK_BA_GUARD=$(grep -cE 'FIX TK.*Skipping FIX BA.*relay_broken_non_mmio' "$CLEAN" 2>/dev/null; :)
# "Fabric health check failed" — the TT_THROW from wait_for_fabric_router_sync() when channels
# fail to reach READY_FOR_TRAFFIC.  The breakdown (corrupt/still-initializing/degraded) indicates
# whether FIX TH2 (still-initializing) or tt-smi reset (corrupt) is needed.
HEALTH_CHECK_FAILED=$(grep -cE 'Fabric health check failed.*did not reach READY_FOR_TRAFFIC' "$CLEAN" 2>/dev/null; :)
STILL_INITIALIZING_COUNT=$(grep -oE '[0-9]+ still-initializing' "$CLEAN" 2>/dev/null | head -1 | grep -oE '[0-9]+' || echo "0")
# FIX TN (#42429): run_t3000_unit_tests.sh gtest_filter wildcard fix — removes leading '*' from
# 'Fabric2DFixture.TestUnicast*' to prevent matching T3kCustomMeshGraphFabric2DFixture by accident.
# Regression evidence: T3kCustomMeshGraph fixture crashes with 'Fabric node id not found in mapping'.
FIX_TN_WILDCARD_CRASH=$(grep -cE 'T3kCustomMeshGraph.*Fabric node id not found in mapping' "$CLEAN" 2>/dev/null; :)
# FIX TL-bash (#42429): run_t3000_unit_tests.sh topology recovery reset — warm-up atexit left
# non-MMIO chips unreachable.  Fires tt-smi -r + FIX TM warm-up to restore topology.
# Log: "[FIX TL] T3K topology damaged after warm-up (N/8 chips)"
FIX_TL_BASH_FIRES=$(grep -cE '\[FIX TL\] T3K topology damaged' "$CLEAN" 2>/dev/null; :)
FIX_TL_BASH_RECOVERED=$(grep -cE '\[FIX TL/TM\] topology recovered' "$CLEAN" 2>/dev/null; :)
FIX_TL_BASH_STILL_DEGRADED=$(grep -cE 'T3K topology still degraded after recovery' "$CLEAN" 2>/dev/null; :)
# FIX TM-bash (#42429): post-TL warm-up after tt-smi -r — opens/closes mesh to re-establish relay.
# Log: "[FIX TM] post-TL warm-up complete" / "[FIX TM] WARNING: post-TL warm-up failed"
FIX_TM_BASH_COMPLETE=$(grep -cE '\[FIX TM\] post-TL warm-up complete' "$CLEAN" 2>/dev/null; :)
FIX_TM_BASH_FAILED=$(grep -cE '\[FIX TM\] WARNING.*post-TL warm-up failed' "$CLEAN" 2>/dev/null; :)
# FIX TN-bash (#42429): || true on topology-check cmd-substitutions so set -eo pipefail doesn't
# abort the script when Python crashes (GetNumAvailableDevices throws on dead relay).
# No direct log — detect via "T3K topology check failed to query device count" error message.
FIX_TN_BASH_MISSING=$(grep -cE 'T3K topology check failed to query device count' "$CLEAN" 2>/dev/null; :)
# FIX DT-1 (#42429): dispatch ERISC teardown timeout (1000ms) in warm-up → rescue_stuck_dispatch_cores →
# ERISCs left with stale go_msg=0x02. Warm-up now detects this and triggers remedial tt-smi -r (FIX UP path).
# Log: "[FIX DT-1 (#42429)] Device N: dispatch ERISC teardown timeout (1000ms) — rescue_stuck_dispatch_cores firing"
FIX_DT1_FIRES=$(grep -cE '\[FIX DT-1 \(#42429\)\].*dispatch ERISC teardown timeout' "$CLEAN" 2>/dev/null; :)
# FIX TO (#42429): remedial tt-smi -r after warm-up >= 120s or ring-sync timeout detected (FIX TH3 threshold).
FIX_TO_BASH_FIRES=$(grep -cE '\[FIX TO\] warm-up ran' "$CLEAN" 2>/dev/null; :)
# FIX UP (#42429): ring-sync timeout marker detected in warm-up output (Python exits 0 but ring never converged).
FIX_UP_FIRES=$(grep -cE '\[FIX UP\] ring-sync timeout marker detected|post-reset warm-up ring-sync timeout' "$CLEAN" 2>/dev/null; :)
# FIX UP INFRA_ERROR (#42429): 3 consecutive ring-sync timeout warm-ups → abort.
FIX_UP_INFRA_ERROR=$(grep -cE '\[FIX UP\] INFRA_ERROR' "$CLEAN" 2>/dev/null; :)
# FIX UP2 (#42429): pre-test-loop ring-sync health gate — retry after ring-sync timeout in initial/TM warm-up.
FIX_UP2_FIRES=$(grep -cE '\[FIX UP2\]' "$CLEAN" 2>/dev/null; :)
FIX_UP2_INFRA_ERROR=$(grep -cE '\[FIX UP2\] INFRA_ERROR' "$CLEAN" 2>/dev/null; :)
# FIX UP3 (#42429): dispatch-ERISC timeout loop detected in FIX UP2 warm-up.
# The warm-up itself triggers the rescue_stuck_dispatch_cores loop (base-UMD ERISCs
# interfere with dispatch FW during open/close → go_msg=0x02 stale → next warm-up same issue).
# FIX UP3: do final tt-smi -r and skip further warm-ups; tests use FIX SC + FIX M/RZ2 to handle state.
# Log: "LOG_METAL: [FIX UP3] dispatch-ERISC timeout loop detected — running final tt-smi -r to clear rescue_stuck stale state, then proceeding to tests without warm-up."
FIX_UP3_FIRES=$(grep -cE '\[FIX UP3\] dispatch-ERISC timeout loop detected' "$CLEAN" 2>/dev/null; :)
# FIX TM2 (#42429): ring-sync timeout detected in post-TL (FIX TM) warm-up.
FIX_TM2_FIRES=$(grep -cE '\[FIX TM2\] ring-sync timeout' "$CLEAN" 2>/dev/null; :)
# Aggregate counters: ring-sync timeouts, base-UMD channel occurrences, channels_not_ready events
RING_SYNC_TIMEOUT_COUNT=$(grep -ciE 'ring.*timeout|timeout.*ring|Timeout after.*ms on Device.*master chan' "$CLEAN" 2>/dev/null; :)
BASE_UMD_CHAN_COUNT=$(grep -c '0x49706550' "$CLEAN" 2>/dev/null; :)
CHANNELS_NOT_READY_COUNT=$(grep -cE 'channels_not_ready|channels_not_ready_for_traffic' "$CLEAN" 2>/dev/null; :)
# FIX M2 (#42429): Secondary check in compile_and_configure_fabric() — channel showed 0x49706550 (base-UMD relay)
# but peer non-MMIO device is confirmed dead-relay → remove from base_umd_channels so configure_fabric_cores()
# performs a hard soft-reset (no relay reads in flight, safe to reset).
FIX_M2_FIRES=$(grep -cE 'FIX M2.*dead-relay|compile_and_configure_fabric: FIX M2' "$CLEAN" 2>/dev/null; :)
# FIX PL (#42429): opt-in timeout guards on l1_barrier / dram_barrier / read_core for non-MMIO chips.
# Fires when the ERISC relay path is dead and the barrier/read would otherwise hang indefinitely.
FIX_PL_FIRES=$(grep -cE 'clear_l1_state: l1_barrier timed out.*dead ERISC relay|clear_dram_state: dram_barrier timed out|terminate_active_ethernet_cores_on_all_chips: l1_barrier timed out|WriteInitMagic: read_core timed out' "$CLEAN" 2>/dev/null; :)
# FIX TV (#42429): poll MMIO ETH heartbeat in run_launch_phase() after reset_cores().
# Success: "FIX TV — all N MMIO ETH channel(s) confirmed base firmware heartbeat in Xms"
# Timeout: "FIX TV — MMIO ETH heartbeat poll timed out after Xms"
# FIX TW (#42429): inside FIX TV, detects static 0xABCDxxxx UMD marker as immediate-ready.
# Without FIX TW, FIX TV always times out because it waits for heartbeat value change.
FIX_TV_SUCCESS=$(grep -cE 'FIX TV.*all.*confirmed base firmware heartbeat' "$CLEAN" 2>/dev/null; :)
FIX_TV_TIMEOUT=$(grep -cE 'FIX TV.*MMIO ETH heartbeat poll timed out' "$CLEAN" 2>/dev/null; :)
# FIX PF (#42429): UMD base fw heartbeat detected at Metal exit — skip writing Metal exit signal.
# Fires in risc_firmware_initializer.cpp when BRISC is still running base-UMD fw at process shutdown.
# Clears stale fw_launch_addr to unblock next session.  Distinct from FIX PA (which fires during init).
FIX_PF_FIRES=$(grep -cE 'FIX PF.*skipping Metal exit signal|FIX PF.*stale fw_launch_addr' "$CLEAN" 2>/dev/null; :)
# Invalid EDMStatus: ERISC L1 holds a value that is neither a valid EDMStatus enum nor the base-UMD
# sentinel (0x49706550).  Seen as e.g. 0x49705180 (ROM postcode mid-boot) or other partial-write artifacts.
# terminate_stale_erisc_routers zeros the address to break corruption cascade for next session.
# status=0xdeaddead: firmware-written fatal-error sentinel seen in teardown timeout messages.
INVALID_EDMSTATUS=$(grep -cE 'NOT a valid EDMStatus value|zeroed edm_status_address' "$CLEAN" 2>/dev/null; :)
DEADDEAD_STATUS=$(grep -cE 'status=0xdeaddead' "$CLEAN" 2>/dev/null; :)
# FIX RZ (#42429): configure_fabric() sets fabric_stale_base_umd_channels_=true when
# skip_soft_reset_channels is non-empty on a non-MMIO device (FIX M path).  Enables
# is_fabric_degraded() to return true so Python AllGather tests skip instead of hang.
STALE_BASE_UMD_FIRES=$(grep -cE 'Setting fabric_stale_base_umd_channels_=true' "$CLEAN" 2>/dev/null; :)
# Python-level skip in test_gap23 — is_fabric_degraded() returns true (FIX RZ flag set)
# and AllGather is skipped to avoid hang on base-UMD cluster (commit 74cb2aa7591).
FIX_RZ_SKIP_FIRES=$(grep -cE 'fabric degraded.*base-UMD channels.*skipping AllGather|GAP-23.*fabric degraded.*base-UMD' "$CLEAN" 2>/dev/null; :)
# FIX QW-B (#42429): C++ fixture skip guard fires because stale_base_umd_channels=true.
# Combined check (relay_broken || channels_not_ready || stale_base_umd) caught the FIX M
# degraded state that the old guard (relay_broken || channels_not_ready) missed.
FIX_QW_B_SKIP=$(grep -cE 'stale_base_umd_channels=true.*skipping to avoid dispatch core hang|skipping to avoid dispatch core hang on base-UMD cluster' "$CLEAN" 2>/dev/null; :)
# FIX QW (#42429): Metal C++ is_fabric_degraded() API call found the cluster degraded.
# This fires in any context where Python or C++ calls MeshDevice::is_fabric_degraded()
# and it returns true (relay_broken, channels_not_ready, OR stale_base_umd_channels set).
# The log line format is: "FIX QW (#42429): cluster degraded (device N ...=true)"
FIX_QW_FIRES=$(grep -cE 'FIX QW \(#42429\): cluster degraded' "$CLEAN" 2>/dev/null; :)
# FIX BB (#42429): GAP-38 Testee-2 subprocess detected degraded cluster and bailed out.
# Testee-2 writes {"error": "SKIP:fabric_degraded..."} and exits 2; parent detects
# error.startswith("SKIP:") and calls pytest.skip() instead of pytest.fail() to avoid
# the 60s AllGather hang that previously produced a false test failure.
FIX_BB_FIRES=$(grep -cE 'FIX BB \(#42429\)|SKIP:fabric_degraded.*FIX BB' "$CLEAN" 2>/dev/null; :)
# FIX RX (#42429): fixture TearDown skip-quiesce guard.  Fires in both the base class
# (MeshDeviceFixtureBase::TearDown, added commit 374e43aff1b) and the sub-class
# (MultiCQFabricMeshDevice2x4Fixture::TearDown) when fabric is broken.
# Before the base-class commit, only the sub-class had this guard; any other
# MeshDeviceFixtureBase-derived fixture (MeshDevice1x4Fixture etc.) would still
# call quiesce_devices() on a broken cluster → ~72s hang + dispatch corruption.
FIX_RX_FIRES=$(grep -cE 'FIX RX \(#42429\).*fabric broken|FIX RX.*skipping quiesce_devices' "$CLEAN" 2>/dev/null; :)
# FIX BC (#42429): MeshDeviceFixtureBase::SetUp() catch — MeshDevice::create() threw
# "Device N is not active" because prior session left ETH relay dead (FIX AQ drops
# non-MMIO devices from UMD TopologyDiscovery) → initialize_fabric_and_dispatch_fw()
# TT_FATALs. FIX BC catches and converts to SetFabricConfig(DISABLED) + GTEST_SKIP.
# Without FIX BC: AllGatherPersistentOutput/ReduceScatter/AllReduce FAIL (SIGABRT)
# instead of SKIP, hiding the hardware-degraded root cause.
FIX_BC_FIRES=$(grep -cE 'FIX BC \(#42429\).*not active|FIX BC \(#42429\).*ETH relay dead' "$CLEAN" 2>/dev/null; :)
# FIX PG (#42429): skip FIX AY when ALL MMIO ETH heartbeats timed out (ac_heartbeat_any_ready=false).
# Fires in RiscFirmwareInitializer::teardown() when no MMIO ETH core confirmed its heartbeat.
# If none do, the relay is NOT restored and FIX AY (deferred non-MMIO ERISC reset) is skipped.
# Without FIX PG: teardown enters FIX AY for every non-MMIO device, polling up to 5s × N channels
# × M non-MMIO devices (e.g., 5s × 4ch × 4 non-MMIO = 80s wasted per session) with no progress.
# Log patterns:
#   "teardown: FIX PG (#42429): ALL MMIO ETH heartbeats timed out — relay NOT restored; skipping FIX AY"
#   "teardown: FIX PG: skipping FIX AY — relay not restored, N non-MMIO device(s) will be handled by FIX BC on next SetUp."
FIX_PG_FIRES=$(grep -cE 'FIX PG.*ALL MMIO ETH heartbeats timed out|FIX PG.*skipping FIX AY' "$CLEAN" 2>/dev/null; :)
# FIX PY (#42429): Phase 2.5 fast-skip when relay already marked broken by a prior channel
# on the same non-MMIO device.  Without FIX PY, each remaining channel on a dead relay
# device also burns 3×retry×5s≈21s before the Phase 2.5 loop exits.  FIX PY skips them.
# Log: "relay already marked broken by prior channel — skipping (FIX PY #42429)"
FIX_PY_FIRES=$(grep -cE 'relay already marked broken by prior channel.*FIX PY|FIX PY.*relay already marked broken' "$CLEAN" 2>/dev/null; :)
# FIX QU (#42429): FabricFirmwareInitializer::configure() re-asserts per-device degraded flags
# after Device::configure_fabric() resets them.  configure_fabric() unconditionally clears
# fabric_relay_path_broken_ and fabric_channels_not_ready_for_traffic_ at its top, but for
# devices in dead_relay_devices_ or mmio_dead_master_chan_devices_ no fresh firmware was loaded —
# the relay or master ERISC is still dead — so FIX QU restores both flags so test guards
# (FIX QS/QW/RX) can detect the degraded cluster and SKIP instead of hanging in AllGather.
# Without FIX QU: guards see false-healthy cluster → AllGather hangs for OPERATION_TIMEOUT.
# Log (relay broken re-assert):  "FabricFirmwareInitializer::configure: FIX QU (#42429) — re-asserting fabric_relay_path_broken_ for Device N"
# Log (not-ready re-assert):     "FabricFirmwareInitializer::configure: FIX QU (#42429) — setting fabric_channels_not_ready_for_traffic_ for Device N"
FIX_QU_FIRES=$(grep -cE 'FIX QU \(#42429\).*re-asserting fabric_relay_path_broken_|FIX QU \(#42429\).*setting fabric_channels_not_ready_for_traffic_' "$CLEAN" 2>/dev/null; :)
# FIX QV (#42429): Phase 4 MUX poll skip when fabric_channels_not_ready_for_traffic_=true
# (MMIO dead-master-chan device).  Without FIX QV: Phase 4 times out (5000ms × N channels)
# then throws, marking UDM tests FAILED instead of SKIPPED.
# Log: "wait_for_fabric_workers_ready: Device N has channels not ready for traffic ... FIX QV"
FIX_QV_FIRES=$(grep -cE 'FIX QV|channels not ready for traffic.*skipping Phase 4.*FIX QV|MMIO dead-master-chan device.*FIX QV' "$CLEAN" 2>/dev/null; :)
# FIX QE (test_multi_tensor_ccl.cpp): GTest SKIP guard — emits GTEST_SKIP() with message
# "FIX QE: fabric not ready (stale ETH firmware from prior teardown); skipping to avoid dispatch timeout."
# This fires in t3k_ttnn_tests multi-tensor CCL tests when fabric is degraded.  The message is in
# GTest format (no Metal log prefix), so the standard TIMELINE/PHASES greps won't catch it.
# Counter searches raw cleaned log for the skip message text.
FIX_QE_FIRES=$(grep -cE 'FIX QE.*fabric not ready|GTEST_SKIP.*FIX QE|fabric not ready.*stale ETH.*FIX QE' "$CLEAN" 2>/dev/null; :)
# FIX SB (GAP-76): guard not_done_cores.insert for IDLE_ETH with INIT_FABRIC flag check.
# Fires in initialize_and_launch_firmware() when IDLE_ETH cores are present but INIT_FABRIC
# is not set in FabricManagerMode (e.g., FabricManagerMode::TERMINATE_FABRIC only).
# Without FIX SB: IDLE_ETH core added to not_done_cores unconditionally → deassert_risc_reset_at_core
# fires → stale L1 firmware writes 0x55 to run_mailbox → wait_until_cores_done TT_FATAL (SIGABRT).
# No log when FIX SB fires (silent guard) — its absence is detected by SIGABRT (exit 134).
# Counter always 0 if FIX SB present and working correctly.
FIX_SB_FIRES=$(grep -cE 'FIX SB.*IDLE_ETH.*not_done_cores|FIX SB.*init_fabric.*guard' "$CLEAN" 2>/dev/null; :)
# FIX GS (#42429): SIGALRM timeout guard for ensure_cluster_healthy open_mesh_device().
# Fires in conftest.py when open_mesh_device() hangs > 30s (UMD relay blocking without exception).
# Log (Python print): "[conftest] FIX GS (#42429): open_mesh_device() did not complete within 30 s"
FIX_GS_FIRES=$(grep -cE 'FIX GS.*open_mesh_device.*did not complete|FIX GS.*dirty post-crash state' "$CLEAN" 2>/dev/null; :)
# FIX RM (#42429): open_mesh_device try/except for "failed to initialize FW" / "run_mailbox"
# in conftest.py mesh_device fixture.  Fires when hardware cores stuck in unexpected mailbox state.
# Log: "FIX RM (#42429): open_mesh_device threw FW init failure"
FIX_RM_FIRES=$(grep -cE 'FIX RM.*open_mesh_device threw FW init failure|FIX RM.*hardware cores in unexpected mailbox' "$CLEAN" 2>/dev/null; :)
# FIX CD (#42429): TTSwitch explicit-ID test skip guards — per-device fabric state flags.
# Log: "Skipping: device N has degraded fabric (#42429)"
FIX_CD_FIRES=$(grep -cE 'FIX CD.*degraded fabric|has degraded fabric \(#42429\)' "$CLEAN" 2>/dev/null; :)
# FIX CD-4 (#42429): SetFabricConfig(FABRIC_2D) topology mapping failure in SetUp — GTEST_SKIP.
# Log: "FIX CD-4 (#42429): FABRIC_2D init failed in SetUp — degraded cluster"
# Also: TearDown swallowed exception log "FIX CD-4 (#42429): SetFabricConfig(DISABLED) threw"
FIX_CD4_FIRES=$(grep -cE 'FIX CD-4.*FABRIC_2D init failed|FIX CD-4.*SetFabricConfig.*threw' "$CLEAN" 2>/dev/null; :)
# FIX GS-2b (#42429): warm-up open/close cycle after tt-smi -r before FABRIC_2D tests.
# Log: "[conftest] FIX GS-2b (#42429): warm-up open/close cycle"
# Also: warm-up degraded warning "FIX GS-2b: WARNING: warm-up mesh still reports degraded"
FIX_GS2B_FIRES=$(grep -cE 'FIX GS-2b.*warm-up' "$CLEAN" 2>/dev/null; :)
FIX_GS2B_DEGRADED=$(grep -cE 'FIX GS-2b.*WARNING.*warm-up.*still reports degraded' "$CLEAN" 2>/dev/null; :)
# FIX SA (GAP-76, llrt.cpp): unknown run_mailbox value → warning + return-false (not TT_FATAL).
# Log: "FIX SA (GAP-76): core X run_mailbox=0xNN is not a known RUN_MSG_* value"
FIX_SA_LLRT_FIRES=$(grep -cE 'FIX SA \(GAP-76\).*run_mailbox.*is not a known RUN_MSG' "$CLEAN" 2>/dev/null; :)
# FIX CD-4b (#42429): FabricSwitchManager::setup() timed out on rank 1 (switch node) because
# rank 0 (compute node) crashed before broadcasting chip info. Wraps setup() in try/catch → GTEST_SKIP.
# Log: "FIX CD-4b (#42429): FabricSwitchManager setup failed in SetUp"
FIX_CD4B_FIRES=$(grep -cE 'FIX CD-4b.*FabricSwitchManager.*setup failed' "$CLEAN" 2>/dev/null; :)
# FIX CD-5 (#42429): open_devices_internal() wrapped in try/catch to handle ControlPlane
# validate_requested_intermesh_connections unordered_map::at on degraded runner.
# Log: "open_devices_internal failed (degraded runner / topology mismatch)"
FIX_CD5_FIRES=$(grep -cE 'open_devices_internal failed.*degraded runner|open_devices_internal failed.*topology mismatch' "$CLEAN" 2>/dev/null; :)
# FIX CD-6 (#42429): After FIX CD-5 catches an open_devices_internal exception, the test loop
# must break (not continue) because the peer MPI rank may have aborted. Continuing to the next
# test group would hang in collective control-plane reinit waiting for the dead rank.
# Log: "Hardware fault during open_devices — aborting remaining test groups (#42429 FIX CD-6)"
FIX_CD6_FIRES=$(grep -cE 'Hardware fault during open_devices.*aborting remaining test groups|hardware fault.*aborting.*FIX CD-6' "$CLEAN" 2>/dev/null; :)
# FIX CD-7 (#42429): MeshDeviceTTSwitchFixture TearDown — skip FabricSwitchManager::teardown()
# when setup_failed_=true (FIX CD-4b caught exception in SetUp). Prevents tearing down
# uninitialized switch manager state which may crash or hang.
# Log: "FIX CD-7 (#42429): FabricSwitchManager::teardown() threw — swallowed:"
FIX_CD7_FIRES=$(grep -cE 'FIX CD-7.*FabricSwitchManager.*teardown.*threw' "$CLEAN" 2>/dev/null; :)
# FIX GS-3 (#42429): FABRIC_1D warm-up open/close after every tt-smi -r in run_t3000_unit_tests.sh.
# Prevents base-UMD reset cycle where GTest SKIPs → tt-smi -r → base-UMD reloaded → FIX M → SKIP → loop.
# Log: "[FIX GS-3] initial warm-up complete" / "[FIX GS-3] post-reset warm-up complete"
# Warning: "[FIX GS-3] WARNING: initial warm-up failed" / "[FIX GS-3] WARNING: post-reset warm-up failed"
FIX_GS3_FIRES=$(grep -cE '\[FIX GS-3\].*warm-up complete' "$CLEAN" 2>/dev/null; :)
FIX_GS3_WARN=$(grep -cE '\[FIX GS-3\] WARNING.*warm-up failed' "$CLEAN" 2>/dev/null; :)
# FIX GS-3b (#42429): conftest warm-up fails with "failed to initialize FW" → pytest.exit()
# to abort cleanly (exit 1) instead of yielding and hitting outer bash 300s timeout (exit 124).
# Log: "[conftest] FIX GS-3b (#42429): warm-up failed with fatal FW init error"
# pytest.exit message: "FIX GS-3b: hardware FW init failed after tt-smi -r — board needs physical reset"
FIX_GS3B_FIRES=$(grep -cE 'FIX GS-3b.*warm-up failed.*fatal FW init|FIX GS-3b.*hardware FW init failed' "$CLEAN" 2>/dev/null; :)

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
    echo "  => [FIX AU/AU-2] relay-broken non-MMIO TERMINATE/l1_barrier attempted (${FIX_AU_FIRES} event(s))"
    echo "     FIX AU-2: attempt is now made even when relay is broken (was silently skipped in FIX AU)."
    echo "     Failure is caught and logged; channels proceed to force-reset second pass."
fi
if [ "${FIX_AX_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AX/AX-2] relay-confirmed-dead non-MMIO ETH: assert_risc_reset_at_core attempted (${FIX_AX_FIRES} event(s))"
    echo "     FIX AX-2: reset is now attempted even when relay is dead (was silently skipped in FIX AX)."
    echo "     On success: FIX XY-2 clears relay_broken (relay restored). On failure: logged and noted."
fi
if [ "${FIX_XY2_RELAY_RESTORED:-0}" -gt 0 ]; then
    echo "  => [FIX XY-2] relay_broken cleared after successful ERISC force-reset (${FIX_XY2_RELAY_RESTORED} event(s))"
    echo "     Relay path restored — subsequent multicast writes and AllGather should succeed."
fi
if [ "${FIX_SB2R_CLEAN_BOOT:-0}" -gt 0 ]; then
    echo "  => [FIX SB2-R] relay_broken NOT set (${FIX_SB2R_CLEAN_BOOT} device(s)) — probe was clean, this was a normal boot"
    echo "     FIX M channels had 0x49706550 sentinel (mid-init) but probe detected no dead channels."
    echo "     relay_broken suppressed to prevent false-positive topology degradation on clean boot."
fi
if [ "${FIX_AJ_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AJ] relay path confirmed dead during force-reset pass (${FIX_AJ_FIRES} event(s)) — device added to relay_dead_devices set"
fi
if [ "${FIX_AK_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AK] transitive relay guard: l1_barrier skipped for ALL non-MMIO devices (${FIX_AK_FIRES} event(s)) — relay-dead device(s) present"
fi
if [ "${FIX_AQ_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AQ] edm_status_address sentinel poll: ${FIX_AQ_SUCCESS:-0} succeeded, ${FIX_AQ_TIMEOUT:-0} timed out (${FIX_AQ_FIRES} total events)."
    if [ "${FIX_AQ_TIMEOUT:-0}" -gt 0 ]; then
        echo "     TIMED OUT channels: ROM postcode 0x49705180 persisted past 10s (FIX AT timeout)."
        echo "     Next session will see FIX RP polling on these channels."
    fi
fi
if [ "${FIX_RP_SUCCESS:-0}" -gt 0 ] || [ "${FIX_RP_TIMEOUT:-0}" -gt 0 ] || [ "${FIX_RP_UNEXPECTED:-0}" -gt 0 ]; then
    echo "  => [FIX RP] ROM postcode transition poll (non-MMIO): ${FIX_RP_SUCCESS:-0} succeeded, ${FIX_RP_TIMEOUT:-0} timed out, ${FIX_RP_UNEXPECTED:-0} unexpected value."
    if [ "${FIX_RP_TIMEOUT:-0}" -gt 0 ]; then
        echo "     TIMED OUT: 0x49705180 did NOT transition to base-UMD (0x49706550) within 5s."
        echo "     These channels fall through to probe_dead_channels → relay_broken cascade."
        echo "     Possible cause: ROM boot after PCIe hard reset takes >5s; consider increasing kRomPostcodePollTotalMs."
    fi
    if [ "${FIX_RP_UNEXPECTED:-0}" -gt 0 ]; then
        echo "     UNEXPECTED VALUE: channel transitioned to something other than 0x49706550 or 0x49705180."
        echo "     This may indicate firmware crash or L1 corruption mid-boot."
    fi
fi
if [ "${FIX_RR_SUCCESS:-0}" -gt 0 ] || [ "${FIX_RR_FAIL:-0}" -gt 0 ] || [ "${FIX_RS_RECOVERED:-0}" -gt 0 ]; then
    echo "  => [FIX RR/RS] MMIO ROM-postcode channel recovery in configure_fabric_cores:"
    echo "     PCIe-direct soft reset: ${FIX_RR_SUCCESS:-0} succeeded, ${FIX_RR_FAIL:-0} failed."
    echo "     Recovered channels propagated back to configure_fabric (FIX RS): ${FIX_RS_RECOVERED:-0} total."
    if [ "${FIX_RR_SUCCESS:-0}" -gt 0 ]; then
        echo "     GOOD: FIX RR recovered MMIO channels; they get firmware and rejoin healthy fabric."
        echo "     These channels are removed from fabric_pre_dead_channels_ (FIX RS) so FIX QE does NOT skip AllGather."
    fi
    if [ "${FIX_RR_FAIL:-0}" -gt 0 ]; then
        echo "     FAIL: ${FIX_RR_FAIL:-0} MMIO channel(s) did not respond to PCIe-direct reset → moved to newly_dead_channels → TT_THROW."
        echo "     Likely cause: PCIe link error or hardware fault on Device 0."
    fi
fi
if [ "${FIX_M_TRANSITION_COUNT:-0}" -gt 0 ]; then
    echo "  => [FIX M] base-UMD channel transitions via launch_msg: ${FIX_M_TRANSITION_COUNT} device(s) set stale_base_umd=true."
fi
if [ "${FIX_RZ2_CLEAR_COUNT:-0}" -gt 0 ]; then
    echo "  => [FIX RZ2] stale_base_umd flag cleared after ring-sync + health check: ${FIX_RZ2_CLEAR_COUNT} device(s)."
fi
if [ "${FIX_M_TRANSITION_COUNT:-0}" -gt 0 ] && [ "${FIX_RZ2_CLEAR_COUNT:-0}" -eq 0 ]; then
    echo "     WARNING: FIX M set stale_base_umd on ${FIX_M_TRANSITION_COUNT} device(s) but FIX RZ2 never cleared it."
    echo "     Root cause: ring-sync or health check failed, so channels are truly degraded."
fi
if [ "${FIX_QW_SKIP_COUNT:-0}" -gt 0 ]; then
    echo "  => [FIX QW] test skip guards fired: ${FIX_QW_SKIP_COUNT} occurrence(s) (stale_base_umd or channels_not_ready)."
fi
if [ "${FIX_AR2_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AR2] post-deassert heartbeat delay fired: ${FIX_AR2_FIRES} time(s) (prevents stale 0xABCD false positive)."
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
FIX_AV_SKIP=$(grep -cE 'running in degraded mode|configure_fabric.*degraded' "$CLEAN" 2>/dev/null; :)
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
if [ "${FIX_AO_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX AO] ring-sync STARTED early-exit: ${FIX_AO_FIRES} device(s) — master chan stuck at STARTED after 1s (peer not responding, likely out-of-mesh). Ring sync skipped cleanly instead of waiting full timeout."
fi
if [ "${FIX_BG_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX BG] host-pre-launch 0xdeadb07e sentinel: ${FIX_BG_FIRES} device(s) — ERISC soft-reset (FIX RR) succeeded but ERISC never started executing. Marked dead-master-chan; ring sync skipped."
fi
if [ "${FIX_BH_SUCCESS:-0}" -gt 0 ] || [ "${FIX_BH_TIMEOUT:-0}" -gt 0 ]; then
    echo "  => [FIX BH] ERISC ROM-to-UMD boot poll after FIX RR deassert: ${FIX_BH_SUCCESS:-0} succeeded, ${FIX_BH_TIMEOUT:-0} timed out."
    if [ "${FIX_BH_TIMEOUT:-0}" -gt 0 ]; then
        echo "     TIMEOUT: ERISC did not exit ROM phase within 500ms — channel left dead, L1 init skipped."
    fi
fi
if [ "${FIX_BO_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX BO] Phase 5 kSyncTimeoutMs extended 10s→120s on ${FIX_BO_FIRES} device(s) — stale base-UMD channels detected."
fi
if [ "${FIX_BP_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX BP] fabric_context null guard: ${FIX_BP_FIRES} event(s) — teardown ordering race hit and recovered."
    echo "     ControlPlane::get_fabric_context() was called after fabric_context_ destroyed; FIX AQ 10s wait skipped."
fi
if [ "${FIX_BF_MMIO_MASTER:-0}" -gt 0 ] || [ "${FIX_BF_NONMMIO_MASTER:-0}" -gt 0 ]; then
    echo "  => [FIX BD/BF] master chan selection: ${FIX_BF_MMIO_MASTER:-0} MMIO-peer master(s), ${FIX_BF_NONMMIO_MASTER:-0} non-MMIO-peer master(s)."
fi
if [ "${FIX_ST_DEAD:-0}" -gt 0 ]; then
    echo "  => [FIX ST] MMIO master chan confirmed dead (not recovered by FIX RR): ${FIX_ST_DEAD} channel(s) — ring sync skipped."
fi
if [ "${FIX_ST_RECOVERED:-0}" -gt 0 ]; then
    echo "  => [FIX ST] MMIO master chan recovered by FIX RR soft-reset: ${FIX_ST_RECOVERED} channel(s) — ring sync proceeding normally."
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
if [ "${FIX_PF_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX PF] UMD base fw heartbeat detected at Metal exit — Metal exit signal skipped (${FIX_PF_FIRES} occurrence(s))."
    echo "     BRISC was still running base-UMD firmware at process shutdown; writing the Metal exit signal"
    echo "     would overwrite a live base-UMD relay.  FIX PF clears the stale fw_launch_addr instead."
    echo "     Distinct from FIX PA (init cascade): FIX PF fires in risc_firmware_initializer.cpp teardown path."
fi
if [ "${FIX_TV_SUCCESS:-0}" -gt 0 ] || [ "${FIX_TV_TIMEOUT:-0}" -gt 0 ]; then
    echo "  => [FIX TV/TW] MMIO ETH heartbeat poll: ${FIX_TV_SUCCESS:-0} success, ${FIX_TV_TIMEOUT:-0} timeout."
    echo "     FIX TV polls MMIO ETH heartbeat in run_launch_phase() after reset_cores()."
    echo "     FIX TW detects static 0xABCDxxxx UMD marker as immediate-ready (no increment wait)."
    if [ "${FIX_TV_TIMEOUT:-0}" -gt 0 ]; then
        echo "     WARNING: FIX TV timed out — MMIO ETH still rebooting after 3000ms. Probe_dead likely."
        echo "     If FIX TW is present, timeout means hardware reboot genuinely slow (not UMD-marker miss)."
    fi
fi
if [ "${INVALID_EDMSTATUS:-0}" -gt 0 ]; then
    echo "  => [CORRUPT EDMSTATUS] Invalid EDMStatus value(s) in ERISC L1 (${INVALID_EDMSTATUS} occurrence(s))."
    echo "     Value is neither a valid EDMStatus enum nor the base-UMD sentinel (0x49706550)."
    echo "     Examples: 0x49705180 (ROM postcode mid-boot), 0x0 (zeroed), arbitrary partial-write garbage."
    echo "     terminate_stale_erisc_routers zeros the address to break corruption cascade for next session."
    echo "     Indicates ERISC L1 was written partially, mid-boot interrupted, or corrupted by prior crash."
fi
if [ "${DEADDEAD_STATUS:-0}" -gt 0 ]; then
    echo "  => [0xdeaddead] Firmware fatal-error sentinel detected in teardown timeout (${DEADDEAD_STATUS} occurrence(s))."
    echo "     An ERISC did not reach TERMINATED within 5000ms and its status reads as 0xdeaddead."
    echo "     This is a firmware-written sentinel indicating the ERISC encountered a fatal error."
    echo "     The hardware reset (assert_risc_reset_at_core) should follow to recover the channel."
fi
if [ "${STALE_BASE_UMD_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX RZ] configure_fabric() set fabric_stale_base_umd_channels_=true on ${STALE_BASE_UMD_FIRES} non-MMIO device(s)."
    echo "     FIX M (launch_msg transition) detected base-UMD relay firmware retained on non-MMIO ERISCs."
    echo "     FIX RZ sets fabric_stale_base_umd_channels_=true so is_fabric_degraded() returns true,"
    echo "     enabling Python AllGather tests and C++ fixture guards to skip instead of hang."
fi
if [ "${FIX_RZ_SKIP_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX RZ SKIP] is_fabric_degraded() returned true (base-UMD channels) — AllGather skipped ${FIX_RZ_SKIP_FIRES} time(s)."
    echo "     Python test_gap23 guard (commit 74cb2aa7591) detected stale base-UMD state and skipped AllGather."
    echo "     Without this guard, AllGather on stale-firmware devices causes completion_queue_wait_front hang."
fi
if [ "${FIX_QW_B_SKIP:-0}" -gt 0 ]; then
    echo "  => [FIX QW-B] C++ fixture skip guard fired on stale_base_umd_channels=true (${FIX_QW_B_SKIP} occurrence(s))."
    echo "     Combined guard (relay_broken || channels_not_ready || stale_base_umd) caught the FIX M degraded state"
    echo "     that the old guard (relay_broken || channels_not_ready) would have missed."
    echo "     Without FIX QW-B, AllGather would proceed on stale-firmware cluster → dispatch core timeout (~100s)."
fi
if [ "${FIX_QW_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX QW] MeshDevice::is_fabric_degraded() returned true (${FIX_QW_FIRES} call(s))."
    echo "     Metal C++ log: 'FIX QW (#42429): cluster degraded (device N relay_broken=... stale_base_umd=true)'."
    echo "     Indicates a test or fixture successfully detected the degraded cluster and is about to skip."
    echo "     Each occurrence = one call to is_fabric_degraded() on a degraded device."
    echo "     Correlation: expect GAP-25/27/38/23 Python pytest.skip() counts to match or exceed this."
fi
if [ "${FIX_BB_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX BB] GAP-38 Testee-2 subprocess detected degraded cluster (${FIX_BB_FIRES} occurrence(s))."
    echo "     Testee-2 opens MeshDevice, calls is_fabric_degraded() after set_fabric_config()."
    echo "     FIX BB writes {\"error\": \"SKIP:fabric_degraded...\"} + sys.exit(2); parent sees SKIP prefix"
    echo "     and calls pytest.skip() instead of pytest.fail()."
    echo "     Without FIX BB: AllGather in Testee-2 would hang 60s → parent: pytest.fail(\"GAP-38 HANG\")."
fi
if [ "${FIX_RX_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX RX] fixture TearDown skip-quiesce guard fired (${FIX_RX_FIRES} occurrence(s))."
    echo "     When fabric is broken (relay_broken || channels_not_ready || stale_base_umd),"
    echo "     TearDown skips quiesce_devices() (~72s waste + dispatch corruption) and calls close() directly."
    echo "     Sources: MeshDeviceFixtureBase::TearDown() (base class, commit 374e43aff1b)"
    echo "              MultiCQFabricMeshDevice2x4Fixture::TearDown() (sub-class, prior commit)"
    echo "     Log patterns:"
    echo "       '[fixture_teardown] MeshDeviceFixtureBase::TearDown() FIX RX (#42429): fabric broken'"
    echo "       '[MultiCQFabricMeshDevice2x4Fixture::TearDown] FIX RX (#42429): fabric broken after test body'"
fi
if [ "${FIX_BC_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX BC] MeshDeviceFixtureBase::SetUp() caught 'Device not active' exception (${FIX_BC_FIRES} occurrence(s))."
    echo "     MeshDevice::create() threw 'Device N is not active' because FIX AQ dropped non-MMIO"
    echo "     devices from UMD TopologyDiscovery (prior session left ETH relay dead)."
    echo "     FIX BC catch: SetFabricConfig(DISABLED) + GTEST_SKIP instead of TT_FATAL/SIGABRT."
    echo "     Without FIX BC: AllGatherPersistentOutput/ReduceScatter/AllReduce would FAIL (exit 134)"
    echo "     instead of SKIP, hiding the hardware-degraded root cause."
    echo "     Log pattern: 'FIX BC (#42429): MeshDevice::create() threw Device not active'"
fi
if [ "${FIX_GS_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX GS] conftest SIGALRM timeout: open_mesh_device() hung > 30s (${FIX_GS_FIRES} occurrence(s))."
    echo "     UMD ETH relay blocking without exception (dirty post-crash state)."
    echo "     Action: treat as degraded → tt-smi -r runs immediately instead of 300s UMD timeout."
fi
if [ "${FIX_RM_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX RM] conftest mesh_device fixture: FW init failure caught (${FIX_RM_FIRES} occurrence(s))."
    echo "     open_mesh_device threw 'failed to initialize FW' or 'run_mailbox' — hardware cores stuck."
    echo "     Action: reset_fabric() + pytest.skip() instead of test ERROR."
fi
if [ "${FIX_CD_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX CD] TTSwitch explicit-ID test skip: device has degraded fabric (${FIX_CD_FIRES} occurrence(s))."
    echo "     Per-device check: relay_path_broken || channels_not_ready || stale_base_umd."
    echo "     Both ranks skip independently before MPI comm — prevents cross-rank hang."
fi
if [ "${FIX_CD4_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX CD-4] TTSwitch SetUp: SetFabricConfig(FABRIC_2D) topology mapping failed (${FIX_CD4_FIRES} occurrence(s))."
    echo "     Dead-firmware gateway ETH channels reduced physical mesh connectivity for STRICT inter-mesh mapping."
    echo "     SetUp catches exception + GTEST_SKIP; TearDown skips SetFabricConfig(DISABLED) when setup_failed_=true."
    echo "     TearDown belt-and-suspenders try/catch logs any swallowed exception for diagnosis."
fi
if [ "${FIX_GS2B_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX GS-2b] conftest warm-up open/close cycle fired (${FIX_GS2B_FIRES} occurrence(s))."
    echo "     After tt-smi -r, open/close once to transition base-UMD ETH channels via FIX M launch_msg."
    echo "     Prevents ControlPlane SIGBUS crash on subsequent FABRIC_2D init."
    if [ "${FIX_GS2B_DEGRADED:-0}" -gt 0 ]; then
        echo "     [WARNING] warm-up mesh still reported degraded AFTER tt-smi -r (${FIX_GS2B_DEGRADED} occurrence(s))."
        echo "     Hardware may not fully recover — FABRIC_2D tests may still fail."
    fi
fi
if [ "${FIX_SA_LLRT_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX SA] llrt.cpp: unknown run_mailbox value detected (${FIX_SA_LLRT_FIRES} occurrence(s))."
    echo "     Stale firmware left non-standard run_mailbox (e.g. 0x55) after tt-smi reset or SIGKILL."
    echo "     Treated as not-done — 10s timeout fires and triggers force-reset recovery."
    echo "     Without FIX SA: TT_FATAL crash (SIGABRT) on unexpected run_mailbox value."
fi
FIX_SC_TOTAL=$(( ${FIX_SC_PRESCAN:-0} + ${FIX_SC_INLINE:-0} ))
if [ "${FIX_SC_TOTAL:-0}" -gt 0 ]; then
    echo "  => [FIX SC] Stale go_msg rescue: ${FIX_SC_PRESCAN:-0} pre-scan + ${FIX_SC_INLINE:-0} inline (${FIX_SC_TOTAL} total)."
    echo "     Hard BRISC reset by rescue_stuck_dispatch_cores left stale L1 firmware writing 0x55."
    echo "     FIX SC asserts BRISC reset → writes RUN_MSG_DONE → core marked done immediately."
    echo "     Without FIX SC: 10s × N_cores wasted wait → TT_THROW FW init timeout."
    if [ "${FIX_SC_TENSIX_FIRE:-0}" -gt 0 ] || [ "${FIX_SC_ETH_FIRE:-0}" -gt 0 ]; then
        echo "     Core-type breakdown: TENSIX=${FIX_SC_TENSIX_FIRE:-0}  ETH(ACTIVE+IDLE)=${FIX_SC_ETH_FIRE:-0}."
    fi
    if [ "${FIX_SC_ETH_FIRE:-0}" -gt 0 ]; then
        echo "     [REGRESSION] FIX SC fired on ${FIX_SC_ETH_FIRE} ETH core(s) — this should not happen with FIX SC-ADDR."
        echo "     ETH cores in not_done_cores should use ETH go_msg address (FIX SC-ADDR).  If FIX SC"
        echo "     fires on ETH: wrong address still being used OR ETH core genuinely has stale go_msg."
        echo "     Check commit a50407086ac (FIX SC-ADDR) is present and llrt::get_core_type() is correct."
    fi
    if [ "${FIX_SC_FAIL:-0}" -gt 0 ]; then
        echo "     [WARNING] FIX SC rescue FAILED for ${FIX_SC_FAIL} core(s) — assert_risc_reset threw."
    fi
fi
if [ "${FIX_SC_ADDR_ETH_VALID:-0}" -gt 0 ] || [ "${FIX_SC_ADDR_ETH_STALE:-0}" -gt 0 ]; then
    echo "  => [FIX SC-ADDR] ETH core go_msg debug (requires debug log level):"
    echo "     Valid (no fire): ${FIX_SC_ADDR_ETH_VALID:-0}  STALE (fire): ${FIX_SC_ADDR_ETH_STALE:-0}."
    echo "     FIX SC-ADDR confirmed using per-core-type go_msg address for ETH cores in not_done_cores."
fi
if [ "${FIX_TF_CP_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TF cp] control_plane: excluded chip skipped in routing-plane/routing-table init (${FIX_TF_CP_FIRES} skip(s))."
    echo "     FIX TB excluded chip from topology mapper (degraded ERISC L1). FIX TF guards"
    echo "     initialize_dynamic_routing_plane_counts + convert_fabric_routing_table_to_chip_routing_table"
    echo "     so they skip absent chips instead of throwing before the MPI all_gather barrier."
    echo "     Without FIX TF: std::out_of_range BEFORE all_gather → peer rank hangs ~14 min (CI timeout)."
fi
if [ "${FIX_CD4B_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX CD-4b] TTSwitch rank 1: FabricSwitchManager::setup() failed (${FIX_CD4B_FIRES} occurrence(s))."
    echo "     Rank 0's SetFabricConfig(FABRIC_2D) crashed at topology mapping BEFORE broadcasting chip info."
    echo "     Rank 1's FabricSwitchManager::setup() timed out (~5s) waiting for chip info header."
    echo "     FIX CD-4b wraps the switch-node setup() in try/catch → GTEST_SKIP instead of test FAILED."
fi
if [ "${FIX_CD5_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX CD-5] open_devices_internal() topology mismatch caught (${FIX_CD5_FIRES} occurrence(s))."
    echo "     ControlPlane::validate_requested_intermesh_connections threw std::out_of_range"
    echo "     (unordered_map::at) — physical topology mapping missing expected fabric node IDs."
    echo "     FIX CD-5 wraps open_devices_internal() in try/catch → close_devices() + return false."
fi
if [ "${FIX_CD6_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX CD-6] Hardware fault abort: test loop broke after open_devices exception (${FIX_CD6_FIRES} occurrence(s))."
    echo "     FIX CD-5 caught the exception; FIX CD-6 breaks out of the test loop instead of continuing."
    echo "     Without FIX CD-6: rank 0 continues to next test group, calls SetFabricConfig(FABRIC_2D) which"
    echo "     reinitializes control plane (collective op). Rank 1 already aborted (TT_FATAL) → rank 0 hangs"
    echo "     indefinitely in collective init waiting for dead rank 1 (up to 15-min CI timeout)."
fi
if [ "${FIX_CD7_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX CD-7] FabricSwitchManager::teardown() threw on switch mesh (${FIX_CD7_FIRES} occurrence(s))."
    echo "     FIX CD-4b caught exception in SetUp → setup_failed_=true. Without FIX CD-7: TearDown"
    echo "     calls teardown() on uninitialized switch manager → potential crash/hang on rank 1."
    echo "     FIX CD-7 skips teardown() when setup_failed_ and wraps in try/catch as defense-in-depth."
fi
if [ "${FIX_GS3_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX GS-3] FABRIC_1D warm-up after tt-smi -r succeeded (${FIX_GS3_FIRES} occurrence(s))."
    echo "     open+close cycle transitions base-UMD ETH channels via FIX M launch_msg before GTest runs."
    echo "     Prevents base-UMD reset cycle (GTest SKIP → tt-smi -r → base-UMD → SKIP → loop)."
    if [ "${FIX_GS3_WARN:-0}" -gt 0 ]; then
        echo "     [WARNING] warm-up failed on ${FIX_GS3_WARN} occasion(s) — GTests may still SKIP."
    fi
fi
if [ "${FIX_GS3B_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX GS-3b] conftest warm-up fatal FW init failure → pytest.exit (${FIX_GS3B_FIRES} occurrence(s))."
    echo "     After tt-smi -r, open_mesh_device() failed with 'failed to initialize FW'."
    echo "     Hardware cannot recover in software — board needs physical reset."
    echo "     FIX GS-3b calls pytest.exit(rc=1) to abort cleanly instead of yielding"
    echo "     and hitting the outer bash 300s timeout (exit code 124)."
fi
if [ "${FIX_PG_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX PG] teardown: ALL MMIO ETH heartbeats timed out — FIX AY skipped (${FIX_PG_FIRES} occurrence(s))."
    echo "     ac_heartbeat_any_ready=false: no MMIO ETH core confirmed heartbeat during FIX AC poll."
    echo "     Relay NOT restored by MMIO reset → FIX AY (deferred non-MMIO ERISC reset) skipped."
    echo "     Without FIX PG: FIX AY would poll 5s × N_channels × N_non_mmio_devices with no progress"
    echo "     (relay not restored → all remote reads time out immediately)."
    echo "     Non-MMIO devices handled by FIX BC on next SetUp() instead."
    echo "     Log: 'FIX PG (#42429): ALL MMIO ETH heartbeats timed out — relay NOT restored; skipping FIX AY'"
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
if [ "${FIX_TG_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TG] configure_routing_tables: connected chip excluded by FIX TB has no host rank (${FIX_TG_FIRES} skip(s))."
    echo "     get_host_rank_for_chip() returned nullopt for a chip that FIX TB excluded (degraded cluster)."
    echo "     Before FIX TG: TT_ASSERT (no-op in Release) failed to guard .value() → bad_optional_access in ControlPlane ctor."
    echo "     FIX TG: has_value() guard + warning log + continue, matching FIX TE pattern."
elif grep -qE 'bad optional access' "$CLEAN" 2>/dev/null && [ "${FIX_TE_SKIPS:-0}" -gt 0 ]; then
    echo "  => [FIX TG MISSING?] bad optional access + FIX TE skips present, but no FIX TG warning."
    echo "     Possible regression: control_plane.cpp:1212 TT_ASSERT→has_value() guard missing or reverted."
    echo "     Commit to check: FIX TG in configure_routing_tables_for_fabric_ethernet_channels."
fi
if [ "${FIX_TH_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TH] relay_mux GenerateStaticConfigs: no available dispatch links (${FIX_TH_FIRES} occurrence(s))."
    echo "     get_forwarding_link_indices(src, dst) returned empty — all ETH channels between devices dead/excluded."
    echo "     Device marked channels_not_ready_for_traffic — FIX SA will GTEST_SKIP the affected test(s)."
    echo "     Before FIX TH: get_dispatch_link_index() would TT_FATAL with 'No links available from ... to ...'."
elif grep -qE 'No links available from' "$CLEAN" 2>/dev/null; then
    echo "  => [FIX TH MISSING] 'No links available from' TT_FATAL found — FIX TH may be absent or reverted."
    echo "     Source: relay_mux.cpp get_dispatch_link_index() TT_FATAL(!available_links.empty())."
    echo "     Occurs when MMIO device IS in fabric cluster but all ETH links to downstream device are dead."
    echo "     Fix: add FIX TH preflight check in GenerateStaticConfigs() before get_dispatch_link_index() call."
    echo "     See relay_mux.cpp — call get_forwarding_link_indices(src,dst); if empty, set channels_not_ready."
fi
if [ "${FIX_TH_CP_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TH-CP] convert_fabric_routing_table: excluded chip skipped in num_ports_per_chip loop (${FIX_TH_CP_FIRES} occurrence(s))."
    echo "     get_chip_ids() returned chips excluded by FIX TB (degraded topology). Without FIX TH:"
    echo "     TT_FATAL 'Fabric node id (M0, D0) not found in mapping' → both ranks crash, rank 1 hangs 15 min."
elif grep -qE 'Fabric node id.*not found in mapping' "$CLEAN" 2>/dev/null; then
    echo "  => [FIX TH-CP MISSING] 'Fabric node id not found in mapping' TT_FATAL detected — FIX TH guard absent/reverted."
    echo "     Source: control_plane.cpp convert_fabric_routing_table_to_chip_routing_table(). Excluded chips in get_chip_ids()"
    echo "     loop must be skipped with try_get_asic_id_from_fabric_node_id() guard."
fi
if [ "${FIX_TJ_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TJ] topology_mapper: WH_B0-invalid mesh shape skipped (${FIX_TJ_FIRES} occurrence(s))."
    echo "     Shape like 3x1 (both dims odd, not 1x1) rejected by MeshGraph for WH_B0."
    echo "     Degraded cluster reduced chip count below 4; topology mapper fell through to invalid shape."
    echo "     FIX TJ prefilter skips it; mapper continues to 2x1 or 1x1."
fi
if [ "${FIX_TK_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TK] DoSetUpTestSuite: ${FIX_TK_FIRES} chip(s) excluded — not in fabric cluster after topology discovery."
    echo "     Topology mapper degraded to 1x1 (all ETH links dead on T3K after progressive SIGKILL teardowns)."
    echo "     cluster_degraded_skip_ set to true — all per-test SetUp() calls will GTEST_SKIP."
    echo "     Before FIX TK: create_unit_meshes called with all chip IDs → TT_FATAL for chips not in 1x1 cluster."
elif grep -qE 'Physical chip id [0-9]+ not found in control plane chip mapping' "$CLEAN" 2>/dev/null; then
    echo "  => [FIX TK MISSING?] 'Physical chip id not found in control plane chip mapping' found — FIX TK may be absent."
    echo "     Source: create_unit_meshes → get_fabric_node_id_from_physical_chip_id() TT_FATAL."
    echo "     Occurs when DoSetUpTestSuite passes ALL chip IDs to create_unit_meshes on a 1x1 degraded cluster."
    echo "     Fix: filter chip IDs via is_physical_chip_in_fabric_cluster() after SetFabricConfig() in DoSetUpTestSuite()."
fi
if [ "${FIX_TL_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TL] DoSetUpTestSuite: bailed before create_unit_meshes (${FIX_TL_FIRES} occurrence(s)) — cluster degraded."
    echo "     FIX TK excluded chips → cluster_degraded_skip_=true → FIX TL returns early to avoid UDM builder crash."
    echo "     Without FIX TL: create_unit_meshes with 1-chip set in UDM mode → TT_FATAL at fabric_tensix_builder.cpp:874."
elif grep -qE 'not found in worker tensix info map|device_it != worker_to_tensix_info_map_' "$CLEAN" 2>/dev/null; then
    echo "  => [FIX TL MISSING?] 'not found in worker tensix info map' found — FIX TL may be absent or reverted."
    echo "     Source: fabric_tensix_builder.cpp:874 TT_FATAL when create_unit_meshes called with partial chip set."
    echo "     FIX TL prevents this by returning early from DoSetUpTestSuite when cluster_degraded_skip_=true."
    echo "     Check: does DoSetUpTestSuite return before create_unit_meshes when any chip is excluded by FIX TK?"
fi
if [ "${FIX_TM_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TM] expand_one_or_all_to_all_unicast: skipped — all_pairs empty (${FIX_TM_FIRES} occurrence(s))."
    echo "     get_all_to_all_unicast_pairs() returned empty vector (0 routing planes on degraded cluster)."
    echo "     Before FIX TM: code accessed all_pairs[0].first without empty check → UB/crash."
fi
if [ "${FIX_TN_WILDCARD_CRASH:-0}" -gt 0 ]; then
    echo "  => [FIX TN MISSING?] T3kCustomMeshGraph fixture 'Fabric node id not found in mapping' crash seen."
    echo "     Possible regression: leading '*' in gtest_filter '*Fabric2DFixture.TestUnicast*' matches"
    echo "     T3kCustomMeshGraphFabric2DFixture (class ends with Fabric2DFixture) on degraded cluster."
    echo "     Fix (FIX TN): remove leading '*' → 'Fabric2DFixture.TestUnicast*' in run_t3000_unit_tests.sh."
fi
if [ "${FIX_TL_BASH_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TL-bash] T3K topology damaged after warm-up (${FIX_TL_BASH_FIRES} occurrence(s))."
    echo "     Warm-up subprocess atexit left non-MMIO chips unreachable — triggered recovery tt-smi -r."
    if [ "${FIX_TL_BASH_RECOVERED:-0}" -gt 0 ]; then
        echo "     Recovery succeeded — topology restored after reset+warm-up."
    fi
    if [ "${FIX_TL_BASH_STILL_DEGRADED:-0}" -gt 0 ]; then
        echo "     [FAIL] Recovery FAILED — topology still degraded after tt-smi -r + warm-up."
        echo "     Hardware needs host reboot or engineer attention."
    fi
fi
if [ "${FIX_TM_BASH_FAILED:-0}" -gt 0 ]; then
    echo "  => [FIX TM-bash] post-TL warm-up FAILED (${FIX_TM_BASH_FAILED} occurrence(s))."
    echo "     open_mesh_device() threw after tt-smi -r — relay not re-established."
    echo "     Topology check will likely still see degraded state → exit 1."
elif [ "${FIX_TM_BASH_COMPLETE:-0}" -gt 0 ]; then
    echo "  => [FIX TM-bash] post-TL warm-up completed (${FIX_TM_BASH_COMPLETE} occurrence(s))."
    echo "     Relay re-established after recovery reset; topology check should see 8 chips."
fi
if [ "${FIX_TN_BASH_MISSING:-0}" -gt 0 ]; then
    echo "  => [FIX TN-bash?] Topology check Python query failed (${FIX_TN_BASH_MISSING} occurrence(s))."
    echo "     GetNumAvailableDevices() threw on dead relay. With || true the script handles it;"
    echo "     without || true, set -eo pipefail aborts the shell silently before recovery."
fi
if [ "${FIX_DT1_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX DT-1] Dispatch ERISC teardown timeout (${FIX_DT1_FIRES} device(s)) in warm-up."
    echo "     wait_for_dispatch_cores() hit 1000ms limit → rescue_stuck_dispatch_cores fired."
    echo "     ERISCs left with stale go_msg=0x02. Warm-up grep detects this and triggers FIX UP → tt-smi -r."
    echo "     If FIX_UP_FIRES=0 but FIX_DT1_FIRES>0: grep pattern mismatch — FIX DT-1 may not have triggered reset."
fi
if [ "${FIX_TO_BASH_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TO-bash] Remedial tt-smi -r after warm-up (${FIX_TO_BASH_FIRES} occurrence(s))."
    echo "     Triggered when warm-up >= 120s (FIX TH3 threshold) or ring-sync timeout markers detected."
    echo "     dispatch-ERISC hard resets leave ETH cores in corrupted go_msg=0x02 state."
    echo "     Remedial reset clears this before the topology check / GTest execution."
fi
if [ "${FIX_UP_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX UP] Ring-sync timeout detected in warm-up output (${FIX_UP_FIRES} occurrence(s))."
    echo "     Python exits 0 but Metal logs show ring never converged (FIX TK / ring_sync_already_timed_out)."
    echo "     Hardware is NOT ready for traffic despite successful open/close cycle."
fi
if [ "${FIX_UP_INFRA_ERROR:-0}" -gt 0 ]; then
    echo "  => [FIX UP INFRA_ERROR] Ring-sync timeout on 3+ consecutive warm-ups — test run aborted."
    echo "     Hardware requires reboot. This prevents infinite SKIP→reset→warm-up loops."
fi
if [ "${FIX_TM2_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TM2] Ring-sync timeout detected in post-TL warm-up (${FIX_TM2_FIRES} occurrence(s))."
    echo "     FIX TM recovery warm-up completed but ring-sync never converged."
fi
if [ "${FIX_UP2_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX UP2] Pre-test-loop ring-sync health gate fired (${FIX_UP2_FIRES} occurrence(s))."
    echo "     Ran extra reset+warm-up cycle because initial/TM warm-up had ring-sync timeout."
    if [ "${FIX_UP2_INFRA_ERROR:-0}" -gt 0 ]; then
        echo "     => INFRA_ERROR: ring-sync timeout persisted after 2 reset+warm-up cycles — run aborted."
    fi
fi
if [ "${FIX_UP3_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX UP3] Dispatch-ERISC timeout loop detected in FIX UP2 warm-up (${FIX_UP3_FIRES} occurrence(s))."
    echo "     base-UMD ERISCs interfere with dispatch FW during warm-up open/close:"
    echo "     rescue_stuck_dispatch_cores fires → go_msg=0x02 stale → next warm-up triggers same loop."
    echo "     FIX UP3: final tt-smi -r + skip further warm-ups; tests use FIX SC + FIX M/RZ2 for state."
    echo "     This means first test session starts WITHOUT a fabric warm-up — AllGather relies on"
    echo "     FIX SC (stale go_msg clear) and FIX M base-UMD channel transition to complete cleanly."
fi
# ─── FABRIC TELEMETRY DUMP (from dump_fabric_telemetry_on_failure in multi_device_fixture.hpp) ───
echo ""
echo "=== FABRIC TELEMETRY AT FAILURE ==="

TELEMETRY_DUMP_BEGIN=$(grep -c '\[fabric_telemetry_dump\] BEGIN' "$CLEAN" 2>/dev/null; :)
TELEMETRY_DUMP_END=$(grep -c '\[fabric_telemetry_dump\] END' "$CLEAN" 2>/dev/null; :)
TELEMETRY_UNHEALTHY=$(grep -c '\[fabric_telemetry_dump\].*\[UNHEALTHY\]' "$CLEAN" 2>/dev/null; :)
TELEMETRY_THREW=$(grep -c '\[fabric_telemetry_dump\].*threw' "$CLEAN" 2>/dev/null; :)
BASELINE_DEGRADED=$(grep -c '\[fabric_baseline_compare\].*DEGRADED' "$CLEAN" 2>/dev/null; :)
BASELINE_HB_STALLED=$(grep -c '\[fabric_baseline_compare\].*HEARTBEAT_STALLED' "$CLEAN" 2>/dev/null; :)

if [ "${TELEMETRY_DUMP_BEGIN:-0}" -gt 0 ]; then
    echo "  Telemetry dumps found: ${TELEMETRY_DUMP_BEGIN}"
    echo ""

    # Extract per-channel telemetry entries
    echo "  --- Per-channel telemetry ---"
    grep '\[fabric_telemetry_dump\]' "$CLEAN" | \
        grep -v 'BEGIN\|END\|threw\|outer exception\|no dynamic info' | \
        sed 's/^.*\[fabric_telemetry_dump\] /  /' | \
        sort -t= -k2,2n -k4,4n | \
        head -100
    echo ""

    if [ "${TELEMETRY_UNHEALTHY:-0}" -gt 0 ]; then
        echo "  => ${TELEMETRY_UNHEALTHY} channel(s) in UNHEALTHY state (router_state != Active)"
    fi
    if [ "${TELEMETRY_THREW:-0}" -gt 0 ]; then
        echo "  => ${TELEMETRY_THREW} telemetry read(s) threw exceptions (relay path broken?)"
    fi

    # Show channels with no dynamic info
    NO_DYNAMIC=$(grep -c '\[fabric_telemetry_dump\].*no dynamic info' "$CLEAN" 2>/dev/null; :)
    if [ "${NO_DYNAMIC:-0}" -gt 0 ]; then
        echo "  => ${NO_DYNAMIC} channel(s) had no dynamic telemetry info"
    fi
else
    echo "  (no fabric telemetry dumps found in log)"
fi

echo ""
echo "  --- Baseline comparison (SetUp vs TearDown) ---"
if [ "${BASELINE_DEGRADED:-0}" -gt 0 ]; then
    grep '\[fabric_baseline_compare\]' "$CLEAN" | \
        grep 'DEGRADED' | \
        sed 's/^.*\[fabric_baseline_compare\] /  /' | \
        head -50
    echo ""
    echo "  => ${BASELINE_DEGRADED} channel(s) degraded between SetUp and TearDown"
    if [ "${BASELINE_HB_STALLED:-0}" -gt 0 ]; then
        echo "  => ${BASELINE_HB_STALLED} of those had stalled heartbeats (tx+rx unchanged)"
    fi
else
    echo "  (no channel degradation detected between SetUp and TearDown)"
fi

echo ""
echo "=== AGGREGATE COUNTERS ==="
echo "  RING_SYNC_TIMEOUT_COUNT:   ${RING_SYNC_TIMEOUT_COUNT:-0}"
echo "  BASE_UMD_CHAN_COUNT:        ${BASE_UMD_CHAN_COUNT:-0}  (0x49706550 occurrences)"
echo "  CHANNELS_NOT_READY_COUNT:   ${CHANNELS_NOT_READY_COUNT:-0}"
echo "  FIX_DT1_FIRES:             ${FIX_DT1_FIRES:-0}  (dispatch ERISC teardown timeout → rescue_stuck_dispatch in warm-up)"
echo "  FIX_UP_FIRES:              ${FIX_UP_FIRES:-0}  (ring-sync/dispatch timeout in warm-up → remedial tt-smi -r)"
echo "  FIX_UP_INFRA_ERROR:        ${FIX_UP_INFRA_ERROR:-0}  (3+ consecutive ring-sync timeouts in warm-up → INFRA_ERROR abort)"
echo "  FIX_UP2_FIRES:             ${FIX_UP2_FIRES:-0}  (pre-test-loop ring-sync health gate)"
echo "  FIX_UP2_INFRA_ERROR:       ${FIX_UP2_INFRA_ERROR:-0}  (ring-sync persisted after 2 reset+warm-up cycles → run aborted)"
echo "  FIX_UP3_FIRES:             ${FIX_UP3_FIRES:-0}  (dispatch-ERISC loop in UP2 warm-up → final tt-smi -r + skip warm-up)"
echo "  FIX_TM2_FIRES:             ${FIX_TM2_FIRES:-0}  (ring-sync timeout in post-TL warm-up)"
echo "  FIX_TH3_FIRES:             ${FIX_TH3_FIRES:-0}  (120s/12x ring-sync timeout extension)"
echo "  FIX_TG2_SYNC_CLEARS:       ${FIX_TG2_SYNC_CLEARS:-0}  (stale sync-address clears for base-UMD channels)"
echo "  FIX_EXT_FIRES:             ${FIX_EXT_FIRES:-0}  (external ETH channel skipped in ring-sync/health-check)"
echo "  FIX_AQ_SUCCESS:            ${FIX_AQ_SUCCESS:-0}  (edm_status sentinel reached 0x49706550 in teardown)"
echo "  FIX_AQ_TIMEOUT:            ${FIX_AQ_TIMEOUT:-0}  (edm_status sentinel poll timed out at 10s — ROM postcode persisted)"
echo "  FIX_RP_SUCCESS:            ${FIX_RP_SUCCESS:-0}  (non-MMIO ROM postcode transitioned to base-UMD)"
echo "  FIX_RP_TIMEOUT:            ${FIX_RP_TIMEOUT:-0}  (non-MMIO ROM postcode did NOT transition within 5s)"
echo "  FIX_RP_UNEXPECTED:         ${FIX_RP_UNEXPECTED:-0}  (non-MMIO ROM postcode transitioned to unexpected value)"
echo "  FIX_RR_SUCCESS:            ${FIX_RR_SUCCESS:-0}  (MMIO ROM-postcode chan recovered by FIX RR PCIe-direct soft reset)"
echo "  FIX_RR_FAIL:               ${FIX_RR_FAIL:-0}  (MMIO ROM-postcode chan FIX RR reset FAILED → newly_dead → TT_THROW)"
echo "  FIX_RS_RECOVERED:          ${FIX_RS_RECOVERED:-0}  (total channels propagated back to configure_fabric as recovered by FIX RS)"
echo "  FIX_M_TRANSITION_COUNT:    ${FIX_M_TRANSITION_COUNT:-0}  (base-UMD channels transitioned via launch_msg — stale_base_umd set)"
echo "  FIX_RZ2_CLEAR_COUNT:      ${FIX_RZ2_CLEAR_COUNT:-0}  (stale_base_umd cleared after ring-sync + health check)"
echo "  FIX_QW_SKIP_COUNT:        ${FIX_QW_SKIP_COUNT:-0}  (test skip guards fired on stale_base_umd or channels_not_ready)"
echo "  FIX_AR2_FIRES:            ${FIX_AR2_FIRES:-0}  (post-deassert delay before heartbeat poll — prevents stale 0xABCD)"
echo "  FIX_AU_AX_FIRES:          ${FIX_AU_FIRES:-0}/${FIX_AX_FIRES:-0}  (FIX AU-2 TERMINATE attempts / FIX AX-2 assert_risc_reset attempts on relay-dead)"
echo "  FIX_XY2_RELAY_RESTORED:   ${FIX_XY2_RELAY_RESTORED:-0}  (relay_broken cleared after successful ERISC force-reset — relay path restored)"
echo "  FIX_SB2R_CLEAN_BOOT:      ${FIX_SB2R_CLEAN_BOOT:-0}  (relay_broken NOT set: probe was clean, FIX M was mid-init sentinel not a real failure)"
echo "  FIX_SC_PRESCAN:           ${FIX_SC_PRESCAN:-0}  (FIX SC stale go_msg pre-scan fires; TENSIX=${FIX_SC_TENSIX_FIRE:-0} ETH=${FIX_SC_ETH_FIRE:-0})"
echo "  FIX_SC_ETH_FIRE:          ${FIX_SC_ETH_FIRE:-0}  ([REGRESSION if > 0] FIX SC fired on ETH core — FIX SC-ADDR regression?)"
echo "  FIX_SC_ADDR_ETH_VALID:    ${FIX_SC_ADDR_ETH_VALID:-0}  (ETH cores in not_done_cores scanned at correct address; valid signal (debug log))"
echo "  FIX_AO_FIRES:              ${FIX_AO_FIRES:-0}  (ring-sync STARTED early-exit at 1s — out-of-mesh/non-MMIO peer not responding)"
echo "  FIX_BG_FIRES:              ${FIX_BG_FIRES:-0}  (host-pre-launch 0xdeadb07e — ERISC not executing after FIX RR soft-reset)"
echo "  FIX_BH_SUCCESS:            ${FIX_BH_SUCCESS:-0}  (ERISC booted from ROM after FIX RR deassert)"
echo "  FIX_BH_TIMEOUT:            ${FIX_BH_TIMEOUT:-0}  (ERISC stuck in ROM phase after FIX RR deassert — channel dead)"
echo "  FIX_BO_FIRES:              ${FIX_BO_FIRES:-0}  (Phase 5 kSyncTimeoutMs extended 10s→120s for stale base-UMD channels)"
echo "  FIX_BP_FIRES:              ${FIX_BP_FIRES:-0}  (fabric_context null guard — teardown ordering race recovered)"
echo "  FIX_BF_MMIO_MASTER:        ${FIX_BF_MMIO_MASTER:-0}  (master chans selected with MMIO peer — FIX BD/BF)"
echo "  FIX_BF_NONMMIO_MASTER:     ${FIX_BF_NONMMIO_MASTER:-0}  (master chans selected with non-MMIO peer — FIX BD/BF fallback)"
echo "  FIX_BE_STRUCTURAL:         active (per-cycle state cleared at compile_and_configure_fabric entry)"
echo "  FIX_ST_DEAD:               ${FIX_ST_DEAD:-0}  (MMIO master chan dead, ring sync skipped — FIX ST)"
echo "  FIX_ST_RECOVERED:          ${FIX_ST_RECOVERED:-0}  (MMIO master chan recovered by FIX RR, ring sync proceeding — FIX ST)"
echo "  FIX_TV_SUCCESS:            ${FIX_TV_SUCCESS:-0}  (MMIO ETH heartbeat confirmed after reset_cores)"
echo "  FIX_TV_TIMEOUT:            ${FIX_TV_TIMEOUT:-0}  (MMIO ETH heartbeat poll timed out — probe_dead likely)"
echo "  TELEMETRY_DUMPS:           ${TELEMETRY_DUMP_BEGIN:-0}  (fabric telemetry failure dumps)"
echo "  TELEMETRY_UNHEALTHY_CHANS: ${TELEMETRY_UNHEALTHY:-0}  (channels with router_state != Active at failure)"
echo "  TELEMETRY_READ_ERRORS:     ${TELEMETRY_THREW:-0}  (telemetry reads that threw exceptions)"
echo "  BASELINE_DEGRADED_CHANS:   ${BASELINE_DEGRADED:-0}  (channels degraded between SetUp and TearDown)"
echo "  BASELINE_HB_STALLED:       ${BASELINE_HB_STALLED:-0}  (degraded channels with stalled heartbeats)"
echo ""
if [ "${FIX_TH3_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TH3] fabric_router_sync_timeout extended from 10s to 120s (12x) (${FIX_TH3_FIRES} occurrence(s))."
    echo "     Base-UMD channels need much longer: T3K worst-case 16 channels × 7.5s each."
    echo "     FIX TH2 (30s/3x) was insufficient — channels stuck at REMOTE_HANDSHAKE_COMPLETE throughout."
elif [ "${FIX_TH2_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX TH2] fabric_router_sync_timeout extended from 10s to 30s (${FIX_TH2_FIRES} occurrence(s))."
    echo "     Superseded by FIX TH3 (120s/12x). If TH2 fires but not TH3, FIX TH3 code may be missing."
elif [ "${HEALTH_CHECK_FAILED:-0}" -gt 0 ] && [ "${STILL_INITIALIZING_COUNT:-0}" -gt 0 ]; then
    echo "  => [FIX TH2 MISSING?] Fabric health check failed with ${STILL_INITIALIZING_COUNT} still-initializing."
    echo "     Channels stuck at REMOTE_HANDSHAKE_COMPLETE (ring barrier incomplete within timeout)."
    echo "     FIX TH2 extends timeout 3x when has_base_umd_channels_ is true. Check commit 9798a110021."
fi
if [ "${FIX_TG_L1_FIRES:-0}" -gt 0 ] || [ "${FIX_TG2_SYNC_CLEARS:-0}" -gt 0 ]; then
    echo "  => [FIX TG/TG2 L1] configure_fabric_cores: ${FIX_TG_L1_FIRES} edm_status preserves, ${FIX_TG2_SYNC_CLEARS} sync-address clears."
    echo "     FIX TG2: partial L1 clear — zeros stale sync addrs (edm_local_sync, edm_local_tensix_sync,"
    echo "     termination_signal) while preserving edm_status_address (0x49706550 sentinel)."
    echo "     Before FIX TG2: stale REMOTE_HANDSHAKE_COMPLETE (0xa1b1c1d1) survived smi resets → hang."
fi
if [ "${FIX_EXT_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX EXT] External ETH channel(s) detected and skipped cleanly (${FIX_EXT_FIRES} occurrence(s))."
    echo "     Channel holds base-UMD sentinel (0x49706550) but has no in-cluster peer (e.g. wired to out-of-mesh host)."
    echo "     Excluded from soft-reset, firmware launch, ring-sync, AND health-check by FIX EXT."
    echo "     Without FIX EXT: these channels block ring-sync barrier or trip health-check as dead → false degraded state."
fi
if [ "${FIX_TI_RING_FIRES:-0}" -gt 0 ] || [ "${FIX_TJ_RING_FIRES:-0}" -gt 0 ] || [ "${FIX_TI_FIRST_TIMEOUT:-0}" -gt 0 ]; then
    echo "  => [FIX TI/TJ/TK ring-sync] base-UMD ring barrier timeout cascade detected:"
    if [ "${FIX_TI_FIRST_TIMEOUT:-0}" -gt 0 ]; then
        echo "     - First timeout: ${FIX_TI_FIRST_TIMEOUT} device(s) waited full 30s (ring barrier failed)"
    fi
    if [ "${FIX_TJ_RING_FIRES:-0}" -gt 0 ]; then
        echo "     - Fast-skipped: ${FIX_TJ_RING_FIRES} device(s) skipped instantly (FIX TJ — ring already failed)"
    fi
    if [ "${FIX_TI_RING_FIRES:-0}" -gt 0 ]; then
        echo "     - Health-check skipped: ${FIX_TI_RING_FIRES} device(s) (FIX TI — channels stuck at REMOTE_HANDSHAKE_COMPLETE)"
        echo "     - fabric_ring_sync_timed_out_ set on skipped devices (FIX TK — prevents FIX BA relay_broken_non_mmio)"
    fi
    echo "     Without FIX TI: verify_all_fabric_channels_healthy() fails in 150ms → false health-check failure."
    echo "     Without FIX TJ: N×30s sequential timeouts (e.g. 8 devices × 30s = 4 min wasted)."
    echo "     Without FIX TK: FIX BA adds devices to relay_broken_non_mmio → FIX AC PCIe reset → all MMIO ETH dead."
    if [ "${FIX_TK_BA_GUARD:-0}" -gt 0 ]; then
        echo "     - FIX TK guard active: ${FIX_TK_BA_GUARD} device(s) had FIX BA skipped in teardown"
    fi
fi
if [ "${FIX_PY_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX PY] Phase 2.5 fast-skip: ${FIX_PY_FIRES} channel(s) skipped because relay already marked broken by prior channel."
    echo "     Without FIX PY: each remaining channel on a dead non-MMIO device burns 3×retry×5s≈21s before Phase 2.5 exits."
fi
if [ "${FIX_QU_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX QU] Per-device degraded flags re-asserted after configure_fabric() reset: ${FIX_QU_FIRES} device(s)."
    echo "     FIX QU restores fabric_relay_path_broken_ / fabric_channels_not_ready_for_traffic_ for dead-relay/MMIO-dead-master-chan"
    echo "     devices so test guards (FIX QS/QW/RX) detect degraded cluster and SKIP instead of hanging in AllGather."
    echo "     Without FIX QU: guards see false-healthy cluster → AllGather hangs for OPERATION_TIMEOUT."
fi
if [ "${FIX_QV_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX QV] Phase 4 MUX poll skipped for ${FIX_QV_FIRES} MMIO dead-master-chan device(s) (channels_not_ready=true)."
    echo "     Without FIX QV: Phase 4 times out (5000ms × N channels) then throws → UDM tests FAILED instead of SKIPPED."
fi
if [ "${FIX_QE_FIRES:-0}" -gt 0 ]; then
    echo "  => [FIX QE] multi-tensor CCL tests skipped due to stale ETH firmware: ${FIX_QE_FIRES} occurrence(s)."
    echo "     FIX QE guard in test_multi_tensor_ccl.cpp fires when is_fabric_degraded() returns true."
    echo "     Note: FIX QE emits GTest SKIP format — not captured in TIMELINE/PHASES (Metal-log format only)."
fi
if [ "${HEALTH_CHECK_FAILED:-0}" -gt 0 ]; then
    echo "  => [HEALTH CHECK] Fabric health check failed — ${HEALTH_CHECK_FAILED} occurrence(s)."
    grep -m3 'Fabric health check failed' "$CLEAN" | sed 's/^/     /' | cut -c1-140
fi

# ─── PROCESS CRASH / SIGNAL CLASSIFICATION ───
echo ""
echo "=== PROCESS CRASH / SIGNAL CLASSIFICATION ==="
# SIGBUS (exit 135 = 128+7): typically SiliconSysmemManager destructor throw after ENODEV
SIGBUS_COUNT=$(grep -ciE 'Bus error|SIGBUS|signal 7|exit.*135' "$CLEAN" 2>/dev/null; :)
# SIGABRT (exit 134 = 128+6): TT_FATAL, std::terminate, destructor throw
SIGABRT_COUNT=$(grep -ciE 'SIGABRT|Aborted|signal 6|exit.*134' "$CLEAN" 2>/dev/null; :)
# SIGSEGV (exit 139 = 128+11)
SIGSEGV_COUNT=$(grep -ciE 'Segmentation fault|SIGSEGV|signal 11|exit.*139' "$CLEAN" 2>/dev/null; :)
# SIGKILL (exit 137 = 128+9): watchdog, OOM, CI timeout
SIGKILL_COUNT=$(grep -ciE 'SIGKILL|Killed|signal 9|exit.*137' "$CLEAN" 2>/dev/null; :)
# UMD destructor throw pattern: SiliconSysmemManager ENODEV
UMD_DTOR_THROW=$(grep -ciE 'unpin_or_unmap_sysmem|TENSTORRENT_IOCTL_UNPIN|SiliconSysmemManager|ENODEV.*unmap' "$CLEAN" 2>/dev/null; :)
# Exit code 77: skip sentinel from subprocess tests (GAP-22)
EXIT_77=$(grep -ciE 'exit.*77|sys\.exit\(77\)' "$CLEAN" 2>/dev/null; :)
# Exit code 124: timeout(1) killed subprocess
EXIT_124=$(grep -ciE 'exit.*124|Command timed out' "$CLEAN" 2>/dev/null; :)
# Core dump
CORE_DUMP=$(grep -ciE 'core dump|core\..*dumped' "$CLEAN" 2>/dev/null; :)

if [ "${SIGBUS_COUNT:-0}" -gt 0 ]; then
    echo "  => SIGBUS (exit 135) detected: ${SIGBUS_COUNT} occurrence(s)."
    echo "     Common cause: SiliconSysmemManager::~SiliconSysmemManager() throws UmdException"
    echo "     when unpin_or_unmap_sysmem() -> TENSTORRENT_IOCTL_UNPIN_PAGES returns ENODEV."
    echo "     Destructor throw -> std::terminate() -> process crash."
    echo "     Fix: FIX SA (UMD submodule) wraps destructor with try/catch."
fi
if [ "${SIGABRT_COUNT:-0}" -gt 0 ]; then
    echo "  => SIGABRT (exit 134) detected: ${SIGABRT_COUNT} occurrence(s)."
    echo "     Common causes: TT_FATAL, std::terminate from destructor throw, or assert()."
fi
if [ "${SIGSEGV_COUNT:-0}" -gt 0 ]; then
    echo "  => SIGSEGV (exit 139) detected: ${SIGSEGV_COUNT} occurrence(s)."
    echo "     Common cause: null dereference on closed device, stale pointer after partial teardown."
fi
if [ "${SIGKILL_COUNT:-0}" -gt 0 ]; then
    echo "  => SIGKILL (exit 137) detected: ${SIGKILL_COUNT} occurrence(s)."
    echo "     Common causes: watchdog timeout, OOM, CI timeout kill, or test_budget_ms exceeded."
fi
if [ "${UMD_DTOR_THROW:-0}" -gt 0 ]; then
    echo "  => UMD destructor throw pattern: ${UMD_DTOR_THROW} occurrence(s)."
    echo "     SiliconSysmemManager::~SiliconSysmemManager() threw exception."
    echo "     Fix: FIX SA (UMD submodule at 5a2e723c) wraps unpin_or_unmap_sysmem() in try/catch."
    echo "     If SIGBUS/SIGABRT also present: FIX SA may be missing or reverted."
fi
if [ "${EXIT_77:-0}" -gt 0 ]; then
    echo "  => Exit code 77 (skip sentinel): ${EXIT_77} occurrence(s)."
    echo "     Subprocess tests (GAP-22) use exit(77) to signal hardware degradation skip."
fi
if [ "${EXIT_124:-0}" -gt 0 ]; then
    echo "  => Exit code 124 (timeout): ${EXIT_124} occurrence(s)."
    echo "     bash timeout(1) killed the subprocess — likely open_mesh_device() or AllGather hang."
fi
if [ "${CORE_DUMP:-0}" -gt 0 ]; then
    echo "  => Core dump detected: ${CORE_DUMP} occurrence(s)."
fi
if [ "${SIGBUS_COUNT:-0}" -eq 0 ] && [ "${SIGABRT_COUNT:-0}" -eq 0 ] && [ "${SIGSEGV_COUNT:-0}" -eq 0 ] && [ "${SIGKILL_COUNT:-0}" -eq 0 ] && [ "${UMD_DTOR_THROW:-0}" -eq 0 ]; then
    echo "  (no crash signals detected)"
fi

# ─── FIX TF (test_tt_fabric degraded skip) ───
FIX_TF_SKIP=$(grep -cE 'Skipping Test Group.*degraded fabric detected' "$CLEAN" 2>/dev/null; :)
if [ "${FIX_TF_SKIP:-0}" -gt 0 ]; then
    echo ""
    echo "  => [FIX TF SKIP] test_tt_fabric skipped ${FIX_TF_SKIP} test group(s) due to degraded fabric."
    echo "     has_degraded_fabric() returned true (relay_broken || channels_not_ready || stale_base_umd)."
    echo "     Without FIX TF: compile_programs() would dispatch to non-MMIO device with broken relay"
    echo "     -> FIX Z throw in enqueue_write_shards_nolock -> std::terminate -> SIGABRT."
fi

echo ""
echo "========================================================================"
