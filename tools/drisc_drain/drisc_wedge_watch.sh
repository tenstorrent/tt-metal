#!/bin/bash
# drisc_wedge_watch.sh -- run the DRISC workload in a loop ON THE BOX and monitor for the PCIe wedge.
#
# Self-contained replacement for the Mac-driven pair (drisc_hang_harness.sh + drisc_hang_compare.sh):
# no ssh, no second machine, nothing to get out of step. Run it on the IRD box.
#
# Everything it needs lives on the HOST side of an IRD box:
#   /dev/tenstorrent/0            crw-rw-rw- -> the workload can open the device from the host
#   /sys/bus/pci/devices/...      link state for both endpoint and root port
#   lspci / setpci / dmesg / last diagnostics
#   ~/.local/bin/tt-smi           recovery (NOT on PATH -- resolved by full path below)
#
# WHAT IT DOES PER RUN
#   1. clears the endpoint's DevSta sticky bits, so they become a per-run PCIe error probe
#   2. runs test_perf_debug_zones, timed, with the log kept
#   3. reads CARD STATE, then classifies:
#        WEDGE     endpoint link state reads Unknown (all-ones config space) -> genuine PCIe wedge
#        TEARDOWN  run failed / core-wait signature, card HEALTHY -> wait_until_cores_done, not a wedge
#        MASKED    rc=0 but the log shows a CAUGHT teardown timeout -> would have hung unarmed
#        OTHER     failed, card healthy, no teardown signature
#        CLEAN     rc=0, no masked signature
#      CARD STATE IS AUTHORITATIVE, NEVER EXIT CODE. Pooling wedge with teardown produced and then
#      destroyed four findings in this investigation; a wedged run can still exit 0.
#   4. on WEDGE: dumps a full diagnostic bundle BEFORE touching the card, then stops by default so
#      the wedged card is preserved for inspection. That is the "monitor" half of this script.
#
# TWO DELIBERATE DIFFERENCES FROM THE OLD MAC HARNESS
#   * It does NOT reboot the box on its own (ALLOW_REBOOT=0). An unclean/watchdog reboot is exactly
#     what leaves the card in the DEGRADED state that only a cold power cycle clears -- so automatic
#     rebooting can manufacture a worse failure than the one being studied. Opt in explicitly.
#   * Output goes to /localdev, which survives a VM freeze + watchdog reboot. /tmp and /home do not.
#     Rows are appended as they happen, so if the box freezes mid-sweep the trail shows where.
#
# USAGE
#   ./drisc_wedge_watch.sh                              # 60 runs, delay 125, stop on first wedge
#   N=200 DELAY=150 ./drisc_wedge_watch.sh              # longer sweep
#   STOP_ON_WEDGE=0 N=400 ./drisc_wedge_watch.sh        # measure a RATE (expect ~2-3% per run)
#   ARMED=1 ./drisc_wedge_watch.sh                      # with TT_METAL_OPERATION_TIMEOUT_SECONDS=45
#   DISPATCH=slow DRAINER=tensix ./drisc_wedge_watch.sh # another 2x2 cell
#
# The wedge rate is ~2-3% per run and does NOT depend on producer delay, so budget runs, not delays.

TAG=${TAG:-watch}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BIN=${BIN:-$REPO/build_Release/programming_examples/test_perf_debug_zones}
OUT=${OUT_DIR:-/localdev/$LOGNAME/drisc_wedge}/$TAG
N=${N:-60}
DELAY=${DELAY:-125}
ITERS=${ITERS:-500}
# Producer grid. NEVER "--gx 0" (= whatever the device offers): under slow dispatch that returns the
# full 12x10 while the drainer holds the last column back, so producers land on a column nothing polls,
# block forever in ring_ensure_room, and the run dies in wait_until_cores_done. That harness bug was
# misread as 26/26 slow-dispatch TEARDOWNs. 11x10 also matches the fast-dispatch grid, so every 2x2
# cell offers the identical 110-core / 550-lane load.
GX=${GX:-11}; GY=${GY:-10}
ARMED=${ARMED:-0}
STOP_ON_WEDGE=${STOP_ON_WEDGE:-1}
ALLOW_REBOOT=${ALLOW_REBOOT:-0}
DISPATCH=${DISPATCH:-fast}
DRAINER=${DRAINER:-drisc}
RUN_TIMEOUT=${RUN_TIMEOUT:-300}
# Stop after this many consecutive non-CLEAN runs. Once the box enters a persistent state every further
# run just pays the timeout again: a real sweep sat at 16 consecutive MASKED (45 s each) learning nothing,
# because MASKED did not trigger recovery and nothing watched for a streak. 0 disables the bail-out.
MAX_CONSEC_FAIL=${MAX_CONSEC_FAIL:-5}
EP=${EP:-0000:01:00.0}          # Blackhole endpoint
RP=${RP:-0000:00:01.1}          # its root port -- the ONLY reliable link-state witness once wedged
TT_SMI=${TT_SMI:-$(command -v tt-smi || echo "$HOME/.local/bin/tt-smi")}

mkdir -p "$OUT" || exit 1
SUM=$OUT/summary.txt
CSV=$OUT/runs.csv

log(){ echo "$*" | tee -a "$SUM"; }

[ -x "$BIN" ] || { log "FATAL: workload not found: $BIN"; log "  build it first: ./build_metal.sh --release --build-programming-examples"; exit 1; }
[ -c /dev/tenstorrent/0 ] || log "WARN: /dev/tenstorrent/0 missing -- is this the host, and is the driver loaded?"
[ -x "$TT_SMI" ] || log "WARN: tt-smi not found at '$TT_SMI' -- recovery will be manual"

# PCIe diagnostics preflight. These need the HOST: /sys is bind-mounted into the dev container so link
# state still reads there, but sudo/setpci config-space access does not -- which is how a whole sweep can
# look instrumented while its DevSta column is meaningless.
[ -f /.dockerenv ] && log "WARN: running INSIDE the container -- PCIe config-space diagnostics need the host"
DEVSTA_OK=0
if sudo -n setpci -s "${EP#0000:}" CAP_EXP+0A.w >/dev/null 2>&1; then
    DEVSTA_OK=1
else
    log "FATAL: cannot read endpoint DevSta (sudo -n setpci -s ${EP#0000:} CAP_EXP+0A.w failed)."
    log "  DevSta at PCIe cap + 0x0A is the per-run PCIe error probe -- AER reads clean while the"
    log "  endpoint carries a real UnsupReq, so without it a wedge has no error evidence at all."
    log "  Run this on the HOST (not the container), or set ALLOW_NO_DEVSTA=1 to sweep without it."
    [ "${ALLOW_NO_DEVSTA:-0}" = "1" ] || exit 1
    log "  ALLOW_NO_DEVSTA=1 -> continuing; the devsta column will read 'unavail', not a value."
fi

# Refuse to append to an existing sweep. Reusing a TAG used to silently concatenate sweeps into one
# runs.csv with restarting k values and different ARMED/DISPATCH settings -- which makes any rate
# computed off that file wrong, and is unrecoverable after the fact because the settings are not in
# the rows. Pick a new TAG, or opt in with APPEND=1 if you really are continuing the same sweep.
if [ -f "$CSV" ] && [ "${APPEND:-0}" != "1" ]; then
    echo "FATAL: $CSV already exists -- that sweep's rows would be mixed with this one's."
    echo "  use a fresh TAG=...   (or APPEND=1 to deliberately continue the same sweep)"
    exit 1
fi
[ -f "$CSV" ] || echo "k,delay,armed,rc,dur_s,ep_link,rp_link,devsta,class" > "$CSV"
# Settings live next to the rows, so a csv is still interpretable months later.
{ echo "date=$(date -Is) host=$(hostname) tag=$TAG"
  echo "N=$N DELAY=$DELAY ITERS=$ITERS GX=$GX GY=$GY ARMED=$ARMED"
  echo "DISPATCH=$DISPATCH DRAINER=$DRAINER STOP_ON_WEDGE=$STOP_ON_WEDGE ALLOW_REBOOT=$ALLOW_REBOOT"
  echo "MAX_CONSEC_FAIL=$MAX_CONSEC_FAIL RUN_TIMEOUT=$RUN_TIMEOUT BIN=$BIN"
} >> "$OUT/params.txt"


# Endpoint config space reads ALL-ONES once wedged, so its own sysfs cannot tell you anything: an
# all-ones link speed surfaces as "Unknown". That is the wedge signature, not downtraining -- always
# corroborate with the ROOT PORT, which stays linked at 32 GT/s x16 through a wedge.
link_state(){ local d=$1
  local s w
  s=$(cat "/sys/bus/pci/devices/$d/current_link_speed" 2>/dev/null | awk '{print $1}')
  w=$(cat "/sys/bus/pci/devices/$d/max_link_width" 2>/dev/null)
  echo "${s:-NA}|${w:-NA}"; }

# DevSta lives at PCIe cap + 0x0A -- a DIFFERENT register from AER, which reads clean while the
# endpoint carries a real UnsupReq. Bits are RW1C, so clearing before each run turns it into a
# per-run probe. 0x8 = UnsupReq, 0x1 = CorrErr.
# DEVSTA_OK is resolved ONCE at startup (see the preflight below). Without it these are no-ops that
# report "unavail" rather than "NA": a per-row "NA" reads like a measured value and is exactly the
# "coarse observable standing in for the real one" mistake this investigation keeps paying for. An
# entire 316-run sweep once recorded NA in every row and nobody noticed.
devsta_clear(){ [ "$DEVSTA_OK" = 1 ] || return 0
  sudo -n setpci -s "${EP#0000:}" CAP_EXP+0A.w=0x000f >/dev/null 2>&1; }
devsta_read(){ [ "$DEVSTA_OK" = 1 ] || { echo "unavail"; return 0; }
  sudo -n setpci -s "${EP#0000:}" CAP_EXP+0A.w 2>/dev/null || echo "readfail"; }

# Everything worth knowing about a wedged card, captured BEFORE recovery touches it.
wedge_dump(){ local k=$1 d=$OUT/wedge_${k}.txt
  {
    echo "=== wedge dump: run k=$k  $(date -Is) ==="
    echo "--- link state (endpoint is unreliable once wedged; root port is the witness) ---"
    echo "endpoint  $EP : $(link_state "$EP")"
    echo "root port $RP : $(link_state "$RP")"
    echo "--- DevSta (PCIe cap + 0x0A; 0x8=UnsupReq 0x1=CorrErr) ---"
    echo "endpoint DevSta: $(devsta_read)"
    echo "--- lspci endpoint ---"; sudo -n lspci -vvv -s "$EP" 2>&1 | sed -n '1,40p'
    echo "--- lspci root port (LnkSta / DevCtl2 completion timeout) ---"
    sudo -n lspci -vvv -s "$RP" 2>&1 | grep -iE "LnkCap|LnkSta|DevCtl2|CESta|UESta" | sed 's/^/  /'
    echo "--- dmesg tail (AER / IO_PAGE_FAULT / tenstorrent) ---"
    sudo -n dmesg 2>/dev/null | grep -iE "aer|io_page_fault|tenstorrent|dpc" | tail -20
    echo "--- did the BOX freeze rather than the card wedge? ---"
    echo "uptime: $(cat /proc/uptime)"
    last -x 2>/dev/null | head -5
  } > "$d" 2>&1
  log "  [dump] $d"; }

# tt-smi -r clears a WEDGE in seconds with the box intact. It does NOT clear DEGRADED (that needs a
# cold power cycle), and a reboot can convert one degraded state into another -- hence no auto-reboot.
recover(){ local st
  if [ -x "$TT_SMI" ]; then log "  [recover] tt-smi -r"; "$TT_SMI" -r >/dev/null 2>&1; sleep 8; fi
  st=$(link_state "$EP")
  log "  [recover] endpoint now: $st"
  if [ "${st%%|*}" = "Unknown" ] || [ "${st%%|*}" = "NA" ]; then
    if [ "$ALLOW_REBOOT" = "1" ]; then
      log "  [recover] still wedged -> rebooting (ALLOW_REBOOT=1)"; sudo -n reboot
    else
      log "  [recover] STILL WEDGED and ALLOW_REBOOT=0 -- stopping. Reboot by hand if you accept that"
      log "            an unclean reboot can leave the card DEGRADED (cold power cycle to clear)."
      return 1
    fi
  fi
  return 0; }

ENVX=""; [ "$ARMED" = "1" ] && ENVX="TT_METAL_OPERATION_TIMEOUT_SECONDS=45"
CELLX=""
[ "$DISPATCH" = "slow" ] && CELLX="$CELLX TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_PERF_DEBUG_RESERVE_COLUMN=1"
[ "$DRAINER" = "tensix" ] && CELLX="$CELLX TT_METAL_PERF_DEBUG_DRAIN_TENSIX=1"
# rpath points into build_Release/lib, which lacks the installed libs; the repo-root lib/ has them.
LIBS=$REPO/lib; [ -d "$LIBS" ] || LIBS=$REPO/build_Release/lib

log "=== drisc_wedge_watch: N=$N delay=$DELAY armed=$ARMED grid=${GX}x${GY} dispatch=$DISPATCH drainer=$DRAINER ==="
log "    host=$(hostname) out=$OUT stop_on_wedge=$STOP_ON_WEDGE allow_reboot=$ALLOW_REBOOT"
log "    baseline endpoint=$(link_state "$EP") root_port=$(link_state "$RP")"
log "    NOTE 2.5GT/s on either side = a DEGRADED link (cold power cycle), a different failure to WEDGE"

nclean=0; nwedge=0; nteardown=0; nmasked=0; nother=0; nconsec=0
for k in $(seq 1 "$N"); do
  RUNLOG=$OUT/${k}.log
  devsta_clear
  t0=$SECONDS
  ( cd "$REPO" && timeout -k 15 "$RUN_TIMEOUT" env \
      TT_METAL_PERF_DEBUG_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 TT_METAL_PERF_DEBUG_NO_DECODE=1 \
      LD_LIBRARY_PATH="$LIBS" $ENVX $CELLX \
      "$BIN" --gx "$GX" --gy "$GY" --iters "$ITERS" --delay "$DELAY" ) > "$RUNLOG" 2>&1
  rc=$?
  dur=$((SECONDS - t0))

  ep=$(link_state "$EP"); rp=$(link_state "$RP"); ds=$(devsta_read)

  # How far the log got is the ROBUST discriminator: the teardown-timeout message is only emitted when
  # the timeout is ARMED, and an unarmed run hangs at the same place silently. Keying off the signature
  # alone dumps every unarmed teardown into OTHER.
  teardown_sig=0
  grep -q "waiting for physical cores to finish\|Continuing with cleanup" "$RUNLOG" 2>/dev/null && teardown_sig=1
  reached_end=0
  grep -q "Cluster destructor completed" "$RUNLOG" 2>/dev/null && reached_end=1

  class=CLEAN
  if [ "${ep%%|*}" = "Unknown" ] || [ "${ep%%|*}" = "NA" ]; then
    class=WEDGE; nwedge=$((nwedge+1))
  elif [ $rc -ne 0 ]; then
    if [ $teardown_sig -eq 1 ] || [ $reached_end -eq 0 ]; then class=TEARDOWN; nteardown=$((nteardown+1))
    else class=OTHER; nother=$((nother+1)); fi
  elif [ $teardown_sig -eq 1 ]; then
    class=MASKED; nmasked=$((nmasked+1))
  else
    nclean=$((nclean+1))
  fi

  echo "$k,$DELAY,$ARMED,$rc,$dur,$ep,$rp,$ds,$class" >> "$CSV"
  [ "$class" != CLEAN ] && log "k=$k rc=$rc dur=${dur}s ep=$ep rp=$rp devsta=$ds -> $class"

  if [ "$class" = CLEAN ]; then
    nconsec=0
  else
    nconsec=$((nconsec+1))
  fi

  if [ "$class" = WEDGE ]; then
    wedge_dump "$k"
    if [ "$STOP_ON_WEDGE" = "1" ]; then
      log "=== WEDGE CAPTURED at k=$k -- stopping with the card left wedged for inspection ==="
      log "    dump: $OUT/wedge_${k}.txt     recover with: $TT_SMI -r"
      break
    fi
    recover || break
  elif [ "$class" = TEARDOWN ] || [ "$class" = OTHER ] || [ "$class" = MASKED ]; then
    # MASKED recovers too: it IS a teardown hang, just one the armed timeout caught. Leaving it
    # unrecovered is what let a sweep grind out 16 identical 46 s masked runs in a row.
    recover || break
  fi

  if [ "$MAX_CONSEC_FAIL" != "0" ] && [ "$nconsec" -ge "$MAX_CONSEC_FAIL" ]; then
    log "=== STOPPING: $nconsec consecutive non-clean runs (last=$class) ==="
    log "    The box is in a persistent state; sweeping further re-pays the timeout and learns nothing."
    log "    Next: $TT_SMI -r, then a few probe runs to see whether it clears or survives a reset."
    break
  fi

  [ $((k % 20)) -eq 0 ] && log "progress k=$k clean=$nclean wedge=$nwedge teardown=$nteardown masked=$nmasked other=$nother"
done

log "=== DONE: clean=$nclean wedge=$nwedge teardown=$nteardown masked=$nmasked other=$nother ==="
python3 - "$CSV" <<'PY' 2>/dev/null | tee -a "$SUM"
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
d = sorted(float(r['dur_s']) for r in rows if r['dur_s'])
if d:
    print(f"duration: min={d[0]:.0f}s median={d[len(d)//2]:.0f}s max={d[-1]:.0f}s n={len(d)}")
n = len(rows)
w = sum(1 for r in rows if r['class'] == 'WEDGE')
if n:
    print(f"wedge rate: {w}/{n} = {100.0*w/n:.1f}%  (expect ~2-3%; no delay dependence)")
PY
log "csv=$CSV"
