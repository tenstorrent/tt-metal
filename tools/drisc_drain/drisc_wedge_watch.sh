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
#   2. runs test_streaming_profiler_zones, timed, with the log kept
#   3. reads CARD STATE, then classifies:
#        WEDGE      endpoint link state reads Unknown (all-ones config space) -> hard PCIe wedge
#        MMIO_STALL "MMIO per-op timeout" in the log: a ~220 ms 4B load = the root-port completion
#                   timeout firing. The wedge mechanism, transient -- the card answers again by the time
#                   sysfs is sampled, so link state CANNOT see it. Not reset by design (see below).
#        TEARDOWN  run failed / core-wait signature, card HEALTHY -> wait_until_cores_done, not a wedge
#        MASKED    rc=0 but the log shows a CAUGHT teardown timeout -> would have hung unarmed
#        ABORT     failed some other way, card healthy, no core-wait evidence
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
# Stamped on every row. summary.txt once held two sweeps while params.txt held one block, so the sweep a
# row belonged to was unrecoverable -- and mixing a host sweep (devsta readable) with a container one
# (devsta unavail) makes any pooled rate meaningless. Per-row is the only version that cannot be lost.
SWEEP_ID=${SWEEP_ID:-$(date +%Y%m%dT%H%M%S)}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
BIN=${BIN:-$REPO/build_Release/programming_examples/test_streaming_profiler_zones}
# Derive the output root from the REPO path, not from $LOGNAME: the dev container runs as root with no
# passwd entry, so $LOGNAME/$USER are EMPTY there and "/localdev/$LOGNAME/..." collapsed to
# "/localdev//drisc_wedge" -- writing sweep results into the SHARED /localdev root instead of the user's
# own directory. $REPO is resolved from this script's own location, so it is right either side.
OUT=${OUT_DIR:-$(dirname "$REPO")/drisc_wedge}/$TAG
N=${N:-60}
DELAY=${DELAY:-125}
ITERS=${ITERS:-500}
# DECODE=0 (default) sets TT_METAL_STREAMING_PROFILER_NO_DECODE=1: the reader does read()+ack and NOTHING else --
# no decode, no publish. That is right for hunting the wedge (cheaper runs, and the wedge is classified
# from the process outcome and card state, neither of which needs decode) but it makes the HOST-DERIVED
# stats void: marker count and producer stall zones are produced BY the decode, so with it off they read 0
# no matter what the device did. Set DECODE=1 to measure those. This script records them as "nodecode"
# rather than 0 when decode is off, because a 0 in a data column is indistinguishable from a measurement.
DECODE=${DECODE:-0}
# Producer grid. NEVER "--gx 0" (= whatever the device offers): under slow dispatch that returns the
# full 12x10 while the drainer holds the last column back, so producers land on a column nothing polls,
# block forever in ring_ensure_room, and the run dies in wait_until_cores_done. That harness bug was
# misread as 26/26 slow-dispatch TEARDOWNs. 11x10 also matches the fast-dispatch grid, so every 2x2
# cell offers the identical 110-core / 550-lane load.
ARMED=${ARMED:-0}
STOP_ON_WEDGE=${STOP_ON_WEDGE:-1}
ALLOW_REBOOT=${ALLOW_REBOOT:-0}
DISPATCH=${DISPATCH:-fast}
DRAINER=${DRAINER:-drisc}

# Grid defaults follow what the DRAINER actually polls, which differs by dispatch mode:
#   fast dispatch  -> compute_with_storage_grid_size() is 11x10, so 110 cores / 550 lanes.
#   slow dispatch  -> the full 12x10 is available, and the DRISC drainer polls all of it, so 120 cores /
#                     600 lanes. Pinning 11x10 here (and forcing RESERVE_COLUMN) was for A/B parity with
#                     the Tensix drainer, which must hold its own column back -- it is NOT the right
#                     default for a DRISC measurement, which should use the whole grid.
# NEVER pass --gx 0: it returns whatever the device offers, which under slow dispatch put producers on a
# column no drainer polls -- they block forever in ring_ensure_room and the run dies in the core wait.
if [ "$DISPATCH" = "slow" ] && [ "$DRAINER" = "drisc" ]; then
    GX=${GX:-12}; GY=${GY:-10}
    RESERVE_COLUMN=${RESERVE_COLUMN:-0}
else
    GX=${GX:-11}; GY=${GY:-10}
    # The Tensix drainer lives in the last column, so it must always reserve it.
    RESERVE_COLUMN=${RESERVE_COLUMN:-1}
fi
RUN_TIMEOUT=${RUN_TIMEOUT:-300}
# Stop after this many consecutive non-CLEAN runs. Once the box enters a persistent state every further
# run just pays the timeout again: a real sweep sat at 16 consecutive MASKED (45 s each) learning nothing,
# because MASKED did not trigger recovery and nothing watched for a streak. 0 disables the bail-out.
MAX_CONSEC_FAIL=${MAX_CONSEC_FAIL:-5}
# Degradation thresholds for the in-band probes the profiler prints every run. Healthy on this box is
# ack ~170-190 ns and read ~780-790 ns. The two bad states separate cleanly:
#   ack normal + read ~2700 ns  -> Gen1 downtrain (posted writes barely care about link rate, reads pay
#                                  the round trip). Needs a COLD power cycle; a warm reboot will not retrain.
#   ack ~2300 ns + read ~2900   -> the 13x MMIO state (follows a box freeze + watchdog reboot).
# These are checked because current_link_speed is NOT trustworthy on its own: a hung card reads healthy
# for several runs, and a degraded card can read 32.0 GT/s while every read costs 3.5x.
ACK_MAX_NS=${ACK_MAX_NS:-600}
READ_MAX_NS=${READ_MAX_NS:-1500}
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
[ -f /.dockerenv ] && log "NOTE: running INSIDE the container -- link state works here, DevSta does not (see below)"

# Link state is the wedge DISCRIMINATOR: a wedged endpoint's config space reads all-ones, which surfaces
# as "Unknown". It reads fine even in the container, where /sys is mounted ro. Without it there is
# nothing to classify, so this one IS fatal.
for _d in "$EP" "$RP"; do
    [ -r "/sys/bus/pci/devices/$_d/current_link_speed" ] || {
        log "FATAL: cannot read link state for $_d -- the wedge classifier has nothing to key on."
        exit 1; }
done

# DevSta is CORROBORATING error evidence, not the classifier, so losing it degrades a sweep rather than
# invalidating it. An earlier version made this fatal and blocked the normal in-container workflow for no
# good reason. In the dev container it is unavailable two ways over: pciutils is not installed, and /sys
# is ro so the RW1C clear could not work even if it were. Run on the HOST when you specifically need
# per-run PCIe error evidence; set REQUIRE_DEVSTA=1 to insist on it.
DEVSTA_OK=0
devsta_why=""
if ! command -v setpci >/dev/null 2>&1; then
    devsta_why="setpci not installed (pciutils missing -- typical in the dev container)"
elif ! sudo -n setpci -s "${EP#0000:}" CAP_EXP+0A.w >/dev/null 2>&1; then
    devsta_why="sudo -n setpci failed (config-space access denied; /sys is mounted ro in the container)"
else
    DEVSTA_OK=1
fi
if [ "$DEVSTA_OK" = 1 ]; then
    log "    devsta probe: ok (per-run PCIe error evidence enabled)"
else
    log "WARN: DevSta unavailable -- $devsta_why"
    log "      The devsta column will read 'unavail'. Wedge CLASSIFICATION is unaffected: it keys on"
    log "      link state. What you lose is the per-run PCIe error probe, and that matters because AER"
    log "      reads clean while the endpoint carries a real UnsupReq -- so a wedge caught here will"
    log "      have no error evidence attached. Re-run on the HOST for that."
    if [ "${REQUIRE_DEVSTA:-0}" = "1" ]; then
        log "FATAL: REQUIRE_DEVSTA=1 and DevSta is unavailable."
        exit 1
    fi
fi

# Refuse to append to an existing sweep. Reusing a TAG silently concatenated sweeps into one runs.csv
# with restarting k values and different settings, which makes any rate computed off it wrong and
# unrecoverable after the fact. Deliberately AFTER every check that can exit, so a failed preflight (e.g.
# DevSta unreadable) leaves at most an empty directory rather than poisoning the tag for the retry.
if [ -f "$CSV" ] && [ "${APPEND:-0}" != "1" ]; then
    echo "FATAL: $CSV already exists -- that sweep's rows would be mixed with this one's."
    echo "  use a fresh TAG=...   (or APPEND=1 to deliberately continue the same sweep)"
    exit 1
fi
# The HEADER matters beyond tidiness: without it every "awk NR>1" silently drops the FIRST DATA ROW, and
# a 6-run sample quietly became 5.
[ -f "$CSV" ] || echo "k,delay,armed,rc,dur_s,ep_link,rp_link,devsta,class,sweep,ack_ns,read_ns,markers,l1_stalls,l1_cores,l1_worst,cw_us" > "$CSV"
# Settings live next to the rows, so a csv is still interpretable months later -- decode state above all,
# since with DECODE=0 the marker and stall columns are not measurements.
{ echo "date=$(date -Is) host=$(hostname) tag=$TAG sweep=$SWEEP_ID"
  echo "N=$N DELAY=$DELAY ITERS=$ITERS GX=$GX GY=$GY ARMED=$ARMED DECODE=$DECODE"
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
wedge_dump(){ local k=$1 d=$OUT/wedge_${SWEEP_ID}_${k}.txt
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
DECODEX=""; [ "$DECODE" != "1" ] && DECODEX="TT_METAL_STREAMING_PROFILER_NO_DECODE=1"
CELLX=""
[ "$DISPATCH" = "slow" ] && CELLX="$CELLX TT_METAL_SLOW_DISPATCH_MODE=1"
[ "$RESERVE_COLUMN" = "1" ] && CELLX="$CELLX TT_METAL_STREAMING_PROFILER_RESERVE_COLUMN=1"
[ "$DRAINER" = "tensix" ] && CELLX="$CELLX TT_METAL_STREAMING_PROFILER_RELAY_TENSIX=1"
# rpath points into build_Release/lib, which lacks the installed libs; the repo-root lib/ has them.
LIBS=$REPO/lib; [ -d "$LIBS" ] || LIBS=$REPO/build_Release/lib

log "=== drisc_wedge_watch: N=$N delay=$DELAY armed=$ARMED grid=${GX}x${GY} ($((GX*GY)) cores) dispatch=$DISPATCH drainer=$DRAINER reserve_col=$RESERVE_COLUMN ==="
log "    host=$(hostname) out=$OUT stop_on_wedge=$STOP_ON_WEDGE allow_reboot=$ALLOW_REBOOT"
log "    baseline endpoint=$(link_state "$EP") root_port=$(link_state "$RP")"
log "    NOTE 2.5GT/s on either side = a DEGRADED link (cold power cycle), a different failure to WEDGE"
log "    health thresholds: ack >= ${ACK_MAX_NS} ns => 13x state | read >= ${READ_MAX_NS} ns => Gen1 downtrain"
if [ "$DECODE" = "1" ]; then
  log "    DECODE=1: marker counts measured too (producer stalls always come from the L1 counters)"
else
  log "    DECODE=0 (NO_DECODE): read+ack only -- markers/stalls recorded as 'nodecode', NOT as 0"
  log "    producer stalls still MEASURED via the L1 control-buffer counters; cw_us is the drainer's own wait"
fi

nclean=0; nwedge=0; nteardown=0; nmasked=0; nmmio=0; nabort=0; nconsec=0
for k in $(seq 1 "$N"); do
  # Namespaced by sweep: two sweeps in one dir both numbered k from 1, so bare ${k}.log meant the second
  # silently overwrote the first's evidence and reclassification could no longer attribute a log to a row.
  RUNLOG=$OUT/${SWEEP_ID}_${k}.log
  devsta_clear
  t0=$SECONDS
  ( cd "$REPO" && timeout -k 15 "$RUN_TIMEOUT" env \
      TT_METAL_STREAMING_PROFILER=1 TT_METAL_DEVICE_PROFILER=1 $DECODEX \
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

  # A 4-byte MMIO load that took ~220 ms is the ROOT-PORT COMPLETION TIMEOUT firing (DevCtl2 on the root
  # port is set to 65-210 ms): the endpoint never completed the read and the root port abandoned it, so
  # the UMD's 2 ms per-op budget throws and the process aborts. That is the WEDGE MECHANISM in transient
  # form. It is invisible to the link-state test below, because by the time we sample sysfs the endpoint
  # is answering again -- the evidence exists only in the log.
  mmio_sig=0
  grep -q "MMIO per-op timeout" "$RUNLOG" 2>/dev/null && mmio_sig=1

  class=CLEAN
  if [ "${ep%%|*}" = "Unknown" ] || [ "${ep%%|*}" = "NA" ]; then
    class=WEDGE; nwedge=$((nwedge+1))          # hard wedge: card stuck all-ones. Most severe, wins.
  elif [ $mmio_sig -eq 1 ]; then
    class=MMIO_STALL; nmmio=$((nmmio+1))       # transient wedge: completion timeout, card recovered
  elif [ $rc -ne 0 ]; then
    # TEARDOWN means the core-wait never returned, so require EVIDENCE of that: its log signature, or a
    # run that actually sat there (>= 40 s; armed the wait is 45 s, unarmed it runs to RUN_TIMEOUT).
    # Never infer it from "did not reach the end marker" alone -- an early abort also fails to reach it,
    # and filing a 3 s crash as a hang pools two distinct failures, which is the one thing this
    # investigation cannot afford. Everything else that failed is ABORT.
    if [ $teardown_sig -eq 1 ] || { [ $reached_end -eq 0 ] && [ "$dur" -ge 40 ]; }; then
      class=TEARDOWN; nteardown=$((nteardown+1))
    else
      class=ABORT; nabort=$((nabort+1))
    fi
  elif [ $teardown_sig -eq 1 ]; then
    class=MASKED; nmasked=$((nmasked+1))
  else
    nclean=$((nclean+1))
  fi

  # In-band latency probes. The profiler prints these once per socket on every run that gets far enough
  # to create one, so they cost nothing and they are the only cheap way to see the degraded states.
  ack_ns=$(grep -ahoE "ACK-WRITE probe: [0-9]+ ns" "$RUNLOG" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  read_ns=$(grep -ahoE "DEVICE-READ probe: [0-9]+ ns" "$RUNLOG" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  ack_ns=${ack_ns:-NA}; read_ns=${read_ns:-NA}

  # PRODUCER STALLS come from the CONTROL BUFFER, not from decode. Each RISC bumps
  # profiler_control_buffer[SPSC_STALL_COUNT_0 + risc] in ring_ensure_room, and the host sums them per
  # core with read_core() at teardown -- so this is valid with DECODE=0 too, and profiler_common.h calls
  # it "the knee metric ... rather than being counted from decoded PROFILER_STALL_ZONE". The decoded
  # "producer stall zones" line is the decode-dependent one; do NOT use it as the knee signal.
  l1=$(grep -aoE "L1 STALL COUNTERS -- [0-9]+ producer stalls across [0-9]+ of [0-9]+ cores \(worst core [0-9]+\)" "$RUNLOG" 2>/dev/null | head -1)
  if [ -n "$l1" ]; then
    l1_stalls=$(printf '%s' "$l1" | sed -E 's/.*-- ([0-9]+) producer.*/\1/')
    l1_cores=$(printf '%s' "$l1" | sed -E 's/.*across ([0-9]+) of.*/\1/')
    l1_worst=$(printf '%s' "$l1" | sed -E 's/.*worst core ([0-9]+).*/\1/')
    [ "$l1_stalls" != "0" ] && log "  [stalls] k=$k L1 producer stalls=$l1_stalls across $l1_cores cores (worst core $l1_worst) -- capture perturbed the workload"
  else
    l1_stalls=NA; l1_cores=NA; l1_worst=NA
  fi

  # Marker count IS decode-derived, so it stays gated -- recorded as "nodecode" rather than a 0 that
  # would read like a measurement.
  if [ "$DECODE" = "1" ]; then
    markers=$(grep -aoE "socket [01] drained: .*markers" "$RUNLOG" 2>/dev/null \
              | awk '!seen[$2]++ {m+=$6} END{print (m==""?0:m)}')
    markers=${markers:-0}
  else
    markers=nodecode
  fi

  # Worst credit-wait, in us: the DRISC's own back-pressure counter, living in device L1 and read at
  # teardown -- so unlike markers/stalls it stays VALID with DECODE=0. That makes it the only usable knee
  # signal for a no-decode sweep. Max across the drainers (worst of the worsts). ~0.1 us = no back-pressure.
  cw_us=$(grep -aoE "worst credit-wait [0-9.]+ us" "$RUNLOG" 2>/dev/null | grep -oE "[0-9.]+" | sort -g | tail -1)
  cw_us=${cw_us:-NA}

  echo "$k,$DELAY,$ARMED,$rc,$dur,$ep,$rp,$ds,$class,$SWEEP_ID,$ack_ns,$read_ns,$markers,$l1_stalls,$l1_cores,$l1_worst,$cw_us" >> "$CSV"

  # A degraded card makes every subsequent number describe the LINK, not the bug -- so stop rather than
  # keep sweeping. Reads inflate 3.5x on a Gen1 downtrain, which changes durations, poll costs and very
  # likely the stall rate being measured.
  if [ "$ack_ns" != NA ] && [ "$read_ns" != NA ]; then
    degraded=""
    if [ "$ack_ns" -ge "$ACK_MAX_NS" ]; then
      degraded="13x MMIO state (ack ${ack_ns} ns, read ${read_ns} ns) -- follows a box freeze + watchdog reboot"
    elif [ "$read_ns" -ge "$READ_MAX_NS" ]; then
      degraded="Gen1 read-latency state (ack ${ack_ns} ns normal, read ${read_ns} ns = ~3.5x) -- link downtrained"
    fi
    if [ -n "$degraded" ]; then
      log "=== DEGRADED CARD at k=$k: $degraded ==="
      log "    ep=$ep rp=$rp  (link speed alone is NOT a reliable check -- these probes are)"
      log "    Needs a COLD POWER CYCLE; a warm reboot will not retrain and can convert one bad state"
      log "    into the other. Any rate measured from here describes the link, not the failure."
      if [ "${ALLOW_DEGRADED:-0}" != "1" ]; then
        log "    stopping (ALLOW_DEGRADED=1 to sweep anyway)"
        break
      fi
    fi
  fi
  [ "$class" != CLEAN ] && log "k=$k rc=$rc dur=${dur}s ep=$ep rp=$rp devsta=$ds ack=${ack_ns}ns read=${read_ns}ns -> $class"

  if [ "$class" = CLEAN ]; then
    nconsec=0
  else
    nconsec=$((nconsec+1))
  fi

  if [ "$class" = WEDGE ]; then
    wedge_dump "$k"
    if [ "$STOP_ON_WEDGE" = "1" ]; then
      log "=== WEDGE CAPTURED at k=$k -- stopping with the card left wedged for inspection ==="
      log "    dump: $OUT/wedge_${SWEEP_ID}_${k}.txt     recover with: $TT_SMI -r"
      break
    fi
    recover || break
  elif [ "$class" = TEARDOWN ] || [ "$class" = ABORT ] || [ "$class" = MASKED ]; then
    # MASKED recovers too: it IS a teardown hang, just one the armed timeout caught. Leaving it
    # unrecovered is what let a sweep grind out 16 identical 46 s masked runs in a row.
    recover || break
  elif [ "$class" = MMIO_STALL ]; then
    # DO reset, despite the card reading healthy afterwards. FINDINGS is explicit that a hung card reads
    # HEALTHY for several runs -- current_link_speed stayed 32.0 GT/s through four cascade runs and only
    # collapsed to Unknown after the block ended -- so "Unknown" is a LATE symptom and a harness that
    # keys recovery off card state UNDER-resets and lets cascades run. Key off the PROCESS outcome, which
    # for this class is the ~210 ms non-completing read that aborted the run.
    # MMIO_NO_RESET=1 skips it, for deliberately testing whether the mode self-clears -- but expect a
    # cascade if it does not, and read the successor runs, not the link speed.
    if [ "${MMIO_NO_RESET:-0}" = "1" ]; then
      log "  [mmio] MMIO_NO_RESET=1 -- not resetting; watch the NEXT runs, not the link speed"
    else
      recover || break
    fi
  fi

  if [ "$MAX_CONSEC_FAIL" != "0" ] && [ "$nconsec" -ge "$MAX_CONSEC_FAIL" ]; then
    log "=== STOPPING: $nconsec consecutive non-clean runs (last=$class) ==="
    log "    The box is in a persistent state; sweeping further re-pays the timeout and learns nothing."
    log "    Next: $TT_SMI -r, then a few probe runs to see whether it clears or survives a reset."
    break
  fi

  [ $((k % 20)) -eq 0 ] && log "progress k=$k clean=$nclean wedge=$nwedge mmio=$nmmio teardown=$nteardown masked=$nmasked abort=$nabort"
done

log "=== DONE: clean=$nclean wedge=$nwedge mmio_stall=$nmmio teardown=$nteardown masked=$nmasked abort=$nabort ==="
python3 - "$CSV" "$SWEEP_ID" <<'PY' 2>/dev/null | tee -a "$SUM"
import sys, csv
allrows = list(csv.DictReader(open(sys.argv[1])))
# THIS sweep only: a csv may hold several, and pooling a host sweep with a container one (different
# devsta availability, possibly different dispatch) yields a rate that describes neither.
rows = [r for r in allrows if r.get('sweep') == sys.argv[2]] or allrows
d = sorted(float(r['dur_s']) for r in rows if r['dur_s'])
if d:
    print(f"duration: min={d[0]:.0f}s median={d[len(d)//2]:.0f}s max={d[-1]:.0f}s n={len(d)}")
n = len(rows)
w = sum(1 for r in rows if r['class'] == 'WEDGE')
m = sum(1 for r in rows if r['class'] == 'MMIO_STALL')
if n:
    print(f"hard wedge:  {w}/{n} = {100.0*w/n:.2f}%   (card stuck all-ones)")
    print(f"mmio stall:  {m}/{n} = {100.0*m/n:.2f}%   (transient: completion timeout, card recovered)")
    print(f"combined:    {w+m}/{n} = {100.0*(w+m)/n:.2f}%   (compare against the recorded ~2-3%/run)")
if len(allrows) != n:
    print(f"note: csv holds {len(allrows)} rows across several sweeps; the above is sweep {sys.argv[2]} only")
PY
log "csv=$CSV"
