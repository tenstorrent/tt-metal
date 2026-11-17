#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
# repro.sh: fast repro of one known chip fault.
# Run it as run.sh --repro <name> [<bdf>] [options] (or ./repro.sh the same way). Each try is one plain run.sh
# (gates, C0 preflight, evidence logging, C2 start warm reset with health recount, C9 end reset; the serving and
# peer gates included) cut to that fault's stages through FLEET_REPRO and the FLEET_BUSBAR_* knobs. It stops at
# the first hit with a FAIL line naming the chip by BDF and tray 1-4, the exact finding under it and the time to
# failure (exit 8), or prints PASS after the tries (exit 0). It changes no power limit, runs no link check or
# fabric change and resets only this host's own chips, as plain run.sh does.
#
#   repro.sh numeric <bdf> [--way heavy|dram|dram-mix|busbar|all] [--tries N] [--known <bdf>[,<bdf>]] [--dry-run]
#     The chip's numeric fault (NUMERIC FAIL naming <bdf>). Ways, by fault type (k/N and times: the notes above):
#       heavy   the bus-bar heavy chunk alone, lofi_all x16 back to back for 90 s, the chip's tray only
#               (#2487: 88:00.0 3/3, first bad verify at 126 s, about 2.5 min of load per try)
#       dram    #2499's C4d 120 s shape: right after the C2 reset, 120 s of the x/w/y-in-DRAM matmul
#               (numeric_dram.py, a readback every 1.5 s of load), the whole box, no CCL mix chunk first (#2499:
#               03:00.0 4/4 at 0-30 s of load, 5.4-6.8 min per try). Every 4th try (4, 8, ..) is the old way, dram-mix
#               (#1062: one old-way try in four until the targeted way hits at least as often per try)
#       dram-mix the old dram way: one 1 min CCL mix chunk, then 120 s of G3 (#2784: 2/4 tries hit by G3, 9.9-21.8 min)
#       busbar  the bus-bar CCL mix alone, chunks 1,2,3,4,4,4,4 min, the chip's tray only (#2489 TB: 83:00.0 4/9,
#               median first hit 11.8 min)
#       all     heavy, dram, busbar in turn, one way per try
#     No --way: the fastest recorded way for that bdf (83:00.0 busbar, 88:00.0 heavy, 03:00.0 dram); any other bdf
#     all. The first line says which way was chosen and why.
#     Default tries: heavy 3, dram 4 (try 4 the old way), dram-mix 3, busbar 4, all 6.
#   repro.sh warm-reset <bdf> [--tries N] [--known <bdf>[,<bdf>]] [--dry-run]
#     The chip lost or failing on a warm reset (RESET FAIL naming <bdf>): no load; each try is the C2 start warm
#     reset (tt-smi -glx_reset) and the C9 end warm reset, so 2 resets per try, with run.sh's own
#     recovery (root-port rescan, then glx_reset of this host's own trays) after a loss. Default tries 5.
#     (#2487: 47:00.0 lost at the start reset 5/5, 2.3 min after start; #2488 c4:00.0 is rarer, about 1 run in 14.)
#   repro.sh tray-drop <bdf> [--shape h3-replay|h3-cadence|imbalance|weights|cold-full|all] [--tries N] [--dry-run]
#     The tray of <bdf> losing chips under load (POWER DROP or CHIP DROP naming that tray:
#     a quad 3 host's tray 2, 41:00.0-48:00.0, lost its 12 V under H3 2/2 times, under plain run.sh 0/9). Each try is the
#     bus-bar stage's mix alone on all 4 trays (one submesh per tray) in one H3-shaped load, fleet/drop_shapes.py:
#       h3-replay   each tray on its own: matmul+CCL ~1-3 s, busy-clock CCL+DRAM+PCIe ~1.5-5 s, random phase (12 min)
#       h3-cadence  all trays together: matmul+CCL 2 s, busy-clock CCL+DRAM+PCIe 4-8 s (12 min)
#       imbalance   <bdf>'s tray at matmul while the other three run the busy-clock mix, swapped every 3 s (12 min)
#       weights     matmul on all trays with 64 MB x4 DRAM streams and a 16 MB PCIe write per chip per launch (12 min)
#       cold-full   clock-checked idle, then straight to the full mix, 2 min, 6 times (control)
#       all         the five in that order, one per try (the default; default tries 5, one shape 3)
#     It stops at the first drop of <bdf>'s tray and prints the minutes from the shape's load start to it, the tray and
#     its BDFs. The load runs at the box's own TDP: no power limit is read for it or written.
#   --known <bdf>[,<bdf>]  known standing faults on this host: a try whose only fault lines name these chips (and no
#     other) is a miss and the loop goes on, instead of stopping (exit 9). E.g. --repro numeric 88:00.0 --known
#     47:00.0 on a host where 47:00.0 (tray 2) is lost at every start reset. Not the repro's own bdf.
#   run.sh options, given before or after --repro, go to every try's run.sh: --known-dropped-tray N (its tray's
#     KNOWN DROP lines are then known faults too; not the repro chip's tray), --since T (--history-since),
#     --hwmon-guard, --hold, --hold-timeout M. --heavy-logs passes through as
#     TRAY_REPRO_HEAVY_LOGS=1 and --dry-run prints the plan. --mode, --tries and --ag-tries before --repro are
#     refused (repro.sh picks the stages; its own --tries comes after the bdf).
#   A hit also names the time of the chip's first bad readback in the try's rows.jsonl (chunk-end verify, or the
#     G3 check's own time), next to the time its FAIL line arrived.
#   A numeric try that finished (run.sh exit 8) with the repro chip's own check done and clean (its NUMERIC RATE line,
#     its tray checked) but a NUMERIC FAIL on another chip is a MISS on the repro chip plus a FAIL line per other chip
#     naming it by BDF, tray 1-4 and DRAM bank, exit 8 as run.sh's own verdict.
# Exit: 0 no hit, 8 hit (or a numeric FAIL on another chip after the repro chip's check finished clean), 10 host busy
# or serving (run.sh's P0 gate), 9 another fault or an incomplete try stopped the loop (its findings are printed),
# 1 usage.
set -uo pipefail
SELF=$(readlink -f "$0"); cd "$(dirname "$SELF")" || exit 1
RUNSH=${TRAY_REPRO_RUNSH:-./run.sh}   # tests stub it
usage() { sed -n '2,/^set -u/p' "$SELF" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "repro.sh: $*" >&2; exit 1; }
tray_of() { case $(( 0x${1%%:*} & 0xF0 )) in 0) echo 1 ;; 64) echo 2 ;; 192) echo 3 ;; 128) echo 4 ;; *) return 1 ;; esac; }   # as run.sh rk_tray
# the fastest recorded way per chip: 83:00.0 bus-bar mix (#2489 TB), 88:00.0 the
# heavy chunk (#2487), 03:00.0 the DRAM readback (#2499)
declare -A FASTEST=([83:00.0]=busbar [88:00.0]=heavy [03:00.0]=dram)

NAME=${1:-}; [ $# -gt 0 ] && shift
case $NAME in -h|--help) usage; exit 0 ;; numeric|warm-reset|tray-drop) ;; busbar-confirm) die "repro busbar-confirm was removed: run.sh --mode=full runs the extra number formats over its whole heavy chunk (a feed peak alone is never a verdict)" ;; '') usage >&2; exit 1 ;;
  *) die "unknown repro '$NAME' (numeric, warm-reset, tray-drop)" ;; esac
SHAPES=""; [ "$NAME" = tray-drop ] && { SHAPES=$(python3 -I fleet/drop_shapes.py list | cut -f1 | paste -sd' ')   # its shapes in try order
  [ -n "$SHAPES" ] || die "fleet/drop_shapes.py lists no tray-drop shape"; }
BDF=""; SHAPE=""; WAY=""; TRIES=""; DRY=0; KNOWN=""; KD=${TRAY_TEST_KNOWN_DROPPED:-}; RUN_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --way=*) WAY=${1#*=} ;;
    --way) WAY=${2:-}; shift ;;
    --shape=*) SHAPE=${1#*=} ;;
    --shape) SHAPE=${2:-}; shift ;;
    --tries=*) TRIES=${1#*=} ;;
    --tries) TRIES=${2:-}; shift ;;
    --known=*) KNOWN="$KNOWN,${1#*=}" ;;
    --known) [ $# -ge 2 ] || die "--known needs a bdf, e.g. 47:00.0"; KNOWN="$KNOWN,$2"; shift ;;
    --dry-run) DRY=1 ;;
    --heavy-logs) TRAY_REPRO_HEAVY_LOGS=1 ;;
    # run.sh's own options, passed on to every try
    --hwmon-guard|--hold) RUN_ARGS+=("$1") ;;
    --known-dropped-tray=*) KD=${1#*=} ;;
    --known-dropped-tray) [ $# -ge 2 ] || die "$1 needs a tray number 1-4"; KD=$2; shift ;;
    --since=*|--history-since=*) RUN_ARGS+=("--since=${1#*=}"); SINCE=${1#*=} ;;
    --since|--history-since) [ $# -ge 2 ] || die "$1 needs a time"; RUN_ARGS+=("--since=$2"); SINCE=$2; shift ;;
    --hold-timeout=*) HT=${1#*=}; RUN_ARGS+=("--hold-timeout=$HT") ;;
    --tdp=*) RUN_ARGS+=("--tdp=${1#*=}") ;;   # each try's run.sh sets and restores the limit
    --tdp) [ $# -ge 2 ] || die "--tdp needs watts, e.g. --tdp=160"; RUN_ARGS+=("--tdp=$2"); shift ;;
    --hold-timeout) [ $# -ge 2 ] || die "$1 needs minutes"; HT=$2; RUN_ARGS+=("--hold-timeout=$HT"); shift ;;
    --mode|--mode=*|--fleet|--fleet=*|--ag-tries|--repro|--repro=*)
      die "$1 does not go with --repro $NAME (repro.sh picks the stages; run.sh --help)" ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option $1 (repro.sh --help)" ;;
    *) [ -z "$BDF" ] || die "one bdf only (got '$BDF' and '$1')"; BDF=$1 ;;
  esac; shift
done
[ -n "$BDF" ] || die "$NAME needs the chip's bdf, e.g. 83:00.0"
BDF=${BDF,,}; BDF=${BDF#0000:}
[[ $BDF =~ ^[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$ ]] || die "bad bdf '$BDF' (want bus:dev.fn, e.g. 83:00.0)"
TRAY=$(tray_of "$BDF") || die "bdf $BDF is on no tray 1-4 (bus & 0xF0 must be 00, 40, c0 or 80)"
KL=""; IFS=, read -ra ks <<< "${KNOWN#,}"
for k in "${ks[@]}"; do k=${k,,}; k=${k#0000:}; [ -n "$k" ] || continue
  [[ $k =~ ^[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$ ]] || die "--known: bad bdf '$k' (want bus:dev.fn, e.g. 47:00.0)"
  tray_of "$k" > /dev/null || die "--known: bdf $k is on no tray 1-4"
  [ "$k" != "$BDF" ] || die "--known $k is the repro's own chip: its fault is the hit"; KL="$KL $k"; done
KL=${KL# }
if [ -n "$KD" ]; then
  [[ $KD =~ ^[1-4]$ ]] || die "--known-dropped-tray takes a tray number 1-4 (got '$KD')"
  [ "$KD" != "$TRAY" ] || die "--known-dropped-tray=$KD is the tray of $BDF: that tray would not be tested"
  RUN_ARGS=("--known-dropped-tray=$KD" "${RUN_ARGS[@]}")
fi
if [ -n "${SINCE:-}" ] && [[ ! $SINCE =~ ^[[:space:]]*$ ]]; then
  s=$(date -d "$SINCE" +%s 2> /dev/null) || die "--since: not a time: '$SINCE'"
  [ "$s" -le "$(date +%s)" ] || die "--since: '$SINCE' is in the future"; fi
[ -z "${HT:-}" ] || [[ $HT =~ ^[1-9][0-9]*$ ]] || die "--hold-timeout takes whole minutes >= 1 (got '$HT')"
WHY="--way given"
if [ "$NAME" = numeric ]; then
  if [ -z "$WAY" ]; then
    if [ -n "${FASTEST[$BDF]:-}" ]; then WAY=${FASTEST[$BDF]}; WHY="the fastest recorded way for $BDF"
    else WAY=all; WHY="no recorded way for $BDF: heavy, dram, busbar in turn"; fi
  fi
  case $WAY in heavy|dram|dram-mix|busbar|all) ;; *) die "--way takes heavy, dram, dram-mix, busbar or all (got '$WAY')" ;; esac
elif [ "$NAME" = tray-drop ]; then
  [ -z "$WAY" ] || die "tray-drop takes --shape, not --way"
  [ -n "$SHAPE" ] || { SHAPE=all; WHY="no --shape: every shape in try order, the one the H3 drops point to first"; }
  [ "$SHAPE" = all ] || [[ " $SHAPES " == *" $SHAPE "* ]] || die "--shape takes ${SHAPES// /, } or all (got '$SHAPE')"
  [ "$SHAPE" = all ] || WHY="--shape given"; WAY=$SHAPE
else [ -z "$WAY$SHAPE" ] || die "warm-reset takes no --way or --shape"; WAY=warm-reset; WHY="no load, the start and end warm resets"; fi
[ "$NAME" = tray-drop ] || [ -z "$SHAPE" ] || die "--shape goes with tray-drop only"
[ -n "$TRIES" ] || case $NAME:$WAY in tray-drop:all) TRIES=$(wc -w <<< "$SHAPES") ;; tray-drop:*) TRIES=3 ;;
  *:heavy|*:dram-mix) TRIES=3 ;; *:dram|*:busbar) TRIES=4 ;; *:all) TRIES=6 ;; *) TRIES=5 ;; esac
[[ $TRIES =~ ^[1-9][0-9]?$ ]] || die "--tries takes 1-99 (got '$TRIES')"

way_of() { local all=(heavy dram busbar); [ "$NAME" = tray-drop ] && all=($SHAPES); if [ "$WAY" = all ]; then echo "${all[$(( ($1 - 1) % ${#all[@]} ))]}"
  elif [ "$WAY" = dram ] && [ $(( $1 % 4 )) = 0 ]; then echo dram-mix; else echo "$WAY"; fi; }   # dram: every 4th try the old way (#1062)
env_of() {  # the env of one try's run.sh
  case $1 in
    heavy) echo "FLEET_REPRO=numeric FLEET_BUSBAR_ORDER=heavy FLEET_BUSBAR_HEAVY_S=90 FLEET_LOAD_TRAYS=$TRAY" ;;
    dram) echo "FLEET_REPRO=numeric FLEET_BUSBAR_ORDER=rw FLEET_BUSBAR_RW_S=120 FLEET_BUSBAR_RW_WALL_S=390 FLEET_BUSBAR_RW_CHECK_S=1.5" ;;
    dram-mix) echo "FLEET_REPRO=numeric FLEET_BUSBAR_ORDER=mix FLEET_BUSBAR_CHUNK_MIN=1 FLEET_BUSBAR_RW_AFTER=1 FLEET_BUSBAR_RW_S=120 FLEET_BUSBAR_RW_WALL_S=390" ;;
    busbar) echo "FLEET_REPRO=numeric FLEET_BUSBAR_ORDER=mix FLEET_BUSBAR_CHUNK_MIN=1,2,3,4,4,4,4 FLEET_LOAD_TRAYS=$TRAY" ;;
    warm-reset) echo "FLEET_REPRO=warm-reset" ;;
    *) echo "FLEET_REPRO=tray-drop $(python3 -I fleet/drop_shapes.py env "$1") FLEET_BUSBAR_SHAPE_TRAY=$TRAY" ;;   # a tray-drop shape
  esac; }
names() {  # <bdf> in a finding line (word bounded, 0000: or not)
  [[ $2 =~ (^|[^0-9a-f:])(0000:)?$1([^0-9]|$) ]]; }
hit() {  # finding line -> 0 if it is this repro's fault on this chip
  if [ "$NAME" = numeric ]; then [[ $1 == NUMERIC* && $1 != *PASS* ]] && names "$BDF" "$1"
  elif [ "$NAME" = tray-drop ]; then [[ $1 == "POWER DROP tray $TRAY "* || $1 == "CHIP DROP tray $TRAY "* ]]
  else [[ $1 == "RESET FAIL "* ]] && names "$BDF" "$1"; fi; }
fault() { [[ $1 =~ (FAIL|HANG|DROP|LOST|MISSING|INCOMPLETE) && ( $1 != *WARN* || $1 == "HSC FAIL "* ) ]]; }   # an HSC FAIL latch line names VIN_UV_WARN
known() {  # fault line -> 0 if it is a declared known fault: every bdf it names is a --known chip or on the
  # --known-dropped-tray, or it is that tray's own KNOWN DROP line; a line naming no chip is never known
  local l=$1 b n=0 r='([0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]'
  [ -n "$KD" ] && [[ $l == "KNOWN DROP tray $KD:"* || $l == "PREFLIGHT KNOWN-DROP "*"(tray $KD) "* ]] && return 0
  while [[ $l =~ $r ]]; do b=${BASH_REMATCH[0]#0000:}; l=${l#*"${BASH_REMATCH[0]}"}; n=$((n + 1))
    [[ " $KL " == *" $b "* ]] || { [ -n "$KD" ] && [ "$(tray_of "$b")" = "$KD" ]; } || return 1; done
  [ "$n" -gt 0 ]; }
checked() {  # 0 if the try's output shows $BDF's numeric check done: its NUMERIC RATE line with launches, its tray checked
  local b=${BDF//./\\.}
  grep -aqE "NUMERIC RATE (0000:)?$b \(tray $TRAY\): [1-9][0-9]* launches" "$OUT" && ! grep -aq "NUMERIC NOT CHECKED tray $TRAY " "$OUT"; }
bank_of() {  # NUMERIC FAIL line -> its DRAM bank(s): the banks its histogram counts, else its first wrong element's bank
  local h b i=0 bs=""
  if [[ $1 =~ DRAM\ bank\ histogram\ \[([0-9,\ ]*)\] ]]; then IFS=', ' read -ra h <<< "${BASH_REMATCH[1]}"
    for b in "${h[@]}"; do [ "$b" -gt 0 ] 2> /dev/null && bs="$bs,$i"; i=$((i + 1)); done; fi
  [ -n "$bs" ] || { [[ $1 =~ DRAM\ bank\ ([0-9]+) ]] && bs=",${BASH_REMATCH[1]}"; }
  echo "${bs#,}"; }
core_of() {  # NUMERIC FAIL line -> its owning compute core(s) 'X,Y': a matmul's
  # 'DRAM bank' is its tile index mod 8, not a bank, so a named compute core goes first
  local l=$1 cs=""
  while [[ $l =~ compute\ core\ \(([0-9]+,[0-9]+)\) ]]; do cs="$cs ${BASH_REMATCH[1]}"; l=${l#*"${BASH_REMATCH[0]}"}; done
  echo "${cs# }"; }
mins() { awk -v s="$1" 'BEGIN { printf "%.1f", s / 60 }'; }
first_bad() {  # <run dir> <try start epoch>: the first bad readback of $BDF in the try's rows.jsonl (label and log only)
  find "$1" -name rows.jsonl 2> /dev/null | python3 -I -c '
import json, sys, time
bdf, t0, best = sys.argv[1], float(sys.argv[2]), None
for p in sys.stdin.read().split():
    for x in open(p, errors="replace"):
        try: r = json.loads(x)
        except ValueError: continue
        ev = r.get("event")
        if ev not in ("verify", "rw", "dbi", "noc") or not isinstance(r.get("start"), (int, float)): continue
        for c in (r.get("chips") or {}).values():
            pci = str(c.get("pci") or "").lower() if isinstance(c, dict) else ""
            if pci.removeprefix("0000:") != bdf: continue
            if ev == "rw": bad, t = c.get("bad_checks"), c.get("first_t") or r["start"]
            elif ev in ("dbi", "noc"): bad, t = c.get("bad_passes"), c.get("first_t") or r["start"]
            else: bad, t = c.get("mismatched") or c.get("bad_parts") or c.get("nonfinite"), r["start"]
            if not bad: continue
            k = r.get("k", -1)
            what = (f"G3 check after chunk {k + 1}" if ev == "rw" else f"G3b DRAM byte/DBI check after chunk {k + 1}" if ev == "dbi" else
                    f"N2 network-on-chip all-to-all check after chunk {k + 1}" if ev == "noc" else
                    "pre-load readback" if k < 0 else f"chunk-end verify of chunk {k + 1}")
            if best is None or t < best[0]: best = (t, what, p)
if best:
    t, what, p = best
    at = time.strftime("%H:%M:%S %Z", time.localtime(t))
    print(f"first bad readback {at}, {(t - t0) / 60:.1f} min into the try ({what}, {p})")
' "$BDF" "$2" 2> /dev/null; }

drop_at() {  # <run dir>: "<min> min from the shape's load start to the drop (...)", from the shape row and drop_s.txt
  local f; f=$(find "$1" -name drop_s.txt 2> /dev/null | head -n1); [ -n "$f" ] || return 0
  find "$1" -name rows.jsonl 2> /dev/null | python3 -I -c '
import json, sys, time, datetime
t0, el, pg = (open(sys.argv[1]).read().split() + ["", "", ""])[:3]
drop = float(pg) if pg else float(t0) + float(el)
st = None
for p in sys.stdin.read().split():
    for x in open(p, errors="replace"):
        try: r = json.loads(x)
        except ValueError: continue
        if r.get("event") == "shape":
            st = datetime.datetime.strptime(r["t"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
at = time.strftime("%H:%M:%S %Z", time.localtime(drop))
how = "first vr_pg fall" if pg else "loss seen"
if st is not None and drop >= st:
    print(f"{(drop - st) / 60:.1f} min from the shape load start to the drop ({how} {at})")
else:
    print(f"{(drop - float(t0)) / 60:.1f} min from the bus-bar stage start to the drop ({how} {at}; no shape row)")
' "$f" 2> /dev/null; }

echo "repro $NAME $BDF (tray $TRAY): way $WAY ($WHY), up to $TRIES tries, each one plain run.sh cut to the fault's stages; stops at the first hit"
[ -z "$KL$KD" ] || echo "known faults (a try with only these is a miss):${KL:+ $KL}${KD:+ tray $KD (--known-dropped-tray)}"
RA=""; [ ${#RUN_ARGS[@]} = 0 ] || RA=" $(printf '%q ' "${RUN_ARGS[@]}")"; RA=${RA% }
if [ "$DRY" = 1 ]; then
  for k in $(seq "$TRIES"); do w=$(way_of "$k"); echo "try $k: way $w: $(env_of "$w") $RUNSH$RA${TRAY_REPRO_HEAVY_LOGS:+ (TRAY_REPRO_HEAVY_LOGS=$TRAY_REPRO_HEAVY_LOGS)}"; done
  exit 0
fi
H=$(hostname -s); T0=$(date +%s); HITS=$(mktemp); OUT=$(mktemp); trap 'rm -f "$HITS" "$OUT"' EXIT
for k in $(seq "$TRIES"); do
  w=$(way_of "$k"); ts=$(date +%s); mark=$(mktemp -p . .repro_mark.XXXXXX)
  echo "$(date '+%F %T %Z') $H: repro try $k of $TRIES, way $w: $(env_of "$w") $RUNSH$RA"
  : > "$HITS"; : > "$OUT"
  # one try: run.sh's output stays on screen; the first hit line's arrival time is the time to failure
  # shellcheck disable=SC2046
  env $(env_of "$w") TRAY_REPRO_HEAVY_LOGS="${TRAY_REPRO_HEAVY_LOGS:-0}" bash "$RUNSH" "${RUN_ARGS[@]}" 2>&1 | while IFS= read -r l; do
    printf '%s\n' "$l"; printf '%s\n' "$l" >> "$OUT"; [ -s "$HITS" ] || { f=${l#* * * * }; hit "$f" && date +%s > "$HITS"; }; done
  rc=${PIPESTATUS[0]}
  d=$(find . -maxdepth 1 -type d -name "run_${H}_*" -newer "$mark" | sort | tail -n1); rm -f "$mark"
  fd=${d:+$d/fleet/findings.txt}; fl=""; [ -n "$fd" ] && [ -f "$fd" ] && fl=$(grep -a . "$fd")
  hl=$(while IFS= read -r l; do hit "$l" && printf '%s\n' "$l"; done <<< "$fl")
  if [ -n "$hl" ]; then
    te=$(cat "$HITS" 2> /dev/null); [ -n "$te" ] || te=$(date +%s)
    echo "FAIL tray $TRAY (1 finding): $BDF (tray $TRAY) $NAME repro hit on try $k of $TRIES (way $w), $(mins $((te - ts))) min into the try, $(mins $((te - T0))) min after the repro started"
    printf '%s\n' "$hl" | sed 's/^/  /'
    if [ "$NAME" = tray-drop ]; then
      tb=$(head -n1 <<< "$hl" | grep -oE '([0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]' | sed 's/^0000://' | paste -sd' ')
      da=$([ -n "$d" ] && drop_at "$d"); echo "  tray $TRAY dropped under shape $w: ${da:-time from the shape start unread (no drop_s.txt)}; tray $TRAY BDFs lost: ${tb:-unnamed}"
    fi
    fb=$([ -n "$d" ] && first_bad "$d" "$ts"); [ -z "$fb" ] || echo "  $fb; the FAIL line came at the chunk end, $(mins $((te - ts))) min into the try"
    if [ "$NAME" = numeric ]; then echo "  cause: $BDF (tray $TRAY) returned wrong matmul results under the $w load (chip compute or its power under load); run dir $d"
    elif [ "$NAME" = tray-drop ]; then echo "  cause: tray $TRAY lost chips under the $w shape ($(head -n1 <<< "$hl" | cut -d' ' -f1-2): POWER DROP = the whole tray or its vr_pg fell, its 12 V side: bus bar, tray power or supply; CHIP DROP = chips left PCIe with power good); run dir $d"
    else echo "  cause: $BDF (tray $TRAY) failed or fell off PCIe on this host's own warm reset (tt-smi -glx_reset); run dir $d"; fi
    other=$(while IFS= read -r l; do hit "$l" || ! fault "$l" || printf '%s\n' "$l"; done <<< "$fl")
    [ -z "$other" ] || { echo "  other findings in the same try:"; printf '%s\n' "$other" | sed 's/^/    /'; }
    # the repro's own verdict line, last: the try's run.sh RESULT line names only that try
    echo "RESULT: $NAME fault $BDF tray $TRAY (hit on try $k of $TRIES, $([ "$NAME" = tray-drop ] && echo shape || echo way) $w, $(mins $((te - ts))) min into the try, $(mins $((te - T0))) min in all)"
    exit 8
  fi
  if [ "$rc" = 10 ]; then echo "GATE repro $NAME $BDF (tray $TRAY): try $k could not start, host busy or serving (run.sh exit 10); stopped after $(mins $(( $(date +%s) - T0 ))) min"; exit 10; fi
  other=$(while IFS= read -r l; do fault "$l" && ! known "$l" && printf '%s\n' "$l"; done <<< "$fl")
  kn=$(while IFS= read -r l; do fault "$l" && known "$l" && printf '%s\n' "$l"; done <<< "$fl")
  # run.sh finished (exit 8) with $BDF's numeric check done and clean, and every other fault is a NUMERIC FAIL
  # on another chip: a miss on $BDF plus that chip's own FAIL line, exit 8 (run.sh's verdict), not an incomplete try
  r='([0-9a-f]{4}:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]'
  if [ "$NAME" = numeric ] && [ "$rc" = 8 ] && [ -n "$other" ] && checked \
      && ! grep -avqE "^NUMERIC FAIL (0000:)?[0-9a-f]{2}:[0-9a-f]{2}\.[0-7] " <<< "$other"; then
    nl=$(grep -aoE "NUMERIC RATE (0000:)?${BDF//./\\.} \(tray $TRAY\): [0-9]+ launches" "$OUT" | head -n1 | grep -oE '[0-9]+ launches$')
    echo "MISS repro $NAME $BDF (tray $TRAY): try $k of $TRIES (way $w): no hit on $BDF, its numeric check finished clean ($nl); run.sh exit 8 with a numeric FAIL on another chip${d:+ (run dir $d)}"
    obs=$(while IFS= read -r l; do [[ $l =~ $r ]] && echo "${BASH_REMATCH[0]#0000:}"; done <<< "$other" | awk '!s[$0]++'); rs=""
    for ob in $obs; do
      ol=$(while IFS= read -r l; do [[ $l =~ $r ]] && [ "${BASH_REMATCH[0]#0000:}" = "$ob" ] && printf '%s\n' "$l"; done <<< "$other")
      ot=$(tray_of "$ob" || echo '?'); bk=$(while IFS= read -r l; do bank_of "$l" | tr , '\n'; done <<< "$ol" | grep . | sort -nu | paste -sd,)
      cc=$(while IFS= read -r l; do core_of "$l" | tr ' ' '\n'; done <<< "$ol" | grep . | awk '!s[$0]++' | sed 's/.*/(&)/' | paste -sd' ')
      bk=${bk:+DRAM bank $bk}; bk=${bk:-DRAM bank not named}; [ -z "$cc" ] || bk="compute core $cc"; nf=$(grep -ac . <<< "$ol")
      echo "FAIL tray $ot ($nf finding$([ "$nf" = 1 ] || echo s)): $ob (tray $ot) $bk: numeric fault on another chip in repro try $k of $TRIES (way $w), not the repro's chip $BDF"
      printf '%s\n' "$ol" | sed 's/^/  /'; rs="$rs, $ob tray $ot $bk"; done
    [ -z "$kn" ] || { echo "  known faults in the same try:"; printf '%s\n' "$kn" | sed 's/^/    /'; }
    echo "RESULT: $NAME miss on $BDF tray $TRAY (try $k of $TRIES, way $w, check finished clean); numeric FAIL on another chip:${rs#,}"
    exit 8
  fi
  # a try with only known faults is a miss when run.sh ended with a fault exit those faults explain (0, 3 tray drop,
  # 5 chip drop, 8 fleet check, 12 KNOWN-DROP, 14 RESET FAIL); 1 unknown, 6 nothing loaded, 7 hang, 9 INCOMPLETE stop
  if [ -n "$other" ] || { [ "$rc" != 0 ] && { [ -z "$kn" ] || [[ ! " 3 5 8 12 14 " == *" $rc "* ]]; }; }; then
    echo "STOPPED repro $NAME $BDF (tray $TRAY): try $k of $TRIES ended with run.sh exit $rc and no hit on $BDF${d:+ (run dir $d)}; another fault or an incomplete try, not this repro's:"
    [ -n "$other$kn" ] && printf '%s\n' "$other" "$kn" | grep -a . | sed 's/^/  /' || echo "  (no fault line in ${fd:-the findings: no run dir found})"
    exit 9
  fi
  if [ -n "$kn" ]; then
    echo "$(date '+%F %T %Z') $H: repro try $k of $TRIES, way $w: no hit ($(mins $(( $(date +%s) - ts ))) min; run.sh exit $rc, known faults only, counted as a miss):"
    printf '%s\n' "$kn" | sed 's/^/  /'
  else echo "$(date '+%F %T %Z') $H: repro try $k of $TRIES, way $w: no hit ($(mins $(( $(date +%s) - ts ))) min)"; fi
done
echo "PASS $BDF (tray $TRAY): no $NAME hit in $TRIES tries (way $WAY)${KL:+, known faults $KL}${KD:+, known dropped tray $KD}, $(mins $(( $(date +%s) - T0 ))) min"
exit 0
