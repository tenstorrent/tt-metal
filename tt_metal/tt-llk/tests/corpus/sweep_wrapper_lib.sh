#!/usr/bin/env bash
# sweep_wrapper_lib.sh — shared helpers for the sweep wrapper entry points
# (weekly_bh_sweep.sh / nightly_bh_sweep.sh / headline_bh_sweep.sh).
# SOURCED by the wrappers, never executed; keep it bash-3-safe and quiet.
#
# Why this file exists (incident, 2026-08-20): the pin-14 weekly's
# date-derived evidence root `weekly-20260820` collided with the existing
# pin-12 full-surface run of the same name — 15 minutes of pin-14 classify
# artifacts were written into pin-12 evidence before a stale-REPORT ping
# exposed it (the root was quarantined as
# weekly-20260820-CONTAMINATED-pin12-plus-15min-pin14).  The guard below
# makes that class impossible: a wrapper REFUSES to write into a root that
# records a different toolchain pin.

# evidence_root_guard EV PIN_SHA WRAPPER_NAME
#   Collision guard for a derived evidence root, run AFTER sourcing the conf
#   (needs the current PINNED_CC1PLUS_SHA256) and BEFORE the first write into
#   the root.  Pin provenance of an existing root is read from, in order:
#     1. EV/PIN_STAMP        — written by this guard at first touch (closes
#                              the window before sweep_2x2.py's preflight);
#     2. EV/preflight.json   — sweep_2x2.py's own record (cc1plus_sha256).
#   Verdicts:
#     - root absent/empty ................ proceed; stamp EV/PIN_STAMP.
#     - recorded pin == current pin ...... proceed (idempotent resume).
#     - recorded pin != current pin ...... REFUSE rc 3, loud, with a
#                                          suggested free SWEEP_DATE.
#     - non-empty root, NO pin record .... REFUSE rc 3 (fail closed: unknown
#                                          provenance is how contamination
#                                          starts).
#   SWEEP_DATE stays the manual root-name override; it does NOT bypass the
#   guard — a same-pin root simply resumes, a foreign-pin root refuses.
evidence_root_guard() {
  local ev=$1 pin=$2 wrapper=${3:-sweep-wrapper}
  local recorded="" src=""

  if [ -z "$pin" ]; then
    echo "FATAL: evidence_root_guard called with an empty pin sha (conf not sourced?)" >&2
    return 3
  fi

  if [ ! -e "$ev" ] || [ -z "$(ls -A -- "$ev" 2>/dev/null)" ]; then
    mkdir -p -- "$ev" || return 3
    printf '%s\n' "$pin" > "$ev/PIN_STAMP"
    return 0
  fi

  if [ -f "$ev/PIN_STAMP" ]; then
    recorded=$(head -n1 -- "$ev/PIN_STAMP" 2>/dev/null) src="PIN_STAMP"
  elif [ -f "$ev/preflight.json" ]; then
    recorded=$(python3 -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("cc1plus_sha256",""))' \
      "$ev/preflight.json" 2>/dev/null) src="preflight.json"
  fi

  if [ -n "$recorded" ] && [ "$recorded" = "$pin" ]; then
    # Same pin: sanctioned idempotent resume.  Backfill the stamp so later
    # resumes never depend on preflight.json alone.
    [ -f "$ev/PIN_STAMP" ] || printf '%s\n' "$pin" > "$ev/PIN_STAMP"
    return 0
  fi

  # ---- refuse: existing root with a foreign or unknown pin ----
  local base date_part kind sfx suggest=""
  base=$(basename -- "$ev")
  kind=${base%%-*}
  date_part=${base#*-}
  for sfx in b c d e f g h i j k l m n; do
    if [ ! -e "$(dirname -- "$ev")/${kind}-${date_part}${sfx}" ]; then
      suggest="${date_part}${sfx}"
      break
    fi
  done
  {
    echo "FATAL: EVIDENCE-ROOT COLLISION — $wrapper refuses to write into $ev"
    if [ -n "$recorded" ]; then
      echo "  root's recorded pin ($src): $recorded"
      echo "  current conf pin (PINNED_CC1PLUS_SHA256): $pin"
      echo "  This root belongs to a DIFFERENT toolchain pin.  Writing into it is the"
      echo "  2026-08-20 cross-contamination class (15 min of pin-14 classify inside the"
      echo "  pin-12 weekly-20260820 root)."
    else
      echo "  root is non-empty but has NO pin record (no PIN_STAMP, no preflight.json)."
      echo "  current conf pin (PINNED_CC1PLUS_SHA256): $pin"
      echo "  Unknown provenance — refusing fail-closed."
    fi
    echo "  -> relaunch under a fresh root:  SWEEP_DATE=${suggest:-<pick-a-new-name>} bash $wrapper ..."
    echo "  -> or quarantine/rename the old root first (append -CONTAMINATED-<why> and"
    echo "     drop a CONTAMINATION-NOTE.md) if it is already known-mixed."
    echo "  -> ONLY if you have hand-verified the root really is this pin's:"
    echo "     echo $pin > $ev/PIN_STAMP    # then rerun"
  } >&2
  return 3
}

# newest_clean_runs EVIDENCE_ROOT CURRENT_EV N PREFIX...
#   Prints (stdout) a comma-chain of the newest N CLEAN run roots under
#   EVIDENCE_ROOT matching PREFIX-* (e.g. weekly nightly headline), newest
#   activity (mtime) first, for the wrappers' --prev-run argument.  Skips:
#     - the current run's own root,
#     - roots whose name marks them dirty (*CONTAMINATED*, *quarantine*),
#     - roots carrying a CONTAMINATION-NOTE.md or QUARANTINED marker file.
#   With N=1 this degrades to the old single-prev behavior (plus the clean
#   filter).  Today sweep_2x2.py's --prev-run feeds only the scoreboard
#   annotator, which probes <prev>/scoreboard.json and silently skips a
#   miss — so a comma-chain is harmless until the cross-pin cell-reuse
#   consumer lands and starts splitting it.
newest_clean_runs() {
  local root=$1 cur=$2 n=$3
  shift 3
  local p d
  local cands=()
  for p in "$@"; do
    for d in "$root/$p"-*/; do
      d=${d%/}
      [ -d "$d" ] && cands+=("$d")
    done
  done
  [ "${#cands[@]}" -eq 0 ] && return 0

  local out=()
  while IFS= read -r d; do
    [ -d "$d" ] || continue
    [ "$d" -ef "$cur" ] 2>/dev/null && continue
    [ "$d" = "$cur" ] && continue
    case "$(basename -- "$d")" in
      *CONTAMINATED*|*Contaminated*|*contaminated*|*QUARANTINE*|*quarantine*|*Quarantine*) continue ;;
    esac
    [ -e "$d/CONTAMINATION-NOTE.md" ] && continue
    [ -e "$d/QUARANTINED" ] && continue
    out+=("$d")
    [ "${#out[@]}" -ge "$n" ] && break
  done < <(ls -1dt -- "${cands[@]}" 2>/dev/null)

  [ "${#out[@]}" -eq 0 ] && return 0
  local IFS=,
  printf '%s' "${out[*]}"
}
