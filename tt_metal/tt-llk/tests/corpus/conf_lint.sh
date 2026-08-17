#!/usr/bin/env bash
# conf_lint.sh — mechanical must-agree linter for the sweep pin audit trail.
#
# Enforcement layer (ledger item 10 / waves 5–6 V1): the baseline header and
# the conf prose repeatedly went stale while the sha values moved ("header
# still lying — now 3 pins stale").  The agreement rules were prose-only;
# this linter makes them a gate.  Both sweep wrappers run it BEFORE anything
# else (before even sourcing the conf), so a lying audit trail refuses the
# whole sweep.
#
# Rules enforced (REFUSES with the exact disagreeing lines):
#   R1  each _REVIEWED_{CC1PLUS,COMPILER,SIM_BH,SIM_WH}_SHA256 is assigned
#       exactly once and is a full 64-hex lowercase sha256;
#   R2  the conf's CURRENT PIN prose block mentions the cc1plus AND driver
#       pins by their 12-hex prefixes (a re-pin that edits only the values
#       leaves the prose lying about what the pin is);
#   R3  the PIN HISTORY has exactly one "(CURRENT)" entry, it is the
#       highest-numbered entry, and its sha prefix equals the cc1plus pin;
#   R4  the sim pins' 12-hex prefixes appear in the conf prose (same rule as
#       the compiler pins);
#   R5  the baseline TSV header names the conf's cc1plus pin as CURRENT via
#       the anchor "CURRENT sweep_2x2.conf PINNED_CC1PLUS_SHA256" — exactly
#       one anchor line, prefix-matched against the conf;
#   R6  the baseline TSV header names the conf's BH sim pin as CURRENT via
#       the anchor "CURRENT sweep_2x2.conf PINNED_SIM_BH_SHA256" (the
#       corrected-sim pairing is part of the measurement identity).
#
# Usage: conf_lint.sh [<sweep_2x2.conf> <baseline.tsv>]
#   Defaults: the checked-in conf beside this script and the baseline for
#   ${CHIP_CLASS:-p150}.  Exit 0 GREEN, exit 1 RED.  Parses only — never
#   sources the conf (a broken conf must still lint).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CONF=${1:-$HERE/sweep_2x2.conf}
BASELINE=${2:-$HERE/sfpu_device_baseline_${CHIP_CLASS:-p150}_v1.tsv}

RED=0
fail() { # fail <rule> <message...>
  local rule=$1; shift
  echo "conf-lint: RED [$rule] $*"
  RED=1
}
show() { # show <file> <grep-pattern> — print the exact lines with location
  grep -nE "$2" "$1" | sed "s|^|    $1:|"
}

[ -f "$CONF" ] || { echo "conf-lint: RED [R0] conf not found: $CONF"; exit 1; }
[ -f "$BASELINE" ] || { echo "conf-lint: RED [R0] baseline not found: $BASELINE"; exit 1; }

# ---- R1: pin values ----
pin_value() { # pin_value <name> -> echoes value; diagnostics on stderr; rc!=0 on failure
  local name=$1 lines n
  lines=$(grep -nE "^_REVIEWED_${name}_SHA256=" "$CONF")
  n=$(printf '%s' "$lines" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    {
      echo "conf-lint: RED [R1] _REVIEWED_${name}_SHA256 must be assigned exactly once in $CONF (found $n):"
      [ -n "$lines" ] && printf '%s\n' "$lines" | sed "s|^|    $CONF:|"
    } >&2
    return 1
  fi
  local val=${lines#*=}
  if ! printf '%s' "$val" | grep -qE '^[0-9a-f]{64}$'; then
    {
      echo "conf-lint: RED [R1] _REVIEWED_${name}_SHA256 is not a full 64-hex lowercase sha256:"
      printf '%s\n' "$lines" | sed "s|^|    $CONF:|"
    } >&2
    return 1
  fi
  printf '%s' "$val"
}

if ! CC1=$(pin_value CC1PLUS); then CC1=""; RED=1; fi
if ! DRV=$(pin_value COMPILER); then DRV=""; RED=1; fi
if ! SBH=$(pin_value SIM_BH); then SBH=""; RED=1; fi
if ! SWH=$(pin_value SIM_WH); then SWH=""; RED=1; fi
[ -n "$CC1$DRV$SBH$SWH" ] || { echo "conf-lint: RED — no pins parsed"; exit 1; }

# Comment-only view of the conf (the prose that must agree with the values).
CONF_PROSE=$(grep -nE '^\s*#' "$CONF")
# The CURRENT PIN narrative block: from the 'CURRENT PIN' line to 'PIN HISTORY'.
CURPIN_BLOCK=$(awk '/^#.*CURRENT PIN \(values below\)/{f=1} /^#.*PIN HISTORY/{f=0} f' "$CONF")

prose_mentions() { # prose_mentions <rule> <12hex> <what> [<scope-text>]
  local rule=$1 pfx=$2 what=$3 scope=${4:-}
  local hay=${scope:-$CONF_PROSE}
  if ! printf '%s\n' "$hay" | grep -q "$pfx"; then
    fail R"${rule#R}" "$what pin value ${pfx}… is NOT mentioned in the conf prose — the values moved but the narrative did not (the audit-trail lie this linter exists to refuse).  Conf prose pin mentions found:"
    show "$CONF" '^[[:space:]]*#.*[0-9a-f]{12}…' | head -20
  fi
}

# ---- R2: the CURRENT PIN block itself mentions both toolchain pins (the
# history mentioning them is not enough — a re-pin must rewrite the
# narrative, not only append a history line) ----
if [ -z "$CURPIN_BLOCK" ]; then
  fail R2 "no 'CURRENT PIN (values below)' narrative block found in $CONF"
else
  if [ -n "$CC1" ]; then prose_mentions R2 "${CC1:0:12}" "cc1plus" "$CURPIN_BLOCK"; fi
  if [ -n "$DRV" ]; then prose_mentions R2 "${DRV:0:12}" "driver" "$CURPIN_BLOCK"; fi
fi

# ---- R3: PIN HISTORY (CURRENT) entry ----
if [ -n "$CC1" ]; then
  # Only history-entry-shaped lines count ("#   N. <12hex>… ... (CURRENT)"):
  # unrelated prose ABOUT the (CURRENT) rule must not trip the counter.
  CUR_LINES=$(printf '%s\n' "$CONF_PROSE" | grep -E '#\s+[0-9]+\.\s+[0-9a-f]{12}[^ ]*.*\(CURRENT\)')
  CUR_N=$(printf '%s' "$CUR_LINES" | grep -c . || true)
  if [ "$CUR_N" -ne 1 ]; then
    fail R3 "PIN HISTORY must carry exactly one (CURRENT) entry (found $CUR_N):"
    show "$CONF" '\(CURRENT\)'
  else
    if ! printf '%s\n' "$CUR_LINES" | grep -q "${CC1:0:12}"; then
      fail R3 "the (CURRENT) history entry does not carry the pinned cc1plus prefix ${CC1:0:12}…:"
      show "$CONF" '\(CURRENT\)'
      echo "    $CONF: _REVIEWED_CC1PLUS_SHA256=$CC1"
    fi
    # (CURRENT) must be the highest-numbered history entry.
    # History entries are the numbered comment lines carrying a sha prefix
    # ("#   N. <12+hex>…") — the sha requirement keeps unrelated numbered
    # prose lists out of the max-entry computation.
    LAST_NUM=$(printf '%s\n' "$CONF_PROSE" | grep -E '^[0-9]+:#\s+[0-9]+\.\s+[0-9a-f]{12}' | sed -E 's/^[0-9]+:#\s+([0-9]+)\..*/\1/' | sort -n | tail -1)
    CUR_NUM=$(printf '%s\n' "$CUR_LINES" | grep -oE '#\s+[0-9]+\.' | grep -oE '[0-9]+' | head -1)
    if [ -n "$LAST_NUM" ] && [ -n "$CUR_NUM" ] && [ "$CUR_NUM" != "$LAST_NUM" ]; then
      fail R3 "(CURRENT) sits on history entry $CUR_NUM but the history reaches entry $LAST_NUM:"
      show "$CONF" '\(CURRENT\)'
    fi
  fi
fi

# ---- R4: sim pin prose ----
if [ -n "$SBH" ]; then prose_mentions R4 "${SBH:0:12}" "libttsim bh"; fi
if [ -n "$SWH" ]; then prose_mentions R4 "${SWH:0:12}" "libttsim wh"; fi

# ---- R5/R6: baseline header anchors ----
anchor_check() { # anchor_check <rule> <anchor-text> <expected-64hex> <what>
  local rule=$1 anchor=$2 pin=$3 what=$4
  local lines n
  lines=$(grep -nE "^#.*[0-9a-f]{12}[^ ]* \($anchor\)" "$BASELINE")
  n=$(printf '%s' "$lines" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    fail "$rule" "baseline header must carry exactly one '$anchor' anchor line (found $n) in $BASELINE"
    [ -n "$lines" ] && printf '%s\n' "$lines" | sed "s|^|    $BASELINE:|"
    return
  fi
  if ! printf '%s\n' "$lines" | grep -q "${pin:0:12}"; then
    fail "$rule" "baseline header's $what anchor disagrees with the conf pin — the header is describing a stale pin (waves 5–6 V1):"
    printf '%s\n' "$lines" | sed "s|^|    $BASELINE:|"
    echo "    $CONF: reviewed $what pin = $pin"
  fi
}
[ -n "$CC1" ] && anchor_check R5 "CURRENT sweep_2x2.conf PINNED_CC1PLUS_SHA256" "$CC1" "cc1plus"
[ -n "$SBH" ] && anchor_check R6 "CURRENT sweep_2x2.conf PINNED_SIM_BH_SHA256" "$SBH" "BH sim"

# ---- R7: LLK-pristine (owner ruling 2026-08-17) ----
# The tt_llk_* library trees must be byte-identical to the reviewed upstream
# base: the compiler proves effects algorithmically — no trusted markers,
# typed shims, or any other source edit in the consumed library.  Semantic
# rewrites live under tests/ only.
LLKBASE=$(sed -n 's/^_REVIEWED_LLK_UPSTREAM_BASE=//p' "$CONF")
if [ -n "$LLKBASE" ]; then
  # Anchor repo discovery at this script's own checkout (fixture confs live
  # in temp dirs outside any repo; the LLK trees under test are always ours).
  REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel 2>/dev/null)
  if [ -n "$REPO_ROOT" ] && git -C "$REPO_ROOT" cat-file -e "$LLKBASE^{commit}" 2>/dev/null; then
    # Diff the WORKTREE against the base (not HEAD) so an uncommitted edit —
    # or an uncommitted revert — is judged by what would actually compile.
    LLK_DIRT=$(git -C "$REPO_ROOT" diff --name-only "$LLKBASE" -- \
      tt_metal/tt-llk/tt_llk_blackhole tt_metal/tt-llk/tt_llk_wormhole_b0 tt_metal/tt-llk/tt_llk_quasar)
    if [ -n "$LLK_DIRT" ]; then
      fail R7 "LLK library trees differ from the reviewed upstream base $LLKBASE (LLK-pristine rule: no edits to consumed LLK headers, ever):"
      printf '%s\n' "$LLK_DIRT" | sed 's|^|    |'
    fi
  else
    fail R7 "reviewed LLK upstream base $LLKBASE is not a commit in this repo (rebase without updating _REVIEWED_LLK_UPSTREAM_BASE?)"
  fi
else
  fail R7 "conf lacks _REVIEWED_LLK_UPSTREAM_BASE (the LLK-pristine rule is unenforceable without it)"
fi

if [ "$RED" -eq 0 ]; then
  echo "conf-lint: GREEN — pin values ↔ conf prose ↔ PIN HISTORY (CURRENT) ↔ baseline header all agree (cc1plus ${CC1:0:12}…, driver ${DRV:0:12}…, sim bh ${SBH:0:12}…, sim wh ${SWH:0:12}…)"
  exit 0
fi
echo "conf-lint: FAILED — the pin audit trail disagrees with itself; fix the prose/header IN THE SAME COMMIT as any pin change (see rules in this script's header)"
exit 1
