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
#       corrected-sim pairing is part of the measurement identity);
#   R7  LLK-PRISTINE (owner ruling 2026-08-17): the tt_llk_* library trees in
#       the worktree are byte-identical to the reviewed upstream base commit
#       (_REVIEWED_LLK_UPSTREAM_BASE in the conf) — the compiler proves
#       effects algorithmically; no trusted markers, typed shims, or any
#       other source edit in the consumed library; semantic rewrites live
#       under tests/ only;
#   R8  (Lane AZ corpus expansion) every corpus manifest row with
#       perf_status=measured has at least one sweep_2x2_ops.tsv row carrying
#       its corpus_id — a measured op that silently drops out of the sweep
#       surface is the omission class that hid welford/recip/bcast/mul_int
#       for two pin cycles; machine-readable kind=skip rows satisfy the rule
#       (visible in every scoreboard), silence does not;
#   R9  (laneBU, the pin-11 lesson) the _REVIEWED_FIRE_WITNESSES table is
#       present, well-formed (flag|pytest-node|dump-flag|required-line), and
#       names only flags in sweep_2x2.py's reviewed ON set — the STRUCTURAL
#       half of the union fire-witness rule; the compile half (each witness
#       node built at the pinned toolchain with the full ON set + dump flag,
#       missing line = RED 'fire witness stale on the union') is
#       witness_preflight.py, run by the nightly wrapper's preflight.
#   R10 (wave-9 quarantine, sfpi 4adac11aac1 HANDOFF item 13) the OPTIONAL
#       _QUARANTINED_FIRE_WITNESSES table (same row format) preserves the
#       witnesses of quarantined flags; its flags must NOT be in the ON
#       set and must not also carry a reviewed-table row — the inverse
#       coherence of R9, so a quarantined flag cannot silently return.
#
# Usage: conf_lint.sh [<sweep_2x2.conf> <baseline.tsv> [<manifest.tsv> <ops.tsv>]]
#   Defaults: the checked-in conf beside this script, the baseline for
#   ${CHIP_CLASS:-p150}, sfpu_corpus_v2.tsv and sweep_2x2_ops.tsv beside this
#   script.  Exit 0 GREEN, exit 1 RED.  Parses only — never
#   sources the conf (a broken conf must still lint).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CONF=${1:-$HERE/sweep_2x2.conf}
BASELINE=${2:-$HERE/sfpu_device_baseline_${CHIP_CLASS:-p150}_v1.tsv}
MANIFEST=${3:-$HERE/sfpu_corpus_v2.tsv}
OPS=${4:-$HERE/sweep_2x2_ops.tsv}

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
anchor_check() { # anchor_check <rule> <anchor-text> <expected-64hex> <what> [<file>]
  local rule=$1 anchor=$2 pin=$3 what=$4 file=${5:-$BASELINE}
  local lines n
  lines=$(grep -nE "^#.*[0-9a-f]{12}[^ ]* \($anchor\)" "$file")
  n=$(printf '%s' "$lines" | grep -c . || true)
  if [ "$n" -ne 1 ]; then
    fail "$rule" "baseline header must carry exactly one '$anchor' anchor line (found $n) in $file"
    [ -n "$lines" ] && printf '%s\n' "$lines" | sed "s|^|    $file:|"
    return
  fi
  if ! printf '%s\n' "$lines" | grep -q "${pin:0:12}"; then
    fail "$rule" "baseline header's $what anchor disagrees with the conf pin — the header is describing a stale pin (waves 5–6 V1):"
    printf '%s\n' "$lines" | sed "s|^|    $file:|"
    echo "    $CONF: reviewed $what pin = $pin"
  fi
}
[ -n "$CC1" ] && anchor_check R5 "CURRENT sweep_2x2.conf PINNED_CC1PLUS_SHA256" "$CC1" "cc1plus"
[ -n "$SBH" ] && anchor_check R6 "CURRENT sweep_2x2.conf PINNED_SIM_BH_SHA256" "$SBH" "BH sim"

# ---- R5b/R6b: KERNEL-scoped (v2) baseline header anchors (lane ET) ----
# The v2 baseline carries the VERDICT anchors (end-to-end KERNEL cells,
# owner ratification 2026-08-21) and is pin-anchored exactly like v1.
# Absent file = v2 not yet seeded (bootstrap): recorded, not RED — the
# sweep's handover rule keeps the v1 checks at full severity until then.
KBASELINE=${BASELINE%_v1.tsv}_v2.tsv
if [ "$KBASELINE" != "$BASELINE" ] && [ -f "$KBASELINE" ]; then
  [ -n "$CC1" ] && anchor_check R5b "CURRENT sweep_2x2.conf PINNED_CC1PLUS_SHA256" "$CC1" "cc1plus" "$KBASELINE"
  [ -n "$SBH" ] && anchor_check R6b "CURRENT sweep_2x2.conf PINNED_SIM_BH_SHA256" "$SBH" "BH sim" "$KBASELINE"
else
  echo "conf-lint: note [R5b/R6b] KERNEL (v2) baseline not present ($KBASELINE) — verdict anchors unseeded; v1 checks stay at full severity"
fi

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
    # Owner ruling 2026-08-18: the rule extends to the metal LLK-API layer
    # (tt_metal/hw/ckernels/*/metal/llk_api) — the consumed kernel surface,
    # same doctrine.  Individually reviewed exceptions are named in the
    # conf's _REVIEWED_LLK_API_EXCEPTIONS (space-separated repo-relative
    # paths, reviewed like a pin; "keep quant.h for now").
    LLK_DIRT=$(git -C "$REPO_ROOT" diff --name-only "$LLKBASE" -- \
      tt_metal/tt-llk/tt_llk_blackhole tt_metal/tt-llk/tt_llk_wormhole_b0 tt_metal/tt-llk/tt_llk_quasar \
      ':(glob)tt_metal/hw/ckernels/*/metal/llk_api/**')
    # Untracked NEW files under the guarded trees evade `git diff` — catch
    # them via status (an added header is as much an edit as a changed one).
    LLK_NEW=$(git -C "$REPO_ROOT" status --porcelain -- \
      tt_metal/tt-llk/tt_llk_blackhole tt_metal/tt-llk/tt_llk_wormhole_b0 tt_metal/tt-llk/tt_llk_quasar \
      ':(glob)tt_metal/hw/ckernels/*/metal/llk_api/**' | sed -n 's/^?? //p')
    [ -n "$LLK_NEW" ] && LLK_DIRT=$(printf '%s\n%s' "$LLK_DIRT" "$LLK_NEW" | sed '/^$/d')
    LLK_EXC=$(sed -n 's/^_REVIEWED_LLK_API_EXCEPTIONS="\(.*\)"$/\1/p' "$CONF")
    if [ -n "$LLK_DIRT" ] && [ -n "$LLK_EXC" ]; then
      LLK_DIRT=$(printf '%s\n' "$LLK_DIRT" | while read -r p; do
        case " $LLK_EXC " in *" $p "*) ;; *) printf '%s\n' "$p";; esac
      done)
    fi
    if [ -n "$LLK_DIRT" ]; then
      fail R7 "LLK library/API trees differ from the reviewed upstream base $LLKBASE (LLK-pristine rule: no edits to the consumed LLK surface; exceptions only via _REVIEWED_LLK_API_EXCEPTIONS):"
      printf '%s\n' "$LLK_DIRT" | sed 's|^|    |'
    fi
  else
    fail R7 "reviewed LLK upstream base $LLKBASE is not a commit in this repo (rebase without updating _REVIEWED_LLK_UPSTREAM_BASE?)"
  fi
else
  fail R7 "conf lacks _REVIEWED_LLK_UPSTREAM_BASE (the LLK-pristine rule is unenforceable without it)"
fi

# ---- R8: every perf_status=measured corpus row is wired into the sweep ----
if [ ! -f "$MANIFEST" ]; then
  fail R8 "corpus manifest not found: $MANIFEST"
elif [ ! -f "$OPS" ]; then
  fail R8 "sweep ops TSV not found: $OPS"
else
  R8_OUT=$(awk -F'\t' '
    FNR==1 { fileno++ }
    /^#/ { next }
    fileno==1 {
      if (!ops_hdr) {
        for (i=1;i<=NF;i++) if ($i=="corpus_id") ci=i
        ops_hdr=1
        if (!ci) { print "HDRFAIL ops: no corpus_id column"; exit 3 }
        next
      }
      wired[$ci]=1; next
    }
    {
      if (!man_hdr) {
        for (i=1;i<=NF;i++) { if ($i=="perf_status") pi=i; if ($i=="id") idi=i }
        man_hdr=1
        if (!pi || !idi) { print "HDRFAIL manifest: no id/perf_status column"; exit 3 }
        next
      }
      if ($pi=="measured" && !($idi in wired)) print $idi
    }' "$OPS" "$MANIFEST")
  if printf '%s' "$R8_OUT" | grep -q '^HDRFAIL'; then
    fail R8 "cannot evaluate: $R8_OUT (manifest $MANIFEST, ops $OPS)"
  elif [ -n "$R8_OUT" ]; then
    fail R8 "corpus rows with perf_status=measured have NO sweep_2x2_ops.tsv row (a measured op silently absent from the sweep surface — wire a row, or a machine-readable kind=skip row with the reason):"
    printf '%s\n' "$R8_OUT" | sed "s|^|    $MANIFEST: id |"
  fi
fi

# ---- R9: union fire-witness table (laneBU, the pin-11 lesson) ----
# Structural half: the _REVIEWED_FIRE_WITNESSES table must exist, parse
# (flag|pytest-node|dump-flag|required-line, 4 non-empty fields), name only
# -mtt-tensix-* flags that are IN sweep_2x2.py's reviewed ON set, use only
# -fdump-{rtl,tree}-rvtt_* dump flags, and carry no exact-duplicate rows.
# The COMPILE half (each witness node compiled at the pinned toolchain with
# the full ON set + dump flag; missing required line = RED "fire witness
# stale on the union") lives in witness_preflight.py — the nightly wrapper
# runs it in preflight (--skip-witness is the loudly-logged escape).
SWEEP_PY="$HERE/sweep_2x2.py"
R9_ASSIGN=$(grep -c '^_REVIEWED_FIRE_WITNESSES="' "$CONF")
if [ "$R9_ASSIGN" -ne 1 ]; then
  fail R9 "conf must assign _REVIEWED_FIRE_WITNESSES exactly once (found $R9_ASSIGN) — the union fire-witness rule is unenforceable without the table (an EMPTY table is legal, a missing one is not)"
else
  R9_TABLE=$(awk '/^_REVIEWED_FIRE_WITNESSES="/{f=1;next} f&&/^"[[:space:]]*$/{f=0;done=1;next} f' "$CONF")
  R9_CLOSED=$(awk '/^_REVIEWED_FIRE_WITNESSES="/{f=1;next} f&&/^"[[:space:]]*$/{f=0;done=1} END{print done?1:0}' "$CONF")
  if [ "$R9_CLOSED" != 1 ]; then
    fail R9 "_REVIEWED_FIRE_WITNESSES table is not terminated by a lone closing quote line"
  fi
  ON_BLOCK=$(awk '/^ON_FLAGS = \(/{f=1;next} f&&/^\)/{f=0} f' "$SWEEP_PY")
  [ -n "$ON_BLOCK" ] || fail R9 "cannot extract the ON_FLAGS block from $SWEEP_PY (layout changed?)"
  R9_SEEN=""
  while IFS= read -r wline; do
    wtrim=$(printf '%s' "$wline" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -z "$wtrim" ] && continue
    case "$wtrim" in \#*) continue;; esac
    wf1=$(printf '%s' "$wtrim" | awk -F'|' '{print $1}')
    wf2=$(printf '%s' "$wtrim" | awk -F'|' '{print $2}')
    wf3=$(printf '%s' "$wtrim" | awk -F'|' '{print $3}')
    wf4=$(printf '%s' "$wtrim" | awk -F'|' '{print $4}')
    wnf=$(printf '%s' "$wtrim" | awk -F'|' '{print NF}')
    if [ "$wnf" -ne 4 ] || [ -z "$wf1" ] || [ -z "$wf2" ] || [ -z "$wf3" ] || [ -z "$wf4" ]; then
      fail R9 "witness table row needs exactly 4 non-empty '|' fields (flag|pytest-node|dump-flag|required-line): $wtrim"
      continue
    fi
    if ! printf '%s' "$wf1" | grep -qE '^-mtt-tensix-[a-z0-9-]+$'; then
      fail R9 "witness table row names a non -mtt-tensix- flag: $wf1"
    elif ! printf '%s\n' "$ON_BLOCK" | grep -qE -- "\"${wf1}[\" ]"; then
      fail R9 "witnessed flag $wf1 is NOT in sweep_2x2.py's reviewed ON set — a witness for a flag outside the union is a stale table row"
    fi
    if ! printf '%s' "$wf3" | grep -qE '^-fdump-(rtl|tree)-rvtt_[a-z_]+$'; then
      fail R9 "witness table row's dump flag is not -fdump-{rtl,tree}-rvtt_*: $wf3"
    fi
    if printf '%s\n' "$R9_SEEN" | grep -qxF -- "$wtrim"; then
      fail R9 "exact duplicate witness table row: $wtrim"
    fi
    R9_SEEN=$(printf '%s\n%s' "$R9_SEEN" "$wtrim")
  done <<EOF_R9
$R9_TABLE
EOF_R9
fi

# ---- R10: quarantined fire-witness table (wave-9, sfpi 4adac11aac1) ----
# The OPTIONAL _QUARANTINED_FIRE_WITNESSES table preserves the fire
# witnesses of flags placed under quarantine (never exercised, never
# promoted, no silicon) without deleting evidence.  witness_preflight.py
# reads only the REVIEWED table, so quarantined rows are never compiled.
# If the table is present it must: be assigned exactly once, terminate
# with a lone closing quote, parse with R9's 4-field row format, name
# only -mtt-tensix-* flags with -fdump-{rtl,tree}-rvtt_* dump flags, and
# — the INVERSE of R9's membership rule — name NO flag that is in
# sweep_2x2.py's reviewed ON set (a quarantined flag reappearing in the
# union without a reviewed row move back to _REVIEWED_FIRE_WITNESSES is
# a silent promotion) and NO flag that also has a reviewed-table row.
R10_ASSIGN=$(grep -c '^_QUARANTINED_FIRE_WITNESSES="' "$CONF")
if [ "$R10_ASSIGN" -gt 1 ]; then
  fail R10 "conf assigns _QUARANTINED_FIRE_WITNESSES more than once (found $R10_ASSIGN)"
elif [ "$R10_ASSIGN" -eq 1 ]; then
  R10_TABLE=$(awk '/^_QUARANTINED_FIRE_WITNESSES="/{f=1;next} f&&/^"[[:space:]]*$/{f=0;done=1;next} f' "$CONF")
  R10_CLOSED=$(awk '/^_QUARANTINED_FIRE_WITNESSES="/{f=1;next} f&&/^"[[:space:]]*$/{f=0;done=1} END{print done?1:0}' "$CONF")
  if [ "$R10_CLOSED" != 1 ]; then
    fail R10 "_QUARANTINED_FIRE_WITNESSES table is not terminated by a lone closing quote line"
  fi
  if [ -z "${ON_BLOCK:-}" ]; then
    ON_BLOCK=$(awk '/^ON_FLAGS = \(/{f=1;next} f&&/^\)/{f=0} f' "$SWEEP_PY")
  fi
  R10_SEEN=""
  while IFS= read -r qline; do
    qtrim=$(printf '%s' "$qline" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    [ -z "$qtrim" ] && continue
    case "$qtrim" in \#*) continue;; esac
    qf1=$(printf '%s' "$qtrim" | awk -F'|' '{print $1}')
    qf2=$(printf '%s' "$qtrim" | awk -F'|' '{print $2}')
    qf3=$(printf '%s' "$qtrim" | awk -F'|' '{print $3}')
    qf4=$(printf '%s' "$qtrim" | awk -F'|' '{print $4}')
    qnf=$(printf '%s' "$qtrim" | awk -F'|' '{print NF}')
    if [ "$qnf" -ne 4 ] || [ -z "$qf1" ] || [ -z "$qf2" ] || [ -z "$qf3" ] || [ -z "$qf4" ]; then
      fail R10 "quarantined witness row needs exactly 4 non-empty '|' fields (flag|pytest-node|dump-flag|required-line): $qtrim"
      continue
    fi
    if ! printf '%s' "$qf1" | grep -qE '^-mtt-tensix-[a-z0-9-]+$'; then
      fail R10 "quarantined witness row names a non -mtt-tensix- flag: $qf1"
    elif printf '%s\n' "$ON_BLOCK" | grep -qE -- "\"${qf1}[\" ]"; then
      fail R10 "QUARANTINED flag $qf1 is in sweep_2x2.py's reviewed ON set — quarantine violated (lifting requires a reviewed commit moving the row back to _REVIEWED_FIRE_WITNESSES with the ON-set line)"
    fi
    if printf '%s\n' "${R9_TABLE:-}" | grep -qE -- "^${qf1}\|"; then
      fail R10 "flag $qf1 appears in BOTH _REVIEWED_FIRE_WITNESSES and _QUARANTINED_FIRE_WITNESSES — a flag is reviewed or quarantined, never both"
    fi
    if ! printf '%s' "$qf3" | grep -qE '^-fdump-(rtl|tree)-rvtt_[a-z_]+$'; then
      fail R10 "quarantined witness row's dump flag is not -fdump-{rtl,tree}-rvtt_*: $qf3"
    fi
    if printf '%s\n' "$R10_SEEN" | grep -qxF -- "$qtrim"; then
      fail R10 "exact duplicate quarantined witness row: $qtrim"
    fi
    R10_SEEN=$(printf '%s\n%s' "$R10_SEEN" "$qtrim")
  done <<EOF_R10
$R10_TABLE
EOF_R10
fi

if [ "$RED" -eq 0 ]; then
  echo "conf-lint: GREEN — pin values ↔ conf prose ↔ PIN HISTORY (CURRENT) ↔ baseline header all agree (cc1plus ${CC1:0:12}…, driver ${DRV:0:12}…, sim bh ${SBH:0:12}…, sim wh ${SWH:0:12}…); LLK trees pristine vs the reviewed upstream base (R7); every measured corpus row is wired into the sweep (R8); fire-witness table well-formed and ON-set-coherent (R9 — compile half in witness_preflight.py); quarantined-witness table coherent with the ON set (R10)"
  exit 0
fi
echo "conf-lint: FAILED — the pin audit trail disagrees with itself; fix the prose/header IN THE SAME COMMIT as any pin change (see rules in this script's header)"
exit 1
