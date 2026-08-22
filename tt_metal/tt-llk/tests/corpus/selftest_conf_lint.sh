#!/usr/bin/env bash
# Self-test for corpus/conf_lint.sh (enforcement layer, ledger item 10).
#
# Proves, against the REAL linter (not a re-implementation):
#   1. a coherent conf+baseline fixture lints GREEN (rc 0);
#   2. sha values changed WITHOUT updating the prose -> RED (the waves-5/6
#      audit-trail lie: values moved, narrative did not);
#   3. (CURRENT) on a non-final PIN HISTORY entry -> RED;
#   4. two (CURRENT) markers -> RED;
#   5. baseline header cc1plus anchor naming a stale pin -> RED (V1);
#   6. baseline header missing the anchor entirely -> RED;
#   7. missing/short sim pin value -> RED;
#   8. THE CHECKED-IN conf + p150 baseline lint GREEN (the shipping state
#      must satisfy its own gate);
#   9. (R7) a bogus LLK upstream base -> RED (LLK-pristine rule);
#  10. (R8) a measured corpus row wired into the ops TSV -> GREEN;
#  11. (R8) a measured corpus row with NO ops TSV row -> RED (the
#      welford/recip/bcast/mul_int omission class).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
LINT="$HERE/conf_lint.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

CC1=aaaaaaaaaaaa1111111111111111111111111111111111111111111111111111
DRV=bbbbbbbbbbbb2222222222222222222222222222222222222222222222222222
SBH=cccccccccccc3333333333333333333333333333333333333333333333333333
SWH=dddddddddddd4444444444444444444444444444444444444444444444444444

write_conf() { # write_conf <file> <cc1> <drv> <sbh> <swh> <current-entry-num> <prose-cc1-prefix>
  local f=$1 cc1=$2 drv=$3 sbh=$4 swh=$5 curnum=$6 prosecc1=$7
  cat > "$f" <<EOF
# fixture sweep conf
# CURRENT PIN (values below): cc1plus ${prosecc1}… / driver ${drv:0:12}…
# sims: bh ${sbh:0:12}…, wh ${swh:0:12}… (corrected oracle pair)
#
# PIN HISTORY:
#   1. 999999999999… — first candidate$( [ "$curnum" = 1 ] && echo ' (CURRENT)' )
#   2. ${cc1:0:12}…$( [ "$curnum" = 2 ] && echo ' (CURRENT)' ) — the build described above.
_REVIEWED_CC1PLUS_SHA256=$cc1
_REVIEWED_COMPILER_SHA256=$drv
_REVIEWED_SIM_BH_SHA256=$sbh
_REVIEWED_SIM_WH_SHA256=$swh
_REVIEWED_LLK_UPSTREAM_BASE=${LLK_BASE_FOR_FIXTURES}
_REVIEWED_LLK_API_EXCEPTIONS="${LLK_EXC_FOR_FIXTURES}"
_REVIEWED_FIRE_WITNESSES="
-mtt-tensix-optimize-ccmask|perf_fixture.py::node[a]|-fdump-tree-rvtt_ccmask|ccmask: folded zeroing CC region
"
EOF
}
# R7 fixtures inherit the real conf's reviewed upstream base (the repo's LLK
# trees are pristine against it, so GREEN fixtures stay GREEN).
LLK_BASE_FOR_FIXTURES=$(sed -n 's/^_REVIEWED_LLK_UPSTREAM_BASE=//p' "$(dirname "$0")/sweep_2x2.conf")
LLK_EXC_FOR_FIXTURES=$(sed -n 's/^_REVIEWED_LLK_API_EXCEPTIONS="\(.*\)"$/\1/p' "$(dirname "$0")/sweep_2x2.conf")

write_baseline() { # write_baseline <file> <cc1-prefix> <sbh-prefix>
  cat > "$1" <<EOF
# fixture baseline
#   2. $2… (CURRENT sweep_2x2.conf PINNED_CC1PLUS_SHA256) — the pin the cells below rode.
# SIM PAIRING: bh $3… (CURRENT sweep_2x2.conf PINNED_SIM_BH_SHA256)
id	arch	cycles
row	bh	1.0
EOF
}

overall=0
check() { # check <name> <expected-rc> <actual-rc>
  if [ "$2" -eq "$3" ]; then
    echo "SELFTEST PASS: $1 (rc=$3 as expected)"
  else
    echo "SELFTEST FAIL: $1 (expected rc=$2, got rc=$3)"
    overall=1
  fi
}

# 1. coherent fixture -> GREEN
write_conf "$TMP/ok.conf" "$CC1" "$DRV" "$SBH" "$SWH" 2 "${CC1:0:12}"
write_baseline "$TMP/ok.tsv" "${CC1:0:12}" "${SBH:0:12}"
"$LINT" "$TMP/ok.conf" "$TMP/ok.tsv" > "$TMP/out1.log" 2>&1
check "coherent fixture lints GREEN" 0 $?

# 2. values moved, prose did not (prose still names an old cc1plus prefix) -> RED
write_conf "$TMP/staleprose.conf" "$CC1" "$DRV" "$SBH" "$SWH" 2 "eeeeeeeeeeee"
"$LINT" "$TMP/staleprose.conf" "$TMP/ok.tsv" > "$TMP/out2.log" 2>&1
check "value-moved-prose-stale refuses RED" 1 $?
grep -q "NOT mentioned in the conf prose" "$TMP/out2.log" || { echo "SELFTEST FAIL: staleprose RED lacks the exact-line diagnosis"; overall=1; }

# 3. (CURRENT) on a non-final history entry -> RED
write_conf "$TMP/wrongcur.conf" "$CC1" "$DRV" "$SBH" "$SWH" 1 "${CC1:0:12}"
"$LINT" "$TMP/wrongcur.conf" "$TMP/ok.tsv" > "$TMP/out3.log" 2>&1
check "(CURRENT) on stale history entry refuses RED" 1 $?

# 4. two (CURRENT) markers -> RED
write_conf "$TMP/twocur.conf" "$CC1" "$DRV" "$SBH" "$SWH" 2 "${CC1:0:12}"
sed -i 's/^#   1. 999999999999… — first candidate$/#   1. 999999999999… — first candidate (CURRENT)/' "$TMP/twocur.conf"
"$LINT" "$TMP/twocur.conf" "$TMP/ok.tsv" > "$TMP/out4.log" 2>&1
check "two (CURRENT) markers refuse RED" 1 $?

# 4b. (R3b) duplicated PIN HISTORY sha (in-place overwrite / global sha
#     replace — the wave-14 audit-corruption class) -> RED
write_conf "$TMP/dupsha.conf" "$CC1" "$DRV" "$SBH" "$SWH" 2 "${CC1:0:12}"
sed -i "s/^#   1. 999999999999…/#   1. ${CC1:0:12}…/" "$TMP/dupsha.conf"
"$LINT" "$TMP/dupsha.conf" "$TMP/ok.tsv" > "$TMP/out4b.log" 2>&1
check "duplicated PIN HISTORY sha (in-place overwrite) refuses RED" 1 $?
grep -q "R3b" "$TMP/out4b.log" || { echo "SELFTEST FAIL: dup-sha RED is not attributed to R3b"; overall=1; }

# 5. baseline anchor names a stale pin -> RED with both disagreeing lines shown
write_baseline "$TMP/stale.tsv" "ffffffffffff" "${SBH:0:12}"
"$LINT" "$TMP/ok.conf" "$TMP/stale.tsv" > "$TMP/out5.log" 2>&1
check "stale baseline anchor refuses RED" 1 $?
grep -q "describing a stale pin" "$TMP/out5.log" || { echo "SELFTEST FAIL: stale-anchor RED lacks diagnosis"; overall=1; }
grep -q "ffffffffffff" "$TMP/out5.log" || { echo "SELFTEST FAIL: stale-anchor RED does not print the disagreeing header line"; overall=1; }

# 6. baseline anchor missing -> RED
printf '# fixture baseline, no anchor\nid\tarch\n' > "$TMP/noanchor.tsv"
"$LINT" "$TMP/ok.conf" "$TMP/noanchor.tsv" > "$TMP/out6.log" 2>&1
check "missing baseline anchor refuses RED" 1 $?

# 7. short/invalid sim pin value -> RED
write_conf "$TMP/badsim.conf" "$CC1" "$DRV" "deadbeef" "$SWH" 2 "${CC1:0:12}"
"$LINT" "$TMP/badsim.conf" "$TMP/ok.tsv" > "$TMP/out7.log" 2>&1
check "non-64-hex sim pin refuses RED" 1 $?

# 8. the CHECKED-IN files must satisfy their own gate
"$LINT" > "$TMP/out8.log" 2>&1
check "checked-in conf+baseline lint GREEN" 0 $?
grep -q "conf-lint: GREEN" "$TMP/out8.log" || { echo "SELFTEST FAIL: shipping-state lint did not report GREEN"; overall=1; }

# 9. R7 LLK-pristine: a bogus upstream base -> RED (base must be a real commit)
sed 's/^_REVIEWED_LLK_UPSTREAM_BASE=.*/_REVIEWED_LLK_UPSTREAM_BASE=1111111111111111111111111111111111111111/' \
  "$TMP/ok.conf" > "$TMP/badllkbase.conf"
"$LINT" "$TMP/badllkbase.conf" "$TMP/ok.tsv" > "$TMP/out9.log" 2>&1
check "R7 bogus LLK upstream base refuses RED" 1 $?
grep -q "R7" "$TMP/out9.log" || { echo "SELFTEST FAIL: bogus-base RED is not attributed to R7"; overall=1; }

# R8 fixtures: a two-row manifest (one measured, one not_run) and an ops TSV.
write_manifest() { # write_manifest <file>
  cat > "$1" <<'EOF'
# fixture corpus manifest
id	surface	perf_status	notes
fixture__op_measured	legacy	measured	measured fixture row
fixture__op_idle	legacy	not_run	unmeasured fixture row
EOF
}
write_ops() { # write_ops <file> <corpus-id-to-wire>
  cat > "$1" <<EOF
# fixture ops
op	corpus_id	kind
fixture-op	$2	semantic
EOF
}
write_manifest "$TMP/man.tsv"

# 10. (R8) measured row wired -> GREEN
write_ops "$TMP/ops-ok.tsv" fixture__op_measured
"$LINT" "$TMP/ok.conf" "$TMP/ok.tsv" "$TMP/man.tsv" "$TMP/ops-ok.tsv" > "$TMP/out10.log" 2>&1
check "R8 measured-row-wired lints GREEN" 0 $?

# 11. (R8) measured row NOT wired -> RED naming the missing id
write_ops "$TMP/ops-miss.tsv" fixture__op_idle
"$LINT" "$TMP/ok.conf" "$TMP/ok.tsv" "$TMP/man.tsv" "$TMP/ops-miss.tsv" > "$TMP/out11.log" 2>&1
check "R8 measured-row-unwired refuses RED" 1 $?
grep -q "\[R8\]" "$TMP/out11.log" || { echo "SELFTEST FAIL: R8 RED lacks the rule tag"; overall=1; }
grep -q "fixture__op_measured" "$TMP/out11.log" || { echo "SELFTEST FAIL: R8 RED does not name the missing corpus id"; overall=1; }

# 12. (R9) missing witness table -> RED naming R9
grep -v "_REVIEWED_FIRE_WITNESSES\|ccmask" "$TMP/ok.conf" > "$TMP/nowit.conf"
"$LINT" "$TMP/nowit.conf" "$TMP/ok.tsv" > "$TMP/out12.log" 2>&1
check "R9 missing witness table refuses RED" 1 $?
grep -q "\[R9\]" "$TMP/out12.log" || { echo "SELFTEST FAIL: missing-table RED lacks the R9 tag"; overall=1; }

# 13. (R9) malformed witness row (3 fields) -> RED
sed 's/^-mtt-tensix-optimize-ccmask|perf_fixture.py::node\[a\]|-fdump-tree-rvtt_ccmask|ccmask: folded zeroing CC region$/-mtt-tensix-optimize-ccmask|perf_fixture.py::node[a]|-fdump-tree-rvtt_ccmask/' \
  "$TMP/ok.conf" > "$TMP/badwit.conf"
"$LINT" "$TMP/badwit.conf" "$TMP/ok.tsv" > "$TMP/out13.log" 2>&1
check "R9 malformed witness row refuses RED" 1 $?
grep -q "4 non-empty" "$TMP/out13.log" || { echo "SELFTEST FAIL: malformed-row RED lacks the field diagnosis"; overall=1; }

# 14. (R9) witnessed flag not in the reviewed ON set -> RED naming the flag
sed 's/^-mtt-tensix-optimize-ccmask|/-mtt-tensix-optimize-not-a-real-flag|/' \
  "$TMP/ok.conf" > "$TMP/offsetwit.conf"
"$LINT" "$TMP/offsetwit.conf" "$TMP/ok.tsv" > "$TMP/out14.log" 2>&1
check "R9 witness for a non-ON-set flag refuses RED" 1 $?
grep -q "not-a-real-flag" "$TMP/out14.log" || { echo "SELFTEST FAIL: non-ON-set RED does not name the flag"; overall=1; }

# 15. (R9) exact duplicate witness row -> RED
awk '1; /^-mtt-tensix-optimize-ccmask\|/ && !d {print; d=1}' "$TMP/ok.conf" > "$TMP/dupwit.conf"
"$LINT" "$TMP/dupwit.conf" "$TMP/ok.tsv" > "$TMP/out15.log" 2>&1
check "R9 duplicate witness row refuses RED" 1 $?
grep -q "duplicate witness" "$TMP/out15.log" || { echo "SELFTEST FAIL: dup-row RED lacks the duplicate diagnosis"; overall=1; }

# R10 fixtures (wave-9 quarantine): the quarantined table is optional; when
# present its flags must NOT be in the real sweep_2x2.py ON set and must not
# duplicate a reviewed-table row.  int-abs is a real non-ON-set flag
# (pin-14 knob-leg-only, deliberately outside the reviewed ON set);
# ccmask is a real ON-set flag.  (crosscall-hoist served as the fixture
# quarantined flag until the pin-14 lift returned it to the ON set.)
add_quarantine() { # add_quarantine <src> <dst> <flag>
  cp "$1" "$2"
  cat >> "$2" <<EOF
_QUARANTINED_FIRE_WITNESSES="
$3|perf_fixture.py::node[q]|-fdump-tree-rvtt_crosscall|hoisted 6 contract materializations
"
EOF
}

# 16. (R10) quarantined table whose flag is outside the ON set -> GREEN
add_quarantine "$TMP/ok.conf" "$TMP/q-ok.conf" "-mtt-tensix-optimize-int-abs"
"$LINT" "$TMP/q-ok.conf" "$TMP/ok.tsv" > "$TMP/out16.log" 2>&1
check "R10 quarantined non-ON-set flag lints GREEN" 0 $?

# 17. (R10) quarantined flag that IS in the reviewed ON set -> RED naming R10
add_quarantine "$TMP/ok.conf" "$TMP/q-viol.conf" "-mtt-tensix-optimize-ccmask"
"$LINT" "$TMP/q-viol.conf" "$TMP/ok.tsv" > "$TMP/out17.log" 2>&1
check "R10 quarantined flag in the ON set refuses RED" 1 $?
grep -q "\[R10\]" "$TMP/out17.log" || { echo "SELFTEST FAIL: quarantine-violation RED lacks the R10 tag"; overall=1; }
grep -q "quarantine violated" "$TMP/out17.log" || { echo "SELFTEST FAIL: quarantine-violation RED lacks the violation diagnosis"; overall=1; }

# 18. (R10) flag carrying rows in BOTH tables -> RED (ccmask is in the
# fixture reviewed table; quarantining it also trips the ON-set rule, so
# assert the both-tables diagnosis specifically)
grep -q "appears in BOTH" "$TMP/out17.log" || { echo "SELFTEST FAIL: both-tables RED lacks the dual-listing diagnosis"; overall=1; }

if [ "$overall" -eq 0 ]; then
  echo "conf-lint self-test: ALL GREEN (coherent->GREEN, stale-prose->RED, stale-(CURRENT)->RED, dup-(CURRENT)->RED, stale-anchor->RED, no-anchor->RED, bad-sim-pin->RED, shipping-state->GREEN, bogus-llk-base->RED, R8-wired->GREEN, R8-unwired->RED, R9 missing/malformed/non-ON/dup->RED, R10 quarantine ok->GREEN / ON-set-violation+both-tables->RED)"
else
  echo "conf-lint self-test: FAILED"
fi
exit $overall
