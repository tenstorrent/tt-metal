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
#   9. (R7) a measured corpus row wired into the ops TSV -> GREEN;
#  10. (R7) a measured corpus row with NO ops TSV row -> RED (the
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
EOF
}

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

# R7 fixtures: a two-row manifest (one measured, one not_run) and an ops TSV.
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

# 9. measured row wired -> GREEN
write_ops "$TMP/ops-ok.tsv" fixture__op_measured
"$LINT" "$TMP/ok.conf" "$TMP/ok.tsv" "$TMP/man.tsv" "$TMP/ops-ok.tsv" > "$TMP/out9.log" 2>&1
check "R7 measured-row-wired lints GREEN" 0 $?

# 10. measured row NOT wired -> RED naming the missing id
write_ops "$TMP/ops-miss.tsv" fixture__op_idle
"$LINT" "$TMP/ok.conf" "$TMP/ok.tsv" "$TMP/man.tsv" "$TMP/ops-miss.tsv" > "$TMP/out10.log" 2>&1
check "R7 measured-row-unwired refuses RED" 1 $?
grep -q "\[R7\]" "$TMP/out10.log" || { echo "SELFTEST FAIL: R7 RED lacks the rule tag"; overall=1; }
grep -q "fixture__op_measured" "$TMP/out10.log" || { echo "SELFTEST FAIL: R7 RED does not name the missing corpus id"; overall=1; }

if [ "$overall" -eq 0 ]; then
  echo "conf-lint self-test: ALL GREEN (coherent->GREEN, stale-prose->RED, stale-(CURRENT)->RED, dup-(CURRENT)->RED, stale-anchor->RED, no-anchor->RED, bad-sim-pin->RED, shipping-state->GREEN, R7-wired->GREEN, R7-unwired->RED)"
else
  echo "conf-lint self-test: FAILED"
fi
exit $overall
