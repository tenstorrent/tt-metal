#!/usr/bin/env bash
# Host-only parser regression: no toolchain, simulator, or hardware.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=sweep_args_lib.sh
source "$HERE/sweep_args_lib.sh" || exit 2
FAILS=0

fail() { echo "  FAIL: $1"; FAILS=$((FAILS + 1)); }
pass() { echo "  PASS: $1"; }

consume_evidence_root_args --ops exp --evidence-root "/tmp/evidence root" --phases classify
if [ "$EVIDENCE_ROOT_OVERRIDE" = "/tmp/evidence root" ] \
   && [ "${EVIDENCE_ARGS[*]}" = "--ops exp --phases classify" ]; then
  pass "space form is consumed and unrelated args are preserved"
else
  fail "space form parse"
fi

consume_evidence_root_args --dry-run --evidence-root=/tmp/exact-root --force
if [ "$EVIDENCE_ROOT_OVERRIDE" = /tmp/exact-root ] \
   && [ "${EVIDENCE_ARGS[*]}" = "--dry-run --force" ]; then
  pass "equals form is consumed"
else
  fail "equals form parse"
fi

consume_evidence_root_args --ops exp
if [ -z "$EVIDENCE_ROOT_OVERRIDE" ] && [ "${EVIDENCE_ARGS[*]}" = "--ops exp" ]; then
  pass "absent override preserves default selection"
else
  fail "default parse"
fi

if consume_evidence_root_args --evidence-root >/dev/null 2>&1; then
  fail "missing value accepted"
else
  pass "missing value refused"
fi
if consume_evidence_root_args --evidence-root= >/dev/null 2>&1; then
  fail "empty equals value accepted"
else
  pass "empty equals value refused"
fi
if consume_evidence_root_args --evidence-root /tmp/a --evidence-root=/tmp/b >/dev/null 2>&1; then
  fail "duplicate override accepted"
else
  pass "duplicate override refused"
fi

if [ "$FAILS" -eq 0 ]; then
  echo "sweep argument self-test: ALL PASS"
  exit 0
fi
echo "sweep argument self-test: $FAILS FAILURE(S)"
exit 1
