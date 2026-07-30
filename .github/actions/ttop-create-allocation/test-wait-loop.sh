#!/usr/bin/env bash
#
# Regression test for the "Wait for allocation" poll loop in action.yaml.
#
#   ./test-wait-loop.sh
#
# No cluster, no credentials, no network: fake kubectl/curl/jq go on PATH and the
# loop body is lifted straight out of action.yaml, so this tests the shipped code
# rather than a copy that can drift.
#
# Each scenario reproduces a way the loop has actually failed or could fail. Every
# one of the first three HANGS on the pre-2026-07-30 version of this action, which
# is why they are here: the bug class is "the loop spins with its success condition
# already met, or with an unrecoverable error it will never classify", and nothing
# in CI could see it. tt-blaze run 30573758666 burned 79 minutes on a superpod that
# way and was only diagnosed by hand from kubectl after the fact.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="$HERE/action.yaml"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Lift the `run:` block of the "Wait for allocation" step. awk rather than a YAML
# parser so this needs nothing but bash+awk -- a test that is awkward to run is a
# test nobody runs.
awk '
  /^    - name: Wait for allocation$/ { instep = 1; next }
  instep && /^      run: \|$/         { inrun = 1; next }
  inrun {
    if ($0 != "" && $0 !~ /^        /) exit   # dedent => end of the block
    sub(/^        /, "")
    print
  }
' "$ACTION" > "$WORK/loop.sh"

[ -s "$WORK/loop.sh" ] || { echo "FAIL: could not extract the wait loop from $ACTION"; exit 1; }

# Speed the loop up and shrink the give-up threshold so the suite runs in seconds.
sed -i.bak -e 's/^POLL_INTERVAL=.*/POLL_INTERVAL=1/' \
           -e 's/^MAX_CONSECUTIVE_ERRORS=.*/MAX_CONSECUTIVE_ERRORS=4/' \
           -e 's/^HEARTBEAT_EVERY=.*/HEARTBEAT_EVERY=2/' "$WORK/loop.sh"

mkdir -p "$WORK/bin"

# curl is only ever piped into jq here; jq is where mint success/failure is decided.
printf '#!/usr/bin/env bash\ncat >/dev/null 2>&1 || true\necho "{}"\n' > "$WORK/bin/curl"

# Minting returns a fresh token each call, or nothing once SCENARIO says it dies.
cat > "$WORK/bin/jq" <<'EOF'
#!/usr/bin/env bash
cat >/dev/null
m=$(cat "$MINTS" 2>/dev/null || echo 0); m=$((m + 1)); echo "$m" > "$MINTS"
if [ "$SCENARIO" = "mint_dies" ] && [ "$m" -gt 1 ]; then echo ""; exit 0; fi
echo "tok-$m"
EOF

cat > "$WORK/bin/kubectl" <<'EOF'
#!/usr/bin/env bash
tok=""; prev=""
for a in "$@"; do [ "$prev" = "--token" ] && tok="$a"; prev="$a"; done
n=$(cat "$COUNTER" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" > "$COUNTER"
[ -s "$FIRSTTOK" ] || printf '%s' "$tok" > "$FIRSTTOK"
first="$(cat "$FIRSTTOK")"

case "$SCENARIO" in
  # A warning on an otherwise-SUCCESSFUL call. Observed live in run 30573758666 at
  # 19:41 and 19:46. Fails if the phase capture merges stderr (`2>&1`).
  stderr_noise)
    echo 'E0730 19:41:24.707535 3470 memcache.go:265] "Unhandled Error" err="couldn'"'"'t get current server API group list"' >&2
    if [ "$n" -ge 3 ]; then printf 'Allocated'; else printf 'NotAvailable'; fi ;;

  # The real 30573758666 failure: once minting dies the token goes empty, and
  # `kubectl --token ""` does NOT error -- it silently falls back to ambient
  # in-cluster credentials (verified: `kubectl --token "" auth whoami` exits 0 and
  # reports the kubeconfig identity). On that runner it landed on the pod's own
  # service account, which has no allocation RBAC, so every poll was Forbidden.
  mint_dies)
    if [ -z "$tok" ]; then
      echo 'Error from server (Forbidden): allocations.tenstorrent.com "a" is forbidden: User "system:serviceaccount:ttop-ci:multihost-with-nfs" cannot get resource "allocations" in API group "tenstorrent.com" in the namespace "ttop-ci"' >&2
      exit 1
    fi
    if [ "$n" -gt 2 ]; then
      echo 'error: the server has asked for the client to provide credentials' >&2; exit 1
    fi
    printf 'NotAvailable' ;;

  # Token expires mid-wait and ONLY a reissued one is accepted. Fails if reissue is
  # gated on an error-message regex, since this text matches no obvious auth pattern.
  expired_token)
    if [ "$n" -gt 2 ] && [ "$tok" = "$first" ]; then
      echo 'error: the server has asked for the client to provide credentials' >&2; exit 1
    fi
    if [ "$n" -gt 2 ]; then printf 'Allocated'; else printf 'NotAvailable'; fi ;;

  # Unrecoverable. Must give up rather than spin to the caller's step timeout.
  always_fail)
    echo 'Error from server (Forbidden): nope' >&2; exit 1 ;;

  # Ordinary happy path -- guards against the fix breaking normal operation.
  happy)
    if [ "$n" -ge 2 ]; then printf 'Allocated'; else printf 'NotAvailable'; fi ;;
esac
EOF
chmod +x "$WORK/bin/"*

failures=0

run_scenario() {
  local scenario="$1" want_rc="$2" want_text="$3"
  local out="$WORK/$scenario.out"

  (
    SCENARIO="$scenario" \
    COUNTER="$(mktemp)" MINTS="$(mktemp)" FIRSTTOK="$(mktemp)" \
    NAMESPACE=ttop-ci ALLOCATION_NAME=a \
    ACTIONS_ID_TOKEN_REQUEST_TOKEN=x ACTIONS_ID_TOKEN_REQUEST_URL=http://example.invalid \
    PATH="$WORK/bin:/usr/bin:/bin" \
    bash -eo pipefail "$WORK/loop.sh"
  ) > "$out" 2>&1 &
  local pid=$! waited=0

  # Not `timeout`: absent on macOS, where this gets run before it gets pushed.
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 20 ]; do sleep 1; waited=$((waited + 1)); done
  local rc verdict
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    rc="hung"; verdict="HUNG after ${waited}s -- the loop never reached a decision"
  else
    wait "$pid"; rc=$?; verdict="exit $rc"
  fi

  if [ "$rc" != "$want_rc" ]; then
    printf 'FAIL  %-14s expected exit %s, got %s\n' "$scenario" "$want_rc" "$verdict"
    sed 's/^/        /' "$out" | tail -6
    failures=$((failures + 1))
    return
  fi
  if [ -n "$want_text" ] && ! grep -q "$want_text" "$out"; then
    printf 'FAIL  %-14s exit %s as expected, but no %s in output\n' "$scenario" "$want_rc" "$want_text"
    sed 's/^/        /' "$out" | tail -6
    failures=$((failures + 1))
    return
  fi
  printf 'ok    %-14s exit %s\n' "$scenario" "$rc"
}

echo "== ttop-create-allocation: wait-loop scenarios =="
run_scenario happy         0 '::notice::'
run_scenario stderr_noise  0 '::notice::'
run_scenario expired_token 0 '::notice::'
run_scenario mint_dies     1 'Forbidden'
run_scenario always_fail   1 '::error::'

echo
if [ "$failures" -ne 0 ]; then
  echo "$failures scenario(s) failed"
  exit 1
fi
echo "all scenarios passed"
