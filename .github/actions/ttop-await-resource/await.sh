#!/usr/bin/env bash
# OIDC-authenticated kubectl loop: poll a resource until ready, or delete it.
#
# Reads everything from the environment: NAMESPACE, RESOURCE, MODE, JSONPATH,
# EXPECTED, OVERALL_TIMEOUT, RETRIES.

set -eo pipefail

POLL_INTERVAL=5
# Was MAX_AUTH_FAILURES=3; any failure now counts, and 3 would abort on a
# 15s blip. Bounds a hopeless case to ~5 min vs timeoutSeconds (999m).
MAX_CONSECUTIVE_ERRORS=60
# Default 5 min: long enough to ride out a token expiry / transient blip,
# short enough not to pin a runner. Override via the timeoutSeconds input.
OVERALL_TIMEOUT="${OVERALL_TIMEOUT:-300}"
RETRIES="${RETRIES:-3}"
# Mirrors the namespace input default, for callers invoking this script directly.
NAMESPACE="${NAMESPACE:-ttop-ci}"

case "$MODE" in poll|delete) ;; *) echo "::error::mode must be poll or delete"; exit 1 ;; esac
if [[ "$MODE" == poll && ( -z "$JSONPATH" || -z "$EXPECTED" ) ]]; then
  echo "::error::poll mode requires jsonpath and expected"; exit 1
fi

# Mints a fresh OIDC token into IDTOKEN and masks it, so every caller gets
# a masked token without having to remember to mask it. Retries in-place so
# a brief token-endpoint blip (e.g. a DNS "could not resolve host", exit 6 —
# the failure that left allocations dangling) is absorbed.
mint_token() {
  local t i
  for ((i=1; i<=RETRIES; i++)); do
    t=$(curl -s --max-time 30 \
      -H "Authorization: Bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
      "$ACTIONS_ID_TOKEN_REQUEST_URL" \
      -H "Accept: application/json; api-version=2.0" \
      -H "Content-Type: application/json" -d "{}" | jq -r '.value') || t=""
    if [[ -n "$t" && "$t" != "null" ]]; then
      IDTOKEN="$t"
      echo "::add-mask::${IDTOKEN}"
      return 0
    fi
    sleep 2
  done
  IDTOKEN=""
  return 1
}

mint_token || true

errors=0
start=$SECONDS
errfile=$(mktemp)
while true; do
  if (( SECONDS - start >= OVERALL_TIMEOUT )); then
    echo "::error::Timed out after $((OVERALL_TIMEOUT / 60))m waiting for ${RESOURCE} to be (${MODE})."
    exit 1
  fi

  # A failed mint leaves IDTOKEN empty; kubectl --token "" prints a
  # confusing "Please enter Username" error that would NOT match the
  # auth branch below, so we'd never re-mint. Re-mint here instead of
  # ever calling kubectl with an empty token.
  if [[ -z "$IDTOKEN" ]]; then
    echo "::warning::No valid token yet; (re)minting before the ${MODE}."
    mint_token || true
    if [[ -z "$IDTOKEN" ]]; then
      sleep "$POLL_INTERVAL"
      continue
    fi
  fi

  # stderr to a file, NOT 2>&1: merged, a warning on a successful call lands
  # in $output and the $EXPECTED compare below fails on a ready resource.
  case "$MODE" in
    delete)
      output=$(kubectl --token "$IDTOKEN" -n "$NAMESPACE" delete "$RESOURCE" --ignore-not-found 2>"$errfile") && rc=0 || rc=$?
      if ((rc==0)); then
        echo "::notice::$RESOURCE is deleted."
        exit 0
      fi
      ;;
    poll)
      output=$(kubectl --token "$IDTOKEN" -n "$NAMESPACE" get "$RESOURCE" -o jsonpath="$JSONPATH" 2>"$errfile") && rc=0 || rc=$?
      if (( rc == 0 )); then
        errors=0                                         # successful read → reset the failure streak
        if [[ "$output" == "$EXPECTED" ]]; then
          echo "::notice::$RESOURCE is ready ($EXPECTED) after $(( (SECONDS - start) / 60 ))m."; exit 0
        fi
        echo "[info] $RESOURCE not ready (got '${output:-<empty>}', want '$EXPECTED') — waiting..."
        sleep "$POLL_INTERVAL"; continue                 # <-- the "waiting" case: loop, DON'T fall into errors
      fi
      ;;                                                 # rc != 0 falls through to the shared error path
  esac

  # Re-mint on ANY failure, no error-message classifier. An expired token
  # often says "the server has asked for the client to provide credentials",
  # which the old auth regex missed, so it kept polling with a dead token.
  # The empty-token guard above can't catch it: not empty, just dead.
  # The token lives ~5 min, so this path runs on every non-trivial wait.
  errors=$((errors + 1))
  echo "::warning::${MODE} failed on ${RESOURCE} (${errors}/${MAX_CONSECUTIVE_ERRORS}, rc=${rc}); reissuing token: $(tr '\n' ' ' < "$errfile" | cut -c1-300)"
  if (( errors >= MAX_CONSECUTIVE_ERRORS )); then
    echo "::error::Giving up on ${RESOURCE} after ${errors} consecutive failures ($(( errors * POLL_INTERVAL ))s). Last error: $(tr '\n' ' ' < "$errfile" | cut -c1-300)"
    exit 1
  fi
  mint_token || true
  sleep "$POLL_INTERVAL"
done
