#!/usr/bin/env bash
# setup_state_lib.sh -- fail-closed validation of craq-sfpi SETUP-STATE.env.
# Sourced by sweep.sh and its host-only self-test.

# verify_craq_setup_state STATE_FILE
#
# Every recorded checkout must still be at its recorded commit.  SFPI,
# tt-metal and tt-blaze must remain on the campaign branch.  GCC may be
# detached because setup.sh pins an immutable fetched commit; a detached GCC
# is accepted only when its HEAD, the recorded GCC SHA and SFPI's gcc gitlink
# are identical.
verify_craq_setup_state() {
  local state=$1 setup_bad=0 v sha br dir actual_sha actual_br sfpi_gcc

  if [ ! -f "$state" ]; then
    echo "FATAL: no setup state at $state" >&2
    return 2
  fi

  # shellcheck source=/dev/null
  . "$state"

  for v in SFPI GCC METAL BLAZE; do
    eval "sha=\${CRAQ_${v}_SHA:-}"
    eval "br=\${CRAQ_${v}_BRANCH:-}"
    eval "dir=\${CRAQ_${v}_DIR:-}"
    if [ -z "$sha" ] || [ "$sha" = "-" ] || [ -z "$dir" ] || \
       ! git -C "$dir" rev-parse --git-dir >/dev/null 2>&1; then
      echo "FATAL: setup state has no usable checkout for $v (dir='$dir')" >&2
      setup_bad=1
      continue
    fi

    actual_sha=$(git -C "$dir" rev-parse HEAD 2>/dev/null) || actual_sha=""
    actual_br=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null) || actual_br=""
    if [ "$actual_sha" != "$sha" ]; then
      echo "FATAL: $v checkout moved: state=$sha actual=${actual_sha:--}" >&2
      setup_bad=1
    fi

    if [ "$v" = GCC ]; then
      if [ "$br" != "$actual_br" ]; then
        echo "FATAL: GCC branch state is stale: state='$br' actual='${actual_br:--}'" >&2
        setup_bad=1
      elif [ "$actual_br" != HEAD ] && [ -n "${CRAQ_SETUP_BRANCH:-}" ] && \
           [ "$actual_br" != "$CRAQ_SETUP_BRANCH" ]; then
        echo "FATAL: GCC is on '$actual_br', not '$CRAQ_SETUP_BRANCH' or detached HEAD" >&2
        setup_bad=1
      fi
    elif [ -n "${CRAQ_SETUP_BRANCH:-}" ] && \
         { [ "$br" != "$CRAQ_SETUP_BRANCH" ] || [ "$actual_br" != "$CRAQ_SETUP_BRANCH" ]; }; then
      echo "FATAL: $v is on state='$br' actual='${actual_br:--}', not '$CRAQ_SETUP_BRANCH'" >&2
      setup_bad=1
    fi
  done

  sfpi_gcc=$(git -C "${CRAQ_SFPI_DIR:-/nonexistent}" rev-parse HEAD:gcc 2>/dev/null) || sfpi_gcc=""
  if [ -z "${CRAQ_GCC_SHA:-}" ] || [ "$sfpi_gcc" != "$CRAQ_GCC_SHA" ]; then
    echo "FATAL: SFPI gcc gitlink ${sfpi_gcc:--} != recorded GCC ${CRAQ_GCC_SHA:--}" >&2
    setup_bad=1
  fi
  if [ -n "${CRAQ_GCC_PIN_SHA:-}" ] && [ "${CRAQ_GCC_PIN_SHA}" != "${CRAQ_GCC_SHA:-}" ]; then
    echo "FATAL: recorded GCC pin ${CRAQ_GCC_PIN_SHA} != recorded GCC ${CRAQ_GCC_SHA:--}" >&2
    setup_bad=1
  fi

  if [ ! -x "${CRAQ_COMPILER:-}" ]; then
    echo "FATAL: no built craq compiler at '${CRAQ_COMPILER:-}' (run setup.sh --stage sfpi)" >&2
    setup_bad=1
  fi

  [ "$setup_bad" = 0 ] || {
    echo "  re-run: bash craq-sfpi/scripts/setup.sh" >&2
    return 2
  }
  return 0
}
