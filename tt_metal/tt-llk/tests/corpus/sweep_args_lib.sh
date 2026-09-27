#!/usr/bin/env bash
# Parse the one engine option whose value the wrapper must also own.
# Outputs globals EVIDENCE_ROOT_OVERRIDE and EVIDENCE_ARGS.

consume_evidence_root_args() {
  EVIDENCE_ROOT_OVERRIDE=""
  EVIDENCE_ARGS=()
  local value=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --evidence-root)
        if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ $2 == --* ]]; then
          echo "FATAL: --evidence-root requires a non-empty path" >&2
          return 2
        fi
        value=$2
        shift 2
        ;;
      --evidence-root=*)
        value=${1#--evidence-root=}
        if [ -z "$value" ]; then
          echo "FATAL: --evidence-root requires a non-empty path" >&2
          return 2
        fi
        shift
        ;;
      *) EVIDENCE_ARGS+=("$1"); shift ;;
    esac
    if [ -n "$value" ]; then
      if [ -n "$EVIDENCE_ROOT_OVERRIDE" ]; then
        echo "FATAL: --evidence-root may be specified only once" >&2
        return 2
      fi
      EVIDENCE_ROOT_OVERRIDE=$value
      value=""
    fi
  done
}
