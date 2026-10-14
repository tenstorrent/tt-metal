#!/usr/bin/env bash
# #58817 legacy bmm port: in-process alternation of the opt-ins (three processes), then the guard.
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 ALT_ROUNDS=${ALT_ROUNDS:-11} ALT_CALLS=${ALT_CALLS:-8} CI_LIMIT=1500
bash scratch_r3/run_ci.sh scratch_r3/plan_leg_alt.txt
