#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass: exhaustive bit dumps of the switch's operand-pass block pack in the broadcast sections.
cd /work
EB_RUN_LIMIT=10000 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/sw5_dump.spec
