#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth pass, merged head: exhaustive bit dumps of the operand pass and the native classes.
cd /work
EB_RUN_LIMIT=10000 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/r9b_dump.spec
