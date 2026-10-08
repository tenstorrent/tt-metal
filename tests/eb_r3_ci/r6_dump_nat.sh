#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fifth pass, final code: exhaustive bit dumps (nat).
cd /work
EB_RUN_LIMIT=10000 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/r6_dump_nat.spec
