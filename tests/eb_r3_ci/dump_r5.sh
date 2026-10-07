#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: exhaustive bit dumps of the PR's two binary_ng changes on the head (ci9).
cd /work
EB_RUN_LIMIT=10000 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/r5_dump.spec
