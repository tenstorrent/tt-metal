#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
cd /work
EB_RUN_LIMIT=3000 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/spec_ilv.txt
