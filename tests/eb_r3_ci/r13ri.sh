#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth review (#58725 at gemma4_d_p's multi-section post-activation add): outputs.
cd /work
EB_RUN_LIMIT=3000 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/spec_ri.txt
