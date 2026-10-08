#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: the head merged with origin/main 304e70ceb65, whose #59697 rewrote
# layernorm_pre_allgather_2d.cpp's merge-core path (the 2D hang fix). Its modules whole, the per-tile define removed against
# kept, bit for bit, and the 2D core grid cases' device time, three passes.
cd /work
bash tests/eb_r3_ci/ln2d.sh
