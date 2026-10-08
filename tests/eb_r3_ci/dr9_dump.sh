#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58724 fourth review): exhaustive bit dump of the dest-reuse forms on 16x32, 16x16, 8x32 and 32x32 tiles,
# per-face against per-tile, with the one-face rule (16x16 takes the per-face program under both). P150.
cd /work
EB_ELFAB="eb_dump_reuse=EB_DUMP_PER_TILE" EB_RUN_LIMIT=5400 bash tests/eb_r3_ci/dump_run.sh tests/eb_r3_ci/spec_dr9.txt
