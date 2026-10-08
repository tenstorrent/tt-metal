#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, final head merged with main: the kernels that keep an opt-in define, every define removed against
# kept, their tests seeded with outputs compared (the r9 selection, ids_bits.txt, where those kernels run).
cd /work
wc -l < tests/eb_r3_ci/r12/kern_final.txt
EB_NODES_FILE=/work/tests/eb_r3_ci/r9opt/ids_bits.txt bash tests/eb_r3_ci/bits_strip.sh tests/eb_r3_ci/r12/kern_final.txt -p eb_select_plugin -p eb_unskip_plugin -p eb_seed_plugin $(cat tests/eb_r3_ci/r12/bits_files.txt)
