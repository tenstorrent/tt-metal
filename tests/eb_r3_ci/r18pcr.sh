#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, re-review of 20:55: prepare_chunk_recurrence.cpp (#59473 on main) with the PR's defines removed
# against kept, on the PR head merged with main bad2b9d0b79, its test module seeded with outputs compared.
cd /work
bash tests/eb_r3_ci/bits_strip.sh tests/eb_r3_ci/r18/kern_pcr.txt -p eb_unskip_plugin -p eb_seed_plugin tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence.py
