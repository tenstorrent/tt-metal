#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, fourth pass: the opted-in kernel main changed since the last merge (chunk_gated_delta_rule.cpp, with
# chunk_gdn_math.hpp), its module whole on the merged head, the per-tile defines removed against as committed.
cd /work
printf '%s\n' ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gated_delta_rule.cpp ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_prep.cpp > /tmp/mrg2_files.txt
F=$(grep -rlE "chunk_gated_delta_rule|chunk_gdn" tests/ttnn --include=test_*.py | tr '\n' ' ')
echo "modules: $F"
echo "##### merge: chunk_gated_delta_rule"; EB_SHOW_ERR=1 EB_RUN_LIMIT=3300 bash tests/eb_r3_ci/bits_strip.sh /tmp/mrg2_files.txt -p eb_seed_plugin $F
