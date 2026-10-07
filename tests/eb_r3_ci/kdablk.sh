#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (#58722 review): the KDA kernels that call add_block / sub_block / mul_block, with the per-tile hand-off
# against the per-tile hand-off plus the block unpack (ELTWISE_BINARY_BLOCK_UNPACK), three passes; then bits.
cd /work
for i in 1 2 3; do
  echo "##### pass $i T vs T+block unpack"; bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_kdablk.txt --nodes-file tests/eb_r3_ci/nodes_kda.txt
done
export EB_NODES_FILE=$(readlink -f tests/eb_r3_ci/nodes_kda.txt)
for i in 1 2; do echo "##### pass $i main vs T"; bash tests/eb_r3_ci/ab_off.sh tests/eb_r3_ci/optin_kdat.txt -p eb_select_plugin $(cut -d: -f1 tests/eb_r3_ci/nodes_kda.txt | sort -u | grep -v sigmoid); done
unset EB_NODES_FILE
echo "##### bits"; bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_kdablk.txt -p eb_seed_plugin tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py
