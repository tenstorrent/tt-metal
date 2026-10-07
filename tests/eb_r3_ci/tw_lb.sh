#!/usr/bin/env bash
# Round 3 eltwise binary (#58723 review, twin round): Blackhole LoudBox with the JIT reading every header from /work
# (TT_METAL_RUNTIME_ROOT): zero_padded_kv_cache (CI's torus-y-8x1 and fabric2d-2x4 cases; its TILE path runs the compute
# kernel) and ring joint MLA SDPA (CI's 4x2 pcc_check fabric2d selection; CI unset, which the test's uncollect_if keys on),
# device time A/B with the opt-in, three passes each, and bits.
cd /work
export TT_METAL_RUNTIME_ROOT=/work EB_SHOW_ERR=1
ZP=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_zero_padded_kv_cache.py
RJ=models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_joint_mla.py
for p in 1 2 3; do
  echo "##### zpkv pass $p"; EB_RUN_LIMIT=900 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_zpkv.txt $ZP -k "torus-y-8x1 or fabric2d-2x4"
  [[ $p == 1 ]] && { echo "##### elf zpkv"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebset/cache_main /tmp/ebset/cache_optin 2>&1 | grep -v "ELFSET identical" | head -30; }
done
echo "##### bits zpkv"; EB_RUN_LIMIT=900 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_zpkv.txt -p eb_seed_plugin $ZP -k "torus-y-8x1 or fabric2d-2x4"
echo "##### ring mla pass 1"; env -u CI -u TT_GH_CI_INFRA EB_REPS=2 EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "test_mla_sdpa and 4x2 and pcc_check and fabric2d and single_run"
echo "##### elf ring mla"; python3 tests/eb_r3_ci/elf_set_diff.py /tmp/ebset/cache_main /tmp/ebset/cache_optin 2>&1 | grep -v "ELFSET identical" | head -30
echo "##### bits ring mla"; env -u CI -u TT_GH_CI_INFRA EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/bits_ab.sh tests/eb_r3_ci/optin_ring.txt -p eb_seed_plugin $RJ -k "test_mla_sdpa and 4x2 and pcc_check and fabric2d and single_run"
for p in 2 3; do
  echo "##### ring mla pass $p"; env -u CI -u TT_GH_CI_INFRA EB_REPS=2 EB_RUN_LIMIT=1200 bash tests/eb_r3_ci/ab_set.sh tests/eb_r3_ci/optin_ring.txt $RJ -k "test_mla_sdpa and 4x2 and pcc_check and fabric2d and single_run"
done
