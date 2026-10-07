#!/usr/bin/env bash
# Round 3 eltwise binary (#58722 review): ELF identity of the block unpack's LLK and compute API additions when no kernel uses
# them: the same tests (block section off) built with the three headers before and after the change, every ELF compared.
cd /work
export EB_R3_NO_BLOCK=1 PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
E=tests/ttnn/unit_tests/operations/eltwise
K=tests/ttnn/nightly/unit_tests/operations/experimental/kda
SEL="$E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binaryng_fp32.py $K/test_prepare_chunk_recurrence.py $K/test_recurrent_chunk_scan.py tests/ttnn/unit_tests/operations/fused/test_softmax.py tests/eb_r3_ci/test_eb_block2.py"
for side in post pre; do
  [[ $side == pre ]] && tar xzf tests/eb_r3_ci/pre_blk.tar.gz -C /work
  export TT_METAL_CACHE=/tmp/elfid_$side; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
  timeout 2400 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $SEL > /tmp/elfid_$side.txt 2>&1
  echo "##### $side: $(grep -E 'passed|failed' /tmp/elfid_$side.txt | tail -1)"
done
python3 tests/eb_r3_ci/elf_cache_diff.py /tmp/elfid_pre /tmp/elfid_post
