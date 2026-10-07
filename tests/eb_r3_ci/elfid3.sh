#!/usr/bin/env bash
# Round 3 eltwise binary, third pass: ELF identity of this branch's device code against origin/main (main_dev3.tar.gz: main's
# version of every device file the branch changes), the main side swapped into both /work and the installed package, each side
# from an empty kernel cache, the same host. Positive control: the kernels the branch opts in must differ (Blackhole).
cd /work
export PYTHONPATH=/work:/work/tests/eb_r3_ci:${PYTHONPATH:-}
RT=$(python3 -c "import ttnn, os; print(os.path.dirname(ttnn.__file__))")
E=tests/ttnn/unit_tests/operations/eltwise
FU=tests/ttnn/unit_tests/operations/fused
SEL="$E/test_add.py $E/test_mul.py $E/test_binary_bcast.py $E/test_binary_scalar.py $E/test_binary_ng_activation_mixed_dtype.py $FU/test_softmax.py $FU/test_layer_norm.py $FU/test_rms_norm.py $FU/test_group_norm.py"
for side in branch main; do
  [[ $side == main ]] && python3 tests/eb_r3_ci/swap_tree.py tests/eb_r3_ci/main_dev3.tar.gz "$RT"
  export TT_METAL_CACHE=/tmp/elfid3_$side; rm -rf $TT_METAL_CACHE; mkdir -p $TT_METAL_CACHE
  timeout 2700 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread $SEL > /tmp/elfid3_$side.txt 2>&1
  echo "##### $side: $(grep -E 'passed|failed' /tmp/elfid3_$side.txt | tail -1)"
done
python3 tests/eb_r3_ci/elf_key_diff.py /tmp/elfid3_main /tmp/elfid3_branch
