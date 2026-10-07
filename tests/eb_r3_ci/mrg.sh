#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, third pass: the opted-in kernels that main changed since the last merge (#57495's hw_startup cleanup,
# #59364's QKV conv), their test modules whole on the merged head, the per-tile defines removed against as committed.
cd /work
U=tests/ttnn/unit_tests/operations
N=tests/ttnn/nightly/unit_tests/operations
echo "##### merge: opted-in kernels main changed"; EB_SHOW_ERR=1 bash tests/eb_r3_ci/bits_strip.sh tests/eb_r3_ci/mrg_files.txt -p eb_seed_plugin $U/fused/test_layer_norm.py $U/fused/test_layer_norm_sharded.py $U/reduce/test_moe.py $U/reduce/test_manual_seed.py $U/reduce/test_tiebreak_input_adjust.py $N/reduction/test_deepseek_grouped_gate.py $N/experimental/kda/test_qkv_causal_conv1d_silu.py $U/fused/test_distributed_layernorm_sharded.py
