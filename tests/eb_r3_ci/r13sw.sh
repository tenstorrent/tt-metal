#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary (ci16sw = final head + origin/main + #58816 + the switch): binary_ng's block sections on #58816's
# pack_block_mop (add and sub at any size) against the head's rule with main's pack (EB_R3_NO_SWITCH), at the shapes
# Blackhole models run; then outputs. usage: r13sw.sh <ab|bits>
cd /work
export EB_RUN_LIMIT=2400 EB_REPS=4 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
T=tests/eb_r3_ci
case $1 in
  ab) for i in 1 2; do
        echo "##### $(date -u +%T) sw: off=- main_env=EB_R3_NO_SWITCH=1"
        EB_K_EXPR="(llama8b_decode and performance) or sdxl_resnet or sdxl_transformer or sdxl_refiner_geglu" bash $T/ab_r10.sh - EB_R3_NO_SWITCH=1 -p eb_k_plugin $T/test_eb_r11.py
      done ;;
  bits) printf '%s\n' 'EB_R3_NO_SWITCH=1|default|||tests/eb_r3_ci/test_eb_dump_sec.py -k "test_block and not post and not scalar"' 'EB_R3_NO_BLOCK=1|default|||tests/eb_r3_ci/test_eb_dump_sec.py -k "test_block and not post and not scalar"' > /tmp/sw.spec
        EB_RUN_LIMIT=3000 bash $T/dump_run.sh /tmp/sw.spec
        E=tests/ttnn/unit_tests/operations/eltwise
        echo "##### modules: head without the switch vs with it"; bash $T/bits_envs.sh "EB_R3_NO_SWITCH=1" "EB_DUMMY=1" -p eb_seed_plugin $E/test_add.py $E/test_mul.py $E/test_binary_ng_sharded_fp32_batch.py $E/test_binary_ng_width_padded_stride.py ;;
esac
echo "##### end $(date -u +%T)"
