#!/usr/bin/env bash
[[ -n ${HWLOCK_HELD:-} || -n ${GITHUB_ACTIONS:-} ]] || { echo "not under hwlock" >&2; exit 2; }
# Round 3 eltwise binary, sixth review (#58722): ci16sw2 = the head with the block unpack alone for a post activation, merged
# with #58816 (73f70cdb02d) and the switch to pack_block_mop (add and sub at any size, a post activation included), at the
# QuietBox 2 decode residual adds on one chip: main's program (EB_R3_NO_BLOCK) against the switch, three passes; then outputs.
# usage: r13sw2.sh <ab|bits>
cd /work
export EB_RUN_LIMIT=2400 EB_REPS=4 TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
T=tests/eb_r3_ci
case $1 in
  dp) for i in 1 2 3; do
        echo "##### $(date -u +%T) dpsw: off=- main_env=EB_R3_NO_BLOCK=1"
        EB_K_EXPR="test_qb2_add and (dp_post or dp_add or llama_p150 or qwen_qb2_40 or llama_qb2)" bash $T/ab_r10.sh - EB_R3_NO_BLOCK=1 -p eb_k_plugin $T/test_eb_r11.py
      done ;;
  ab) for i in 1 2 3; do
        echo "##### $(date -u +%T) qb2sw: off=- main_env=EB_R3_NO_BLOCK=1"
        EB_K_EXPR="test_qb2_add" bash $T/ab_r10.sh - EB_R3_NO_BLOCK=1 -p eb_k_plugin $T/test_eb_r11.py
      done ;;
  bits) printf '%s\n' 'EB_R3_NO_BLOCK=1|default|||tests/eb_r3_ci/test_eb_dump_sec.py -k "test_qb2"' 'EB_R3_NO_SWITCH=1|default|||tests/eb_r3_ci/test_eb_dump_sec.py -k "test_qb2"' > /tmp/sw2.spec
        EB_RUN_LIMIT=3000 bash $T/dump_run.sh /tmp/sw2.spec ;;
esac
echo "##### end $(date -u +%T)"
