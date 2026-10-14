#!/usr/bin/env bash
# #58817 fork audit at the decode matmuls Blackhole models run (fork_decode_cases.py): bits of every variant against
# main, the head's variant 0 against main's ELFs, then the in-process alternation of the fork's opt-ins (three processes).
# usage: ci_fd.sh [asserts 0|1]
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 BITS_ASSERTS=${1:-0} ALT_ROUNDS=11 ALT_CALLS=8 CI_LIMIT=3600 ALT_VARIANTS=0,1,5,6,4
export GUARD_KERNEL=bmm_large_block_zm_fused_bias_activation_metal2 GUARD_DEFINE=FORK_VARIANT
export FORK_CASES_OUT=/work/generated/fork_cases.txt
export ALT_NAMES="1=the head;5=narrowed (tail rule at every fidelity);6=narrowed (tail rule at LoFi);4=neither, A/A copy"
mkdir -p generated; rm -f $FORK_CASES_OUT
bash scratch_r3/run_ci.sh scratch_r3/plan_fd.txt
OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1)
READELF=$(dirname "$OBJ")/riscv-tt-elf-readelf python scratch_r3/tools/fork_main_guard.py $HOME/r3cache_bits_head $HOME/r3cache_bits_main
echo "== cases"; sort -u $FORK_CASES_OUT
