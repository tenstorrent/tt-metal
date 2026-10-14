#!/usr/bin/env bash
# #58817 fork rule narrowing (c76624ced8c, current main merged): bits and ELF identity of the head against main and the
# previous head (decode cases, the legacy port's cases, the broad set), then on Blackhole the in-process alternation at
# the decode matmuls models run (fork_decode_cases.py). usage: ci_fd4.sh bh|wh [asserts 0|1]
set +e +o pipefail
cd "$(git rev-parse --show-toplevel)"
export CI_FRESH=1 CI_ELFCFG=1 BITS_ASSERTS=${2:-0} ALT_ROUNDS=13 ALT_CALLS=8 CI_LIMIT=3600 ALT_VARIANTS=0,1,7,8,4 ALT_BASE2=7
export GUARD_KERNEL=bmm_large_block_zm_fused_bias_activation_metal2 GUARD_DEFINE=FORK_VARIANT
export FORK_CASES_OUT=$PWD/generated/fork_cases.txt
export ALT_NAMES="1=the previous head;7=the head;8=the head, no row MOP below 4-tile one-row sub blocks;4=neither, A/A copy"
mkdir -p generated; rm -f $FORK_CASES_OUT
bash scratch_r3/run_ci.sh scratch_r3/plan_fd4_$1.txt
if [[ $1 == bh ]]; then
  OBJ=$(find / -name riscv-tt-elf-objcopy -type f 2>/dev/null | head -1); export READELF=$(dirname "$OBJ")/riscv-tt-elf-readelf
  for pair in alt,main alt,head alt,old head,old; do
    GUARD_NAMES=$pair python scratch_r3/tools/fork_main_guard.py $HOME/r3cache_bits_${pair%,*} $HOME/r3cache_bits_${pair#*,}
  done
  echo "== cases"; sort -u $FORK_CASES_OUT
fi
