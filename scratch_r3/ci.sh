#!/usr/bin/env bash
# Scratch CI proofs of the unary datacopy family (not for merge): ELF hashes on the Wormhole and Blackhole mocks, then the
# device bit identity cases. usage: bash scratch_r3/ci.sh [trial]
cd /work
EC=$(cat scratch_r3/elf_cases.txt); BC=$(cat scratch_r3/bitid_cases.txt); CC=$(cat scratch_r3/bitid_conv3d.txt)
if [[ ${1:-} == trial ]]; then EC="add_fp32_1024x1024 where_ttt_fp32_1024x1024 mul_nob_1x1x32x4096"; BC="f32_upper_both_add_dram where_ttt_f32_upper"; CC=""; fi
python3 scratch_r3/run_proofs.py elf wh $EC
python3 scratch_r3/run_proofs.py elf bh $EC
python3 scratch_r3/run_proofs.py bitid $BC
[[ -n $CC ]] && python3 scratch_r3/run_proofs.py bitid $CC
echo "CI PROOFS DONE"
