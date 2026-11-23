#!/bin/bash
# Independent checks after runs 7b-7e (same as postcheck_7a.sh): the reset / no-reset sequence from the
# run-6b prompt, plus tt_tti_counts (elfray TT/TTI) on the harness codesize build.
# Run in the bh-30 IRD container after the harness has finished.
# Usage: postcheck_7x.sh <run dir> <name, e.g. 7b>
set -uo pipefail
R=${1:?run dir}; N=${2:?name}
B=/proj_sw/user_dev/ctr-sbratic
WS=$R/ws; T=$WS/tt_metal/tt-llk/tests; L=$R/postcheck_$N
mkdir -p $L
source $B/hal-env-ird/.venv/bin/activate; export PATH=$PATH:$HOME/.local/bin; export TMPDIR=$R/tmp
run() { (cd $T && timeout 60m pytest python_tests/$1 -k add -q > $L/$2 2>&1); echo "$2: exit $?, $(tail -1 $L/$2)"; }
{
echo "== 1. tt-smi -r, HAL test";            tt-smi -r > $L/reset1.txt 2>&1; run test_eltwise_binary_hal.py check1_hal_after_reset.txt
echo "== 2. no reset, original test";        run test_eltwise_binary.py     check2_original_noreset.txt
echo "== 3. no reset, HAL test";             run test_eltwise_binary_hal.py check3_hal_noreset.txt
echo "== 4. tt-smi -r, HAL test";            tt-smi -r > $L/reset4.txt 2>&1; run test_eltwise_binary_hal.py check4_hal_after_reset.txt
} | tee $L/checks_$N.txt
TOOLS=$T/sfpi/compiler/bin
ELFRAY=$B/tools/elfray python3 $B/tools/tt_tti_run6/tt_tti_counts.py $R --objdump $TOOLS/riscv-tt-elf-objdump \
  --md $L/tt_tti_counts.md --json $L/tt_tti_counts.json > $L/tt_tti_counts_stdout.txt 2>&1 || echo "tt_tti_counts failed"
echo "POSTCHECK DONE $(date -u +%FT%TZ)" | tee -a $L/checks_$N.txt
