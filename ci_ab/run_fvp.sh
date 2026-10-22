#!/usr/bin/env bash
# Scratch job: fast untilize against pack untilize of raw block-float tiles on arm C, tests chosen by $FVP_K (pytest -k).
source /work/ci_ab/common.sh
OUT=/work/generated/utfvp; rm -rf $OUT; mkdir -p $OUT
use fvp_C C
timeout -s INT -k 60 3300 python3 -m pytest -q -p no:cacheprovider -o timeout_method=thread -o addopts= -rfE -k "$FVP_K" tests/ttnn/unit_tests/operations/data_movement/test_untilize_fast_vs_pack.py > $OUT/fvp_C.log 2>&1
echo "== fast_vs_pack C ($FVP_K) rc=$? $(grep -aE '[0-9]+ (passed|failed)' $OUT/fvp_C.log | tail -1)"
grep -a -E '^(FAILED|ERROR)' $OUT/fvp_C.log | cut -c1-250 | head -20
echo "RAB ELF C $(elfsum $TT_METAL_CACHE) $(ls -d $TT_METAL_CACHE/*/kernels/*/* 2>/dev/null | wc -l) builds"
use A A
echo "== done $(date -u +%FT%TZ)"
