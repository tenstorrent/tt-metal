#!/usr/bin/env bash
# #59149 stack round, scratch: the untilize callers this round reaches (pack_untilize_block callers and the fast
# untilize helper), LLK asserts on; sharded group norm without them (its Welford fp32 configs trip main's unpack format
# assert on main's code too).
set +e +o pipefail
R=$(git rev-parse --show-toplevel); cd $R
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH} HWLOCK_HELD=ci
echo "== head $(git rev-parse HEAD) $(hostname)"
mapfile -t A < ci_bits/ids_a.txt
TT_METAL_LLK_ASSERTS=1 timeout -s INT -k 60 3000 python -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE "${A[@]}" ci_bits/test_r3_chunk_callers.py scratch_hr/tools/test_hr_logits.py > /tmp/a.log 2>&1
echo "== callers, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/a.log | tail -1)"
grep -aE "^(FAILED|ERROR) " /tmp/a.log | cut -c1-300
TT_METAL_LLK_ASSERTS=1 timeout -s INT -k 60 900 python -m pytest -q -p no:cacheprovider -rfE models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py -k "scaled_fp8_packed_row or single_device" > /tmp/d.log 2>&1
echo "== deepseek padded kv cache, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/d.log | tail -1)"
mapfile -t G < ci_bits/ids_gn.txt
timeout -s INT -k 60 2400 python -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE "${G[@]}" > /tmp/g.log 2>&1
echo "== group norm, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/g.log | tail -1)"
grep -aE "^(FAILED|ERROR) " /tmp/g.log | cut -c1-300
exit 0
