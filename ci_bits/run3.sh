#!/usr/bin/env bash
# #59149 headroom round, scratch: the DeepSeek padded KV cache untilize callers, LLK asserts on, after main packaged
# packed_kv_layout.hpp (#60308).
set +e +o pipefail
R=$(git rev-parse --show-toplevel); cd $R
export PYTHONPATH=$R${PYTHONPATH:+:$PYTHONPATH}
echo "== head $(git rev-parse HEAD) $(hostname)"
A=(models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py -k "scaled_fp8_packed_row or single_device")
TT_METAL_LLK_ASSERTS=1 timeout -s INT -k 60 1800 python -m pytest -q -p no:cacheprovider -o timeout_method=thread -rfE "${A[@]}" > /tmp/a.log 2>&1
echo "== deepseek padded kv cache, LLK asserts on, rc=$? $(date -u +%T): $(grep -aE '[0-9]+ (passed|failed)' /tmp/a.log | tail -1)"
grep -aE "^(FAILED|ERROR)|fatal error|not found" /tmp/a.log | cut -c1-300 | head -20
exit 0
