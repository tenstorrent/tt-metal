#!/usr/bin/env bash
set -euo pipefail
export QWEN_OPT_POLICY="$(python -c 'import json;print(json.dumps(json.load(open("models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/reader_padding_candidates.json"))["role_winners_padmix"]))')"
bash models/autoports/qwen_qwen3_8_27b/tests/profile_multichip_readers.sh final_padded
