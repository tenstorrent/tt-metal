#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
name=${1:?name}
layer=${2:?layer}
policy=${3:-\{\}}
batch=${4:-32}
shift "$(( $# < 4 ? $# : 4 ))"
model=models/autoports/qwen_qwen3_6_35b_a3b
doc=$model/doc/optimized_multichip_decoder
[[ ! -e "$doc/STOP" ]] || exit 3
if fuser /dev/tenstorrent/*; then echo 'Target mesh is owned' >&2; exit 2; fi
extra=()
if [[ $batch == 32 ]]; then
    extra=(--activation-file "$model/doc/multichip_decoder/natural_b32.pt"
        --baseline-load "$model/doc/multichip_decoder/baseline_natural_l${layer}_b32.pt")
elif [[ $batch == 1 ]]; then
    extra=(--baseline-load "$doc/baseline_l${layer}_b1.pt")
fi
timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder \
    --layer "$layer" --length 32 --batch "$batch" --decode-steps 8 --repeats 60 \
    --prefill-repeats 3 --stress-repeats 3 --mesh-build-kwargs "$policy" \
    "${extra[@]}" --output "$doc/$name.json" "$@" > "$doc/logs/$name.log" 2>&1
python - "$doc/$name.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
print(sys.argv[1], d['passed'], d['prefill_pcc'], min(d['decode_pcc']),
      {k:v for k,v in d['multichip'].items() if k in ('prefill_warmed_ms', 'decode_traced_warmed_ms')})
PY
