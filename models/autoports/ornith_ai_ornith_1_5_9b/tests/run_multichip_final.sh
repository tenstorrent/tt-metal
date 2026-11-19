#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
D=models/autoports/ornith_ai_ornith_1_5_9b/doc/multichip_decoder
G=models/autoports/ornith_ai_ornith_1_5_9b/generated/multichip
M=models.autoports.ornith_ai_ornith_1_5_9b.tests
lengths=1,2,3,4,31,32,33,1023,1024,1025,2051
run() {
  local name=$1 seconds=$2
  shift 2
  if [[ -e "$D/STOP" ]]; then
    printf 'STOP requested between jobs before %s\n' "$name"
    exit 75
  fi
  if python - "$D/$name.json" <<'CHECK'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
s = Path("models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py")
try:
    assert json.loads(p.read_text())["source_sha256"] == hashlib.sha256(s.read_bytes()).hexdigest()
except (OSError, ValueError, KeyError, AssertionError):
    sys.exit(1)
CHECK
  then
    printf 'EXISTING_PASS %s\n' "$name"
    return
  fi
  printf 'START %s\n' "$name"
  timeout "$seconds" "$@" > "$D/$name.log" 2>&1
  printf 'PASS %s\n' "$name"
}
for layer in 0 3; do
  run "baseline_final_$layer" 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    python -m "$M.multichip_probe" --baseline --layer "$layer" --lengths "$lengths" --trace --refresh \
    --reference "$G/final_$layer.pt" --output "$D/baseline_final_$layer.json"
  run "final_contracts_$layer" 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    python -m "$M.multichip_probe" --layer "$layer" --layout replicated --lengths "$lengths" --trace --refresh --positive-control \
    --reference "$G/final_$layer.pt" --output "$D/final_contracts_$layer.json"
  run "final_perf_$layer" 600 python -m "$M.multichip_probe" --layer "$layer" --layout replicated --lengths 1024 --trace --benchmark \
    --reference "$G/perf_$layer.pt" --output "$D/final_perf_$layer.json"
  for mode in baseline final; do
    extra=()
    if [[ "$mode" == baseline ]]; then extra+=(--baseline); fi
    run "${mode}_stress_$layer" 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
      python -m "$M.multichip_probe" "${extra[@]}" --layer "$layer" --layout replicated --lengths 128 --trace --steps 128 \
      --reference "$G/stress_$layer.pt" --output "$D/${mode}_stress_$layer.json"
    run "${mode}_long_$layer" 900 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
      python -m "$M.multichip_long" "${extra[@]}" --layer "$layer" --layout replicated \
      --reference "$G/long_$layer.pt" --output "$D/${mode}_long_$layer.json"
  done
  run "watcher_$layer" 600 env TT_METAL_WATCHER=10 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_LOGS_PATH="$PWD/$D/watcher/$layer" \
    python -m "$M.multichip_probe" --layer "$layer" --layout replicated --lengths "$lengths" --trace --refresh \
    --reference "$G/final_$layer.pt" --output "$D/watcher_$layer.json"
done
for batch in 1 32; do
  for mode in baseline final; do
    extra=()
    if [[ "$mode" == baseline ]]; then extra+=(--baseline); fi
    run "${mode}_stack_$batch" 600 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
      python -m "$M.multichip_stack" "${extra[@]}" --batch "$batch" --layout replicated \
      --reference "$G/stack_$batch.pt" --output "$D/${mode}_stack_$batch.json"
  done
done
for batch in 1 32; do
  for mode in baseline final; do
    extra=()
    if [[ "$mode" == baseline ]]; then extra+=(--baseline); fi
    run "${mode}_stack_perf_$batch" 600 python -m "$M.multichip_stack" "${extra[@]}" \
      --batch "$batch" --layout replicated --benchmark \
      --reference "$G/stack_$batch.pt" --output "$D/${mode}_stack_perf_$batch.json"
  done
done
bash models/autoports/ornith_ai_ornith_1_5_9b/tests/run_multichip_profiles.sh
printf 'MULTICHIP_FINAL_PASS\n'
