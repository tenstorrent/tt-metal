#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Stage 07 datatype sweep: {LLM MLP bfp4 / bfp8 / bf16} x {KV bfp8 / bf16} x {DiT bfp8 / bf16}, one process per
# policy (the backbone cannot be reloaded in-process after the traces exist), scored by scripts/measure_perf.py on the
# golden 10 s replay (frame-hidden PCC, latent PCC vs golden, log-mel distance) with the teacher-forced AR frames/s and
# DiT chunk time. Results: doc/optimize/perf_runs/sweep_*.json; table: scripts/dtype_sweep_table.py.
#
#   source ~/mm3-bringup/common.sh && cd $MM3_WT
#   nohup bash models/autoports/minimaxai_minimax_music3/scripts/dtype_sweep.sh > models/autoports/minimaxai_minimax_music3/generated/dtype_sweep.log 2>&1 &
set -uo pipefail
source ~/mm3-bringup/common.sh
cd "$MM3_WT"
G="$MM3_MODEL_DIR/generated"
EXTRA="${MM3_SWEEP_EXTRA:-}"
for mlp in bfp8 bfp4 bf16; do
  for kv in bfp8 bf16; do
    for dit in bfp8 bf16; do
      label="sweep_mlp-${mlp}_kv-${kv}_dit-${dit}"
      if [ -f "$MM3_MODEL_DIR/doc/optimize/perf_runs/${label}.json" ] && [ -z "${MM3_SWEEP_FORCE:-}" ]; then
        echo "+ ${label}: exists, skipping"; continue
      fi
      echo "+ ${label} $(date +%T)"
      with_hw_lock timeout 3000 "$MM3_PY" "$MM3_MODEL_DIR/scripts/measure_perf.py" --label "$label" --policy optimized \
        --llm-policy "opt_mlp-${mlp}_kv-${kv}" --dit-dtype "$dit" --skip-free-running $EXTRA > "$G/${label}.log" 2>&1
      echo "  exit $? $(grep -o 'golden replay: .*' "$G/${label}.log" | cut -c1-200)"
    done
  done
done
echo "+ sweep done $(date +%T)"
