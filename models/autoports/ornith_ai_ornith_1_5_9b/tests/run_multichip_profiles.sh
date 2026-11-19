#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repo_root=$(git rev-parse --show-toplevel)
model_root="$repo_root/models/autoports/ornith_ai_ornith_1_5_9b"
art="$model_root/doc/multichip_decoder"
cd "$model_root"
for layer in 0 3; do
  timeout 600 env TRACY_NO_WEB_SERVER=1 TT_METAL_HOME="$repo_root" \
    TT_METAL_KERNEL_PATH="$model_root/doc/optimized_decoder/profiler_kernel_overlay" \
    PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}" \
    python -m tracy -r -p -v -o "$art/tracy/$layer" \
    -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_probe \
    --layer "$layer" --lengths 1024 --layout replicated --trace --profile \
    --reference "$model_root/generated/multichip/perf_$layer.pt" --output "$art/profile_$layer.json" \
    > "$art/profile_$layer.log" 2>&1
done
python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_perf_reports
