#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# The local TP4 decoder benefits from direct host dispatch; preserve explicit controls.
export TT_MESH_PASS_THROUGH_THREAD_POOL=${TT_MESH_PASS_THROUGH_THREAD_POOL:-1}
export TT_MODEL_BRINGUP_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4
export TT_AUTODEBUG_ROOT=/home/mvasiljevic/qwen38-full-rerun/codex-home/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.5
BRINGUP_EXPORTS=$(python_env/bin/python "$TT_MODEL_BRINGUP_ROOT/scripts/environment.py")
eval "$BRINGUP_EXPORTS"
export PYTHONPATH=.:$PYTHONPATH
experiment_name=$1
shift
experiment_doc=models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder
trap 'experiment_status=$?; printf "%s\n" "$experiment_status" > "$experiment_doc/$experiment_name.exit_status"' EXIT
mkdir -p "$experiment_doc/sources"
printf '%q ' "$0" "$experiment_name" "$@" >> "$experiment_doc/commands.log"
printf '\n' >> "$experiment_doc/commands.log"
git rev-parse HEAD > "$experiment_doc/$experiment_name.commit"
python_env/bin/python - "$experiment_doc/$experiment_name.environment.json" <<'PY'
import json
import os
import sys
from pathlib import Path

names = (
    "TT_METAL_CACHE", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DISABLE_ETH",
    "TT_METAL_WATCHER_NOINLINE", "TT_METAL_FABRIC_OPT_LEVEL", "TT_METAL_FORCE_JIT_COMPILE",
    "TT_METAL_LOG_KERNELS_COMPILE_COMMANDS", "TT_METAL_DEVICE_PROFILER", "TT_MESH_PASS_THROUGH_THREAD_POOL",
    "QWEN_EXPERIMENT_ENTRY", "QWEN_GDN_PHASED", "QWEN_GDN_SCAN_SERIAL", "QWEN_GDN_PREP_SERIAL", "QWEN_GDN_DUMP",
)
Path(sys.argv[1]).write_text(json.dumps({name: os.getenv(name) for name in names}, indent=2) + "\n")
PY
for source in models/autoports/qwen_qwen3_8_27b/tt/*decoder.py models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.{cpp,hpp} ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp models/autoports/qwen_qwen3_8_27b/tests/gdn_multichip_probe.py; do
    source_hash=$(sha256sum "$source" | cut -d ' ' -f 1)
    cp "$source" "$experiment_doc/sources/$source_hash.py.txt"
    printf '%s  %s\n' "$source_hash" "$source" >> "$experiment_doc/$experiment_name.source.sha256"
done
sha256sum ttnn/ttnn/_ttnn.so build/lib/_ttnncpp.so >> "$experiment_doc/$experiment_name.source.sha256"
timeout -k 10 900 python_env/bin/python "${QWEN_EXPERIMENT_ENTRY:-models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py}" --output "$experiment_doc/$experiment_name.json" "$@" > "$experiment_doc/$experiment_name.log" 2>&1
