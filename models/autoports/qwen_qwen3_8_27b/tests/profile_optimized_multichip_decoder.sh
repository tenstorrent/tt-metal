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
profile_name=$1
shift
profile_evidence=models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder
trap 'profile_status=$?; printf "%s\n" "$profile_status" > "$profile_evidence/$profile_name.exit_status"' EXIT
printf '%q ' "$0" "$profile_name" "$@" >> "$profile_evidence/commands.log"
printf '\n' >> "$profile_evidence/commands.log"
git rev-parse HEAD > "$profile_evidence/$profile_name.commit"
python_env/bin/python - "$profile_evidence/$profile_name.environment.json" <<'ENV_PY'
import json
import os
import sys
from pathlib import Path

names = ("TT_METAL_CACHE", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DISABLE_ETH", "TT_METAL_WATCHER_NOINLINE", "TT_METAL_FABRIC_OPT_LEVEL", "TT_METAL_DEVICE_PROFILER", "TT_MESH_PASS_THROUGH_THREAD_POOL", "QWEN_PROFILE_MODULE")
Path(sys.argv[1]).write_text(json.dumps({name: os.getenv(name) for name in names}, indent=2) + "\n")
ENV_PY
sha256sum models/autoports/qwen_qwen3_8_27b/tt/*decoder.py models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.{cpp,hpp} ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp ttnn/ttnn/_ttnn.so build/lib/_ttnncpp.so > "$profile_evidence/$profile_name.source.sha256"
timeout -k 10 900 python_env/bin/python -m tracy -r -p -v \
 -o "$profile_evidence/tracy/$profile_name" \
 -m "${QWEN_PROFILE_MODULE:-models.autoports.qwen_qwen3_8_27b.tests.run_multichip_decoder}" \
 --profile --output "$profile_evidence/$profile_name.json" "$@" > "$profile_evidence/$profile_name.log" 2>&1
test -s "$profile_evidence/$profile_name.json"
python_env/bin/python models/autoports/qwen_qwen3_8_27b/tests/multichip_profile_tables.py "$profile_evidence/tracy/$profile_name"
