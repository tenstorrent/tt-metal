#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export TT_METAL_HOME="$PWD"
export PYTHONPATH="${QWEN_CI_PLUGIN_DIR:-/tmp/qwen-ci-vllm-plugin}/src:$PWD"
export HF_HUB_OFFLINE=1
export QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B
export QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export EXTRA_MODELS_DIR="$PWD/models/autoports/vllm_bundles"
export QWEN36_PREFILL_PER_REQUEST=1
export QWEN36_PREFILL_LOG_K=1
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
export QWEN36_TRACE_REUSE="${QWEN36_TRACE_REUSE:-1}"
export QWEN36_WARMUP_PREFILL_LENGTHS="${QWEN36_WARMUP_PREFILL_LENGTHS:-128,4096}"
export TT_MESH_GRAPH_DESC_PATH="$PWD/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
# The shared runner uses model-dir only as its output root. Model selection is
# the actual registered autoport bundle above; keep preexisting readiness files.
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis \
  --hf-model Qwen/Qwen3.8-27B --mesh-device P300x2 \
  --stages serve,sampling,qualitative,benchmark --sampling-profile smoke \
  --max-num-seqs 32 --max-model-len 262144 --port 8023 \
  --benchmark-prompt-len 128 --benchmark-output-len 252 \
  --benchmark-concurrency 1 --benchmark-num-requests 4 \
  --ci-benchmark-prompt-len 4096 --ci-benchmark-output-len 252 \
  --ci-benchmark-concurrency 8 --ci-benchmark-num-requests 8 \
  --tt-config '{"trace_region_size":200000000,"fabric_config":"FABRIC_1D_RING"}' \
  "$@"
