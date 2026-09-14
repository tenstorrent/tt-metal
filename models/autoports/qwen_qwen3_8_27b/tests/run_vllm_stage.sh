#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /home/mvasiljevic/qwen38-full-rerun/run-env.sh
export PYTHONPATH="$VLLM_ROOT:$PYTHONPATH"
export PATH="$TT_METAL_ROOT/python_env/bin:$PATH"
export TT_MESH_PASS_THROUGH_THREAD_POOL=1
export HF_HUB_OFFLINE=1
exec python "$TT_METAL_ROOT/models/autoports/qwen_qwen3_8_27b/tests/vllm_process_guard.py" -- \
 python -m readiness_check.run_vllm_server \
 --model-dir models/autoports/qwen_qwen3_8_27b --hf-model Qwen/Qwen3.8-27B \
 --mesh-device P300x2 --max-num-seqs "${QWEN_VLLM_MAX_SEQS:-4}" \
 --max-model-len 262144 --block-size 32 \
 --tt-config '{"trace_region_size":134217728,"fabric_config":"FABRIC_1D_RING","fabric_max_packet_payload_size_bytes":8192,"trace_mode":"decode_only"}' \
 --additional-server-args="--hf-overrides '{\"architectures\":[\"TTQwen38ForCausalLM\"]}' --no-enable-prefix-caching --max-logprobs -1" "$@"
