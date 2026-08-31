#!/usr/bin/env bash
cd ${TT_METAL_HOME:-/home/mvasiljevic/tt-metal}
export TT_METAL_HOME=${TT_METAL_HOME:-/home/mvasiljevic/tt-metal}
export PYTHONPATH=${TT_METAL_HOME:-/home/mvasiljevic/tt-metal}
export QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B
export EXTRA_MODELS_DIR=${TT_METAL_HOME:-/home/mvasiljevic/tt-metal}/models/autoports/vllm_bundles
# From doc/qwen38_checkpoint_swap/tt-inference-server-onboard-qwen38-autoport.patch:
# the pinned default revision is Qwen3.6's, so 3.8 must name its own.
export QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
export HF_MODEL=Qwen/Qwen3.8-27B
export QWEN36_MAX_TOKENS_ALL_USERS=525312
export TT_MESH_GRAPH_DESC_PATH=${TT_METAL_HOME:-/home/mvasiljevic/tt-metal}/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
python_env/bin/python -m models.common.readiness_check.run_vllm_server \
  --stages serve,benchmark \
  --model-dir models/autoports/qwen_qwen3_6_27b \
  --hf-model Qwen/Qwen3.8-27B --mesh-device P300x2 \
  --max-num-seqs 32 --max-model-len 262144 \
  --sampling-profile full \
  --benchmark-prompt-len 128 --benchmark-output-len 128 \
  --benchmark-concurrency 32 --benchmark-num-requests 32 \
  --no-benchmark-ci-serving \
  --tt-config '{"trace_region_size": 200000000, "fabric_config": "FABRIC_1D_RING"}'
