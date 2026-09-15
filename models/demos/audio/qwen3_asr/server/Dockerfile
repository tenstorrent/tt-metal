# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Dedicated Qwen3-ASR API server image (Tenstorrent P150 / Blackhole).
# Boots straight into the OpenAI /v1/audio/transcriptions server.
# tt-metal (TT_METAL_HOME) + model weights mount at runtime as volumes
# (same model as the CPU Qwen3-ASR; this is the TT-accelerated server).
#
# Build from a clean checkout — the build context is this model's directory, so every
# file the image needs is tracked in the repo:
#   docker build -f models/demos/audio/qwen3_asr/server/Dockerfile \
#                -t qwen3-asr-server models/demos/audio/qwen3_asr
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
SHELL ["/bin/bash", "-c"]

# server + audio deps and the pinned Qwen3-ASR processor INTO the dev image's /opt/venv
# (system pip would miss it). The processor goes in with --no-deps so its own
# transformers pin cannot displace tt-metal's — see requirements-processor.txt and
# reference/qwen_asr_processor.py.
COPY requirements.txt requirements-processor.txt /tmp/qwen3asr/
RUN source /opt/venv/bin/activate && \
    uv pip install --python /opt/venv/bin/python3 -r /tmp/qwen3asr/requirements.txt && \
    uv pip install --python /opt/venv/bin/python3 --no-deps -r /tmp/qwen3asr/requirements-processor.txt && \
    uv pip install --python /opt/venv/bin/python3 requests

COPY server/entrypoint.sh /usr/local/bin/qwen3asr-entrypoint.sh
RUN chmod +x /usr/local/bin/qwen3asr-entrypoint.sh

ENV ARCH_NAME=blackhole \
    TT_METAL_HOME=/work \
    TT_MESH_GRAPH_DESC_PATH=/work/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=/models/qwen3_asr_text_decoder \
    TT_LOGGER_LEVEL=ERROR \
    QWEN3ASR_PORT=8002

EXPOSE 8002
ENTRYPOINT ["/usr/local/bin/qwen3asr-entrypoint.sh"]
