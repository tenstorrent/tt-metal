# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Dedicated Qwen3-ASR API server image (Tenstorrent P150 / Blackhole).
# Boots straight into the OpenAI /v1/audio/transcriptions server.
# tt-metal (TT_METAL_HOME) + model weights mount at runtime as volumes
# (same model as the CPU Qwen3-ASR; this is the TT-accelerated server).
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
SHELL ["/bin/bash", "-c"]

# server + audio deps INTO the dev image's /opt/venv (system pip would miss it)
RUN source /opt/venv/bin/activate && \
    uv pip install --python /opt/venv/bin/python3 \
      fastapi "uvicorn[standard]" python-multipart librosa requests

# bake the Qwen3-ASR processor (prompt build + mel); __init__ pre-minimized so it
# loads with stock transformers (skips the newer-transformers modeling import)
COPY qwen_asr /opt/venv/lib/python3.10/site-packages/qwen_asr
COPY entrypoint.sh /usr/local/bin/qwen3asr-entrypoint.sh
RUN chmod +x /usr/local/bin/qwen3asr-entrypoint.sh

ENV ARCH_NAME=blackhole \
    TT_METAL_HOME=/work \
    TT_MESH_GRAPH_DESC_PATH=/work/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    HF_MODEL=/models/qwen3_asr_text_decoder \
    TT_LOGGER_LEVEL=ERROR \
    QWEN3ASR_PORT=8002

EXPOSE 8002
ENTRYPOINT ["/usr/local/bin/qwen3asr-entrypoint.sh"]
