# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# llama-nemotron-rerank-1b-v2 rerank server image (Blackhole, one worker per chip): /v1/rerank, /v1/score,
# /health, /metrics (serve/rerank_server.py). Same layout as models/demos/audio/qwen3_asr/server/Dockerfile:
# the server code is in the image; a built tt-metal (TT_METAL_HOME=/work) and the caches (/cache) mount at runtime.
#
# Build from a clean checkout; the build context is this model's directory:
#   docker build --network host -f models/demos/blackhole/llama_nemotron_rerank_1b/serve/Dockerfile \
#                -t nemotron-rerank-server models/demos/blackhole/llama_nemotron_rerank_1b
ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-c"]

COPY serve/requirements.txt /tmp/nemotron-rerank/requirements.txt
RUN source /opt/venv/bin/activate && \
    uv pip install --python /opt/venv/bin/python3 -r /tmp/nemotron-rerank/requirements.txt

COPY serve/rerank_server.py serve/rerank_batcher.py serve/rerank_client.py /opt/nemotron-rerank/
COPY serve/entrypoint.sh /usr/local/bin/nemotron-rerank-entrypoint.sh
RUN chmod +x /usr/local/bin/nemotron-rerank-entrypoint.sh

ENV PATH=/opt/venv/bin:$PATH \
    ARCH_NAME=blackhole \
    TT_METAL_HOME=/work \
    PYTHONPATH=/opt/nemotron-rerank:/work/ttnn:/work:/work/tools \
    LD_LIBRARY_PATH=/work/build/lib \
    TT_CACHE_PATH=/cache/weights \
    TT_METAL_CACHE=/cache/tt-metal \
    HF_HOME=/cache/hf \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    TT_MM_THROTTLE_PERF=0 \
    TT_METAL_TDP_LIMIT_WATTS=160 \
    TT_LOGGER_LEVEL=ERROR \
    NEMOTRON_RERANK_REVISION=828765652b05bd439c9789d2a6d093db1caa1443 \
    RR_PORT=8112

EXPOSE 8112
ENTRYPOINT ["/usr/local/bin/nemotron-rerank-entrypoint.sh"]
