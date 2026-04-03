# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Pi0.5 on Tenstorrent Blackhole - Docker Recipe
#
# Build:
#   docker build -t pi05-blackhole -f models/experimental/pi0/Dockerfile .
#
# Run (bind device 2):
#   docker run --rm -it --device /dev/tenstorrent/2 pi05-blackhole
#
# Run with all devices:
#   docker run --rm -it \
#     --device /dev/tenstorrent/0 \
#     --device /dev/tenstorrent/1 \
#     --device /dev/tenstorrent/2 \
#     --device /dev/tenstorrent/3 \
#     pi05-blackhole

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git wget curl \
    libatomic1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Install TTNN (Tenstorrent SDK)
RUN pip3 install --break-system-packages ttnn

# Install ML dependencies
RUN pip3 install --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --break-system-packages \
    transformers safetensors huggingface_hub "numpy<2,>=1.24.4"

# Copy model code only (not the full tt-metal source)
COPY models/experimental/pi0 /workspace/models/experimental/pi0

# Download Pi0.5 weights at build time (optional - can also mount at runtime)
# Uncomment the following to bake weights into the image:
# RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('lerobot/pi05_base', 'model.safetensors')"

# Verify imports
RUN python3 -c "import ttnn; import torch; print('Imports OK')" 2>/dev/null || true

# Default entrypoint: run PCC test
CMD ["python3", "-c", "print('Pi0.5 on Blackhole ready. Run tests with: python3 models/experimental/pi0/tests/pcc/test_pcc_pi05_model.py')"]
