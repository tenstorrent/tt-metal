# GR00T N1.6 on Tenstorrent Blackhole p150a
#
# Build:
#   docker build -t groot-n16-bh -f models/experimental/groot_n16/Dockerfile .
#
# Run (bind Tenstorrent device 2):
#   docker run --rm -it \
#     --device /dev/tenstorrent/2 \
#     -v /dev/hugepages:/dev/hugepages \
#     -v $(pwd):/workspace \
#     groot-n16-bh
#
# The container expects tt-metal to be built and available at /workspace.

ARG BASE_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-amd64:latest

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV ARCH_NAME=blackhole
ENV TT_METAL_HOME=/workspace
ENV PYTHONPATH=/workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3-dev \
    python3-pip \
    python3-venv \
    libhwloc-dev \
    libyaml-cpp-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies for GR00T N1.6
RUN pip3 install --no-cache-dir \
    torch>=2.1.0 \
    transformers>=4.40.0 \
    safetensors \
    huggingface_hub \
    numpy \
    Pillow \
    scipy \
    einops \
    accelerate \
    pytest \
    pytest-timeout

# Install Isaac-GR00T for reference implementation
RUN pip3 install --no-cache-dir isaac-groot || \
    echo "Isaac-GR00T not available via pip, clone manually if needed"

WORKDIR /workspace

# Verify device access at runtime
COPY <<'EOF' /usr/local/bin/check-tt-device.sh
#!/bin/bash
if [ -e /dev/tenstorrent ]; then
    echo "Tenstorrent device found:"
    ls -la /dev/tenstorrent/
    echo ""
    if command -v tt-smi &>/dev/null; then
        tt-smi -l
    fi
else
    echo "WARNING: No Tenstorrent device found at /dev/tenstorrent"
    echo "Run container with: --device /dev/tenstorrent/2"
fi
EOF
RUN chmod +x /usr/local/bin/check-tt-device.sh

# Entry point: activate venv if available, check device
COPY <<'ENTRY' /usr/local/bin/entrypoint.sh
#!/bin/bash
set -e

# Activate tt-metal venv if present
if [ -f /workspace/python_env/bin/activate ]; then
    source /workspace/python_env/bin/activate
fi

# Check device
check-tt-device.sh

exec "$@"
ENTRY
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
