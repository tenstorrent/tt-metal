# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Dockerfile for Face Matching Solution on Tenstorrent QuietBox
#
# This Dockerfile builds a containerized face verification API using:
# - YuNet (TTNN) for face detection
# - SFace (TTNN) for face recognition
# - FastAPI for REST API
#
# Requirements:
# - Tenstorrent QuietBox with tt-metal installed
# - Access to TT device (/dev/tenstorrent)

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY server/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY server/ /app/server/

# Copy model files (these should be mounted or pre-built)
# Note: In production, tt-metal and models should be mounted from the host
COPY ../../common.py /app/models/experimental/sface/common.py
COPY ../../reference/ /app/models/experimental/sface/reference/
COPY ../../tt/ /app/models/experimental/sface/tt/

# Create directories for persistent data
RUN mkdir -p /app/server/registered_faces /app/server/mock_database

# Environment variables
ENV PYTHONPATH=/app:/tt-metal
ENV TT_METAL_HOME=/tt-metal

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Default command
CMD ["python3", "-m", "uvicorn", "server.fast_api_face_recognition:app", "--host", "0.0.0.0", "--port", "8000"]
