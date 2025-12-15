#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

################################################################################
# ComfyUI Bridge Server Launch Script
#
# Starts the Unix socket bridge between ComfyUI frontend and tt-metal backend.
#
# Usage:
#   ./launch_comfyui_bridge.sh [--dev] [--socket-path PATH] [--device-id ID]
#
# Options:
#   --dev               Enable dev mode (single worker, fast warmup)
#   --socket-path PATH  Unix socket path (default: /tmp/tt-comfy.sock)
#   --device-id ID      Device ID to use (default: 0)
#
# Environment variables:
#   TT_COMFY_SOCKET     Socket path (default: /tmp/tt-comfy.sock)
#   SDXL_DEV_MODE       Enable dev mode (set to "true")
#   TT_VISIBLE_DEVICES  Comma-separated device IDs for T3K
#
# Examples:
#   # Start with defaults
#   ./launch_comfyui_bridge.sh
#
#   # Dev mode (fast startup for testing)
#   ./launch_comfyui_bridge.sh --dev
#
#   # Custom socket path
#   ./launch_comfyui_bridge.sh --socket-path /tmp/my-bridge.sock
#
#   # Use specific device
#   ./launch_comfyui_bridge.sh --device-id 1
#
################################################################################

set -e  # Exit on error

# Determine script directory (absolute path)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo "ComfyUI Bridge Server Launch Script"
echo "======================================"
echo "Script directory: ${SCRIPT_DIR}"
echo ""

# Check if python_env exists
PYTHON_ENV="${SCRIPT_DIR}/python_env"
if [ ! -d "${PYTHON_ENV}" ]; then
    echo "ERROR: Python environment not found at ${PYTHON_ENV}"
    echo "Please run: python3 -m venv python_env && source python_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating Python environment..."
source "${PYTHON_ENV}/bin/activate"

# Verify required packages
echo "Verifying dependencies..."
python3 -c "import msgpack; import torch; import ttnn" 2>/dev/null || {
    echo "ERROR: Required packages not installed"
    echo "Please install: pip install msgpack torch ttnn"
    exit 1
}

# Set working directory to script directory
cd "${SCRIPT_DIR}"

# Default values
SOCKET_PATH="${TT_COMFY_SOCKET:-/tmp/tt-comfy.sock}"
DEVICE_ID=0
DEV_MODE=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            DEV_MODE=true
            export SDXL_DEV_MODE=true
            shift
            ;;
        --socket-path)
            SOCKET_PATH="$2"
            shift 2
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dev] [--socket-path PATH] [--device-id ID]"
            echo ""
            echo "Options:"
            echo "  --dev               Enable dev mode (single worker, fast warmup)"
            echo "  --socket-path PATH  Unix socket path (default: /tmp/tt-comfy.sock)"
            echo "  --device-id ID      Device ID to use (default: 0)"
            echo ""
            echo "Environment variables:"
            echo "  TT_COMFY_SOCKET     Socket path (default: /tmp/tt-comfy.sock)"
            echo "  SDXL_DEV_MODE       Enable dev mode (set to 'true')"
            echo "  TT_VISIBLE_DEVICES  Comma-separated device IDs for T3K"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo ""
echo "Configuration:"
echo "  Socket path: ${SOCKET_PATH}"
echo "  Device ID:   ${DEVICE_ID}"
echo "  Dev mode:    ${DEV_MODE}"
echo ""

# Remove old socket if it exists
if [ -S "${SOCKET_PATH}" ]; then
    echo "Removing existing socket: ${SOCKET_PATH}"
    rm -f "${SOCKET_PATH}"
fi

# Start bridge server
echo "Starting ComfyUI Bridge Server..."
echo "Press Ctrl+C to stop"
echo ""

exec python3 -m comfyui_bridge.server \
    --socket-path "${SOCKET_PATH}" \
    --device-id "${DEVICE_ID}" \
    ${DEV_MODE:+--dev}
