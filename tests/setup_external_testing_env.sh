#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# --- Configuration ---
TT_SMI_REPO="https://github.com/tenstorrent/tt-smi"
EXALENS_WHEEL="ttexalens-0.1.250626+dev.7538f60-cp310-cp310-linux_x86_64.whl"
VENV_DIR=".venv"

# --- Functions ---

# Print usage info
usage() {
    echo "Usage: $0 [--reuse] [--clean] [--help]"
    echo "  --reuse      Skip setup steps if a virtual environment already exists."
    echo "  --clean      Remove existing setup (virtual environment, temporary files)."
    echo "  --help       Display this help message."
    exit 1
}

# Cleanup function for temporary files
cleanup() {
    if [[ -n "${TEMP_DIR:-}" && -d "$TEMP_DIR" ]]; then
        rm -rf "$TEMP_DIR"
    fi
}

# Check for required system commands
check_deps() {
    for cmd in python3 git wget sudo; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: Required command '$cmd' is not installed." >&2
            exit 1
        fi
    done
}

# Check for supported Python version
check_python_version() {
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    if [[ "$PYTHON_VERSION" != "3.8"* && "$PYTHON_VERSION" != "3.10"* ]]; then
        echo "Error: Only Python 3.8 or 3.10 are supported. Detected: $PYTHON_VERSION" >&2
        exit 1
    fi
    echo "Supported Python version detected: $PYTHON_VERSION"
}

# --- Main Script ---

# Set up trap for cleanup on exit
trap cleanup EXIT

# Parse arguments
REUSE=false
CLEAN=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reuse) REUSE=true ;;
        --clean) CLEAN=true ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Handle --clean flag
if [[ "$CLEAN" == true ]]; then
    echo "Cleaning up environment..."
    rm -rf tt-smi sfpi "$VENV_DIR" arch.dump sfpi-release.tgz
    echo "Cleanup complete."
    exit 0
fi

# Initial checks
check_deps
check_python_version

# Deactivate any active virtual environment
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Deactivating current Python environment: $VIRTUAL_ENV"
    deactivate
fi

# Main setup logic
if [[ "$REUSE" == false || ! -d "$VENV_DIR" ]]; then
    echo "Performing full setup..."

    # Install system packages
    echo "Updating and installing system packages..."
    sudo apt-get update
    sudo apt-get install -y curl cmake software-properties-common build-essential libyaml-cpp-dev libhwloc-dev libzmq3-dev git-lfs xxd wget

    # Clone tt-smi repository
    if [ ! -d "tt-smi" ]; then
        echo "Cloning tt-smi repository..."
        git clone "$TT_SMI_REPO"
    else
        echo "tt-smi repository already exists. Skipping clone."
    fi

    # Create virtual environment
    echo "Creating Python virtual environment..."
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip

    # Download tt-exalens wheel to a temporary directory
    TEMP_DIR=$(mktemp -d)
    EXALENS_TAG_PLUS_DEV=${EXALENS_WHEEL#ttexalens-}
    EXALENS_TAG=${EXALENS_TAG_PLUS_DEV%%+*}
    DOWNLOAD_URL="https://github.com/tenstorrent/tt-exalens/releases/download/${EXALENS_TAG}/${EXALENS_WHEEL}"

    echo "Downloading tt-exalens from ${DOWNLOAD_URL}"
    wget -O "$TEMP_DIR/$EXALENS_WHEEL" "$DOWNLOAD_URL"

    # Install all Python dependencies in one command for better resolution
    echo "Installing Python dependencies..."
    pip install -r requirements.txt "./tt-smi" "$TEMP_DIR/$EXALENS_WHEEL"

    # Download and extract SFPI release
    ./setup_testing_env.sh
else
    echo "Reusing existing virtual environment."
fi

# Activate the virtual environment for the current shell
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
