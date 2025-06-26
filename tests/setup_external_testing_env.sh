#!/usr/bin/env bash

set -o pipefail  # Catch errors in pipelines

# Function to print usage info
usage() {
    echo "Usage: $0 [--reuse] [--clean] [--help]"
    echo "  --reuse      Skip some setup steps if a virtual environment already exists"
    echo "  --clean      Remove existing setup (virtual environment, temporary files)"
    echo "  --help       Display this help message"
    exit 1
}

# Variables
REUSE=false
CLEAN=false
TT_SMI_REPO="https://github.com/tenstorrent/tt-smi"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --reuse) REUSE=true ;;
        --clean) CLEAN=true ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Cleanup logic
if [[ "$CLEAN" == true ]]; then
    echo "Cleaning up environment..."
    rm -rf tt-smi sfpi .venv arch.dump sfpi-release.tgz
    echo "Cleanup complete."
    exit 0
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
if [[ "$PYTHON_VERSION" != "3.8"* && "$PYTHON_VERSION" != "3.10"* ]]; then
    echo "Error: Only Python 3.8 or 3.10 are supported. Detected version: $PYTHON_VERSION"
    exit 1
else
    echo "Supported version of Python detected: $PYTHON_VERSION"
fi

# Deactivate any active virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Deactivating current Python environment: $VIRTUAL_ENV"
    deactivate
fi

# If not reusing, perform setup
if [[ "$REUSE" == false ]]; then
    echo "Updating system packages..."
    sudo apt update
    sudo apt install -y curl cmake software-properties-common build-essential libyaml-cpp-dev libhwloc-dev libzmq3-dev git-lfs xxd wget

    # Clone tt-smi repository if not already cloned
    if [ ! -d "tt-smi" ]; then
        echo "Cloning tt-smi repository..."
        git clone "$TT_SMI_REPO"
    else
        echo "tt-smi repository already exists. Skipping clone."
    fi

    # Create virtual environment in the current directory if not already created
    if [ ! -d ".venv" ]; then
        echo "Creating Python virtual environment in the current directory..."
        python3 -m venv .venv
        if [ ! -d ".venv" ]; then
            echo "Failed to create virtual environment"
            exit 1
        fi
    else
        echo "Virtual environment already exists in the current directory. Skipping creation."
    fi
    source .venv/bin/activate

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing tt-smi"
    cd tt-smi
    pip install .
    cd ..

    echo "Installing required Python packages..."
    pip install -r requirements.txt

    # Install tt-exalens
    echo "Installing tt-exalens..."
    if [[ $PYTHON_VERSION == "3.10"* ]]; then
        EXALENS_WHEEL="ttexalens-0.1.250626+dev.7538f60-cp310-cp310-linux_x86_64.whl"
        echo "Python 3.10 detected, using pre-built wheel for tt-exalens $EXALENS_WHEEL"
    else
        echo "Unsupported Python version: $PYTHON_VERSION"
        exit 1
    fi
    wget -O $EXALENS_WHEEL https://github.com/tenstorrent/tt-exalens/releases/download/0.1.250626/$EXALENS_WHEEL
    if [ ! -f $EXALENS_WHEEL ]; then
        echo "Failed to download tt-exalens wheel file"
        exit 1
    fi
    pip install $EXALENS_WHEEL

    # Download and extract SFPI release
    ./setup_testing_env.sh
else   # Reuse existing environment setup
    # Deactivate any active virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo "Deactivating current Python environment: $VIRTUAL_ENV"
        deactivate
    fi
    echo "Reusing existing virtual environment setup..."
fi

# Activate the virtual environment
source .venv/bin/activate

echo "Setup completed successfully!"
