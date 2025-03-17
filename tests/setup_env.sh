#!/bin/bash

# Function to print usage info
usage() {
    echo "Usage: $0 [--reuse]"
    echo "  --reuse      Skip some setup steps if a virtual environment already exists"
    exit 1
}

# Default behavior: Assume not reusing if no argument is passed
REUSE=false

# Check if we passed command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reuse)
            REUSE=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Check if a Python virtual environment is activated
# This can cause crash of script so first deactivate any existing environments
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Deactivating current Python environment: $VIRTUAL_ENV"
    deactivate
fi

# If we're not reusing, continue with setup
if [[ "$REUSE" == false ]]; then
    # Update packages and install gawk (if necessary)
    echo "Updating system packages..."
    sudo apt update
    sudo apt install software-properties-common build-essential libyaml-cpp-dev libhwloc-dev libzmq3-dev git-lfs xxd

    pip install --upgrade pip

    # **************** DOWNLOAD & INSTALL TT-SMI ****************************
    echo "Cloning tt-smi repository..."
    git clone https://github.com/tenstorrent/tt-smi
    cd tt-smi

    echo "Creating Python virtual environment..."
    python3 -m venv .venv
    if [ ! -d ".venv" ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
    source .venv/bin/activate

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing required packages..."
    pip install .
    pip install pytest pytest-cov pytest-repeat pytest-timeout

    # Detect architecture for chip
    echo "Running tt-smi -ls to detect architecture..."
    tt-smi -ls > ../arch.dump
    echo "tt-smi -ls completed. Running find_arch.py..."
    cd ..
    result=$(python3 helpers/find_arch.py "Wormhole" "Blackhole" "Grayskull" arch.dump)
    echo "Detected architecture: $result"

    if [ -z "$result" ]; then
        echo "Error: Architecture detection failed!"
        exit 1
    fi

    echo "Setting CHIP_ARCH variable..."
    export CHIP_ARCH="$result"
    echo "CHIP_ARCH is: $CHIP_ARCH"

    # Install torch and related packages
    echo "Installing PyTorch and related packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

    # **************** DOWNLOAD & INSTALL TT-EXALENS ****************************
    pip install git+https://github.com/tenstorrent/tt-exalens.git@cdca310241827b05a1752db2a15edd11e89a9712

    # **************** DOWNLOAD & INSTALL SFPI ****************************
    echo "Downloading SFPI release..."
    wget https://github.com/tenstorrent/sfpi/releases/download/v6.5.0-sfpi/sfpi-release.tgz -O sfpi-release.tgz
    if [ ! -f "sfpi-release.tgz" ]; then
        echo "SFPI release not found!"
        exit 1
    fi
    echo "Extracting SFPI release..."
    tar -xzvf sfpi-release.tgz
    rm -f sfpi-release.tgz

    # **************** SETUP PYTHON VENV **********************************
    # Ensure python3.10-venv is installed, fallback to python3.8-venv
    echo "Checking python3.10-venv..."
    if ! dpkg -l | grep -q python3.10-venv; then
        echo "python3.10-venv not found, attempting to install python3.8-venv..."
        sudo apt install -y python3.8-venv || { echo "Failed to install python3.8-venv."; exit 1; }
    else
        sudo apt install -y python3.10-venv
    fi

    # Set up Python virtual environment if not already set
    echo "Ensuring virtual environment is set up..."
    python3 -m ensurepip

    # Install needed packages
    pip install -U pytest pytest-cov
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# **************** REUSE ENVIRONMENT SETUP ****************************
if [[ "$REUSE" == true ]]; then
    echo "Reusing existing virtual environment setup..."

    # Activate the existing virtual environment
    source tt-smi/.venv/bin/activate

    # Detect architecture for chip
    echo "Running tt-smi -ls to detect architecture..."
    tt-smi -ls > arch.dump
    echo "tt-smi -ls completed. Running find_arch.py..."
    result=$(python3 helpers/find_arch.py "Wormhole" "Blackhole" "Grayskull" arch.dump)
    echo "Detected architecture: $result"

    if [ -z "$result" ]; then
        echo "Error: Architecture detection failed!"
        exit 1
    fi

    echo "Setting CHIP_ARCH variable..."
    export CHIP_ARCH="$result"
    echo "CHIP_ARCH is: $CHIP_ARCH"

fi

echo "Script completed successfully!"
