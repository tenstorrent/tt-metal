#!/bin/bash
# TT-Metal Development Environment Setup Script
# This script sets up the complete development environment for tt-metal

set -e  # Exit on any error

echo "=========================================="
echo "TT-Metal Development Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Step 1: Installing Python dependencies${NC}"
echo "Installing from tt_metal/python_env/requirements-dev.txt..."
pip3 install -r tt_metal/python_env/requirements-dev.txt

echo ""
echo -e "${GREEN}Step 2: Installing missing dependencies${NC}"
echo "Installing psutil (missing from requirements)..."
pip3 install psutil

echo "Installing pytest plugins..."
pip3 install pytest-xdist pytest-timeout

echo ""
echo -e "${GREEN}Step 3: Initializing git submodules${NC}"
if [ ! -d "tt_metal/third_party/umd/.git" ]; then
    echo "Initializing submodules..."
    git submodule update --init --recursive
else
    echo "Submodules already initialized, skipping..."
fi

echo ""
echo -e "${GREEN}Step 4: Building TT-Metal${NC}"
echo "This will take 20-30 minutes..."
if [ ! -d "build_Release" ]; then
    echo "Running full build..."
    ./build_metal.sh --build-all
else
    echo -e "${YELLOW}Build directory exists. Skipping build.${NC}"
    echo -e "${YELLOW}To rebuild, delete build_Release/ and run this script again.${NC}"
fi

echo ""
echo -e "${GREEN}Step 5: Installing ttnn from source${NC}"
# Uninstall any pip-installed ttnn
pip3 uninstall -y ttnn 2>/dev/null || true

# Install workspace ttnn in editable mode
echo "Installing ttnn in editable mode..."
pip3 install -e . --no-deps

echo ""
echo -e "${GREEN}Step 6: Setting up environment variables${NC}"
# Export environment variables for current session
export TT_METAL_HOME="$SCRIPT_DIR"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
export ARCH_NAME=wormhole_b0
export LD_LIBRARY_PATH="${TT_METAL_HOME}/build_Release/lib:${LD_LIBRARY_PATH}"

# Create a persistent env file
cat > "${SCRIPT_DIR}/env_setup.sh" << 'EOF'
#!/bin/bash
# Source this file to set up your TT-Metal development environment
# Usage: source env_setup.sh

export TT_METAL_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=${TT_METAL_HOME}:${PYTHONPATH}
export ARCH_NAME=wormhole_b0
export LD_LIBRARY_PATH=${TT_METAL_HOME}/build_Release/lib:${LD_LIBRARY_PATH}

echo "TT-Metal environment configured:"
echo "  TT_METAL_HOME=${TT_METAL_HOME}"
echo "  ARCH_NAME=${ARCH_NAME}"
echo "  PYTHONPATH includes workspace"
echo "  LD_LIBRARY_PATH includes build libraries"
EOF

chmod +x "${SCRIPT_DIR}/env_setup.sh"

echo ""
echo -e "${GREEN}=========================================="
echo -e "Setup Complete!"
echo -e "==========================================${NC}"
echo ""
echo "Environment variables have been set for this session."
echo ""
echo -e "${YELLOW}IMPORTANT:${NC} For future sessions, run:"
echo -e "  ${GREEN}source $(pwd)/env_setup.sh${NC}"
echo ""
echo "Then you can run your tests with:"
echo -e "  ${GREEN}pytest ./models/demos/patchtsmixer/tests/pcc/test_modules.py -v${NC}"
echo ""
echo -e "${YELLOW}Quick start alias (add to your ~/.bashrc):${NC}"
echo "  alias tt-env='source ${SCRIPT_DIR}/env_setup.sh'"
echo ""
