#!/bin/bash

# Setup script to run Open WebUI OUTSIDE Docker, connecting to vLLM server INSIDE Docker
# This works when vLLM is running in a container and exposing port 8000

set -e

echo "==== Open WebUI Setup (External - No Docker) ===="
echo ""
echo "This will install and run Open WebUI using pip (not Docker)"
echo "and connect it to your vLLM server running in a Docker container."
echo ""

# Check if the vLLM container is running and exposing port 8000
echo "🔍 Checking for vLLM server..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM server found at http://localhost:8000"
    echo ""
    curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null | head -20 || echo "Models endpoint responded"
else
    echo "⚠️  WARNING: Cannot reach vLLM server at http://localhost:8000"
    echo "   Make sure your vLLM Docker container is:"
    echo "   1. Running"
    echo "   2. Exposing port 8000 with: -p 8000:8000"
    echo ""
    echo "   Example vLLM container start command:"
    echo "   docker run -d --name vllm-server -p 8000:8000 ..."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "📦 Checking Open WebUI installation..."

# Check if Open WebUI is installed
if ! python3 -c "import open_webui" 2>/dev/null; then
    echo "Installing Open WebUI..."
    echo ""
    echo "⚠️  NOTE: If pip installation fails, you can use Docker instead:"
    echo "   ./run_open_webui.sh  (Open WebUI in Docker, connects to vLLM on host)"
    echo ""

    # Try pip3 install
    if ! pip3 install open-webui; then
        echo ""
        echo "❌ pip installation failed!"
        echo ""
        echo "Alternative options:"
        echo ""
        echo "Option 1: Use Docker (recommended):"
        echo "  ./run_open_webui.sh"
        echo ""
        echo "Option 2: Try installing with specific Python version:"
        echo "  python3.10 -m pip install open-webui"
        echo ""
        echo "Option 3: Use Docker with both containers:"
        echo "  ./run_open_webui_docker_network.sh"
        echo ""
        exit 1
    fi
else
    echo "✅ Open WebUI is already installed"
fi

echo ""
echo "🚀 Starting Open WebUI..."
echo "  - Connecting to vLLM server at: http://localhost:8000"
echo "  - Open WebUI will be available at: http://localhost:3000"
echo "  - Authentication is DISABLED"
echo ""

# Set environment variables and run Open WebUI
export OPENAI_API_BASE_URL=http://localhost:8000/v1
export WEBUI_AUTH=False
export ENABLE_SIGNUP=False
export PORT=3000

echo "Press Ctrl+C to stop Open WebUI"
echo ""

# Run Open WebUI
open-webui serve --port 3000

# This will only be reached if the server is stopped
echo ""
echo "Open WebUI stopped."
