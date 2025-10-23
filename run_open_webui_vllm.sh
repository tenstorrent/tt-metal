#!/bin/bash

# Setup script to run Open WebUI with vLLM server (no authentication, no API keys)
# Based on official vLLM documentation: https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html

set -e

echo "==== Open WebUI Setup for vLLM (No API Keys) ===="
echo ""

# Check if docker is available
if command -v docker &> /dev/null; then
    echo "Docker found, using Docker method..."
    echo ""

    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q '^open-webui-vllm$'; then
        echo "Container 'open-webui-vllm' already exists."
        echo "Stopping and removing old container..."
        docker stop open-webui-vllm 2>/dev/null || true
        docker rm open-webui-vllm 2>/dev/null || true
    fi

    echo "Starting Open WebUI with Docker..."
    echo "  - Connecting to vLLM server at: http://host.docker.internal:8000"
    echo "  - Open WebUI will be available at: http://localhost:3000"
    echo "  - Authentication is DISABLED (WEBUI_AUTH=False)"
    echo "  - No API keys required"
    echo ""

    # Run Open WebUI with Docker
    # Based on official vLLM docs with no-auth modifications
    # WEBUI_AUTH=False enables auto-login (no password required)
    # OPENAI_API_BASE_URLS (plural) points to vLLM server(s)
    # No OPENAI_API_KEYS - connecting without authentication

    docker run -d \
        --name open-webui-vllm \
        -p 3000:8080 \
        --add-host=host.docker.internal:host-gateway \
        -e OPENAI_API_BASE_URLS="http://host.docker.internal:8000/v1" \
        -e WEBUI_AUTH=False \
        -e ENABLE_SIGNUP=False \
        -v open-webui:/app/backend/data \
        --restart always \
        ghcr.io/open-webui/open-webui:main

    echo ""
    echo "✅ Open WebUI started successfully!"
    echo ""
    echo "📍 Access it at: http://localhost:3000"
    echo "🔓 No login required - authentication is disabled"
    echo ""
    echo "📝 Useful commands:"
    echo "  View logs:       docker logs -f open-webui-vllm"
    echo "  Stop container:  docker stop open-webui-vllm"
    echo "  Start container: docker start open-webui-vllm"
    echo "  Remove container: docker rm open-webui-vllm"
    echo ""
    echo "📚 Official docs: https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html"

elif command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
    echo "Docker not found, using pip installation method..."
    echo ""

    # Check if Open WebUI is installed
    if ! python -c "import open_webui" 2>/dev/null; then
        echo "Installing Open WebUI..."
        pip install open-webui
    else
        echo "Open WebUI is already installed"
    fi

    echo ""
    echo "Starting Open WebUI..."
    echo "  - Connecting to vLLM server at: http://localhost:8000"
    echo "  - Open WebUI will be available at: http://localhost:3000"
    echo "  - Authentication is DISABLED"
    echo ""

    # Set environment variables and run Open WebUI
    export OPENAI_API_BASE_URL=http://localhost:8000/v1
    export WEBUI_AUTH=False
    export PORT=3000

    open-webui serve --port 3000

else
    echo "ERROR: Neither Docker nor pip found!"
    echo "Please install Docker or pip to continue."
    exit 1
fi
