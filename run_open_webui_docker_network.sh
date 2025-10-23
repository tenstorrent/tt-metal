#!/bin/bash

# Setup script to run Open WebUI in Docker, connecting to vLLM server also in Docker
# This handles networking between two Docker containers

set -e

echo "==== Open WebUI Setup (Docker to Docker) ===="
echo ""

# Get the vLLM container name from user
echo "What is your vLLM Docker container name?"
echo "(Press Enter to use default: vllm-server)"
read -p "Container name: " VLLM_CONTAINER
VLLM_CONTAINER=${VLLM_CONTAINER:-vllm-server}

echo ""
echo "🔍 Checking for vLLM container: $VLLM_CONTAINER"

# Check if vLLM container exists
if ! docker ps --format '{{.Names}}' | grep -q "^${VLLM_CONTAINER}$"; then
    echo "❌ ERROR: Container '$VLLM_CONTAINER' is not running!"
    echo ""
    echo "Available running containers:"
    docker ps --format "  - {{.Names}}"
    echo ""
    exit 1
fi

echo "✅ Found container: $VLLM_CONTAINER"

# Get the network of the vLLM container
VLLM_NETWORK=$(docker inspect $VLLM_CONTAINER --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}' | head -1)
echo "🌐 vLLM container is on network: $VLLM_NETWORK"

# Get the vLLM container's IP or use container name for DNS
VLLM_HOST=$VLLM_CONTAINER
echo "🔗 Will connect to vLLM at: http://${VLLM_HOST}:8000"

echo ""
echo "🐳 Starting Open WebUI Docker container..."

# Remove old container if exists
if docker ps -a --format '{{.Names}}' | grep -q '^open-webui-vllm$'; then
    echo "Removing old Open WebUI container..."
    docker stop open-webui-vllm 2>/dev/null || true
    docker rm open-webui-vllm 2>/dev/null || true
fi

# Run Open WebUI on the same network as vLLM
docker run -d \
    --name open-webui-vllm \
    --network $VLLM_NETWORK \
    -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://${VLLM_HOST}:8000/v1 \
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
echo "🔗 Connection details:"
echo "  - Open WebUI container: open-webui-vllm"
echo "  - vLLM container: $VLLM_CONTAINER"
echo "  - Shared network: $VLLM_NETWORK"
echo "  - API endpoint: http://${VLLM_HOST}:8000/v1"
echo ""
echo "📝 Useful commands:"
echo "  View logs:       docker logs -f open-webui-vllm"
echo "  Stop container:  docker stop open-webui-vllm"
echo "  Test connection: docker exec open-webui-vllm curl http://${VLLM_HOST}:8000/v1/models"
echo ""
