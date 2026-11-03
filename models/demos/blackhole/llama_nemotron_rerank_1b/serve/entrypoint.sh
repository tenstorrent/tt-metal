#!/bin/bash
# Entrypoint of the llama-nemotron-rerank-1b-v2 server image: checks the mounts, then starts serve/rerank_server.py.
# tt-metal (built, TT_METAL_HOME=/work) and the caches (/cache) are mounted at runtime; RR_CHIPS is the chip's PCI address.
set -e
if [ ! -d "$TT_METAL_HOME/ttnn" ] || [ ! -d "$TT_METAL_HOME/build/lib" ]; then
    echo "mount a built tt-metal at $TT_METAL_HOME (needs ttnn/ and build/lib/)" >&2
    exit 2
fi
if [ -z "$RR_CHIPS" ]; then
    echo "set RR_CHIPS to the chip's PCI address, e.g. -e RR_CHIPS=0000:86:00.0" >&2
    exit 2
fi
mkdir -p "$TT_CACHE_PATH" "$TT_METAL_CACHE" "$HF_HOME"
cd /opt/nemotron-rerank
# the container binds all interfaces; publish the port on the host's 127.0.0.1 only (docker run -p 127.0.0.1:...)
exec python3 rerank_server.py --host 0.0.0.0 --port "$RR_PORT" --chips "$RR_CHIPS" "$@"
