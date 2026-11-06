#!/bin/bash
# Entrypoint of the self-contained rerank server image: checks the cache volume, finds the chip, starts the server.
# RR_CHIPS (PCI addresses, comma-separated) is optional when exactly one /dev/tenstorrent/N is passed in.
set -e
if ! mkdir -p "$HOME" "$TT_CACHE_PATH" "$TT_METAL_CACHE" "$HF_HOME" 2>/dev/null || [ ! -w "$HF_HOME" ]; then
    echo "the cache volume at /cache is not writable by user id $(id -u): run 'chmod a+rwx' on the host directory you mount there" >&2
    exit 2
fi
if [ -z "$RR_CHIPS" ]; then
    devs=(/dev/tenstorrent/[0-9]*)
    if [ ${#devs[@]} -ne 1 ] || [ ! -e "${devs[0]}" ]; then
        echo "pass exactly one chip (--device /dev/tenstorrent/N), or set RR_CHIPS to the chips' PCI addresses; found: ${devs[*]}" >&2
        exit 2
    fi
    n=${devs[0]##*/}
    RR_CHIPS=$(basename "$(readlink -f "/sys/class/tenstorrent/tenstorrent!$n/device")")
    echo "chip /dev/tenstorrent/$n = $RR_CHIPS"
fi
# one worker per PCI address in RR_CHIPS
exec python3 rerank_server.py --host 0.0.0.0 --port "$RR_PORT" --chips "$RR_CHIPS" "$@"
