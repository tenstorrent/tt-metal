#!/bin/bash

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

# tt-smi Telemetry Polling Script
# This script polls tt-smi periodically and logs the output to a file

SCRIPT_NAME=$(basename "$0")
PID_FILE="/tmp/poll_telemetry.pid"
POLLING_INTERVAL_MS=1000

usage() {
    echo "Usage: $SCRIPT_NAME <output_file>"
    echo "       $SCRIPT_NAME stop"
    echo ""
    echo "Start polling tt-smi telemetry data to a file, or stop an existing polling process."
    echo ""
    echo "Arguments:"
    echo "  <output_file>  Path to the file where telemetry data will be logged"
    echo "  stop           Stop the currently running polling process"
    echo ""
    echo "The polling process runs in the background and can be stopped using:"
    echo "  $SCRIPT_NAME stop"
    exit 1
}

stop_polling() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Stopping telemetry polling (PID: $PID)..."
            kill "$PID"
            rm -f "$PID_FILE"
            echo "Telemetry polling stopped."
        else
            echo "Process $PID is not running. Cleaning up PID file."
            rm -f "$PID_FILE"
        fi
    else
        echo "No polling process is currently running (PID file not found)."
    fi
    exit 0
}

start_polling() {
    OUTPUT_FILE="$1"

    # Check if bc is installed (required for sleep interval calculation)
    if ! command -v bc &> /dev/null; then
        echo "Error: 'bc' command not found. Please install it to use this script."
        echo "Install with: sudo apt-get install bc (Debian/Ubuntu)"
        exit 1
    fi

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Error: Telemetry polling is already running (PID: $PID)"
            echo "Stop it first using: $SCRIPT_NAME stop"
            exit 1
        else
            echo "Cleaning up stale PID file..."
            rm -f "$PID_FILE"
        fi
    fi

    # Create output file if it doesn't exist
    touch "$OUTPUT_FILE" || {
        echo "Error: Cannot create output file: $OUTPUT_FILE"
        exit 1
    }

    echo "Starting telemetry polling to: $OUTPUT_FILE"
    echo "Polling interval: ${POLLING_INTERVAL_MS}ms"

    # Start background polling process
    (
        while true; do
            # Run tt-smi -s and suppress stderr (to avoid pkg_resources warnings)
            tt-smi -s 2>/dev/null >> "$OUTPUT_FILE"

            # Add separator between entries
            echo "" >> "$OUTPUT_FILE"

            # Sleep for the specified interval (convert ms to seconds)
            sleep $(echo "scale=3; $POLLING_INTERVAL_MS / 1000" | bc)
        done
    ) &

    # Save the PID
    echo $! > "$PID_FILE"

    echo "Telemetry polling started (PID: $(cat "$PID_FILE"))"
    echo "To stop polling, run: $SCRIPT_NAME stop"
}

# Main script logic
if [ $# -eq 0 ]; then
    usage
fi

if [ "$1" == "stop" ]; then
    stop_polling
elif [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
else
    start_polling "$1"
fi
