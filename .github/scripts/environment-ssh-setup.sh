#!/bin/bash

# Script to setup SSH configuration for an environment from configmap data via stdin
# Usage: kubectl get cm <configmap-name> -oyaml | yq .data | ./environment-ssh-setup.sh

set -euo pipefail

# Function to display usage
usage() {
    echo "Usage: kubectl get cm <configmap-name> -oyaml | yq .data | $0"
    echo ""
    echo "Reads the configmap .data YAML from stdin and sets up SSH configuration."
    echo ""
    echo "Example:"
    echo "  kubectl get cm my-env-connection -n ttop -oyaml | yq .data | $0"
    exit 1
}

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check for help flag
if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
fi

# Read stdin
if [ -t 0 ]; then
    echo "ERROR: No input on stdin. Pipe configmap data into this script."
    usage
fi

INPUT=$(cat)

if [[ -z "$INPUT" ]]; then
    echo "ERROR: Empty input on stdin"
    exit 1
fi

# Create SSH config content
ssh_config=""
current_host=""
current_ip=""

while IFS= read -r line; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Match host line (no leading spaces, ends with colon)
    if [[ "$line" =~ ^([^[:space:]]+):[[:space:]]*$ ]]; then
        current_host="${BASH_REMATCH[1]}"
        current_ip=""
    # Match IP line (has leading spaces, contains 'ip:')
    elif [[ "$line" =~ ^[[:space:]]+ip:[[:space:]]*(.+)[[:space:]]*$ ]]; then
        current_ip="${BASH_REMATCH[1]}"

        # When we have both host and IP, create the SSH config entry
        if [[ -n "$current_host" && -n "$current_ip" ]]; then
            ssh_config+="Host ${current_host}
    HostName ${current_ip}
    User user
    Port 2223

"
            log "Mapped ${current_host} -> ${current_ip}"
        fi
    fi
done <<< "$INPUT"

if [[ -z "$ssh_config" ]]; then
    log "ERROR: No valid hostname-IP mappings found in input"
    exit 1
fi

# Print the SSH config
log "Generated SSH configuration:"
echo "$ssh_config"

# Ensure SSH directory exists
mkdir -p /home/runner/.ssh

# Write SSH config to file
echo "$ssh_config" > /home/runner/.ssh/config

# Set appropriate permissions
chmod 600 /home/runner/.ssh/config
chmod 700 /home/runner/.ssh

log "SSH configuration written to /home/runner/.ssh/config"
log "SSH configuration setup completed!"
