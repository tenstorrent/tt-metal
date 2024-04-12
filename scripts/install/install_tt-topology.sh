#!/bin/bash

source common.sh
source tools_versioning.sh

cleanup() {
    # Check if tt-topology directory exists
    if [ -d "$HOME/tt-topology" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-topology" || { echo "Failed to remove directory tt-topology."; return 1; }
    fi
}

install_tt_topology() {
    if [ -z "$TT_TOPOLOGY_VERSION" ]; then
        echo "Error: TT_TOPOLOGY_VERSION variable is not set."
        return 1
    fi

    # Check if tt-kmd directory exists
    if [ -d "$HOME/tt-topology" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-topology"
    fi

    # Clone tt-topology repository
    git clone https://github.com/tenstorrent/tt-topology.git "$HOME/tt-topology" || { echo "Failed to clone tt-topology repository."; cleanup; return 1; }
    cd "$HOME/tt-topology" || { echo "Failed to change directory to tt-topology."; cleanup; return 1; }
    git checkout $TT_TOPOLOGY_VERSION || { echo "Failed to checkout tt-topology version."; cleanup; return 1; }

    # Upgrade pip3
    pip3 install --upgrade pip || { echo "Failed to upgrade pip."; cleanup; return 1; }

    # Install tt-topology
    pip3 install . || { echo "Failed to install tt-topology."; cleanup; return 1; }

    # Clean up directory
    rm -rf "$HOME/tt-topology" || { echo "Failed to remove directory tt-topology."; cleanup; return 1; }
}

# Install tt-topology
execute_function_with_timer install_tt_topology

if [ $? -ne 0 ]; then
    echo "Error: install_tt_topology failed."
    echo "Abort"
    exit 1
fi
