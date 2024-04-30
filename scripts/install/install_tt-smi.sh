#!/bin/bash

source common.sh
source tools_versioning.sh

install_tt_smi() {
    # Source rust environment
    source "$HOME/.cargo/env"

    # Check if tt-smi directory exists
    if [ -d "$HOME/tt-smi" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-smi"
    fi

    # Clone tt-smi repository
    git clone https://github.com/tenstorrent/tt-smi.git "$HOME/tt-smi" || { echo "Failed to clone tt-smi repository."; return 1; }
    cd "$HOME/tt-smi" || { echo "Failed to change directory to tt-smi."; return 1; }
    git checkout $TT_SMI_VERSION || { echo "Failed to checkout tt-smi version."; return 1; }

    # Upgrade pip3
    pip3 install --upgrade pip || { echo "Failed to upgrade pip."; return 1; }

    # Install tt-smi
    pip3 install . || { echo "Failed to install tt-smi."; return 1; }
}

# Install tt-smi
execute_function_with_timer install_tt_smi

if [ $? -ne 0 ]; then
    echo "Error: install_tt_smi failed."
    echo "Abort"
    exit 1
fi
