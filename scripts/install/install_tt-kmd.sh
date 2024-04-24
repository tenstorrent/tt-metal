#!/bin/bash

source common.sh
source tools_versioning.sh

install_tt_kmd() {
    if [ -z "$TT_KMD_VERSION" ]; then
        echo "Error: TT_KMD_VERSION variable is not set."
        return 1
    fi

    # Install dkms
    sudo apt install -y dkms || { echo "Failed to install dkms."; return 1; }

    # Check if tt-kmd directory exists
    if [ -d "$HOME/tt-kmd" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-kmd"
    fi

    # Clone tt-kmd repository
    git clone https://github.com/tenstorrent/tt-kmd.git "$HOME/tt-kmd" || { echo "Failed to clone tt-kmd repository."; return 1; }
    cd "$HOME/tt-kmd" || { echo "Failed to change directory to tt-kmd."; return 1; }

    # Add tt-kmd to dkms
    sudo dkms add . || { echo "Failed to add tt-kmd to dkms."; return 1; }

    # Install tt-kmd using dkms
    sudo dkms install tenstorrent/"$TT_KMD_VERSION" || { echo "Failed to install tt-kmd using dkms."; return 1; }

    # Load the tenstorrent kernel module
    sudo modprobe tenstorrent || { echo "Failed to load the tenstorrent kernel module."; return 1; }
}

# Install tt-kmd
execute_function_with_timer install_tt_kmd

if [ $? -ne 0 ]; then
    echo "Error: install_tt_kmd failed."
    echo "Abort"
    exit 1
fi
