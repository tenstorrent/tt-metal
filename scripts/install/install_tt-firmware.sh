#!/bin/bash

source common.sh
source tools_versioning.sh

install_tt_flash() {
    # Source rust environment
    source "$HOME/.cargo/env"

    # Check if tt-flash directory exists
    if [ -d "$HOME/tt-flash" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-flash"
    fi

    # Clone tt-flash repository
    git clone https://github.com/tenstorrent/tt-flash.git "$HOME/tt-flash" || { echo "Failed to clone tt-flash repository."; return 1; }
    cd "$HOME/tt-flash" || { echo "Failed to change directory to tt-flash."; return 1; }
    git checkout $TT_FLASH_VERSION || { echo "Failed to checkout tt-flash version."; return 1; }

    # Install tt-flash
    pip3 install . || { echo "Failed to install tt-flash."; return 1; }
}

install_tt_firmware() {
    # Source rust environment
    source "$HOME/.cargo/env"
    
    # Check if tt-firmware directory exists
    if [ -d "$HOME/tt-firmware" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-firmware"
    fi

    # Clone tt-firmware repository
    git clone https://github.com/tenstorrent/tt-firmware.git "$HOME/tt-firmware" || { echo "Failed to clone tt-firmware repository."; return 1; }
    cd "$HOME/tt-firmware" || { echo "Failed to change directory to tt-firmware."; return 1; }
    git checkout $TT_FIRMWARE_VERSION || { echo "Failed to checkout tt-firmware version."; return 1; }

    # Add tools directory to PATH
    export PATH="$PATH:$HOME/.local/bin"

    # Run tt-flash to install firmware
    tt-flash --force --fw-tar $HOME/tt-firmware/patches/$TT_FIRMWARE_BUNDLE_VERSION || { echo "Failed to install firmware using tt-flash."; return 1; }
}

# Install tt-flash
execute_function_with_timer install_tt_flash

if [ $? -ne 0 ]; then
    echo "Error: install_tt_flash failed."
    echo "Abort"
    exit 1
fi

# Install tt-firmware
execute_function_with_timer install_tt_firmware

if [ $? -ne 0 ]; then
    echo "Error: install_tt_firmware failed."
    echo "Abort"
    exit 1
fi
