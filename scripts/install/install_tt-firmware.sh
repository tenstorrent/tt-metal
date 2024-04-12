#!/bin/bash

source common.sh
source tools_versioning.sh

cleanup() {
    # Check if tt-flash directory exists
    if [ -d "$HOME/tt-flash" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-flash" || { echo "Failed to remove directory tt-flash."; return 1; }
    fi

    # Check if tt-firmware directory exists
    if [ -d "$HOME/tt-firmware" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-firmware" || { echo "Failed to remove directory tt-firmware."; return 1; }
    fi
}

install_tt_flash() {
    # Check if tt-flash directory exists
    if [ -d "$HOME/tt-flash" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-flash"
    fi

    # Clone tt-flash repository
    git clone https://github.com/tenstorrent/tt-flash.git "$HOME/tt-flash" || { echo "Failed to clone tt-flash repository."; cleanup; return 1; }
    cd "$HOME/tt-flash" || { echo "Failed to change directory to tt-flash."; cleanup; return 1; }
    git checkout $TT_FLASH_VERSION || { echo "Failed to checkout tt-flash version."; cleanup; return 1; }

    # Install tt-flash
    pip install . || { echo "Failed to install tt-flash."; cleanup; return 1; }
}

install_tt_firmware() {
    if [ -z "$TT_FLASH_VERSION" ]; then
        echo "Error: TT_FLASH_VERSION variable is not set."
        return 1
    fi

    # Check if tt-firmware directory exists
    if [ -d "$HOME/tt-firmware" ]; then
        # If it exists, remove it
        rm -rf "$HOME/tt-firmware"
    fi

    # Clone tt-firmware repository
    git clone https://github.com/tenstorrent/tt-firmware.git "$HOME/tt-firmware" || { echo "Failed to clone tt-firmware repository."; cleanup; return 1; }
    cd "$HOME/tt-firmware" || { echo "Failed to change directory to tt-firmware."; cleanup; return 1; }
    git checkout $TT_FIRMWARE_VERSION || { echo "Failed to checkout tt-firmware version."; cleanup; return 1; }

    # Add tools directory to PATH
    export PATH="$PATH:$HOME/.local/bin"

    # Run tt-flash to install firmware
    tt-flash --force --fw-tar $HOME/tt-firmware/patches/fw_pack-80.8.11.0.fwbundle || { echo "Failed to install firmware using tt-flash."; cleanup; return 1; }
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
