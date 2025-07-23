#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Fail if any command in a pipeline fails

# --- Functions ---

# Function to get chip architecture using tt-smi (borrowed from setup_testing_env.sh)
get_chip_architecture() {
    local smi_output
    if ! smi_output=$(tt-smi -ls 2>/dev/null); then
        echo "WARNING: tt-smi command failed or not found. Architecture detection unavailable." >&2
        return 1
    fi

    if echo "$smi_output" | grep -q "Blackhole"; then
        echo "blackhole"
    elif echo "$smi_output" | grep -q "Wormhole"; then
        echo "wormhole"
    else
        echo "WARNING: Unable to determine chip architecture from tt-smi output." >&2
        return 1
    fi
}

# Function to set architecture-specific variables
set_arch_variables() {
    local arch=$1
    case "$arch" in
        wormhole)
            ARCH_LLK_ROOT="tt_llk_wormhole_b0"
            ARCH_DEFINE="ARCH_WORMHOLE"
            CHIP_ARCH="wormhole"
            ;;
        blackhole)
            ARCH_LLK_ROOT="tt_llk_blackhole"
            ARCH_DEFINE="ARCH_BLACKHOLE"
            CHIP_ARCH="blackhole"
            ;;
        *)
            echo "ERROR: Unsupported architecture: $arch" >&2
            echo "Supported architectures: wormhole, blackhole" >&2
            return 1
            ;;
    esac
}

# Function to validate required directories
validate_directories() {
    local missing_dirs=()

    # Check if git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        echo "ERROR: Not in a git repository. Please run this script from within the project." >&2
        exit 1
    fi

    # Check critical directories
    local dirs_to_check=(
        "$ROOT_DIR/tests"
        "$ROOT_DIR/$ARCH_LLK_ROOT"
        "$ROOT_DIR/tests/firmware/riscv/common"
        "$ROOT_DIR/tests/firmware/riscv/$CHIP_ARCH"
    )

    for dir in "${dirs_to_check[@]}"; do
        if [[ ! -d "$dir" ]]; then
            missing_dirs+=("$dir")
        fi
    done

    if [[ ${#missing_dirs[@]} -gt 0 ]]; then
        echo "ERROR: The following required directories are missing:" >&2
        printf "  %s\n" "${missing_dirs[@]}" >&2
        echo "Please ensure you're in the correct project directory and all dependencies are installed." >&2
        exit 1
    fi
}

# Function to setup testing environment
setup_testing_environment() {
    echo "Setting up testing environment..."
    local setup_script="$ROOT_DIR/tests/setup_testing_env.sh"

    if [[ ! -f "$setup_script" ]]; then
        echo "ERROR: setup_testing_env.sh not found at $setup_script" >&2
        exit 1
    fi

    if ! bash "$setup_script"; then
        echo "ERROR: Failed to execute setup_testing_env.sh" >&2
        exit 1
    fi

    echo "Testing environment setup completed."
}

# Function to validate sfpi installation
validate_sfpi_installation() {
    local sfpi_dirs=(
        "$ROOT_DIR/tests/sfpi/compiler/lib/gcc/riscv32-tt-elf/12.4.0/include/"
        "$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include"
        "$ROOT_DIR/tests/sfpi/include"
    )

    for dir in "${sfpi_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            echo "WARNING: SFPI directory not found: $dir" >&2
            echo "The setup_testing_env.sh script may not have completed successfully." >&2
        fi
    done
}

# Function to generate compile flags
generate_compile_flags() {
    echo "Generating compile_flags.txt for $CHIP_ARCH architecture..."

    cat > "$ROOT_DIR/compile_flags.txt" <<EOF
-D$ARCH_DEFINE
-DTENSIX_FIRMWARE
-DCOMPILE_FOR_TRISC
-std=c++17
-nostdinc++
-nostdinc

-DLLK_TRISC_UNPACK
-DLLK_TRISC_MATH
-DLLK_TRISC_PACK

-isystem
$ROOT_DIR/tests/sfpi/compiler/lib/gcc/riscv32-tt-elf/12.4.0/include/
-isystem
$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include
-isystem
$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include/c++/12.4.0
-isystem
$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include/c++/12.4.0/riscv32-tt-elf
-isystem
$ROOT_DIR/tests/sfpi/include
-I$ROOT_DIR/tests/firmware/riscv/common
-I$ROOT_DIR/tests/firmware/riscv/$CHIP_ARCH
-I$ROOT_DIR/tests/hw_specific/$CHIP_ARCH/inc
-I$ROOT_DIR/$ARCH_LLK_ROOT/common/inc
-I$ROOT_DIR/$ARCH_LLK_ROOT/common/inc/sfpu
-I$ROOT_DIR/$ARCH_LLK_ROOT/llk_lib
-I$ROOT_DIR/tests/helpers/include
EOF

    echo "compile_flags.txt generated successfully at $ROOT_DIR/compile_flags.txt"
}

# Function to restart clangd
restart_clangd() {
    echo "Restarting clangd language server..."

    # More robust clangd restart
    if pgrep -x clangd >/dev/null; then
        echo "Stopping existing clangd process..."
        pkill -x clangd || true
        sleep 1  # Give it time to shut down
    fi

    # Start clangd in background, suppress output
    if command -v clangd >/dev/null 2>&1; then
        nohup clangd >/dev/null 2>&1 &
        echo "clangd restarted successfully."
    else
        echo "WARNING: clangd not found in PATH. Please install clangd for language server support." >&2
    fi
}

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [ARCHITECTURE]

Set up clangd configuration for TT-LLK development.

ARCHITECTURE:
    wormhole    Configure for Wormhole architecture
    blackhole   Configure for Blackhole architecture

If no architecture is specified, the script will attempt to auto-detect
the architecture using tt-smi.

Examples:
    $0 wormhole     # Set up for Wormhole
    $0 blackhole    # Set up for Blackhole
    $0              # Auto-detect architecture
EOF
}

# --- Main Script ---

main() {
    local arch="$1"

    # Handle help requests
    if [[ "$arch" == "-h" || "$arch" == "--help" ]]; then
        usage
        exit 0
    fi

    # Get root directory
    ROOT_DIR=$(git rev-parse --show-toplevel)

    # Determine architecture
    if [[ -z "$arch" ]]; then
        echo "No architecture specified. Attempting auto-detection..."
        if arch=$(get_chip_architecture); then
            echo "Detected architecture: $arch"
        else
            echo "ERROR: Auto-detection failed. Please specify architecture manually." >&2
            usage
            exit 1
        fi
    fi

    # Set architecture-specific variables
    if ! set_arch_variables "$arch"; then
        usage
        exit 1
    fi

    echo "Configuring clangd for $arch architecture..."

    # Validate directories before proceeding
    validate_directories

    # Setup testing environment (downloads SFPI and headers)
    setup_testing_environment

    # Validate SFPI installation
    validate_sfpi_installation

    # Generate compile flags
    generate_compile_flags

    # Restart clangd
    restart_clangd

    echo "Setup complete! clangd is now configured for $arch architecture."
    echo "You can now use IDE features like code completion and navigation."
}

# Execute the main function with all arguments
main "$@"
