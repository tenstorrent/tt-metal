#!/bin/bash
# Install uv package manager with version pinning.
# Run with --help for usage information.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# uv version to install
# Update this version when upgrading uv across all Dockerfiles
UV_VERSION="0.9.26"

# SHA256 hash of the install script for UV_VERSION (for security verification)
# To update: curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sha256sum
# IMPORTANT: Update this hash whenever UV_VERSION changes!
UV_INSTALLER_SHA256="09ace6a888bd5941b5d44f1177a9a8a6145552ec8aa81c51b1b57ff73e6b9e18"

# Installation mode: "system" or "user"
# - system: Install uv to /usr/local/bin, Python to /usr/local/share/uv
# - user: Install to ~/.local/bin (default)
INSTALL_MODE="user"

# Force reinstall even if uv is already installed
FORCE_INSTALL="false"

# ============================================================================
# Help
# ============================================================================

show_help() {
    cat << 'EOF'
Usage: install-uv.sh [OPTIONS]

Install uv package manager with version pinning.
Uses the official uv standalone installer.

OPTIONS:
    --user      Install to user directory (~/.local/bin) (default).
                Updates PATH to include ~/.local/bin.
                Does not require elevated permissions.
    --system    Install at system level (for Docker containers).
                Installs uv binary to /usr/local/bin.
                Sets UV_PYTHON_BIN_DIR to /usr/local/share/uv.
                Requires appropriate permissions (use sudo if needed).
                Does NOT modify shell profiles.
    --force     Reinstall uv even if already installed.
                Useful for upgrading or ensuring a specific version.
    --help, -h  Show this help message and exit.

SUPPORTED PLATFORMS:
    - Linux (any distribution with curl or wget)
    - macOS

REQUIREMENTS:
    - curl or wget (for standalone installer)

BEHAVIOR:
    --user (default):
        - Installs uv to ~/.local/bin
        - Adds ~/.local/bin to PATH if not already present

    --system:
        - Installs uv binary to /usr/local/bin via UV_INSTALL_DIR
        - Configures UV_PYTHON_BIN_DIR to /usr/local/share/uv
        - Uses UV_NO_MODIFY_PATH to avoid shell profile modifications
        - Requires elevated permissions
EOF
}

# ============================================================================
# Argument Parsing
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --system)
            INSTALL_MODE="system"
            shift
            ;;
        --user)
            INSTALL_MODE="user"
            shift
            ;;
        --force)
            FORCE_INSTALL="true"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            echo "Run '${BASH_SOURCE[0]}' --help for usage information." >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Early Exit if Already Installed (unless --force)
# ============================================================================

# Note: We don't require the installed version to match UV_VERSION.
# The pinned version is primarily for reproducibility and security (known-good version),
# not for specific feature requirements. Any working version of uv is acceptable.
# Use --force to reinstall/upgrade to the pinned version.
if command -v uv &>/dev/null; then
    uv_path=$(command -v uv)
    uv_version=$(uv --version)
    if [[ "$FORCE_INSTALL" == "true" ]]; then
        echo "uv is already installed but --force specified, reinstalling..."
        echo "  Current path: ${uv_path}"
        echo "  Current version: ${uv_version}"
        echo "  Target version: ${UV_VERSION}"
    else
        echo "uv is already installed:"
        echo "  Path: ${uv_path}"
        echo "  Version: ${uv_version}"
        echo "Use --force to reinstall/upgrade to version ${UV_VERSION}."
        exit 0
    fi
fi

# ============================================================================
# Helper Functions
# ============================================================================

# Helper: Add ~/.local/bin to PATH if needed
ensure_local_bin_in_path() {
    local user_bin="${HOME}/.local/bin"
    if [[ -d "$user_bin" ]] && [[ ":$PATH:" != *":$user_bin:"* ]]; then
        # Warn if PATH is very long (some systems have PATH length limits)
        if [[ ${#PATH} -gt 4096 ]]; then
            echo "Warning: PATH is very long (${#PATH} characters)" >&2
            echo "Adding ~/.local/bin may cause issues on some systems" >&2
        fi
        export PATH="$user_bin:$PATH"
    fi
}

# ============================================================================
# SHA256 Verification
# ============================================================================

# Verify SHA256 hash of a file
# Arguments: $1 = file path, $2 = expected hash, $3 = install mode (optional, for security enforcement)
verify_sha256() {
    local file="$1"
    local expected_hash="$2"
    local install_mode="${3:-user}"
    local actual_hash

    if command -v sha256sum &>/dev/null; then
        actual_hash=$(sha256sum "$file" | cut -d' ' -f1)
    elif command -v shasum &>/dev/null; then
        actual_hash=$(shasum -a 256 "$file" | cut -d' ' -f1)
    else
        if [[ "$install_mode" == "system" ]]; then
            echo "Error: SHA256 verification required for system installation but no tool available" >&2
            echo "Please install sha256sum or shasum (usually provided by coreutils or openssl)" >&2
            return 1
        else
            echo "Warning: No SHA256 tool available, skipping hash verification" >&2
            echo "Continuing with unverified installer (not recommended for production use)" >&2
            return 0
        fi
    fi

    if [[ "$actual_hash" != "$expected_hash" ]]; then
        echo "Error: Hash verification failed for installer script" >&2
        echo "  Expected: ${expected_hash}" >&2
        echo "  Actual:   ${actual_hash}" >&2
        echo "This could indicate a compromised download or an outdated hash." >&2
        echo "If UV_VERSION was updated, update UV_INSTALLER_SHA256 as well." >&2
        return 1
    fi

    echo "Hash verification passed"
    return 0
}

# ============================================================================
# Standalone Installer
# https://docs.astral.sh/uv/reference/installer/
# ============================================================================

# Install uv using the official standalone installer
# Supports version pinning via URL: https://astral.sh/uv/{version}/install.sh
# Security: Downloads to temp file and verifies SHA256 hash before execution
install_uv_standalone() {
    local installer_url="https://astral.sh/uv/${UV_VERSION}/install.sh"
    local temp_script

    echo "Installing uv ${UV_VERSION} using standalone installer..."

    # Create temp file for the installer script
    temp_script=$(mktemp "${TMPDIR:-/tmp}/uv-install.XXXXXX.sh")
    # shellcheck disable=SC2064
    # SC2064: We intentionally expand $temp_script at trap-setting time (not signal time)
    # so the cleanup uses the actual temp file path, not whatever $temp_script might be later.
    cleanup_temp() {
        rm -f "$temp_script" 2>/dev/null
    }
    trap cleanup_temp EXIT INT TERM

    # Download installer script to temp file
    if command -v curl &>/dev/null; then
        if ! curl -LsSf "$installer_url" -o "$temp_script"; then
            echo "Error: Failed to download uv installer script" >&2
            return 1
        fi
    elif command -v wget &>/dev/null; then
        if ! wget -q "$installer_url" -O "$temp_script"; then
            echo "Error: Failed to download uv installer script" >&2
            return 1
        fi
    else
        echo "Error: Neither curl nor wget is available." >&2
        echo "Please install curl or wget." >&2
        return 1
    fi

    # Verify the hash before executing (pass install mode for security enforcement)
    if ! verify_sha256 "$temp_script" "$UV_INSTALLER_SHA256" "$INSTALL_MODE"; then
        rm -f "$temp_script"
        return 1
    fi

    # Set environment variables based on install mode
    local env_vars=()
    if [[ "$INSTALL_MODE" == "system" ]]; then
        # System install: put uv in /usr/local/bin, don't modify shell profiles
        env_vars=(
            "UV_INSTALL_DIR=/usr/local/bin"
            "UV_NO_MODIFY_PATH=1"
        )
        echo "System installation:"
        echo "  UV_INSTALL_DIR=/usr/local/bin"
        echo "  UV_PYTHON_BIN_DIR=/usr/local/share/uv (set in environment)"
    fi

    # Execute the verified installer with appropriate environment
    if [[ ${#env_vars[@]} -gt 0 ]]; then
        env "${env_vars[@]}" sh "$temp_script"
    else
        sh "$temp_script"
    fi
    local install_result=$?

    rm -f "$temp_script"
    trap - EXIT

    if [[ $install_result -ne 0 ]]; then
        echo "Error: uv installer script failed" >&2
        return 1
    fi

    # For user installs, ensure PATH includes installation directory
    if [[ "$INSTALL_MODE" == "user" ]]; then
        ensure_local_bin_in_path

        # Also check ~/.cargo/bin for older installer versions
        local cargo_bin="${HOME}/.cargo/bin"
        if [[ -d "$cargo_bin" ]] && [[ ":$PATH:" != *":$cargo_bin:"* ]]; then
            export PATH="$cargo_bin:$PATH"
        fi
    fi
}

# ============================================================================
# Main Installation Logic
# ============================================================================

echo "Installing uv ${UV_VERSION} (${INSTALL_MODE} mode)..."

if ! install_uv_standalone; then
    echo "Error: Installation failed" >&2
    if [[ "$INSTALL_MODE" == "system" ]]; then
        echo "System installation requires elevated permissions." >&2
        echo "Run with sudo, or use --user for user-level installation." >&2
    fi
    exit 1
fi

# Final verification
if command -v uv &>/dev/null; then
    echo "uv installed successfully: $(uv --version)"
    if [[ "$INSTALL_MODE" == "system" ]]; then
        echo ""
        echo "For system installs, set UV_PYTHON_BIN_DIR in your environment:"
        echo "  export UV_PYTHON_BIN_DIR=/usr/local/share/uv"
    fi
else
    echo "Error: uv not found after installation" >&2
    echo "This may indicate:" >&2
    echo "  - Installation completed but uv is not in PATH" >&2
    echo "  - The installation method failed silently" >&2
    if [[ "$INSTALL_MODE" == "user" ]]; then
        echo "Try running: export PATH=\"\$HOME/.local/bin:\$PATH\" and verify uv is available." >&2
    else
        echo "Verify /usr/local/bin is in your PATH." >&2
    fi
    exit 1
fi
