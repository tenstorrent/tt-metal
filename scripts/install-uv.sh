#!/bin/bash
# Install uv package manager with version pinning.
# Run with --help for usage information.

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# uv version to install
# Update this version when upgrading uv across all Dockerfiles
UV_VERSION="0.7.12"

# SHA256 hash of the install script for UV_VERSION (for security verification)
# To update: curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sha256sum
# IMPORTANT: Update this hash whenever UV_VERSION changes!
UV_INSTALLER_SHA256="8a348686376016950a5f90a26c8dd7ee35355197b35cf085bdaf96bf8d94bd47"

# Installation mode: "system" or "user"
# - system: Install at system level, fail if not possible
# - user: Install to ~/.local/bin with pip --user or standalone installer (default)
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
Auto-detects the OS/distribution and uses the appropriate installation method.

OPTIONS:
    --user      Install to user directory (~/.local/bin) (default).
                Uses pip --user or standalone installer.
                Updates PATH to include ~/.local/bin.
                Does not require elevated permissions.
    --system    Install at system level (for Docker containers).
                Uses system package manager or pip without --user.
                Requires appropriate permissions (use sudo if needed).
                Does NOT modify PATH.
    --force     Reinstall uv even if already installed.
                Useful for upgrading or ensuring a specific version.
    --help, -h  Show this help message and exit.

SUPPORTED PLATFORMS:
    - Ubuntu 22.04+ (pip or standalone installer)
    - Debian 11+ (pip or standalone installer)
    - Fedora/RHEL/CentOS/Rocky/AlmaLinux (native dnf/yum package)
    - macOS (pip or standalone installer)

REQUIREMENTS:
    One of the following:
    - Python 3 with pip, OR
    - curl or wget (for standalone installer fallback)
    - For Fedora/RHEL: dnf or yum package manager

BEHAVIOR:
    --user (default):
        - Installs directly to ~/.local/bin
        - Uses pip --user if available, otherwise standalone installer
        - Adds ~/.local/bin to PATH if not already present
        - Does not require elevated permissions

    --system:
        - Fedora/RHEL: Uses dnf/yum to install system package
        - Ubuntu/Debian/macOS: Uses pip with --break-system-packages if needed
        - Falls back to standalone installer if pip unavailable
        - Fails with error if installation requires elevated permissions

STANDALONE INSTALLER:
    If pip is not available, the script will attempt to use the official
    uv standalone installer (https://astral.sh/uv/install.sh) via curl or wget.
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

# Helper: Check if pip is available
has_pip() {
    python3 -m pip --version &>/dev/null
}

# Helper: Check if pip supports --break-system-packages
pip_supports_break_system_packages() {
    python3 -m pip install --help 2>&1 | grep -q "\-\-break-system-packages"
}

# Helper: Check if distro requires --break-system-packages (PEP 668)
# PEP 668 is enforced on Ubuntu 23.04+, Debian 12+
requires_break_system_packages() {
    local distro_id="$1"
    local distro_version="$2"

    case "${distro_id}" in
        ubuntu)
            local major_version="${distro_version%%.*}"
            [[ "${major_version}" -ge 23 ]]
            ;;
        debian)
            local major_version="${distro_version%%.*}"
            [[ "${major_version}" -ge 12 ]]
            ;;
        *)
            return 1
            ;;
    esac
}

# ============================================================================
# Standalone Installer (fallback when pip is unavailable)
# https://docs.astral.sh/uv/getting-started/installation/
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
        echo "Please install curl or wget, or install pip to use pip-based installation." >&2
        return 1
    fi

    # Verify the hash before executing (pass install mode for security enforcement)
    if ! verify_sha256 "$temp_script" "$UV_INSTALLER_SHA256" "$INSTALL_MODE"; then
        rm -f "$temp_script"
        return 1
    fi

    # Execute the verified installer
    sh "$temp_script"
    local install_result=$?

    rm -f "$temp_script"
    trap - EXIT

    if [[ $install_result -ne 0 ]]; then
        echo "Error: uv installer script failed" >&2
        return 1
    fi

    # Standalone installer puts uv in ~/.local/bin (or ~/.cargo/bin for older versions)
    ensure_local_bin_in_path

    # Also check ~/.cargo/bin for older installer versions
    local cargo_bin="${HOME}/.cargo/bin"
    if [[ -d "$cargo_bin" ]] && [[ ":$PATH:" != *":$cargo_bin:"* ]]; then
        export PATH="$cargo_bin:$PATH"
    fi
}

# ============================================================================
# User-mode Installation (--user)
# ============================================================================

# Install uv to ~/.local/bin using pip --user or standalone installer
install_uv_user() {
    echo "Installing uv to user directory (~/.local/bin)..."

    if has_pip; then
        local pip_output
        local pip_exit_code
        pip_output=$(python3 -m pip install --user --no-cache-dir "uv==${UV_VERSION}" 2>&1) || pip_exit_code=$?
        pip_exit_code=${pip_exit_code:-0}

        if [[ $pip_exit_code -ne 0 ]]; then
            echo "Error: pip install --user failed (exit code: $pip_exit_code)" >&2
            echo "pip output:" >&2
            echo "$pip_output" >&2
            echo "Trying standalone installer as fallback..." >&2
            if ! install_uv_standalone; then
                echo "Error: Both pip and standalone installation failed" >&2
                return 1
            fi
        fi
        ensure_local_bin_in_path
    else
        echo "pip not available, using standalone installer..."
        if ! install_uv_standalone; then
            echo "Error: Standalone installation failed" >&2
            return 1
        fi
    fi
}

# ============================================================================
# System-mode Installation (--system)
# ============================================================================

# Error out when system install fails
fail_system_install() {
    echo "Error: System-level installation failed." >&2
    echo "Run with sudo, or use --user for user-level installation." >&2
    exit 1
}

# Install uv on Fedora/RHEL family using system package manager
install_uv_fedora_system() {
    local temp_output
    temp_output=$(mktemp "${TMPDIR:-/tmp}/uv-dnf-output.XXXXXX.log" 2>/dev/null || echo "/tmp/uv-dnf-output.log")

    if command -v dnf &>/dev/null; then
        if ! dnf install -y uv 2>&1 | tee "$temp_output"; then
            # Check for specific error conditions
            if grep -qiE "no package.*uv|package.*uv.*not found|nothing provides.*uv" "$temp_output" 2>/dev/null; then
                echo "Error: uv package not available in Fedora repositories" >&2
                echo "The uv package may not be in the default repositories for this Fedora version." >&2
                echo "Falling back to pip-based installation..." >&2
                rm -f "$temp_output"
                # Fall back to pip installation
                install_uv_pip_system "fedora" ""
                return $?
            else
                echo "Error: dnf install failed" >&2
                echo "Check the output above for details." >&2
                rm -f "$temp_output"
                return 1
            fi
        fi
        rm -f "$temp_output"
    elif command -v yum &>/dev/null; then
        if ! yum install -y uv 2>&1 | tee "$temp_output"; then
            # Check for specific error conditions
            if grep -qiE "no package.*uv|package.*uv.*not found|nothing provides.*uv" "$temp_output" 2>/dev/null; then
                echo "Error: uv package not available in yum repositories" >&2
                echo "The uv package may not be in the default repositories for this RHEL/CentOS version." >&2
                echo "Falling back to pip-based installation..." >&2
                rm -f "$temp_output"
                # Fall back to pip installation
                install_uv_pip_system "rhel" ""
                return $?
            else
                echo "Error: yum install failed" >&2
                echo "Check the output above for details." >&2
                rm -f "$temp_output"
                return 1
            fi
        fi
        rm -f "$temp_output"
    else
        echo "Error: No package manager found (dnf/yum)" >&2
        return 1
    fi

    # Verify installation succeeded
    if ! command -v uv &>/dev/null; then
        echo "Error: uv not found after package installation" >&2
        echo "This may indicate:" >&2
        echo "  - The package was installed to a location not in PATH" >&2
        echo "  - The installation completed but the binary is missing" >&2
        echo "  - PATH needs to be updated to include the installation directory" >&2
        return 1
    fi
}

# Install uv via pip at system level
install_uv_pip_system() {
    local distro_id="${1:-}"
    local distro_version="${2:-}"
    local pip_args="--no-cache-dir"

    if [[ -n "$distro_id" ]] && requires_break_system_packages "${distro_id}" "${distro_version}"; then
        pip_args="${pip_args} --break-system-packages"
    fi

    # shellcheck disable=SC2086
    if ! python3 -m pip install ${pip_args} "uv==${UV_VERSION}"; then
        fail_system_install
    fi
}

# Install uv on macOS at system level with PEP 668 handling
install_uv_macos_system() {
    if ! has_pip; then
        echo "pip not available, using standalone installer..."
        if ! install_uv_standalone; then
            fail_system_install
        fi
        return 0
    fi

    local pip_args="--no-cache-dir"

    # First attempt: standard pip install
    # shellcheck disable=SC2086
    local pip_output
    local pip_exit_code
    pip_output=$(python3 -m pip install ${pip_args} "uv==${UV_VERSION}" 2>&1) || pip_exit_code=$?
    pip_exit_code=${pip_exit_code:-0}

    # Check if pip succeeded (exit code 0) AND uv is now available
    if [[ $pip_exit_code -eq 0 ]] && command -v uv &>/dev/null; then
        return 0
    fi

    # Check for PEP 668 restriction
    if echo "$pip_output" | grep -q "externally-managed-environment"; then
        echo "PEP 668 restriction detected, trying --break-system-packages..."
        if pip_supports_break_system_packages; then
            # shellcheck disable=SC2086
            if python3 -m pip install ${pip_args} --break-system-packages "uv==${UV_VERSION}"; then
                if command -v uv &>/dev/null; then
                    return 0
                fi
                echo "Error: pip install succeeded but uv is not available in PATH" >&2
                echo "This may indicate:" >&2
                echo "  - uv was installed to a location not in PATH" >&2
                echo "  - PATH was not updated after installation" >&2
                echo "  - Installation completed but binary is missing" >&2
            fi
        fi
        # Fallback to standalone installer if --break-system-packages doesn't work
        echo "pip installation blocked by PEP 668, trying standalone installer..." >&2
        if install_uv_standalone; then
            return 0
        fi
        fail_system_install
    fi

    # pip failed with non-PEP-668 error
    if [[ $pip_exit_code -ne 0 ]]; then
        echo "Error: pip install failed (exit code: $pip_exit_code)" >&2
        echo "pip output:" >&2
        echo "$pip_output" >&2
        fail_system_install
    fi

    # pip exited 0 but uv not available - unusual case
    if ! command -v uv &>/dev/null; then
        echo "Error: pip install appeared to succeed but uv is not available" >&2
        echo "This may indicate:" >&2
        echo "  - uv was installed to a location not in PATH" >&2
        echo "  - PATH was not updated after installation" >&2
        echo "  - Installation completed but binary is missing" >&2
        echo "pip output: $pip_output" >&2
        fail_system_install
    fi
}

# Install uv on Ubuntu/Debian at system level
install_uv_debian_system() {
    local distro_id="$1"
    local distro_version="$2"

    if ! has_pip; then
        echo "pip not available, using standalone installer..."
        if ! install_uv_standalone; then
            fail_system_install
        fi
        return 0
    fi

    install_uv_pip_system "${distro_id}" "${distro_version}"
}

# Install uv on unknown distro at system level
install_uv_unknown_system() {
    echo "Warning: Unknown distribution, attempting installation..." >&2

    if has_pip; then
        if ! python3 -m pip install --no-cache-dir "uv==${UV_VERSION}"; then
            fail_system_install
        fi
    else
        echo "pip not available, using standalone installer..."
        if ! install_uv_standalone; then
            fail_system_install
        fi
    fi
}

# ============================================================================
# Main Installation Logic
# ============================================================================

# Handle user-mode installation separately (simple, direct path)
if [[ "$INSTALL_MODE" == "user" ]]; then
    echo "Installing uv ${UV_VERSION} (user mode)..."
    install_uv_user
else
    # System-mode installation
    case "$(uname -s)" in
        Darwin)
            echo "Installing uv ${UV_VERSION} on macOS (system mode)..."
            install_uv_macos_system
            ;;
        Linux)
            if [[ ! -f /etc/os-release ]]; then
                echo "Error: /etc/os-release not found" >&2
                exit 1
            fi

            # shellcheck source=/dev/null
            source /etc/os-release
            DISTRO_ID="${ID:-unknown}"
            DISTRO_VERSION="${VERSION_ID:-unknown}"

            echo "Installing uv ${UV_VERSION} on ${DISTRO_ID} ${DISTRO_VERSION} (system mode)..."

            case "${DISTRO_ID}" in
                fedora|rhel|centos|rocky|almalinux)
                    install_uv_fedora_system
                    ;;
                ubuntu|debian)
                    install_uv_debian_system "${DISTRO_ID}" "${DISTRO_VERSION}"
                    ;;
                *)
                    install_uv_unknown_system
                    ;;
            esac
            ;;
        *)
            echo "Error: Unsupported operating system: $(uname -s)" >&2
            exit 1
            ;;
    esac
fi

# Final verification
if command -v uv &>/dev/null; then
    echo "uv installed successfully: $(uv --version)"
else
    echo "Error: uv not found after installation" >&2
    echo "This may indicate:" >&2
    echo "  - Installation completed but uv is not in PATH" >&2
    echo "  - The installation method failed silently" >&2
    echo "  - PATH needs to be updated manually" >&2
    echo "Try running: export PATH=\"\$HOME/.local/bin:\$PATH\" and verify uv is available." >&2
    exit 1
fi
