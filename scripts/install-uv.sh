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

# Installation mode: "system" or "user"
# - system: Install at system level, fail if not possible
# - user: Install to ~/.local/bin with pip --user or standalone installer (default)
INSTALL_MODE="user"

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
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'" >&2
            echo "Run '$0 --help' for usage information." >&2
            exit 1
            ;;
    esac
done

# ============================================================================
# Early Exit if Already Installed
# ============================================================================

if command -v uv &>/dev/null; then
    uv_path=$(command -v uv)
    uv_version=$(uv --version)
    echo "uv is already installed:"
    echo "  Path: ${uv_path}"
    echo "  Version: ${uv_version}"
    exit 0
fi

# ============================================================================
# Helper Functions
# ============================================================================

# Helper: Add ~/.local/bin to PATH if needed
ensure_local_bin_in_path() {
    local user_bin="${HOME}/.local/bin"
    if [[ -d "$user_bin" ]] && [[ ":$PATH:" != *":$user_bin:"* ]]; then
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

# Install uv using the official standalone installer
# Supports version pinning via URL: https://astral.sh/uv/{version}/install.sh
install_uv_standalone() {
    local installer_url="https://astral.sh/uv/${UV_VERSION}/install.sh"

    echo "Installing uv ${UV_VERSION} using standalone installer..."

    if command -v curl &>/dev/null; then
        curl -LsSf "$installer_url" | sh
    elif command -v wget &>/dev/null; then
        wget -qO- "$installer_url" | sh
    else
        echo "Error: Neither curl nor wget is available." >&2
        echo "Please install curl or wget, or install pip to use pip-based installation." >&2
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
        python3 -m pip install --user --no-cache-dir "uv==${UV_VERSION}"
        ensure_local_bin_in_path
    else
        echo "pip not available, using standalone installer..."
        install_uv_standalone
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
    if command -v dnf &>/dev/null; then
        dnf install -y uv
    elif command -v yum &>/dev/null; then
        yum install -y uv
    else
        echo "Error: No package manager found (dnf/yum)" >&2
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
    pip_output=$(python3 -m pip install ${pip_args} "uv==${UV_VERSION}" 2>&1) || true

    if command -v uv &>/dev/null; then
        return 0
    fi

    # Check for PEP 668 restriction
    if echo "$pip_output" | grep -q "externally-managed-environment"; then
        echo "PEP 668 restriction detected, trying --break-system-packages..."
        if pip_supports_break_system_packages; then
            # shellcheck disable=SC2086
            if python3 -m pip install ${pip_args} --break-system-packages "uv==${UV_VERSION}"; then
                return 0
            fi
        fi
        fail_system_install
    fi

    # Other error
    if echo "$pip_output" | grep -qiE "error:|failed|exception"; then
        echo "$pip_output" >&2
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
    exit 1
fi
