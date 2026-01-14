#!/bin/bash
# Install uv package manager with version pinning
# This script centralizes uv installation logic to avoid drift between Dockerfiles
#
# Usage: install-uv.sh
#
# Supports:
#   - Ubuntu 22.04/24.04 (pip-based installation with PEP 668 handling)
#   - Fedora (native dnf package available)
#   - macOS (pip-based installation with PEP 668 handling)
#
# The script auto-detects the distribution and uses the appropriate installation method.
#
# Requirements:
#   - Python 3 with pip installed
#   - For macOS: pip 23.0+ recommended for --break-system-packages support
#   - For Linux: dnf/yum (Fedora/RHEL) or pip (Ubuntu/Debian)
#
# Behavior:
#   - On failure, the script will exit with a non-zero status code
#   - For PEP 668 restrictions, the script will automatically retry with appropriate flags
#   - The script updates PATH to include ~/.local/bin if uv is installed there
#   - Final verification ensures uv is available before exiting

set -euo pipefail

# uv version to install (when using pip)
# Update this version when upgrading uv across all Dockerfiles
UV_VERSION="0.7.12"

# Helper: Add ~/.local/bin to PATH if needed
ensure_local_bin_in_path() {
    local user_bin="${HOME}/.local/bin"
    if [[ -d "$user_bin" ]] && [[ ":$PATH:" != *":$user_bin:"* ]]; then
        export PATH="$user_bin:$PATH"
    fi
}

# Helper: Check if pip supports --break-system-packages
pip_supports_break_system_packages() {
    python3 -m pip install --help 2>&1 | grep -q "\-\-break-system-packages"
}

# Helper: Install uv via pip with PEP 668 fallback handling
# Tries: normal install -> --break-system-packages -> --user
install_uv_with_pip_fallback() {
    local pip_args="--no-cache-dir"

    # First attempt: standard pip install
    # shellcheck disable=SC2086
    local pip_output
    pip_output=$(python3 -m pip install ${pip_args} "uv==${UV_VERSION}" 2>&1) || true
    ensure_local_bin_in_path

    # Success on first try
    if command -v uv &>/dev/null; then
        return 0
    fi

    # Check for PEP 668 restriction
    if echo "$pip_output" | grep -q "externally-managed-environment"; then
        echo "PEP 668 restriction detected, trying alternative installation methods..."

        # Second attempt: --break-system-packages (if available)
        if pip_supports_break_system_packages; then
            echo "Trying --break-system-packages..."
            # shellcheck disable=SC2086
            if python3 -m pip install ${pip_args} --break-system-packages "uv==${UV_VERSION}"; then
                ensure_local_bin_in_path
                return 0
            fi
            echo "Warning: --break-system-packages failed, falling back to --user..." >&2
        fi

        # Third attempt: --user installation
        echo "Trying --user installation..."
        python3 -m pip install --user --no-cache-dir "uv==${UV_VERSION}"
        ensure_local_bin_in_path
        return 0
    fi

    # Non-PEP-668 error - check for actual errors (not just warnings)
    if echo "$pip_output" | grep -qiE "error:|failed|exception"; then
        echo "$pip_output" >&2
        return 1
    fi

    # Unknown state - let final verification handle it
    return 0
}

# Helper: Install uv on Fedora/RHEL family
install_uv_fedora() {
    if command -v dnf &>/dev/null; then
        dnf install -y uv
    elif command -v yum &>/dev/null; then
        yum install -y uv
    else
        echo "Error: No package manager found (dnf/yum)" >&2
        return 1
    fi
}

# Helper: Check if distro requires --break-system-packages (PEP 668)
# PEP 668 is enforced on Ubuntu 23.04+, Debian 12+
requires_break_system_packages() {
    local distro_id="$1"
    local distro_version="$2"

    case "${distro_id}" in
        ubuntu)
            # Ubuntu versions are YY.MM format (e.g., 22.04, 24.04)
            # PEP 668 enforced starting with 23.04
            local major_version="${distro_version%%.*}"
            [[ "${major_version}" -ge 23 ]]
            ;;
        debian)
            # Debian uses major version numbers (e.g., 11, 12)
            # PEP 668 enforced starting with Debian 12
            local major_version="${distro_version%%.*}"
            [[ "${major_version}" -ge 12 ]]
            ;;
        *)
            return 1
            ;;
    esac
}

# Helper: Install uv on Ubuntu/Debian via pip
install_uv_debian() {
    local distro_id="$1"
    local distro_version="$2"
    local pip_args="--no-cache-dir"

    if requires_break_system_packages "${distro_id}" "${distro_version}"; then
        pip_args="${pip_args} --break-system-packages"
    fi

    # shellcheck disable=SC2086
    python3 -m pip install ${pip_args} "uv==${UV_VERSION}"
}

# ============================================================================
# Main Installation Logic
# ============================================================================

# Detect OS and install
case "$(uname -s)" in
    Darwin)
        echo "Installing uv ${UV_VERSION} on macOS..."
        install_uv_with_pip_fallback
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

        echo "Installing uv ${UV_VERSION} on ${DISTRO_ID} ${DISTRO_VERSION}..."

        case "${DISTRO_ID}" in
            fedora|rhel|centos|rocky|almalinux)
                install_uv_fedora
                ;;
            ubuntu|debian)
                install_uv_debian "${DISTRO_ID}" "${DISTRO_VERSION}"
                ;;
            *)
                echo "Warning: Unknown distribution '${DISTRO_ID}', attempting pip installation..." >&2
                python3 -m pip install --no-cache-dir "uv==${UV_VERSION}" || {
                    echo "Error: Failed to install uv" >&2
                    exit 1
                }
                ;;
        esac
        ;;
    *)
        echo "Error: Unsupported operating system: $(uname -s)" >&2
        exit 1
        ;;
esac

# Final verification
if command -v uv &>/dev/null; then
    echo "uv installed successfully: $(uv --version)"
else
    echo "Error: uv not found after installation" >&2
    exit 1
fi
