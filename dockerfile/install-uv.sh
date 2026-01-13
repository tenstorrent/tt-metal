#!/bin/bash
# Install uv package manager with version pinning
# This script centralizes uv installation logic to avoid drift between Dockerfiles
#
# Usage: install-uv.sh
#
# Supports:
#   - Ubuntu 22.04/24.04 (pip-based installation with PEP 668 handling)
#   - Fedora (native dnf package available)
#
# The script auto-detects the distribution and uses the appropriate installation method.

set -euo pipefail

# uv version to install (when using pip)
# Update this version when upgrading uv across all Dockerfiles
UV_VERSION="0.7.12"

# Detect distribution
if [[ -f /etc/os-release ]]; then
    # shellcheck source=/dev/null
    source /etc/os-release
    DISTRO_ID="${ID:-unknown}"
    DISTRO_VERSION="${VERSION_ID:-unknown}"
else
    echo "Error: /etc/os-release not found" >&2
    exit 1
fi

echo "Installing uv ${UV_VERSION} on ${DISTRO_ID} ${DISTRO_VERSION}..."

case "${DISTRO_ID}" in
    fedora|rhel|centos|rocky|almalinux)
        # Fedora and RHEL-family have uv in their repos
        if command -v dnf &>/dev/null; then
            dnf install -y uv
        elif command -v yum &>/dev/null; then
            yum install -y uv
        else
            echo "Error: No package manager found" >&2
            exit 1
        fi
        ;;
    ubuntu|debian)
        # Ubuntu/Debian: Install via pip with version pinning
        # Handle PEP 668 (externally-managed-environment) for Ubuntu 24.04+
        PIP_ARGS="--no-cache-dir"

        # Check if we need --break-system-packages (Ubuntu 24.04+ / Debian 12+)
        if [[ "${DISTRO_ID}" == "ubuntu" && "${DISTRO_VERSION}" == "24.04" ]] || \
           [[ "${DISTRO_ID}" == "debian" && "${DISTRO_VERSION%%.*}" -ge 12 ]]; then
            PIP_ARGS="${PIP_ARGS} --break-system-packages"
        fi

        # Install specific version
        # shellcheck disable=SC2086
        python3 -m pip install ${PIP_ARGS} "uv==${UV_VERSION}"
        ;;
    *)
        echo "Warning: Unknown distribution '${DISTRO_ID}', attempting pip installation..." >&2
        python3 -m pip install --no-cache-dir "uv==${UV_VERSION}" || {
            echo "Error: Failed to install uv" >&2
            exit 1
        }
        ;;
esac

# Verify installation
if command -v uv &>/dev/null; then
    echo "uv installed successfully: $(uv --version)"
else
    echo "Error: uv not found after installation" >&2
    exit 1
fi
