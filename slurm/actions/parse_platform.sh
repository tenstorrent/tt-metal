#!/usr/bin/env bash
# parse_platform.sh - Parse a platform string into its components.
#
# Exports DISTRO, DISTRO_VERSION, TOOLCHAIN, and ENABLE_LTO as environment
# variables and prints them as KEY=VALUE lines.
#
# Ported from: .github/actions/parse-platform/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "$SCRIPT_DIR/../lib/common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PLATFORM=""
DISTRO_IN=""
VERSION_IN=""
DEFAULT_DISTRO="ubuntu"
DEFAULT_VERSION="22.04"
ENABLE_LTO_OVERRIDE="auto"

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") --platform STRING [OPTIONS]

Parse a platform string (e.g. "Ubuntu 22.04") into its constituent parts and
export them as environment variables.

Options:
  --platform STRING       Platform string (e.g. "Ubuntu 22.04", "Ubuntu 24.04")
  --distro DISTRO         Direct distro input (used when --platform is not set)
  --version VERSION       Direct version input (used when --platform is not set)
  --default-distro NAME   Fallback distro [default: $DEFAULT_DISTRO]
  --default-version VER   Fallback version [default: $DEFAULT_VERSION]
  --enable-lto BOOL       Override LTO (true/false/auto) [default: $ENABLE_LTO_OVERRIDE]
  -h, --help              Show this help

Exported variables:
  DISTRO, DISTRO_VERSION, TOOLCHAIN, ENABLE_LTO
EOF
    exit 1
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)        PLATFORM="$2";          shift 2 ;;
        --distro)          DISTRO_IN="$2";         shift 2 ;;
        --version)         VERSION_IN="$2";        shift 2 ;;
        --default-distro)  DEFAULT_DISTRO="$2";    shift 2 ;;
        --default-version) DEFAULT_VERSION="$2";   shift 2 ;;
        --enable-lto)      ENABLE_LTO_OVERRIDE="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *) log_fatal "Unknown argument: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve LTO
# ---------------------------------------------------------------------------

resolve_lto() {
    local platform_default="$1"
    if [[ "$ENABLE_LTO_OVERRIDE" == "true" || "$ENABLE_LTO_OVERRIDE" == "false" ]]; then
        echo "$ENABLE_LTO_OVERRIDE"
    else
        echo "$platform_default"
    fi
}

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

if [[ -n "$PLATFORM" ]]; then
    case "$PLATFORM" in
        "Ubuntu 22.04")
            DISTRO="ubuntu"
            DISTRO_VERSION="22.04"
            TOOLCHAIN="cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
            ENABLE_LTO=$(resolve_lto "false")
            ;;
        "Ubuntu 24.04")
            DISTRO="ubuntu"
            DISTRO_VERSION="24.04"
            TOOLCHAIN="cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
            ENABLE_LTO=$(resolve_lto "false")
            ;;
        *)
            log_warn "Unknown platform '$PLATFORM' — using defaults"
            DISTRO="$DEFAULT_DISTRO"
            DISTRO_VERSION="$DEFAULT_VERSION"
            TOOLCHAIN="cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake"
            ENABLE_LTO=$(resolve_lto "false")
            ;;
    esac
else
    DISTRO="${DISTRO_IN:-$DEFAULT_DISTRO}"
    DISTRO_VERSION="${VERSION_IN:-$DEFAULT_VERSION}"

    if [[ "$DISTRO" == "ubuntu" ]]; then
        TOOLCHAIN="cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
    else
        TOOLCHAIN="cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake"
    fi

    ENABLE_LTO=$(resolve_lto "false")
fi

# ---------------------------------------------------------------------------
# Export and print
# ---------------------------------------------------------------------------

export DISTRO DISTRO_VERSION TOOLCHAIN ENABLE_LTO

echo "DISTRO=${DISTRO}"
echo "DISTRO_VERSION=${DISTRO_VERSION}"
echo "TOOLCHAIN=${TOOLCHAIN}"
echo "ENABLE_LTO=${ENABLE_LTO}"
